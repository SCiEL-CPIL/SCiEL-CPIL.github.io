import numpy as np
import time
from scipy.optimize import minimize

N = 400
_FSIZE = 1
while _FSIZE < 2 * N - 1:
    _FSIZE *= 2

# Precompute basis vectors for efficient FFT updates
# Shape: (FSIZE//2 + 1, N)
_freqs = np.arange(_FSIZE // 2 + 1)
_basis_vectors = np.exp(-2j * np.pi * np.outer(_freqs, np.arange(N)) / _FSIZE)

def compute_raw_value(x):
    """
    Computes the raw value: max(|autoconvolution|) / (sum(x))^2.
    """
    X = np.fft.rfft(x, _FSIZE)
    conv = np.fft.irfft(X * X, _FSIZE)
    conv = conv[:2*N-1]
    max_abs_conv = np.max(np.abs(conv))
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return float('inf')
    return max_abs_conv / (integral * integral)

def _soft_max_abs_and_grad(h, beta=50.0):
    """
    Computes the soft-max absolute autoconvolution value and its analytical gradient.
    Returns: value, gradient
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10, np.zeros_like(h)
    
    H = np.fft.rfft(h, _FSIZE)
    ac = np.fft.irfft(H * H, _FSIZE)[:2*N-1]
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    
    if m < 1e-15:
        S = 0.0
        grad_S = np.zeros_like(h)
    else:
        scaled = beta * (abs_ac - m)
        scaled = np.clip(scaled, -50, 50)
        exp_scaled = np.exp(scaled)
        sum_exp = np.sum(exp_scaled)
        log_sum = np.log(sum_exp)
        S = m + log_sum / beta
        
        # Weights for the soft-max
        w = exp_scaled / sum_exp
        
        # v_k = w_k * sign(ac_k)
        v = w * np.sign(ac)
        
        # Gradient of S w.r.t h
        # S = soft_max(|ac|)
        # dS/dh_i = 2 * sum_k v_k h_{k-i} 
        # This is a cross-correlation between v and h.
        
        fsize_corr = 1
        while fsize_corr < 3 * N:
            fsize_corr *= 2
            
        V = np.fft.rfft(v, fsize_corr)
        Hc = np.fft.rfft(h, fsize_corr)
        # Cross-correlation: V * conj(Hc)
        C = np.fft.irfft(V * np.conj(Hc), fsize_corr)
        
        # C[i] = sum_n v[n+i] conj(h[n])
        # For real h, C[i] = sum_n v[n+i] h[n]
        # We need the first N elements of this correlation to form the gradient vector.
        grad_S = 2 * C[:N]
    
    # Objective f = S / s^2
    # df/dh_i = (dS/dh_i * s^2 - S * 2 * s) / s^4
    # = (dS/dh_i * s - 2 * S) / s^3
    
    grad = (grad_S * s - 2 * S) / (s ** 3)
    
    return S / (s * s), grad

def _simulated_annealing(h, deadline=None, initial_temp=1.0, cooling_rate=0.995, min_temp=0.01):
    """
    Simulated annealing to escape local minima.
    Accepts worse solutions with probability exp(-delta/T) to explore the solution space.
    """
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    best_h = h.copy()
    H = np.fft.rfft(h, _FSIZE)
    integral = np.sum(h)
    
    current_val = best_val
    temp = initial_temp
    
    rng = np.random.default_rng(42)
    
    while temp > min_temp and (deadline is None or time.time() < deadline):
        # Pick a random index to flip
        i = rng.integers(0, N)
        old_val = h[i]
        h[i] = -old_val
        
        delta = -2 * old_val * _basis_vectors[:, i]
        H_new = H + delta
        conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
        new_integral = integral - 2 * old_val
        
        if abs(new_integral) < 1e-12:
            new_val = float('inf')
        else:
            new_val = np.max(np.abs(conv)) / (new_integral * new_integral)
        
        delta_val = new_val - current_val
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta_val < 0 or rng.random() < np.exp(-delta_val / temp):
            current_val = new_val
            H = H_new
            integral = new_integral
            
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            # Revert
            h[i] = old_val
        
        temp *= cooling_rate
    
    return best_h, best_val

def _spectral_peak_targeting(h, deadline=None):
    """
    Targets the spectral peaks of the autoconvolution.
    Identifies the top K lags with highest absolute autoconvolution values.
    For each peak lag, tries to find sign patterns that reduce correlation at that lag.
    Uses a greedy approach: for each peak, try flipping individual bits that most reduce that peak.
    """
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    H = np.fft.rfft(h, _FSIZE)
    integral = np.sum(h)
    
    K = 5  # Number of top peaks to target
    
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        
        # Get current autoconvolution and find top K peaks
        conv = np.fft.irfft(H * H, _FSIZE)[:2*N-1]
        abs_conv = np.abs(conv)
        peak_indices = np.argsort(abs_conv)[-K:][::-1]  # Top K peaks, largest first
        
        for peak_lag in peak_indices:
            if deadline is not None and time.time() > deadline:
                break
            
            # For this peak lag, we want to reduce |conv[peak_lag]|
            # conv[peak_lag] = sum_i h[i] * h[i + peak_lag] (with wrap-around for circular, but we use linear)
            # Actually, for linear autoconvolution: conv[l] = sum_{i=0}^{N-1-l} h[i] * h[i+l]
            
            # Try flipping each bit and see if it reduces the peak at this lag
            best_flip = -1
            best_new_peak_val = abs_conv[peak_lag]
            best_new_val = best_val
            
            for i in range(N):
                if deadline is not None and time.time() > deadline:
                    break
                
                old_val = h[i]
                h[i] = -old_val
                
                delta = -2 * old_val * _basis_vectors[:, i]
                H_new = H + delta
                conv_new = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
                new_integral = integral - 2 * old_val
                
                if abs(new_integral) < 1e-12:
                    new_val = float('inf')
                else:
                    new_val = np.max(np.abs(conv_new)) / (new_integral * new_integral)
                
                # Accept if it reduces the overall value, or if it significantly reduces this specific peak
                if new_val < best_new_val:
                    best_new_val = new_val
                    best_new_peak_val = np.abs(conv_new[peak_lag])
                    best_flip = i
                elif abs(conv_new[peak_lag]) < best_new_peak_val * 0.9 and new_val < best_val * 1.1:
                    # Accept if it reduces this peak significantly and doesn't worsen overall too much
                    best_new_val = new_val
                    best_new_peak_val = np.abs(conv_new[peak_lag])
                    best_flip = i
                
                h[i] = old_val  # Revert for next trial
            
            # Apply the best flip if found
            if best_flip >= 0 and best_new_val < best_val:
                h[best_flip] = -h[best_flip]
                delta = -2 * h[best_flip] * _basis_vectors[:, best_flip]
                H = H + delta
                integral = integral - 2 * h[best_flip]
                best_val = best_new_val
                improved = True
    
    return h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    # 1. Baseline
    best_x = np.ones(N, dtype=float)
    best_val = compute_raw_value(best_x)
    
    # 2. Generate Diverse Candidates
    candidates = [best_x.copy()]
    t = np.arange(N)
    
    # Chirps
    for freq in [0.5, 1.0, 1.5, 2.0, 3.0]:
        phase = 2 * np.pi * freq * (t**2) / N
        x_chirp = np.sign(np.cos(phase))
        x_chirp[x_chirp == 0] = 1
        candidates.append(x_chirp)
        
    # Linear
    for freq in [1.0, 2.0, 4.0]:
        phase = 2 * np.pi * freq * t / N
        x_lin = np.sign(np.cos(phase))
        x_lin[x_lin == 0] = 1
        candidates.append(x_lin)
        
    # Random
    rng = np.random.default_rng(123)
    for _ in range(20):
        x_rand = rng.choice([1.0, -1.0], N)
        candidates.append(x_rand)
        
    # 3. Evaluate Candidates
    for x in candidates:
        val = compute_raw_value(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
            
    # 4. Smooth Surrogate Optimization (Global Phase)
    # Use the best candidate as start
    h_start = best_x.copy()
    s0 = np.sum(h_start)
    if abs(s0) > 1e-6:
        h_norm = h_start / s0
        
        # Multi-stage beta
        for beta in [40, 100, 200, 400]:
            if time.time() > deadline - 40:
                break
            
            try:
                res = minimize(_soft_max_abs_and_grad, h_norm, method='L-BFGS-B', jac=True,
                               options={'maxiter': 200, 'disp': False, 'gtol': 1e-7, 'beta': beta})
                h_opt = res.x
                
                # Binarize and check
                h_bin = np.sign(h_opt)
                h_bin[h_bin == 0] = 1
                val_bin = compute_raw_value(h_bin)
                
                if val_bin < best_val:
                    best_val = val_bin
                    best_x = h_bin.copy()
                    # Update start for next beta
                    h_norm = h_bin / np.sum(h_bin) if np.sum(h_bin) != 0 else h_norm
                    
            except Exception:
                pass

    # 5. Simulated Annealing Refinement
    # Step 5a: Spectral Peak Targeting
    if time.time() < deadline - 30:
        h_work, val_work = _spectral_peak_targeting(best_x, deadline=deadline - 25)
        if val_work < best_val:
            best_val = val_work
            best_x = h_work.copy()

    # Step 5b: Simulated Annealing
    if time.time() < deadline - 10:
        h_work, val_work = _simulated_annealing(best_x, deadline=deadline - 5, 
                                                initial_temp=0.5, cooling_rate=0.998, min_temp=0.005)
        if val_work < best_val:
            best_val = val_work
            best_x = h_work.copy()

    # 6. Final Polish with spectral targeting on random restarts if time permits
    if time.time() < deadline - 20:
        for _ in range(2):
            if time.time() > deadline - 10:
                break
            x_rand = rng.choice([1.0, -1.0], N)
            # Spectral peak targeting on random start
            h_work, val_work = _spectral_peak_targeting(x_rand, deadline=deadline - 8)
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()
            
            # Quick simulated annealing
            if time.time() < deadline - 5:
                h_work2, val_work2 = _simulated_annealing(h_work, deadline=deadline - 3,
                                                          initial_temp=0.3, cooling_rate=0.997, min_temp=0.01)
                if val_work2 < best_val:
                    best_val = val_work2
                    best_x = h_work2.copy()

    return best_x

if __name__ == "__main__":
    x = construct()
    val = compute_raw_value(x)
    print(f"Best raw value: {val}")
    print(f"Sum: {np.sum(x)}")
