import numpy as np
import time
from scipy.optimize import minimize

N = 400
_FSIZE = 1
while _FSIZE < 2 * N - 1:
    _FSIZE *= 2

# Precompute FFT basis for incremental updates
_FREQS = np.arange(_FSIZE // 2 + 1)
_BASIS = np.exp(-2j * np.pi * _FREQS[None, :] * np.arange(N)[:, None] / _FSIZE)  # shape (N, _FSIZE//2+1)

# Precompute basis for FFT updates if needed, but for gradient we use FFT directly
# For local search, we might use incremental FFT, but let's stick to the robust approach first.

def _autoconv_full(h):
    """Full linear autoconvolution via FFT."""
    H = np.fft.rfft(h, _FSIZE)
    ac = np.fft.irfft(H * H, _FSIZE)
    return ac[:2*N-1]

def compute_raw_value(x):
    """
    Computes the raw value: max(|autoconvolution|) / (sum(x))^2.
    Assumes x is a 1D numpy array.
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
    
    ac = _autoconv_full(h)
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
        # dS/dh_i = 2 * sum_k v_k h_{k-i}
        # This is 2 * cross_correlation(v, h)
        
        # Use FFT for cross-correlation
        # corr[i] = sum_k v[k+i] h[k]
        # We need i from 0 to N-1.
        # v has length 2N-1, h has length N.
        
        # To compute cross-correlation efficiently:
        # C = ifft(fft(v) * conj(fft(h)))
        # C[i] = sum_n v[n+i] conj(h[n])
        
        # We need to handle lengths.
        # v_len = 2N-1, h_len = N.
        # Result length is v_len + h_len - 1? No, cross-correlation of lengths L1, L2 has length L1+L2-1.
        # But we only need the first N lags (i=0 to N-1).
        
        # Let's use a large enough FFT size.
        # Max index in v is 2N-2. Max index in h is N-1.
        # We need v[n+i] where n goes 0..N-1 and i goes 0..N-1.
        # Max index accessed in v is (N-1) + (N-1) = 2N-2.
        # So v is fully covered.
        
        # Standard FFT cross-correlation:
        # r[k] = sum_n a[n] conj(b[n-k]) ?
        # Let's stick to the definition: sum_m v[m+i] h[m]
        
        # Let's use np.correlate for simplicity and correctness, as N=400 is small enough.
        # np.correlate(v, h, mode='full')
        # r[k] = sum_n v[n+k] conj(h[n])
        # We want sum_m v[m+i] h[m].
        # This matches r[i] if h is real.
        
        # However, np.correlate is O(N^2) which is 160k ops. This is very fast.
        # Let's try to use FFT for speed if possible, but np.correlate is safer for indexing.
        
        # Actually, let's use the FFT method properly.
        # We want C[i] = sum_{m=0}^{N-1} v[m+i] h[m]
        
        # Pad v and h to size FSIZE_CORR.
        # FSIZE_CORR must be >= (2N-1) + N - 1 = 3N - 2.
        fsize_corr = 1
        while fsize_corr < 3 * N:
            fsize_corr *= 2
            
        V = np.fft.rfft(v, fsize_corr)
        H = np.fft.rfft(h, fsize_corr)
        C = np.fft.irfft(V * np.conj(H), fsize_corr)
        
        # C[i] should be sum_n v[n+i] conj(h[n])
        # Since h is real, conj(h[n]) = h[n].
        # So C[i] = sum_n v[n+i] h[n].
        # This is exactly what we want for i=0..N-1.
        
        grad_S = 2 * C[:N]
    
    # Objective f = S / s^2
    # df/dh_i = (dS/dh_i * s^2 - S * 2 * s) / s^4
    # = (dS/dh_i * s - 2 * S) / s^3
    
    grad = (grad_S * s - 2 * S) / (s ** 3)
    
    return S / (s * s), grad

def _smooth_obj_with_grad(h, beta=50.0):
    """
    Objective and gradient for scipy.optimize.minimize.
    """
    return _soft_max_abs_and_grad(h, beta)

def _local_search_flip(h, max_passes=10, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    Uses precomputed basis for incremental FFT updates.
    """
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    H = np.fft.rfft(h, _FSIZE)
    integral = np.sum(h)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            old_val = h[i]
            h[i] = -old_val
            delta = -2.0 * old_val * _BASIS[i]
            H_new = H + delta
            conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
            new_integral = integral - 2.0 * old_val
            if abs(new_integral) < 1e-12:
                new_val = float('inf')
            else:
                new_val = np.max(np.abs(conv)) / (new_integral * new_integral)
            
            if new_val < best_val:
                best_val = new_val
                H = H_new
                integral = new_integral
                improved = True
            else:
                h[i] = old_val
        if not improved:
            break
    return h, best_val

def _local_search_block(h, deadline=None, block_sizes=None):
    """
    Block flip local search: flip contiguous blocks of signs.
    Uses precomputed basis matrix for vectorized incremental FFT updates.
    """
    if block_sizes is None:
        block_sizes = [2, 3, 4, 5, 6, 8, 10, 13, 17]
    
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    H = np.fft.rfft(h, _FSIZE)
    integral = np.sum(h)
    
    improved = True
    pass_count = 0
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        pass_count += 1
        # Adaptive step: start fine, get coarser
        step = max(1, min(pass_count, 5))
        
        for bs in block_sizes:
            if deadline is not None and time.time() > deadline:
                break
            for start in range(0, N - bs + 1, step):
                if deadline is not None and time.time() > deadline:
                    break
                block = h[start:start+bs].copy()
                h[start:start+bs] = -h[start:start+bs]
                # Vectorized FFT update using precomputed basis rows
                delta = -2.0 * block @ _BASIS[start:start+bs]
                H_new = H + delta
                conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
                new_integral = integral - 2.0 * np.sum(block)
                if abs(new_integral) < 1e-12:
                    new_val = float('inf')
                else:
                    new_val = np.max(np.abs(conv)) / (new_integral * new_integral)
                
                if new_val < best_val:
                    best_val = new_val
                    H = H_new
                    integral = new_integral
                    improved = True
                else:
                    h[start:start+bs] = block
    return h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill
    
    # 1. Baseline: All ones
    best_x = np.ones(N, dtype=float)
    best_val = compute_raw_value(best_x)
    
    # 2. Generate diverse candidates
    candidates = [best_x.copy()]
    t = np.arange(N)
    
    # Quadratic chirp sequences
    for freq in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        phase = 2 * np.pi * freq * (t**2) / N
        x_chirp = np.sign(np.cos(phase))
        x_chirp[x_chirp == 0] = 1
        candidates.append(x_chirp)
    
    # Linear phase (constant frequency)
    for freq in [0.5, 1.0, 1.5, 2.0, 3.0]:
        phase = 2 * np.pi * freq * t / N
        x_lin = np.sign(np.cos(phase))
        x_lin[x_lin == 0] = 1
        candidates.append(x_lin)
    
    # Barker-like structured patterns
    for period in [3, 4, 5, 6, 7, 8]:
        x_barker = np.array([1.0 if (i % period) < period // 2 else -1.0 for i in range(N)])
        candidates.append(x_barker)
    
    # Random sequences with fixed seed for reproducibility
    rng = np.random.default_rng(42)
    for _ in range(20):
        x_rand = rng.choice([1.0, -1.0], N)
        candidates.append(x_rand)
    
    # Golden ratio and other irrational-based
    for ratio in [(1 + np.sqrt(5)) / 2, np.sqrt(2), np.pi / 2, np.e / 2]:
        x_irr = np.sign(np.cos(2 * np.pi * ratio * t))
        x_irr[x_irr == 0] = 1
        candidates.append(x_irr)
    
    # 3. Evaluate all candidates
    for x in candidates:
        val = compute_raw_value(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    # 4. Smooth surrogate optimization from best candidates using analytical gradient
    scored_candidates = []
    for x in candidates:
        val = compute_raw_value(x)
        scored_candidates.append((val, x))
    scored_candidates.sort(key=lambda t: t[0])
    
    # Optimize from top 10 diverse starts
    top_starts = scored_candidates[:10]
    
    for init_val, h_init in top_starts:
        if time.time() > deadline - 50:
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        h_norm = h_init / s0
        
        # Multi-stage optimization with increasing beta for sharper approximation
        for beta in [15, 30, 60, 100]:
            if time.time() > deadline - 25:
                break
            
            try:
                def obj_and_grad(x, b=beta):
                    return _soft_max_abs_and_grad(x, beta=b)
                
                res = minimize(obj_and_grad, h_norm, method='L-BFGS-B',
                               jac=True,
                               options={'maxiter': 400, 'disp': False, 'gtol': 1e-9,
                                        'ftol': 1e-12, 'maxls': 30})
                h_opt = res.x
                
                val = compute_raw_value(h_opt)
                if val < best_val:
                    best_val = val
                    best_x = h_opt.copy()
                
                h_bin = np.sign(h_opt)
                h_bin[h_bin == 0] = 1
                val_bin = compute_raw_value(h_bin)
                if val_bin < best_val:
                    best_val = val_bin
                    best_x = h_bin.copy()
                    
            except Exception:
                pass
    
    # 5. Local search refinement on best result
    if time.time() < deadline - 20:
        h_work = np.sign(best_x)
        h_work[h_work == 0] = 1
        
        # Single-bit flip local search (multiple passes)
        h_work, val_work = _local_search_flip(h_work, max_passes=10, deadline=deadline - 10)
        if val_work < best_val:
            best_val = val_work
            best_x = h_work.copy()
        
        # Block flip local search with wider range of sizes
        if time.time() < deadline - 8:
            h_work, val_work = _local_search_block(h_work, deadline=deadline - 4,
                                                    block_sizes=[2, 3, 4, 5, 6, 8, 10, 13])
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()
    
    # 6. Random restarts with local search if time remains
    if time.time() < deadline - 30:
        for _ in range(10):
            if time.time() > deadline - 15:
                break
            x_rand = rng.choice([1.0, -1.0], N)
            h_work, val_work = _local_search_flip(x_rand, max_passes=4, deadline=deadline - 8)
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()
            # Also try block search from this random start if it was promising
            if val_work < best_val * 1.1 and time.time() < deadline - 10:
                h_work2, val_work2 = _local_search_block(h_work, deadline=deadline - 5,
                                                          block_sizes=[2, 3, 5, 7])
                if val_work2 < best_val:
                    best_val = val_work2
                    best_x = h_work2.copy()
    
    # 7. Final high-precision polish
    if time.time() < deadline - 5:
        h_work = np.sign(best_x)
        h_work[h_work == 0] = 1
        for polish_pass in range(4):
            if time.time() > deadline - 2:
                break
            h_work, val_work = _local_search_flip(h_work, max_passes=3, deadline=deadline - 2)
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()
            if time.time() > deadline - 1:
                break
            h_work, val_work = _local_search_block(h_work, deadline=deadline - 1,
                                                    block_sizes=[2, 3, 4])
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()
    
    # Ensure we return a valid array
    return best_x

if __name__ == "__main__":
    x = construct()
    val = compute_raw_value(x)
    print(f"Best raw value: {val}")
    print(f"Sum: {np.sum(x)}")
