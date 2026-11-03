import numpy as np
from scipy.optimize import minimize
import time

N = 600
TIME_LIMIT = 135  # Safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    # Use FFT for faster convolution
    conv = np.fft.irfft(np.fft.rfft(f) * np.fft.rfft(f), n=2*N-1)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    return np.max(conv) / (s * s)

def smooth_max(x, beta):
    """Smooth approximation to max using log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    if x_max < 1e-15:
        return 0.0
    x_scaled = x / x_max
    shifted = x_scaled - np.max(x_scaled)
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    return x_max * (1.0 + log_sum)

def parametric_objective(coeffs, K, beta=10.0, neg_penalty=100.0):
    """
    Objective for parametric optimization.
    coeffs: [a0, a1, ..., aK]
    f(x) = sum_{k=0}^K a_k cos(2*pi*k*x/L)
    x is in [-0.25, 0.25], L=0.5.
    """
    x = np.linspace(-0.25, 0.25, N)
    # Compute f
    f = np.zeros(N)
    for k in range(K + 1):
        if k == 0:
            f += coeffs[0]
        else:
            f += coeffs[k] * np.cos(2 * np.pi * k * x / 0.5)
    
    # Penalty for negative values
    if np.any(f < 0):
        min_f = np.min(f)
        penalty = neg_penalty * (min_f ** 2)
    else:
        penalty = 0.0
        
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return 1e10
    
    conv = np.fft.irfft(np.fft.rfft(f) * np.fft.rfft(f), n=2*N-1)
    norm_conv = conv / (s * s)
    val = smooth_max(norm_conv, beta)
    return val + penalty

def local_refinement(h, start_time, max_time, alpha_init=0.08, threshold_frac=0.92):
    """Local refinement using sensitivity-based peak reduction."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = alpha_init
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = np.fft.irfft(np.fft.rfft(current_h) * np.fft.rfft(current_h), n=2*N-1)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        threshold = max_c * threshold_frac
        peak_indices = np.where(conv >= threshold)[0]
        
        sensitivity = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            if j_min <= j_max:
                indices = np.arange(j_min, j_max + 1)
                contrib = current_h[m - indices]
                sensitivity[indices] += contrib
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = np.fft.irfft(np.fft.rfft(h_new) * np.fft.rfft(h_new), n=2*N-1)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.03, 0.4)
        else:
            alpha *= 0.88
            if alpha < 1e-8:
                break
    
    return best_h, best_val

def polish_pass(h, start_time, max_time):
    """Final high-precision polish pass."""
    best_h = h.copy()
    best_val = compute_score(h)
    current_h = h.copy()
    alpha = 0.02
    
    while time.time() - start_time < max_time:
        s = np.sum(current_h)
        if s < 1e-12:
            break
        current_h = current_h * (N / s)
        
        conv = np.fft.irfft(np.fft.rfft(current_h) * np.fft.rfft(current_h), n=2*N-1)
        max_c = np.max(conv)
        current_val = max_c / (N * N)
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h.copy()
        
        threshold = max_c * 0.99
        peak_indices = np.where(conv >= threshold)[0]
        
        sensitivity = np.zeros(N)
        for m in peak_indices:
            j_min = max(0, m - (N - 1))
            j_max = min(N - 1, m)
            if j_min <= j_max:
                indices = np.arange(j_min, j_max + 1)
                contrib = current_h[m - indices]
                sensitivity[indices] += contrib
        
        sens_mean = np.mean(sensitivity)
        sens_std = np.std(sensitivity) + 1e-12
        sens_norm = (sensitivity - sens_mean) / sens_std
        
        h_new = current_h * (1.0 - alpha * sens_norm)
        h_new = np.maximum(h_new, 0.0)
        
        s_new = np.sum(h_new)
        if s_new < 1e-12:
            break
        h_new = h_new * (N / s_new)
        
        conv_new = np.fft.irfft(np.fft.rfft(h_new) * np.fft.rfft(h_new), n=2*N-1)
        val_new = np.max(conv_new) / (N * N)
        
        if val_new < current_val:
            current_h = h_new
            alpha = min(alpha * 1.02, 0.15)
        else:
            alpha *= 0.9
            if alpha < 1e-10:
                break
    
    return best_h, best_val

def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    best_f = np.ones(N)
    best_score = compute_score(best_f)
    
    # Phase 1: Parametric Search
    # We search for good shapes in a low-dimensional Fourier space.
    # This provides diverse, smooth initializations that are likely to be in good basins.
    
    max_param_time = 30.0 # Spend 30s on parametric search
    param_end_time = start_time + max_param_time
    
    for K in [5, 10, 20]:
        if time.time() > param_end_time:
            break
            
        # Try a few random initializations in coefficient space
        for restart in range(3):
            if time.time() > param_end_time:
                break
                
            # Initialize coefficients: DC term positive, others small
            coeffs0 = np.zeros(K + 1)
            coeffs0[0] = 1.0
            coeffs0[1:] = rng.normal(0, 0.1, K)
            
            res = minimize(
                parametric_objective, 
                coeffs0, 
                args=(K, 10.0, 100.0),
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-8}
            )
            
            # Evaluate the best result
            coeffs_opt = res.x
            x = np.linspace(-0.25, 0.25, N)
            f_candidate = np.zeros(N)
            for k in range(K + 1):
                if k == 0:
                    f_candidate += coeffs_opt[0]
                else:
                    f_candidate += coeffs_opt[k] * np.cos(2 * np.pi * k * x / 0.5)
            
            f_candidate = np.maximum(f_candidate, 0.0)
            if np.sum(f_candidate) < 1e-12:
                continue
                
            f_candidate = f_candidate * (N / np.sum(f_candidate))
            score_candidate = compute_score(f_candidate)
            
            if score_candidate < best_score:
                best_score = score_candidate
                best_f = f_candidate.copy()
                
    # Phase 2: Local Refinement on the best candidate
    if time.time() - start_time < TIME_LIMIT - 30:
        refined_f, refined_score = local_refinement(
            best_f, start_time, TIME_LIMIT - 30
        )
        if refined_score < best_score:
            best_f = refined_f
            best_score = refined_score

    # Phase 3: Final Polish
    if time.time() - start_time < TIME_LIMIT - 10:
        polished_f, polished_score = polish_pass(
            best_f, start_time, TIME_LIMIT - 2
        )
        if polished_score < best_score:
            best_f = polished_f
            best_score = polished_score
    
    best_f = np.maximum(best_f, 0.0)
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")
