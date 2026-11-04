"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def _autoconv(h):
    """Compute autoconvolution of h on the grid [-1/4, 1/4]."""
    # h has N values on equal intervals of [-1/4, 1/4]
    # interval width = (1/4 - (-1/4)) / (N-1) = (1/2)/(N-1)
    # Autoconvolution: (h*h)(t) = integral h(s)h(t-s) ds
    # For discrete: sum_i h_i * h_j where t_i + t_j = t_k
    # Using FFT for efficiency
    n_fft = 1 << (2 * N - 1).bit_length()  # power of 2 >= 2N-1
    H = np.fft.rfft(h, n_fft)
    conv = np.fft.irfft(H * H, n_fft)[:2 * N - 1]
    return conv

def _ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0) or np.sum(h) < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    l1 = np.sum(h)
    conv = _autoconv(h)
    sup = np.max(conv)
    if sup < 1e-30:
        return 0.0
    return l2_sq / (l1 * sup)

def _gaussian_profile(center, width, amplitude, N, grid_start, grid_end):
    """Generate a Gaussian profile on the grid."""
    x = np.linspace(grid_start, grid_end, N)
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)

def _lorentzian_profile(center, width, amplitude, N, grid_start, grid_end):
    """Generate a Lorentzian (Cauchy) profile on the grid."""
    x = np.linspace(grid_start, grid_end, N)
    return amplitude / (1 + ((x - center) / width) ** 2)

def _local_optimize(h0, max_iters=200, tol=1e-8):
    """Simple gradient-free local optimization using Nelder-Mead style."""
    h = np.array(h0, dtype=float)
    h = np.maximum(h, 0)
    
    best_h = h.copy()
    best_r = _ratio(h)
    
    # Use scipy's minimize if available, else simple coordinate descent
    try:
        from scipy.optimize import minimize
        
        def neg_ratio(h):
            h = np.maximum(h, 0)
            return -_ratio(h)
        
        res = minimize(neg_ratio, h, method='Nelder-Mead', 
                      options={'maxiter': max_iters, 'xatol': tol, 'fatol': tol})
        h_opt = np.maximum(res.x, 0)
        r_opt = _ratio(h_opt)
        if r_opt > best_r:
            best_h = h_opt
            best_r = r_opt
    except:
        # Fallback: coordinate ascent
        for _ in range(50):
            improved = False
            for i in range(N):
                h_save = h[i]
                # Try small perturbations
                for delta in [-0.1, -0.05, 0.05, 0.1]:
                    h[i] = max(0, h_save + delta)
                    r = _ratio(h)
                    if r > best_r:
                        best_r = r
                        best_h = h.copy()
                        improved = True
                h[i] = h_save
            if not improved:
                break
    
    return best_h, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    grid_start = -0.25
    grid_end = 0.25
    
    best_h = np.ones(N)
    best_r = _ratio(best_h)
    
    # Phase 1: Structured parametric search
    centers = np.linspace(-0.2, 0.2, 9)
    widths = np.linspace(0.02, 0.25, 12)
    amplitudes = [1.0]
    
    for profile_func in [_gaussian_profile, _lorentzian_profile]:
        for c in centers:
            for w in widths:
                if time.time() > deadline:
                    break
                for amp in amplitudes:
                    h = profile_func(c, w, amp, N, grid_start, grid_end)
                    # Ensure nonnegative and normalize for comparison
                    h = np.maximum(h, 0)
                    if np.sum(h) < 1e-10:
                        continue
                    r = _ratio(h)
                    if r > best_r:
                        best_r = r
                        best_h = h.copy()
        
        if time.time() > deadline:
            break
    
    # Phase 2: Local optimization from best structured result
    if time.time() < deadline - 10:
        h_opt, r_opt = _local_optimize(best_h, max_iters=300)
        if r_opt > best_r:
            best_h = h_opt
            best_r = r_opt
    
    # Phase 3: Try additional local optimizations from diverse starting points
    if time.time() < deadline - 5:
        # Try a few more diverse initializations
        diverse_starts = []
        # Uniform
        diverse_starts.append(np.ones(N))
        # Triangular
        diverse_starts.append(np.abs(np.linspace(-1, 1, N)))
        # Two peaks
        h2 = np.zeros(N)
        h2[:N//4] = 1.0
        h2[3*N//4:] = 1.0
        diverse_starts.append(h2)
        
        for h0 in diverse_starts:
            if time.time() > deadline - 5:
                break
            h_opt, r_opt = _local_optimize(h0, max_iters=100)
            if r_opt > best_r:
                best_h = h_opt
                best_r = r_opt
    
    # Ensure final result is valid
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)
    
    return list(best_h)
# EVOLVE-BLOCK-END
