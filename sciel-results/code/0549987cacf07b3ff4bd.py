import numpy as np
import time
from scipy.optimize import minimize

N = 50
w = 1.0 / 100.0
DEADLINE = 145.0  # Safety margin from 150s

def compute_value(h):
    """Exact evaluation using np.convolve as per task contract."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return 0.0
    h = np.clip(h, 0.0, None)
    g = np.convolve(h, h)
    m = len(g)
    g_sq = np.sum(g**2)
    cross = np.sum(g[:-1] * g[1:])
    l2 = (w / 3.0) * (2.0 * g_sq + cross)
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    if norm_inf == 0 or norm_1 == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def compute_value_fast(h):
    """FFT-based fast evaluation for optimization (same result, faster)."""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12) or not np.all(np.isfinite(h)):
        return 0.0
    h = np.clip(h, 0.0, None)
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n - 1:
        n_fft <<= 1
    H = np.fft.rfft(h, n_fft)
    G = H * H
    g = np.fft.irfft(G, n_fft)[:2*n-1]
    g = np.real(g)
    g[g < 0] = 0.0
    m = len(g)
    g_sq = np.sum(g**2)
    cross = np.sum(g[:-1] * g[1:])
    l2 = (w / 3.0) * (2.0 * g_sq + cross)
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    if norm_inf == 0 or norm_1 == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def soft_max(x, beta=100.0):
    """Smooth approximation to max(x) using log-sum-exp."""
    x_max = np.max(x)
    if not np.isfinite(x_max) or x_max == 0:
        return 0.0
    scaled = beta * (x - x_max)
    # Clip to prevent overflow in exp, though beta*(x-x_max) <= 0
    scaled = np.clip(scaled, -50.0, 0.0)
    exp_sum = np.sum(np.exp(scaled))
    if exp_sum == 0:
        return x_max
    return x_max - np.log(exp_sum) / beta

def neg_value_smooth(log_h, beta=100.0):
    """Smooth objective: uses soft-max for norm_inf to enable gradient flow."""
    # Center log_h to maintain scale invariance and numerical stability
    centered = log_h - np.mean(log_h)
    h = np.exp(centered)
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n - 1:
        n_fft <<= 1
    H = np.fft.rfft(h, n_fft)
    G = H * H
    g = np.fft.irfft(G, n_fft)[:2*n-1]
    g = np.real(g)
    g[g < 0] = 0.0
    m = len(g)
    g_sq = np.sum(g**2)
    cross = np.sum(g[:-1] * g[1:])
    l2 = (w / 3.0) * (2.0 * g_sq + cross)
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = soft_max(g, beta)
    if norm_inf <= 0 or norm_1 <= 0:
        return 0.0
    return -(l2 / (norm_1 * norm_inf))

def neg_value_direct(log_h):
    """Direct objective with exact max (for final polish)."""
    centered = log_h - np.mean(log_h)
    h = np.exp(centered)
    return -compute_value_fast(h)

def construct():
    start_time = time.time()
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initial guesses in log space
    starts = []
    x = np.arange(N)
    center = N / 2.0
    
    # Uniform
    starts.append(np.zeros(N))
    
    # Gaussians with various widths
    for sigma_sq in [1, 2, 5, 10, 20, 40, 80]:
        g = np.exp(-((x - center)**2) / (2 * sigma_sq))
        starts.append(np.log(g + 1e-9))
        
    # Sinc-like (main lobe)
    g = np.sinc((x - center) / 5.0)
    g = np.maximum(g, 1e-9)
    starts.append(np.log(g))
    
    # Raised Cosine
    g = 0.5 * (1 + np.cos(np.pi * x / (N - 1)))
    starts.append(np.log(g + 1e-9))
    
    # Triangle
    g = 1 - np.abs(x - center) / center
    g[g < 0] = 1e-9
    starts.append(np.log(g))
    
    # Plateau (box filter)
    for width in [10, 20, 30]:
        g = np.zeros(N)
        g[N//2 - width//2 : N//2 + width//2 + 1] = 1.0
        starts.append(np.log(g + 1e-9))
        
    # Bump functions (narrow peaks)
    for peak_pos in [N//4, N//2, 3*N//4]:
        for width in [2, 4, 8]:
            g = np.exp(-((x - peak_pos)**2) / (2 * width**2))
            starts.append(np.log(g + 1e-9))
    
    # Random smooth starts
    for _ in range(30):
        x_rand = rng.randn(N)
        kernel = np.ones(7) / 7.0
        x_rand = np.convolve(x_rand, kernel, mode='same')
        x_rand = x_rand - np.mean(x_rand)
        x_rand = x_rand / (np.max(np.abs(x_rand)) + 1e-9) * 2.0
        starts.append(x_rand)
        
    # Phase 1: Optimize with smooth surrogate (soft-max)
    for i, start_log in enumerate(starts):
        if time.time() > DEADLINE - 20.0:
            break
        try:
            res = minimize(
                neg_value_smooth,
                start_log,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            h_cand = np.exp(res.x - np.mean(res.x))
            v_cand = compute_value(h_cand)
            if v_cand > best_val:
                best_val = v_cand
                best_h = h_cand.copy()
        except Exception:
            pass
    
    # Phase 2: Polish with direct objective around best solution
    if time.time() < DEADLINE - 5.0:
        best_log = np.log(np.maximum(best_h, 1e-9))
        best_log = best_log - np.mean(best_log)
        
        # Refine with direct (nonsmooth) objective
        try:
            res = minimize(
                neg_value_direct,
                best_log,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'maxfun': 20000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            h_cand = np.exp(res.x - np.mean(res.x))
            v_cand = compute_value(h_cand)
            if v_cand > best_val:
                best_val = v_cand
                best_h = h_cand.copy()
        except Exception:
            pass
        
        # Random restarts around best with smooth then direct
        for _ in range(50):
            if time.time() > DEADLINE - 2.0:
                break
            perturb = rng.randn(N) * 0.1
            perturb = np.convolve(perturb, np.ones(3)/3.0, mode='same')
            start_log = best_log + perturb
            
            try:
                # First smooth pass
                res1 = minimize(
                    neg_value_smooth,
                    start_log,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                # Then direct polish
                res2 = minimize(
                    neg_value_direct,
                    res1.x,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'maxfun': 8000, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                h_cand = np.exp(res2.x - np.mean(res2.x))
                v_cand = compute_value(h_cand)
                if v_cand > best_val:
                    best_val = v_cand
                    best_h = h_cand.copy()
                    best_log = res2.x - np.mean(res2.x)
            except Exception:
                pass
    
    # Final verification with exact compute_value
    final_val = compute_value(best_h)
    if final_val > best_val:
        best_val = final_val
    
    return list(np.asarray(best_h, dtype=float))
