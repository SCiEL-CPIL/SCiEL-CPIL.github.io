import numpy as np
from scipy import optimize
import time

N = 600
DELTA = (1.0/2.0) / N  # interval width = 1/1200
x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.5 * DELTA

def compute_autoconv(h):
    """Compute autoconvolution of step function with heights h on N equal intervals."""
    h = np.asarray(h, dtype=np.float64)
    size = 2 * N - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    conv = np.fft.irfft(H * H, n=fft_size)[:size]
    autoconv = DELTA * conv
    return autoconv

def compute_score(h):
    """Compute the exact normalized max autoconvolution score."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-12):
        return np.inf
    h_clipped = np.maximum(h, 0)
    autoconv = compute_autoconv(h_clipped)
    max_ac = np.max(autoconv)
    integral = DELTA * np.sum(h_clipped)
    if integral <= 0:
        return np.inf
    # Normalization factor 0.5 is consistent with seed
    score = max_ac * 0.5 / (integral ** 2)
    return score

def compute_score_smooth(h, p=20):
    """Compute a smooth approximation of the score using a high p-norm."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-12):
        return 1e10
    h_clipped = np.maximum(h, 0)
    autoconv = compute_autoconv(h_clipped)
    # Use mean of AC^p as a proxy for max^p
    # Score is proportional to max(AC). So min mean(AC^p) approximates min max(AC).
    # We want to minimize the score.
    # score = max(AC) * 0.5 / I^2
    # proxy = (mean(AC^p))^(1/p) * 0.5 / I^2
    # To make it differentiable and simple, we can just minimize mean(AC^p) * 0.5 / I^2
    # Since I is constant in Fourier optimization (if we constrain it or if it varies slowly),
    # but here I varies. Let's just minimize the ratio directly with the smooth max.
    if np.max(autoconv) < 1e-15:
        return 0.0
    smooth_max = np.mean(autoconv ** p) ** (1.0 / p)
    integral = DELTA * np.sum(h_clipped)
    if integral <= 0:
        return 1e10
    score = smooth_max * 0.5 / (integral ** 2)
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_score = np.inf
    best_h = np.ones(N)
    
    # --- Phase 1: Fourier Space Optimization ---
    # We parameterize h(x) = sum_{k=1}^K (a_k cos(2pi k x) + b_k sin(2pi k x)) + c
    # We enforce non-negativity by clipping.
    # To help optimization, we can use a soft-plus or just clip. Clipping is non-smooth but L-BFGS-B handles it.
    # Better: use exponential or sigmoid for positive weights? 
    # Let's stick to linear combination with clipping, as it worked before, but use DE for init.
    
    n_modes = 15
    def params_to_h(params):
        # params: [c, a1, b1, ..., aK, bK]
        c = params[0]
        h = np.full(N, c)
        for k in range(1, n_modes + 1):
            a = params[2*k-1]
            b = params[2*k]
            h += a * np.cos(2*np.pi*k*x)
            h += b * np.sin(2*np.pi*k*x)
        return np.maximum(h, 0)
        
    def objective_fourier_smooth(params):
        h = params_to_h(params)
        return compute_score_smooth(h, p=20)
        
    def objective_fourier_exact(params):
        h = params_to_h(params)
        return compute_score(h)

    # Initial guesses
    np.random.seed(42)
    
    # 1a. Differential Evolution for better global start
    # Bounds for Fourier coefficients: 
    # c should be around 1. a,b around 0.5
    bounds = [(-2.0, 3.0)] + [(-1.0, 1.0)] * (2 * n_modes)
    
    if time.time() < deadline - 60:
        try:
            # DE is expensive, limit iterations
            res_de = optimize.differential_evolution(
                objective_fourier_smooth, 
                bounds, 
                seed=42, 
                maxiter=20, 
                tol=1e-6,
                popsize=5,
                init='latinhypercube'
            )
            if res_de.fun < best_score:
                # Evaluate exact score
                h_candidate = params_to_h(res_de.x)
                s = compute_score(h_candidate)
                if s < best_score:
                    best_score = s
                    best_h = h_candidate
        except Exception as e:
            pass

    # 1b. L-BFGS-B Polish in Fourier space from multiple starts
    starts = []
    # Start from uniform (all params 0)
    starts.append(np.zeros(2 * n_modes + 1))
    # Start from DE result if available
    if 'res_de' in locals():
        starts.append(res_de.x)
    # Random starts
    for _ in range(10):
        if time.time() > deadline - 40:
            break
        starts.append(np.random.randn(2 * n_modes + 1) * 0.2)
        
    for x0 in starts:
        if time.time() > deadline - 30:
            break
        try:
            res_lbfgs = optimize.minimize(
                objective_fourier_smooth, 
                x0, 
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            if res_lbfgs.fun < best_score:
                h_candidate = params_to_h(res_lbfgs.x)
                s = compute_score(h_candidate)
                if s < best_score:
                    best_score = s
                    best_h = h_candidate
        except Exception as e:
            pass

    # --- Phase 2: Direct Optimization on 600 bins ---
    # This is the most effective phase.
    # We will use L-BFGS-B on the exact score.
    # To improve robustness, we can run it from the best Fourier solution 
    # and potentially from a few slight perturbations.
    
    def objective_direct(h_vec):
        return compute_score(h_vec)
        
    # We need to ensure h_vec is non-negative. 
    # L-BFGS-B handles bounds.
    bounds_direct = [(0, None)] * N
    
    # Start 1: Best Fourier result
    h_start = best_h.copy()
    
    # Run L-BFGS-B
    if time.time() < deadline - 10:
        try:
            res_direct = optimize.minimize(
                objective_direct, 
                h_start, 
                method='L-BFGS-B',
                bounds=bounds_direct,
                options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res_direct.fun < best_score:
                best_score = res_direct.fun
                best_h = res_direct.x
        except Exception as e:
            pass
            
    # Try a second polish run with tighter tolerances if time permits
    if time.time() < deadline - 5 and best_score < np.inf:
        try:
            res_direct2 = optimize.minimize(
                objective_direct, 
                best_h, 
                method='L-BFGS-B',
                bounds=bounds_direct,
                options={'maxiter': 30, 'ftol': 1e-12, 'gtol': 1e-9}
            )
            if res_direct2.fun < best_score:
                best_score = res_direct2.fun
                best_h = res_direct2.x
        except Exception as e:
            pass
            
    # Final check
    if time.time() < deadline - 1:
        # Ensure non-negative
        best_h = np.maximum(best_h, 0)
        
    return best_h.tolist()
