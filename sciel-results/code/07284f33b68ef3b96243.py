"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize

def construct():
    """
    Minimize value = abs(2*N*max(convolve(h,h)) / (sum(h))^2)
    using generalized quadratic phase sequences and windowing.
    """
    best_h = np.ones(N)
    best_val = 2.0  # seed value
    
    def evaluate(h):
        if not np.all(np.isfinite(h)):
            return float('inf')
        s = np.sum(h)
        if abs(s) < 1e-12:
            return float('inf')
        conv = np.convolve(h, h)
        m = np.max(conv)
        val = abs(2 * N * m / (s * s))
        return val

    t = np.arange(N, dtype=np.float64)
    
    def get_h(params):
        mu, alpha, win_param, dc = params
        
        # Generalized quadratic phase
        # phi(t) = -pi * mu * t * (t+1) / N + pi * alpha * t * t / N
        phase = -np.pi * mu * t * (t + 1) / N + np.pi * alpha * t * t / N
        signal = np.exp(1j * phase)
        
        # Windowing
        # win_param in [0, 1] -> Rectangular to Hann
        # win_param in [1, 2] -> Gaussian with varying width
        if win_param <= 1.0:
            n_norm = (t - (N-1)/2.0) / ((N-1)/2.0) # -1 to 1
            hann = 0.5 * (1 + np.cos(np.pi * n_norm))
            window = (1 - win_param) * 1.0 + win_param * hann
        else:
            # Map win_param (1, 2] to sigma_norm (large, small)
            # win_param=1.0 -> sigma_norm=5.0 (wide)
            # win_param=2.0 -> sigma_norm=0.5 (narrow)
            sigma_norm = 5.0 * (2.0 - win_param) + 0.5
            n_norm = (t - (N-1)/2.0) / ((N-1)/2.0)
            window = np.exp(-0.5 * (n_norm / sigma_norm)**2)
        
        h_real = np.real(signal) * window + dc
        return h_real

    # Define bounds for differential evolution
    # mu: -10 to 10
    # alpha: -5 to 5
    # win_param: 0 to 2.0
    # dc: -1.0 to 1.0
    bounds = [(-10.0, 10.0), (-5.0, 5.0), (0.0, 2.0), (-1.0, 1.0)]
    
    # Objective function wrapper
    def objective(params):
        h_test = get_h(params)
        return evaluate(h_test)
    
    best_candidate = None
    best_candidate_val = float('inf')
    
    # Run differential evolution
    try:
        res = differential_evolution(objective, bounds, maxiter=50, popsize=10, seed=42, tol=1e-6, atol=1e-8)
        if res.success and res.fun < best_candidate_val:
            best_candidate_val = res.fun
            best_candidate = get_h(res.x).copy()
            
            # Refine with Nelder-Mead
            res2 = minimize(
                lambda p: evaluate(get_h(p)),
                res.x,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if res2.success and res2.fun < best_candidate_val:
                best_candidate_val = res2.fun
                best_candidate = get_h(res2.x).copy()
    except Exception:
        pass

    if best_candidate is not None and best_candidate_val < best_val:
        return best_candidate
    return best_h
# EVOLVE-BLOCK-END
