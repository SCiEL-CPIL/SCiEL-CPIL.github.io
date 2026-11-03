"""Second autocorrelation inequality - Basin Hopping global search."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
_DEADLINE = 140.0  # Safety margin before 150s kill

def _autoconv_peak(h):
    """Compute max of linear autoconvolution h*h[0..n-1]."""
    n = len(h)
    fft_h = np.fft.rfft(h, n=2 * n)
    autoconv = np.fft.irfft(fft_h * np.conj(fft_h), n=2 * n)[:n]
    return autoconv.max()

def _score(h):
    """Compute (L2^2) / (L1 * ||autoconv||_inf)."""
    h = np.asarray(h, dtype=float)
    l1 = h.sum()
    if l1 < 1e-15:
        return 0.0
    l2_sq = float(np.dot(h, h))
    peak = _autoconv_peak(h)
    if peak < 1e-15:
        return 0.0
    return l2_sq / (l1 * peak)

def _local_optimize(h0):
    """
    Perform local optimization using L-BFGS-B on the simplex.
    We optimize in log-space or directly? 
    Directly with bounds [0, 1] is safer for L-BFGS-B.
    We normalize h to sum=1 before returning.
    """
    h0 = np.asarray(h0, dtype=float)
    h0 = np.maximum(0, h0)
    s = h0.sum()
    if s < 1e-15:
        return h0, 0.0
    
    h0 = h0 / s
    
    def neg_score(x):
        # x is the vector h, constrained to be non-negative by L-BFGS-B bounds
        # We don't strictly enforce sum=1 in the objective, but we normalize 
        # inside the score calculation to make it scale invariant, 
        # OR we can just let L-BFGS-B optimize the shape.
        # Since the score is scale invariant, we can fix the scale by 
        # normalizing inside the objective.
        x_norm = x / x.sum() if x.sum() > 1e-15 else x
        return -_score(x_norm)

    # Bounds: 0 to 1 (arbitrary upper bound, normalization handles scale)
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    res = minimize(neg_score, h0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
    
    h_opt = res.x
    h_opt = np.maximum(0, h_opt)
    s = h_opt.sum()
    if s < 1e-15:
        return h0, 0.0
    h_opt = h_opt / s
    sc = _score(h_opt)
    return h_opt, sc

def construct():
    t_start = time.time()
    rng = np.random.RandomState(42)
    
    best_h = np.ones(N, dtype=float) / N
    best_score = _score(best_h)
    
    # Initial candidates
    candidates = [
        np.ones(N) / N,
        np.arange(1, N+1, dtype=float),
        np.arange(N, 0, -1, dtype=float),
        np.exp(np.linspace(-2, 2, N)),
        np.exp(-np.linspace(0, 4, N)),
        np.exp(-np.linspace(4, 0, N)),
        np.random.dirichlet(np.ones(N) * 0.5), # Sparse-ish
        np.random.dirichlet(np.ones(N) * 10.0), # Uniform-ish
    ]
    
    # Pre-optimize initial candidates
    for c in candidates:
        if time.time() - t_start > _DEADLINE - 5.0:
            break
        h_opt, sc = _local_optimize(c)
        if sc > best_score:
            best_score = sc
            best_h = h_opt
    
    # Basin Hopping Loop
    T = 0.5 # Initial temperature
    T_min = 0.01
    alpha = 0.95 # Cooling rate
    max_iter = 2000
    
    for i in range(max_iter):
        if time.time() - t_start > _DEADLINE - 5.0:
            break
        
        # Perturb current best
        # Add noise in log-space to preserve positivity and relative structure
        # Or just add Gaussian noise and project to simplex
        h_curr = best_h.copy()
        
        # Perturbation strategy: Mix of small Gaussian and occasional larger jumps
        if rng.random() < 0.8:
            noise = rng.randn(N) * 0.1
        else:
            noise = rng.randn(N) * 0.5
            
        h_pert = h_curr + noise
        h_pert = np.maximum(0, h_pert)
        s = h_pert.sum()
        if s < 1e-15:
            h_pert = rng.dirichlet(np.ones(N))
        else:
            h_pert = h_pert / s
            
        # Local optimize the perturbed point
        h_opt, sc = _local_optimize(h_pert)
        
        # Metropolis Acceptance
        if sc > best_score:
            best_score = sc
            best_h = h_opt
            # Reheat slightly on improvement? Or just continue cooling.
            # T = max(T_min, T * alpha) 
        else:
            delta = sc - best_score
            # delta is negative
            accept_prob = np.exp(-delta / T)
            if rng.random() < accept_prob:
                best_h = h_opt
                # Do not update best_score if we accept a worse solution, 
                # but we do move the current state. 
                # However, for tracking the global best, we keep best_score separate.
                # In standard Basin Hopping, you track the current state and the global best.
                # Here we are effectively doing a random walk on the local minima landscape.
                # Let's stick to: if accepted, update current state. 
                # We always keep track of the absolute best found.
                pass # best_h is the current state, best_score is the global best
        
        # Cooling
        T = max(T_min, T * alpha)
        
        # Optional: If T is very low and no improvement for a while, restart from random
        # For simplicity and time constraints, we rely on the cooling schedule.
        
    # Final check and cleanup
    best_h = np.maximum(0, best_h)
    if best_h.sum() < 1e-15:
        best_h = np.ones(N)
    else:
        best_h = best_h / best_h.sum()
        
    return best_h.tolist()
