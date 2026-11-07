"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, SLSQP

N = 100
DEADLINE = time.time() + 145  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _raw_value(h):
    """Compute the evaluator's value: max correlation / N * 2."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2

def _is_valid(h):
    """Check if h is valid: finite, in [0,1], sum close to N/2."""
    if not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-12) or np.any(h > 1 + 1e-12):
        return False
    if abs(np.sum(h) - N / 2) > 1e-9 * N:
        return False
    return True

def _project_and_refine(h, max_iter=2000):
    """
    Local refinement using projected gradient / SLSQP on the true max-correlation objective.
    We use a smooth surrogate: since max of correlations is non-smooth, we use the fact that
    at the optimum, the max is achieved at some lag k. We can try each lag as the active one
    and optimize the smooth objective for that lag, then take the best.
    
    Actually, a better approach: use SLSQP to minimize the max directly via an epigraph formulation,
    or just try each lag separately with a smooth objective.
    
    For simplicity and speed, we'll use a multi-start local refinement:
    1. Try to identify which lag gives the max correlation
    2. Optimize the smooth objective for that specific lag using SLSQP
    3. Iterate: re-check which lag is max, re-optimize
    """
    h = h.copy()
    h = np.clip(h, 0, 1)
    
    # Ensure sum constraint by projecting
    s = np.sum(h)
    if abs(s - N/2) > 1e-12:
        h = h + (N/2 - s) / N
    
    best_h = h.copy()
    best_val = _raw_value(h)
    
    for iteration in range(50):  # outer iterations
        if time.time() > DEADLINE - 5:
            break
        
        # Find which lag gives the max
        corr = np.correlate(h, 1 - h, mode="full")
        lag = np.argmax(corr)
        
        # Define smooth objective: minimize corr[lag] (since we want to minimize the max,
        # and this lag is currently the max)
        # corr[lag] = sum_i h[i] * (1-h[i+lag-N+1]) for appropriate indices
        # Actually, let's just use SLSQP with the full max via a smooth approximation
        
        # Use log-sum-exp smooth max
        def smooth_max_correlation(h_vec, temp=0.01):
            corr = np.correlate(h_vec, 1 - h_vec, mode="full")
            # Smooth max via log-sum-exp
            max_c = np.max(corr)
            log_sum = np.log(np.sum(np.exp((corr - max_c) / temp)))
            return (max_c + temp * log_sum) / N * 2
        
        # Gradient of the smooth max is expensive; use finite differences or just SLSQP
        # SLSQP with finite differences
        def objective(h_vec):
            return smooth_max_correlation(h_vec)
        
        def objective_grad(h_vec):
            # Use central differences
            eps = 1e-7
            grad = np.zeros_like(h_vec)
            f0 = objective(h_vec)
            for i in range(N):
                h_plus = h_vec.copy()
                h_minus = h_vec.copy()
                h_plus[i] += eps
                h_minus[i] -= eps
                grad[i] = (objective(h_plus) - objective(h_minus)) / (2 * eps)
            return grad
        
        constraints = [
            {'type': 'eq', 'fun': lambda h_vec: np.sum(h_vec) - N/2,
             'jac': lambda h_vec: np.ones(N)}
        ]
        bounds = [(0, 1)] * N
        
        try:
            res = minimize(objective, h, method='SLSQP', jac=objective_grad,
                          bounds=bounds, constraints=constraints,
                          options={'maxiter': 100, 'ftol': 1e-10})
            if res.success and _is_valid(res.x):
                val = _raw_value(res.x)
                if val < best_val - 1e-12:
                    best_val = val
                    best_h = res.x.copy()
            h = res.x if res.success else h
            h = np.clip(h, 0, 1)
            s = np.sum(h)
            if abs(s - N/2) > 1e-12:
                h = h + (N/2 - s) / N
        except Exception:
            break
        
        # Check convergence
        if abs(best_val - _raw_value(h)) < 1e-10:
            break
    
    return best_h

def _generate_candidate(seed):
    """Generate a candidate using sinusoidal basis functions."""
    rng = np.random.RandomState(seed)
    
    # Use a sum of sinusoids with random frequencies and phases
    # This creates smooth patterns that can have low overlap
    n_components = rng.randint(3, 7)  # 3 to 6 sinusoidal components
    
    h = np.full(N, 0.5)  # start at mean
    
    for _ in range(n_components):
        freq = rng.uniform(1, 5)  # frequency in cycles over N
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.1, 0.4)
        h += amp * np.sin(2 * np.pi * freq * np.arange(N) / N + phase)
    
    # Clip to [0, 1]
    h = np.clip(h, 0, 1)
    
    # Project to satisfy sum constraint
    s = np.sum(h)
    if abs(s - N/2) > 1e-12:
        h = h + (N/2 - s) / N
        h = np.clip(h, 0, 1)
        # Re-project if clipping broke the sum
        s = np.sum(h)
        if abs(s - N/2) > 1e-12:
            h = h + (N/2 - s) / N
    
    return h

def construct():
    start_time = time.time()
    
    # Start with the seed
    best_h = np.full(N, 0.5)
    best_val = _raw_value(best_h)  # Should be around 0.5
    
    # Multi-start local refinement
    n_starts = 20
    for i in range(n_starts):
        if time.time() > DEADLINE - 10:
            break
        
        if i == 0:
            h_candidate = best_h.copy()
        else:
            h_candidate = _generate_candidate(i * 12345 + 67890)
        
        # Local refinement
        h_refined = _project_and_refine(h_candidate)
        
        if _is_valid(h_refined):
            val = _raw_value(h_refined)
            if val < best_val:
                best_val = val
                best_h = h_refined.copy()
        
        # Also try a few more random perturbations around the best
        if i > 0 and i % 5 == 0:
            for _ in range(3):
                if time.time() > DEADLINE - 10:
                    break
                perturbation = np.random.RandomState(i * 100 + _).normal(0, 0.1, N)
                h_pert = best_h + perturbation
                h_pert = np.clip(h_pert, 0, 1)
                s = np.sum(h_pert)
                if abs(s - N/2) > 1e-12:
                    h_pert = h_pert + (N/2 - s) / N
                    h_pert = np.clip(h_pert, 0, 1)
                
                h_refined = _project_and_refine(h_pert)
                if _is_valid(h_refined):
                    val = _raw_value(h_refined)
                    if val < best_val:
                        best_val = val
                        best_h = h_refined.copy()
    
    # Final validation
    if not _is_valid(best_h):
        # Fallback to seed
        best_h = np.full(N, 0.5)
    
    return best_h
# EVOLVE-BLOCK-END
