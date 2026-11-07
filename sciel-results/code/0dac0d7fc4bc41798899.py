"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 100

# EVOLVE-BLOCK-START

def _objective(h):
    """Compute the evaluation metric: max(np.correlate(h, 1-h, mode='full')) / N * 2"""
    corr = np.correlate(h, 1 - h, mode="full")
    return float(np.max(corr)) / N * 2

def _smooth_objective(h, beta=50.0):
    """Smooth surrogate using log-sum-exp for the max."""
    corr = np.correlate(h, 1 - h, mode="full")
    # Scale for numerical stability
    max_corr = np.max(corr)
    # Soft max: (1/beta) * log(sum(exp(beta * (corr - max_corr)))) + max_corr
    # But we want the actual max approximated
    shifted = beta * (corr - max_corr)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -100, 100)
    logsumexp = max_corr + (1.0 / beta) * np.log(np.sum(np.exp(shifted)))
    return logsumexp / N * 2

def _construct_with_optimization():
    """Optimize the heights to minimize the max correlation."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Initial guess: all 0.5
    h0 = np.full(N, 0.5)
    
    # Constraints:
    # 1. Sum(h) = N/2 = 50
    # 2. 0 <= h[i] <= 1
    A_eq = np.ones((1, N))
    b_eq = np.array([N / 2.0])
    
    A_ub = np.eye(N)
    b_ub = np.ones(N)
    A_lb = -np.eye(N)
    b_lb = -np.zeros(N)
    
    # Try SLSQP first with smooth objective
    best_h = h0.copy()
    best_val = _objective(h0)
    
    # Multiple random restarts to escape local minima
    rng = np.random.default_rng(42)
    
    # First try from the seed
    try:
        res = minimize(
            _smooth_objective,
            h0,
            args=(50.0,),
            method='SLSQP',
            bounds=[(0, 1)] * N,
            constraints=[
                {'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                 'jac': lambda x: np.ones(N)}
            ],
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        if res.success and res.x is not None:
            h_candidate = np.clip(res.x, 0, 1)
            # Project to satisfy sum constraint exactly
            h_candidate = h_candidate - (np.sum(h_candidate) - N/2.0) / N
            h_candidate = np.clip(h_candidate, 0, 1)
            val = _objective(h_candidate)
            if val < best_val:
                best_val = val
                best_h = h_candidate.copy()
    except Exception:
        pass
    
    # Try a few random restarts if time permits
    n_restarts = 5
    for i in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate a random feasible starting point
        h_rand = rng.uniform(0, 1, N)
        h_rand -= (np.sum(h_rand) - N/2.0) / N
        h_rand = np.clip(h_rand, 0, 1)
        # Re-project to satisfy sum exactly
        diff = N/2.0 - np.sum(h_rand)
        if diff > 0:
            # Add to elements that can accept more
            mask = h_rand < 1.0
            if np.any(mask):
                add_per = diff / np.sum(mask)
                h_rand[mask] += add_per
                h_rand = np.clip(h_rand, 0, 1)
        else:
            mask = h_rand > 0.0
            if np.any(mask):
                sub_per = -diff / np.sum(mask)
                h_rand[mask] -= sub_per
                h_rand = np.clip(h_rand, 0, 1)
        
        try:
            res = minimize(
                _smooth_objective,
                h_rand,
                args=(50.0,),
                method='SLSQP',
                bounds=[(0, 1)] * N,
                constraints=[
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                     'jac': lambda x: np.ones(N)}
                ],
                options={'maxiter': 300, 'ftol': 1e-12}
            )
            if res.success and res.x is not None:
                h_candidate = np.clip(res.x, 0, 1)
                h_candidate = h_candidate - (np.sum(h_candidate) - N/2.0) / N
                h_candidate = np.clip(h_candidate, 0, 1)
                val = _objective(h_candidate)
                if val < best_val:
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Final polish: use Nelder-Mead on the exact (nonsmooth) objective
    if time.time() < deadline:
        try:
            res = minimize(
                _objective,
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                         'adaptive': True}
            )
            if res.x is not None:
                h_candidate = np.clip(res.x, 0, 1)
                # Project to satisfy sum
                h_candidate = h_candidate - (np.sum(h_candidate) - N/2.0) / N
                h_candidate = np.clip(h_candidate, 0, 1)
                val = _objective(h_candidate)
                if val < best_val:
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Ensure final result is valid
    final_h = np.clip(best_h, 0, 1)
    final_h = final_h - (np.sum(final_h) - N/2.0) / N
    final_h = np.clip(final_h, 0, 1)
    
    return final_h

def construct():
    return _construct_with_optimization()

# EVOLVE-BLOCK-END
