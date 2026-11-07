import numpy as np
from scipy.optimize import minimize
import time

def construct():
    N = 100
    deadline = time.time() + 145
    
    def compute_corr(h):
        return np.correlate(h, 1 - h, mode="full")
    
    def objective(h, lam):
        corr = compute_corr(h)
        # Soft max
        # To avoid overflow, shift by max
        m = np.max(corr)
        return (1/lam) * np.log(np.sum(np.exp(lam * (corr - m)))) + m
        
    def objective_grad(h, lam):
        # Numerical gradient is too slow for SLSQP with large N?
        # SLSQP uses finite differences if no grad provided.
        # N=100, so 100 function evals per step.
        # Might be okay.
        pass

    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2}]
    bounds = [(0, 1)] * N
    
    best_h = None
    best_val = np.inf
    
    # Seed: Anti-periodic cosine
    i = np.arange(N)
    h_seed = 0.5 + 0.5 * np.cos(2 * np.pi * i / (N/2))
    # h_seed has period 50.
    # Wait, cos(2 pi i / 50) has period 50.
    # h_seed[i+50] = h_seed[i].
    # This is the periodic case which had peak at k=50?
    # No, for periodic case, C_50 = sum h_i (1-h_i).
    # For cosine, this is ~16.
    # But we found C_k approx 25 - 50 cos(omega k).
    # Max at cos = -1 -> k=25 (half period).
    # So peak at k=25.
    
    # Let's try a different seed: Random
    rng = np.random.RandomState(42)
    h_random = rng.rand(N)
    h_random = h_random / np.sum(h_random) * N/2
    
    # Progressive lambda
    lambdas = [1.0, 5.0, 10.0, 20.0, 50.0]
    
    current_h = h_seed.copy()
    
    for lam in lambdas:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(
                lambda h: objective(h, lam),
                current_h,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 50, 'ftol': 1e-6, 'disp': False}
            )
            if res.success:
                current_h = res.x
        except Exception:
            pass
            
    # Final evaluation
    corr = compute_corr(current_h)
    max_corr = np.max(corr)
    best_val = max_corr / N * 2
    best_h = current_h
    
    # Try a few random restarts with high lambda
    for start in range(3):
        if time.time() > deadline - 10:
            break
        h0 = rng.rand(N)
        h0 = h0 / np.sum(h0) * N/2
        try:
            res = minimize(
                lambda h: objective(h, 20.0),
                h0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 50, 'ftol': 1e-6, 'disp': False}
            )
            if res.success:
                h_cand = res.x
                corr_c = compute_corr(h_cand)
                val_c = np.max(corr_c) / N * 2
                if val_c < best_val:
                    best_val = val_c
                    best_h = h_cand
        except Exception:
            pass
            
    # Ensure validity
    best_h = np.clip(best_h, 0, 1)
    best_h = best_h / np.sum(best_h) * N/2
    
    return best_h
