import numpy as np
from scipy.optimize import minimize
import time

N = 100

def compute_overlap(h):
    """Compute max over shifts k of sum_i h[i] * (1 - h[(i-k) % N]) / N"""
    N = len(h)
    max_overlap = 0.0
    for k in range(N):
        shifted = np.roll(h, k)
        overlap = np.sum(h * (1 - shifted)) / N
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def construct():
    deadline = time.time() + 140  # safety margin
    
    # Start with constant solution
    h0 = np.full(N, 0.5)
    
    # Try to optimize
    def objective(x):
        # x are the heights, need to satisfy sum = N/2 and 0 <= x <= 1
        # Use penalty for constraint violation
        h = x
        penalty = 0.0
        if np.sum(h) != N/2:
            penalty += 100 * (np.sum(h) - N/2)**2
        penalty += 100 * np.sum(np.maximum(0, -h)**2) + 100 * np.sum(np.maximum(0, h-1)**2)
        return compute_overlap(h) + penalty
    
    # Use SLSQP
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2}]
    bounds = [(0, 1)] * N
    
    best_h = h0.copy()
    best_score = compute_overlap(best_h)
    
    start = time.time()
    try:
        result = minimize(objective, h0, method='SLSQP', bounds=bounds, 
                         constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
        if result.success and result.fun < best_score:
            # Project back to feasible set
            h = result.x
            # Ensure sum is exactly N/2
            diff = N/2 - np.sum(h)
            h = h + diff / N
            # Clip to [0,1]
            h = np.clip(h, 0, 1)
            # Re-normalize
            if np.sum(h) > 0:
                h = h * (N/2) / np.sum(h)
                h = np.clip(h, 0, 1)
            score = compute_overlap(h)
            if score < best_score:
                best_h = h
                best_score = score
    except Exception:
        pass
    
    # Check deadline
    if time.time() > deadline - 5:
        return best_h
    
    # Try multiple random restarts if time permits
    np.random.seed(42)
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        # Random initial point
        h_rand = np.random.uniform(0, 1, N)
        h_rand = h_rand * (N/2) / np.sum(h_rand)
        h_rand = np.clip(h_rand, 0, 1)
        # Adjust to get exact sum
        diff = N/2 - np.sum(h_rand)
        h_rand = h_rand + diff / N
        h_rand = np.clip(h_rand, 0, 1)
        
        try:
            result = minimize(objective, h_rand, method='SLSQP', bounds=bounds,
                             constraints=constraints, options={'maxiter': 200, 'ftol': 1e-10})
            if result.success:
                h = result.x
                diff = N/2 - np.sum(h)
                h = h + diff / N
                h = np.clip(h, 0, 1)
                score = compute_overlap(h)
                if score < best_score:
                    best_h = h
                    best_score = score
        except Exception:
            pass
    
    # Final: ensure exact sum
    diff = N/2 - np.sum(best_h)
    best_h = best_h + diff / N
    best_h = np.clip(best_h, 0, 1)
    
    return best_h
