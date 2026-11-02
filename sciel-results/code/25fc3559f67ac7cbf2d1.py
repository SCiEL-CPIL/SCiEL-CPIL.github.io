import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    # Hypothesis: The score is C = (||h||_1 * ||h*h||_inf) / ||h||_2^2
    # We maximize this.
    # Due to scale invariance of C, we can fix ||h||_2 = 1.
    # Then we maximize ||h||_1 * ||h*h||_inf.
    
    def compute_score(h):
        h = np.array(h)
        l2_sq = np.sum(h**2)
        if l2_sq == 0:
            return 0
        l1 = np.sum(h)
        conv = np.convolve(h, h, mode='full')
        max_conv = np.max(conv)
        return (l1 * max_conv) / l2_sq

    # Initial guess: Uniform distribution over a subset
    # Uniform over all 50
    h0 = np.ones(N) / np.sqrt(N)
    
    def objective(params):
        # params are the first N-1 values, last is determined by sphere constraint?
        # Simpler: use L-BFGS-B on unconstrained and project?
        # Or use SLSQP with constraint.
        h = np.zeros(N)
        h[:N-1] = params
        # Calculate last element to satisfy sum h^2 = 1
        s = np.sum(params**2)
        if s >= 1:
            # Project back
            h[:N-1] = params / np.sqrt(s)
            h[N-1] = 0
        else:
            h[N-1] = np.sqrt(1 - s)
        # Ensure non-negative
        h = np.abs(h)
        # Re-normalize
        norm = np.linalg.norm(h)
        if norm > 0:
            h = h / norm
        return -compute_score(h)

    # Use a simpler optimization: Coordinate Descent or Random Restarts
    # Given time budget, try a few random restarts and local polish
    
    best_score = 0
    best_h = h0
    
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    # Try uniform
    best_h = np.ones(N) / np.sqrt(N)
    best_score = compute_score(best_h)
    
    # Try localized uniform (blocks)
    for block_size in [2, 5, 10, 20, 25, 50]:
        h_test = np.zeros(N)
        h_test[:block_size] = 1.0 / np.sqrt(block_size)
        score_test = compute_score(h_test)
        if score_test > best_score:
            best_score = score_test
            best_h = h_test.copy()
            
    # Local optimization using scipy
    # We optimize over h directly, penalizing deviation from unit L2 norm
    
    def obj_constrained(h):
        h = np.abs(h)
        norm = np.linalg.norm(h)
        if norm == 0: return 0
        h_norm = h / norm
        return -compute_score(h_norm)
    
    # Run a few local optimizations from different starts
    starts = [
        np.ones(N) / np.sqrt(N),
        np.zeros(N), # Will be handled by abs/norm, maybe bad start
        np.array([1.0] + [0.0]*(N-1)),
        np.array([1.0/np.sqrt(2)]*2 + [0.0]*(N-2)),
        np.array([1.0/np.sqrt(10)]*10 + [0.0]*(N-10))
    ]
    
    for h_start in starts:
        if time.time() > deadline:
            break
        
        # Normalize start
        norm = np.linalg.norm(h_start)
        if norm == 0:
            continue
        h_start = h_start / norm
        
        res = minimize(obj_constrained, h_start, method='L-BFGS-B', 
                       bounds=[(0, 10)]*N, 
                       options={'maxiter': 500, 'ftol': 1e-9})
        
        h_opt = np.abs(res.x)
        norm = np.linalg.norm(h_opt)
        if norm > 0:
            h_opt = h_opt / norm
            score_opt = compute_score(h_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_h = h_opt.copy()
                
    return list(best_h)
