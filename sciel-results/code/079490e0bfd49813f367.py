"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Define two fixed binary patterns:
    # Pattern A: alternating 0,1,0,1,... (balanced)
    # Pattern B: blocks of 0s and 1s (high-contrast)
    pattern_A = np.zeros(N)
    pattern_A[::2] = 1.0  # 1 at even indices
    
    pattern_B = np.zeros(N)
    # Create a different pattern: e.g., 3 ones followed by 2 zeros, repeating
    # This gives a different autocorrelation structure
    for i in range(N):
        if (i % 5) < 3:
            pattern_B[i] = 1.0
    
    # h = alpha * pattern_A + beta * pattern_B
    # Constraints:
    # 1. sum(h) = N/2
    # 2. 0 <= h[i] <= 1 for all i
    
    sum_A = pattern_A.sum()  # 50
    sum_B = pattern_B.sum()  # 60 (3/5 of 100)
    
    # From sum constraint: alpha * sum_A + beta * sum_B = N/2 = 50
    # So: 50*alpha + 60*beta = 50
    # => alpha = (50 - 60*beta) / 50 = 1 - 1.2*beta
    
    # Bounds: 0 <= alpha <= 1, 0 <= beta <= 1
    # 0 <= 1 - 1.2*beta <= 1 => 0 <= beta <= 1/1.2 ≈ 0.8333
    
    # Also need 0 <= h[i] <= 1 for all i
    # h[i] = alpha * A[i] + beta * B[i]
    # For i where A[i]=1, B[i]=0: h[i] = alpha
    # For i where A[i]=0, B[i]=1: h[i] = beta  
    # For i where A[i]=1, B[i]=1: h[i] = alpha + beta
    # For i where A[i]=0, B[i]=0: h[i] = 0
    
    # So we need: 0 <= alpha, 0 <= beta, 0 <= alpha+beta <= 1
    
    # From alpha = 1 - 1.2*beta:
    # 0 <= 1 - 1.2*beta => beta <= 5/6 ≈ 0.8333
    # 0 <= beta
    # alpha + beta = 1 - 1.2*beta + beta = 1 - 0.2*beta <= 1 (always true for beta >= 0)
    # alpha + beta = 1 - 0.2*beta >= 0 => beta <= 5 (always true)
    
    # So feasible range: beta in [0, 5/6]
    
    def compute_h(beta):
        alpha = 1.0 - 1.2 * beta
        h = alpha * pattern_A + beta * pattern_B
        return h
    
    def compute_max_overlap(h):
        """Compute the maximum overlap of h with shifted 1-h over all non-zero shifts."""
        h = np.asarray(h)
        n = len(h)
        max_ov = 0.0
        for k in range(1, n):
            shifted = np.roll(h, -k)
            ov = np.sum(h * (1.0 - shifted))
            if ov > max_ov:
                max_ov = ov
        return max_ov
    
    # Optimize over beta in [0, 5/6]
    def objective(beta_arr):
        beta = beta_arr[0]
        if beta < 0 or beta > 5.0/6.0 + 1e-9:
            return 1e10
        h = compute_h(beta)
        # Check bounds
        if np.any(h < -1e-9) or np.any(h > 1.0 + 1e-9):
            return 1e10
        return compute_max_overlap(h)
    
    # Scan a grid of beta values to find good starting points
    beta_grid = np.linspace(0, 5.0/6.0, 100)
    best_beta = beta_grid[0]
    best_score = float('inf')
    
    for b in beta_grid:
        if time.time() > deadline:
            break
        score = objective([b])
        if score < best_score:
            best_score = score
            best_beta = b
    
    # Refine with local optimization
    def run_local_optimization(beta0):
        result = minimize(
            objective, 
            [beta0], 
            method='Nelder-Mead',
            bounds=[(0, 5.0/6.0)],
            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        if result.success and result.fun < best_score:
            return result.x[0], result.fun
        return None, None
    
    # Try local optimization from the best grid point
    beta_opt, score_opt = run_local_optimization(best_beta)
    if beta_opt is not None:
        best_beta = beta_opt
        best_score = score_opt
    
    # Also try a few other starting points
    for b in [0.1, 0.3, 0.5, 0.7]:
        if time.time() > deadline:
            break
        beta_opt, score_opt = run_local_optimization(b)
        if beta_opt is not None and score_opt < best_score:
            best_beta = beta_opt
            best_score = score_opt
    
    # Construct final h
    best_h = compute_h(best_beta)
    
    # Ensure exact sum (should be exact by construction, but verify)
    diff = N / 2.0 - best_h.sum()
    if abs(diff) > 1e-10:
        best_h += diff / N
        best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
