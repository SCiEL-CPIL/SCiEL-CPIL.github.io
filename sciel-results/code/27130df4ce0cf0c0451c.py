"""
Strategy: Continuous Optimization (L-BFGS-B) + Discrete Refinement.

1. Precompute the AP incidence matrix M (dense float32).
2. Define a smooth surrogate loss: L(x) = 0.5 * || M @ x ||_2^2, where x is in [-1, 1]^N.
   Gradient: (M.T @ (M @ x))
3. Use scipy.optimize.minimize (L-BFGS-B) to find a continuous vector x that minimizes L.
   This helps navigate the landscape by using gradient information, potentially finding
   better basins than random discrete starts.
4. Discretize x to s (signs of x_i, with 0 mapped to random or previous best).
5. Perform discrete local search (greedy flips) to refine s, starting from the 
   discretized continuous solution.
6. Return the best discrete solution found.
"""

import time
import numpy as np
from scipy.optimize import minimize

N = 256

def _get_ap_indices():
    """Precompute all arithmetic progressions in [1, N] as a list of index lists."""
    aps = []
    for step in range(1, N // 2 + 1):
        for start in range(1, N - step + 1):
            max_len = (N - start) // step + 1
            if max_len < 2:
                continue
            indices = [start - 1 + k * step for k in range(max_len)]
            for length in range(2, max_len + 1):
                aps.append(indices[:length])
    return aps

def _precompute_ap_matrix(aps):
    """Build dense incidence matrix M (num_aps x N) in float32."""
    num_aps = len(aps)
    M = np.zeros((num_aps, N), dtype=np.float32)
    for i, idxs in enumerate(aps):
        for idx in idxs:
            M[i, idx] = 1.0
    return M

def _compute_max_abs_sum(signs, M):
    """Compute max abs sum using matrix multiplication."""
    sums = M @ signs
    return np.max(np.abs(sums))

def _find_best_flip(signs, M, current_max):
    """
    Find the single flip that reduces the max abs sum the most.
    """
    S = M @ signs
    best_k = -1
    best_max = current_max
    
    for k in range(N):
        delta = 2.0 * signs[k] * M[:, k]
        new_S = S + delta
        new_max = np.max(np.abs(new_S))
        if new_max < best_max:
            best_max = new_max
            best_k = k
            
    return best_k, best_max

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 90s limit, keep 5s margin
    
    # 1. Precompute APs and Matrix
    aps = _get_ap_indices()
    M = _precompute_ap_matrix(aps)
    num_aps = len(aps)
    
    # Precompute M.T @ M for gradient computation? 
    # grad = M.T @ (M @ x) = (M.T @ M) @ x
    # M.T @ M is N x N (256 x 256). This is very fast to compute and store.
    MTM = M.T @ M
    
    def loss_and_grad(x):
        # x is continuous vector in [-1, 1]^N
        # Loss = 0.5 * || M @ x ||^2
        # Grad = M.T @ (M @ x)
        # We can compute M @ x (num_aps,) then M.T @ (M @ x) (N,)
        # Or use MTM @ x (N,)
        # MTM @ x is faster if MTM is precomputed.
        # But M @ x is needed for loss value.
        # Let's just do M @ x -> sums, then loss = 0.5 * dot(sums, sums), grad = M.T @ sums
        # Actually, grad = MTM @ x is equivalent and avoids the large intermediate vector if we only need grad.
        # But we need loss too.
        # Let's compute sums = M @ x.
        sums = M @ x
        loss = 0.5 * np.dot(sums, sums)
        # grad = M.T @ sums
        # Since M is float32, let's keep consistent.
        grad = MTM @ x
        return loss, grad

    best_signs = None
    best_val = float('inf')
    
    # 2. Continuous Optimization Phase
    # We will try a few different initializations for the continuous optimizer.
    # L-BFGS-B is bounded, so we use bounds [(-1, 1)] for all variables.
    bounds = [(-1.0, 1.0)] * N
    
    initializations = []
    
    # Random initializations
    rng = np.random.default_rng(42)
    for _ in range(3):
        initializations.append(rng.uniform(-1, 1, size=N))
        
    # Thue-Morse (discrete, but valid start)
    tm = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(N)], dtype=np.float64)
    initializations.append(tm)
    
    # Alternating
    alt = np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=np.float64)
    initializations.append(alt)
    
    # Zero (center)
    initializations.append(np.zeros(N))
    
    for x0 in initializations:
        if time.time() > deadline - 10.0:
            break
            
        try:
            # L-BFGS-B
            # maxiter is limited to ensure we don't spend too long per run
            # We can adjust maxiter based on time remaining, but a fixed reasonable number is safer.
            # 1000 iterations of L-BFGS-B on 256 vars with this cost is likely fast enough.
            res = minimize(
                loss_and_grad, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            x_opt = res.x
            
            # Discretize
            # Map to signs. If x_i is 0, pick randomly or keep previous?
            # Let's pick sign based on value, 0 -> 1 (arbitrary).
            s = np.sign(x_opt)
            s[s == 0] = 1.0
            
            # Ensure float32 for M @ s
            s = s.astype(np.float32)
            
            val = _compute_max_abs_sum(s, M)
            
            if val < best_val:
                best_val = val
                best_signs = s.copy()
                
        except Exception:
            continue

    # 3. Discrete Local Search Refinement
    if best_signs is None:
        # Fallback if continuous phase failed completely
        best_signs = np.array([1.0] * N, dtype=np.float32)
        best_val = _compute_max_abs_sum(best_signs, M)
        
    current_signs = best_signs
    current_max = best_val
    
    # Greedy Local Search
    # We can do multiple passes or until no improvement.
    # Since we started from a better continuous solution, we might converge faster.
    
    while time.time() < deadline - 2.0:
        best_k, new_max = _find_best_flip(current_signs, M, current_max)
        
        if best_k == -1:
            break
            
        current_signs[best_k] *= -1
        current_max = new_max
        
    # 4. Return result
    result = [int(x) for x in current_signs]
    return result
