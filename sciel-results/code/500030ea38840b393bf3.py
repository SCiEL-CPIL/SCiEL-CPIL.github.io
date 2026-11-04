"""
Solver for QAPLIB wil100 using Linearization-Based Heuristic (LBH) 
with Iterative Surrogate Update and 2-Opt Polish.

Strategy:
1. Initialize with a doubly stochastic matrix X (uniform or spectral).
2. Iteratively:
   a. Compute linearized cost matrix C based on current X.
   b. Solve assignment problem on C to get permutation P.
   c. Update X by mixing with P.
   d. Normalize X (Sinkhorn or simple row/col scaling) to maintain properties.
3. Track the best permutation found during iterations.
4. Apply a fast 2-Opt local search to the best permutation to polish.
"""
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j):
    """
    Compute the change in cost when swapping positions i and j in permutation p.
    O(n) vectorized computation.
    """
    pi = p[i]
    pj = p[j]
    
    # Direct terms (i,j pair)
    delta = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    # Diagonal terms
    delta += a[i, i] * (b[pj, pj] - b[pi, pi])
    delta += a[j, j] * (b[pi, pi] - b[pj, pj])
    
    # Cross terms with all other k
    # Term 1: sum_{k != i,j} (a[i,k] - a[j,k]) * (b[pj, p[k]] - b[pi, p[k]])
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_T1 = np.dot(diff_a_row, diff_b_col)
    # Subtract k=i and k=j contributions
    sum_T1 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    sum_T1 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    
    # Term 2: sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k], pj] - b[p[k], pi])
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_T2 = np.dot(diff_a_col, diff_b_row)
    # Subtract k=i and k=j contributions
    sum_T2 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    sum_T2 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    
    delta += sum_T1 + sum_T2
    return int(delta)

def _linearization_solve(a, b, time_limit, initial_x=None):
    """
    Perform iterative linearization of the QAP.
    X is a doubly stochastic matrix.
    C = A X B + A^T X B^T (linearization of the QAP objective gradient)
    Solve assignment on C to get P.
    Update X = (1-alpha)X + alpha P.
    """
    n = len(a)
    if initial_x is None:
        # Initialize X as uniform doubly stochastic matrix
        X = np.ones((n, n)) / n
    else:
        X = initial_x.copy()
    
    best_p = None
    best_cost = np.inf
    start_time = time.time()
    
    # Adaptive step size
    alpha = 0.5
    stagnation_count = 0
    min_alpha = 0.05
    
    while time.time() - start_time < time_limit:
        # 1. Compute Linearized Cost Matrix
        # The objective is tr(A^T P B P^T) = sum_{i,j,k,l} a_{ij} b_{kl} p_{ik} p_{jl}
        # Linearization at X: sum_{i,j,k,l} a_{ij} b_{kl} x_{ik} p_{jl} + ... 
        # A common surrogate is C = A X B + A^T X B^T
        # Let's verify dimensions:
        # A (n,n), X (n,n), B (n,n)
        # A X B: (n,n) @ (n,n) @ (n,n) -> (n,n)
        # Term (i,j) of A X B is sum_{k,l} a_{ik} x_{kl} b_{lj}
        # This matches the structure of the QAP linearization.
        
        C = a @ X @ b + a.T @ X @ b.T
        
        # 2. Solve Assignment Problem
        # We want to minimize sum_{i,j} C[i,j] * P[i,j]
        # linear_sum_assignment minimizes the sum of C[row, col]
        row_ind, col_ind = linear_sum_assignment(C)
        
        P = np.zeros((n, n))
        P[row_ind, col_ind] = 1.0
        
        # Convert to permutation list
        p_list = col_ind.copy() # p[facility] = location
        
        # 3. Evaluate Cost
        cost = _qap_cost(a, b, p_list)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p_list.copy()
            stagnation_count = 0
        else:
            stagnation_count += 1
            # If stuck, reduce alpha to be more exploratory or reset X?
            # Actually, if stuck, we might want to increase alpha to jump more?
            # Or decrease alpha to refine?
            # Let's decrease alpha to allow finer adjustments if we are oscillating.
            if stagnation_count > 5:
                alpha = max(min_alpha, alpha * 0.9)
        
        # 4. Update X
        # X = (1 - alpha) * X + alpha * P
        X = (1 - alpha) * X + alpha * P
        
        # 5. Sinkhorn Normalization (to keep X doubly stochastic)
        # Simple iterative scaling
        for _ in range(5): # 5 iterations is usually enough for stability
            # Row normalize
            row_sums = X.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1e-10
            X = X / row_sums
            # Col normalize
            col_sums = X.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1e-10
            X = X / col_sums
        
        # Check time
        if time.time() - start_time > time_limit:
            break

    return best_p, best_cost

def _two_opt_polish(a, b, p, time_limit):
    """
    Fast 2-opt local search.
    Uses vectorized delta calculation.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    start_time = time.time()
    
    # Precompute row/col sums for potential speedup? 
    # The delta function is O(n). 100*99/2 = 4950 checks per pass.
    # 4950 * 100 ops = ~500k ops per pass. 
    # In Python, this might be slow. Let's try to vectorize the pass.
    
    # Vectorized Best Improvement Pass
    # This is complex to vectorize fully for all pairs.
    # Instead, we do a "First Improvement" with randomization or a "Best Improvement" 
    # but limited to a subset of pairs if time is tight.
    
    # Given the 120s limit, and n=100, a full best-improvement pass takes ~0.5-1s in NumPy if optimized.
    # Let's implement a loop that does best-improvement passes until no improvement or time runs out.
    
    while time.time() - start_time < time_limit:
        improved = False
        best_delta = 0
        best_i, best_j = -1, -1
        
        # We can speed this up by checking pairs in a random order and stopping early 
        # if we find a large improvement? No, best-improvement is safer for convergence.
        # However, for n=100, 4950 pairs is small.
        
        for i in range(n):
            for j in range(i + 1, n):
                delta = _swap_delta(a, b, p, i, j)
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j
        
        if best_i >= 0:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            improved = True
        else:
            break
            
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    deadline = 115.0  # Safety margin
    start = time.time()
    
    np.random.seed(42)
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: Linearization Heuristic
    # Allocate 80% of time to this phase
    phase1_time = 90.0
    p_lin, cost_lin = _linearization_solve(a, b, phase1_time)
    
    if p_lin is not None:
        best_p = p_lin
        best_cost = cost_lin
    
    # Phase 2: Polish with 2-Opt
    time_remaining = deadline - (time.time() - start)
    if time_remaining > 5 and best_p is not None:
        p_polished, cost_polished = _two_opt_polish(a, b, best_p, time_remaining - 2)
        if cost_polished < best_cost:
            best_cost = cost_polished
            best_p = p_polished
    
    # Fallback: If linearization failed or returned None
    if best_p is None:
        # Fallback to simple random + local search
        p_rand = np.arange(n)
        np.random.shuffle(p_rand)
        p_polished, cost_polished = _two_opt_polish(a, b, p_rand, time_remaining - 2)
        best_p = p_polished
        best_cost = cost_polished

    return best_p.tolist()
