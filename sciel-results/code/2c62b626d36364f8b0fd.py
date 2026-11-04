"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit

    # Helper to compute cost
    def calc_cost(p):
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))

    # 1. Initialization: Rank Matching
    # Facilities with high flow sums should be placed at locations with low distance sums (central)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Sort facilities by descending flow sum
    heavy_order = np.argsort(-a_row_sums, kind="stable")
    # Sort locations by ascending distance sum (most central first)
    central_order = np.argsort(b_row_sums, kind="stable")
    
    # Create initial permutation p such that p[i] is the location for facility i
    p = np.empty(n, dtype=np.int64)
    for fac, loc in zip(heavy_order, central_order):
        p[fac] = loc

    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost

    # 2. Continuous Relaxation Phase
    # We minimize f(X) = Tr(X^T A X B) subject to X being doubly stochastic.
    # Gradient: G = A X B + A^T X B^T. If A, B symmetric: G = 2 (A X B).
    # We use a simple projected gradient descent where projection is done via
    # alternating row/col normalization (Sinkhorn-Knopp) which is fast and robust.
    
    # Start from the permutation matrix
    X = np.zeros((n, n))
    for i in range(n):
        X[i, p[i]] = 1.0

    # Adaptive step size
    # Scale is roughly magnitude of A*B. 
    # Let's estimate a safe step size.
    # max_grad = np.max(np.abs(A @ X @ B))
    # step = 1.0 / max_grad
    # But we can just use a fixed small step and decay.
    
    # Precompute A and B as float64 for gradient calculation
    A_float = a.astype(np.float64)
    B_float = b.astype(np.float64)
    
    # Sinkhorn normalization helper
    def normalize(X):
        # Normalize rows
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        X /= row_sums
        # Normalize cols
        col_sums = X.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        X /= col_sums
        return X

    # Gradient Descent Loop
    # We run this for a fixed number of steps or until time runs out (leaving time for polishing)
    # Time budget allocation: 60% for continuous, 40% for discrete polishing
    
    t_cont_end = start_time + (deadline - start_time) * 0.6
    
    # Initial step size
    # Let's estimate the scale of the gradient
    # grad = 2 * (A @ X @ B)
    # We want step * ||grad|| to be small relative to the cost landscape.
    # A rough heuristic: step = 1 / max_sum_A_B
    max_val = np.max(np.abs(A_float)) + np.max(np.abs(B_float))
    if max_val == 0:
        step = 0.1
    else:
        step = 1.0 / (max_val * n) * 10.0 # Tuned heuristic
        
    # Let's try a simpler adaptive approach:
    # Start with a moderate step, reduce if oscillating.
    step = 0.5 / (np.max(np.abs(A_float)) * np.max(np.abs(B_float)) * n)
    if step == 0 or not np.isfinite(step):
        step = 0.001

    # To make it faster, we can batch the normalization.
    # However, simple Sinkhorn is O(n^2) per iteration.
    # n=150, n^2=22500. 1000 iters = 22.5M ops. Very fast.
    
    # We will run until time limit for this phase.
    last_cost = np.inf
    no_improve_count = 0
    
    while time.time() < t_cont_end:
        # Compute Gradient: G = A X B + A^T X B^T
        # Assuming symmetry for speed, but let's be safe:
        # G = A @ X @ B + A.T @ X @ B.T
        AXB = A_float @ X @ B_float
        ATXBT = A_float.T @ X @ B_float.T
        G = AXB + ATXBT
        
        # Update X
        X = X - step * G
        
        # Clip to avoid negatives (Sinkhorn can handle small negatives but it's better to clip)
        X = np.maximum(X, 0)
        
        # Normalize to doubly stochastic
        X = normalize(X)
        
        # Check convergence / stagnation
        # Cost in continuous space: Tr(X^T A X B) = sum(A * (X @ B @ X.T))? 
        # Actually f(X) = sum_{i,j,k,l} A[i,j] X[i,k] B[k,l] X[j,l]
        # = sum_{i,j} A[i,j] (X B X^T)[i,j]
        # Let's compute this to monitor.
        XBXt = X @ B_float @ X.T
        cont_cost = np.sum(A_float * XBXt)
        
        if cont_cost < last_cost:
            last_cost = cont_cost
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 100:
                # Reduce step size if stuck
                step *= 0.9
                no_improve_count = 0
                if step < 1e-8:
                    break

    # 3. Discrete Polishing Phase
    # Convert X back to a permutation using Hungarian Algorithm
    # We want to minimize sum A[i,j] B[p[i], p[j]].
    # This is not a linear assignment on a single matrix.
    # However, we can use the continuous solution X to guide the initialization.
    # A common heuristic: Use the Hungarian algorithm on a matrix derived from X?
    # Or simply use the "closest" permutation to X.
    # The problem is minimizing a quadratic form over permutations.
    # We can use the linearized approximation:
    # At the current best permutation p, the gradient for facility i is:
    # g_i(k) = sum_j A[i,j] B[k, p[j]] + sum_j A[j,i] B[p[j], k] (if symmetric, 2 * sum_j A[i,j] B[k, p[j]])
    # This gives a linear assignment problem: assign facility i to location k minimizing g_i(k).
    # This is essentially one step of Block Coordinate Descent / Linearized Assignment.
    
    # Let's implement a loop of:
    # 1. Compute linearized cost matrix C where C[i, k] = sum_j A[i,j] B[k, p[j]]
    # 2. Solve linear assignment problem on C to get new p.
    # 3. If cost improves, accept. Else, break or do local search.
    
    # First, get an initial discrete p from X.
    # We can just use the previous best_p, or try to extract a permutation from X.
    # Extracting a permutation from a doubly stochastic matrix is itself a hard problem (assignment).
    # We can use the Hungarian algorithm on -X (maximize sum X[i,p[i]]) to get the permutation closest to X.
    # This gives a good starting point for the linearized search.
    
    # Hungarian on -X
    row_ind, col_ind = linear_sum_assignment(-X)
    p_new = np.empty(n, dtype=np.int64)
    for i, j in zip(row_ind, col_ind):
        p_new[i] = j
        
    cost_new = calc_cost(p_new)
    if cost_new < best_cost:
        best_cost = cost_new
        best_p = p_new.copy()
        current_p = p_new.copy()
        current_cost = cost_new
    else:
        current_p = best_p.copy()
        current_cost = best_cost

    # Now, perform Linearized Assignment Descent (LAD)
    # This is a powerful method for QAP.
    
    while time.time() < deadline:
        # Compute linearized cost matrix
        # C[i, k] = sum_j A[i, j] B[k, p[j]]
        # B_p = B[:, p] is the matrix where column j is B[:, p[j]]
        # We want C = A @ B_p ?
        # (A @ B_p)[i, k] = sum_j A[i, j] B_p[j, k] = sum_j A[i, j] B[p[j], k]
        # Since B is symmetric, B[p[j], k] = B[k, p[j]].
        # So C = A @ B[:, p]
        
        B_p = B_float[:, current_p]
        C = A_float @ B_p
        
        # Solve Linear Assignment Problem
        # We want to minimize sum_i C[i, p[i]]
        row_ind, col_ind = linear_sum_assignment(C)
        
        p_candidate = np.empty(n, dtype=np.int64)
        for i, j in zip(row_ind, col_ind):
            p_candidate[i] = j
            
        cost_candidate = calc_cost(p_candidate)
        
        if cost_candidate < current_cost:
            current_p = p_candidate
            current_cost = cost_candidate
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p_candidate.copy()
        else:
            # No improvement from global linearized step.
            # We might be stuck. 
            # Let's try a few random swaps or a local search around current_p.
            # Given the time, a simple 2-opt local search using the vectorized delta from the original code
            # is a good fallback to polish the solution.
            
            # Switch to local search mode for remaining time
            # We can reuse the logic from the seed but adapted to run until deadline.
            
            # To avoid infinite loop if no improvement is possible, we can break or do random restarts.
            # Let's do a limited number of local search steps.
            
            local_steps = 0
            max_local_steps = 100 # Limit to ensure we don't overshoot deadline
            
            # Precompute for fast delta
            # We use the vectorized swap delta logic
            
            while local_steps < max_local_steps and time.time() < deadline:
                best_delta = 0
                best_i = -1
                best_j = -1
                
                # Vectorized search for best swap
                # To save time, we might not check all i every step?
                # With n=150, checking all i is 150 iterations.
                # Each iteration involves O(n^2) operations (vectorized).
                # 150 * (150*150) = 3.3M ops per full scan.
                # In NumPy, this is very fast (< 10ms).
                
                for i in range(n - 1):
                    u = current_p[i]
                    ai = A_float[i]
                    bu = B_float[:, u]
                    
                    # Indices j > i
                    js = np.arange(i + 1, n)
                    vs = current_p[js]
                    ajs = A_float[js]
                    bvs = B_float[:, vs]
                    
                    # Compute terms
                    ai_bu = np.dot(ai, bu)
                    aj_bu = ajs @ bu
                    ai_bv = ai @ bvs
                    aj_bv = np.sum(ajs * bvs.T, axis=1)
                    
                    deltas = ai_bv - ai_bu - aj_bv + aj_bu
                    
                    min_idx_local = np.argmin(deltas)
                    min_delta_local = deltas[min_idx_local]
                    
                    if min_delta_local < best_delta:
                        best_delta = min_delta_local
                        best_i = i
                        best_j = js[min_idx_local]
                
                if best_i == -1 or best_delta >= 0:
                    # No improvement found in 2-opt
                    break
                
                # Apply swap
                current_p[best_i], current_p[best_j] = current_p[best_j], current_p[best_i]
                current_cost += int(best_delta)
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = current_p.copy()
                
                local_steps += 1
            
            # If we broke out of local search, we can try to restart LAD with a perturbed solution
            # or just break if we are close to deadline.
            if time.time() > deadline - 5.0:
                break
            
            # Perturb to escape local minimum for next LAD iteration
            # Swap two random facilities
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i != j:
                current_p[i], current_p[j] = current_p[j], current_p[i]
                current_cost = calc_cost(current_p)
                # Don't update best_p here, just continue searching

    return best_p.tolist()

# EVOLVE-BLOCK-END
