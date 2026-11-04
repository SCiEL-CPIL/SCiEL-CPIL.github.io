"""QAP Solver using Sinkhorn-Knopp Continuous Relaxation and Gradient Descent."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _sinkhorn_step(P, a, b, temp, steps=10):
    """
    Perform Sinkhorn-Knopp normalization to make P doubly stochastic.
    P is the assignment matrix (facilities x locations).
    """
    # P[i, j] ~ probability facility i is at location j
    # We want to minimize sum_{i,j} a[i, j] b[k, l] P[i, k] P[j, l]
    # Gradient descent on the log-sum-exp smoothed assignment energy
    
    # For Sinkhorn, we typically normalize an initial matrix.
    # Here we use a gradient-based update on the logits.
    
    # Let Z be the log-assignment matrix.
    # P = exp(Z) / row_sums, then P = P / col_sums (iteratively)
    
    # To make it differentiable and stable, we work with Z.
    # Update rule for Z to minimize QAP cost:
    # dL/dZ[i, j] = sum_{k, l} a[i, k] b[j, l] P[k, l] P[i, j] (approx)
    # Actually, the gradient of sum_{i,j,k,l} a[i,k] b[j,l] P[i,j] P[k,l] wrt P[i,j] is:
    # G[i,j] = sum_{k,l} a[i,k] b[j,l] P[k,l]
    # Then we update Z[i,j] -= lr * G[i,j]
    # Then re-normalize P from Z.
    
    # However, the prompt suggests "Sinkhorn-Knopp algorithm with a temperature-annealing schedule".
    # Standard Sinkhorn solves the entropy-regularized optimal transport.
    # QAP is: min_P <A, P B P^T> s.t. P in Birkhoff polytope.
    
    # Let's implement a simple gradient descent on the Birkhoff polytope using Sinkhorn normalization.
    
    # 1. Compute Gradient G[i, j] = sum_{k, l} a[i, k] * b[j, l] * P[k, l]
    # Note: b is symmetric in QAP usually, but let's assume general.
    # G = A @ P @ B.T ?
    # Let's check dimensions.
    # A: (n, n) facilities
    # B: (n, n) locations
    # P: (n, n) facilities x locations
    
    # Cost = sum_{i,j} A[i, j] * sum_{k,l} B[k, l] P[i, k] P[j, l]
    # Let M = P @ B @ P.T ? No.
    # Let's define C = P @ B @ P.T. C[i, j] = sum_{k,l} P[i, k] B[k, l] P[j, l].
    # Cost = sum_{i, j} A[i, j] C[i, j] = trace(A.T C) = <A, C>.
    
    # Gradient of C wrt P:
    # d/dP[i, k] (P B P.T) = B P.T[:, i] + P B[:, k] ... this is getting messy.
    
    # Easier: Gradient of Cost wrt P[i, j] is:
    # G[i, j] = sum_k A[i, k] * (sum_l B[j, l] P[k, l]) + sum_l A[l, j] * (sum_k B[k, j] P[i, k])
    # Wait, symmetry. If A and B are symmetric:
    # G[i, j] = 2 * sum_k A[i, k] * sum_l B[k, l] P[i, l] ? No.
    
    # Let's use the vectorized form:
    # G = A @ (P @ B) + A.T @ (P @ B.T) ... roughly.
    
    # Let's compute G = A @ (P @ B) + (A.T @ P @ B.T).T ?
    # Let's stick to the definition:
    # G[i, j] = sum_{k, l} (A[i, k] B[j, l] + A[l, j] B[k, i]) P[k, l]  (if symmetric)
    # Assuming A, B symmetric:
    # G[i, j] = sum_k A[i, k] * (sum_l B[k, l] P[i, l]) ? No, P is i->k.
    
    # Correct Gradient:
    # L = <A, P B P^T>
    # dL/dP = A P B^T + A^T P^T B
    
    # So:
    PB = P @ b
    G = a @ PB + a.T @ PB.T  # If a, b symmetric, a.T = a, b.T = b
    # G[i, j] = sum_k a[i, k] (P @ b)[k, j] + sum_k a.T[i, k] (P @ b.T)[k, j]
    # (P @ b)[k, j] = sum_l P[k, l] b[l, j]
    # So G[i, j] = sum_k sum_l a[i, k] P[k, l] b[l, j] + ...
    
    # Update logits Z
    # Z -= lr * G
    # P = exp(Z / temp)
    # Then Sinkhorn normalize P
    
    # We don't have Z explicitly in the function signature, we have P.
    # We can treat P as the current state and update it directly?
    # No, Sinkhorn requires starting from a positive matrix.
    
    # Alternative: Use the "Sinkhorn iterations" on the gradient?
    # No, standard approach:
    # Maintain Logits Z.
    # Z_new = Z - lr * G
    # P = exp(Z_new / temp)
    # P = Sinkhorn(P)
    
    # Since we only have P, we can't easily recover Z unless we assume P = exp(Z) normalized.
    # Let's assume we keep Z in the main loop.
    
    return P

def _solve_qap_sinkhorn(a, b, time_limit=118.0, start_time=None):
    """
    Solve QAP using Sinkhorn-Knopp relaxation.
    """
    n = len(a)
    if start_time is None:
        start_time = time.time()
    
    # Initialize P as uniform
    P = np.ones((n, n), dtype=np.float64) / n
    
    # Temperature schedule
    temp = 1.0
    min_temp = 0.1
    decay = 0.99
    
    # Learning rate
    lr = 0.01
    
    best_p = None
    best_cost = float('inf')
    
    # Precompute symmetric a
    a_sym = a + a.T
    
    while time.time() - start_time < time_limit:
        # Compute Gradient G
        # G = a @ (P @ b) + a.T @ (P @ b.T)
        PB = P @ b
        G = a @ PB + a.T @ PB.T
        
        # Update P via gradient descent on the relaxed objective
        # P_new = P - lr * G
        # But we must stay on the Birkhoff polytope.
        # We can project or use Sinkhorn.
        
        # Let's use a simple projection:
        # 1. P -= lr * G
        # 2. Clip to 0
        # 3. Sinkhorn normalize
        
        P -= lr * G
        P = np.maximum(P, 1e-10)
        
        # Sinkhorn Normalization
        # Row normalize
        row_sums = P.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        P = P / row_sums
        
        # Column normalize
        col_sums = P.sum(axis=0, keepdims=True)
        col_sums = np.maximum(col_sums, 1e-10)
        P = P / col_sums
        
        # Iterate a few times for better convergence
        for _ in range(5):
            row_sums = P.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-10)
            P = P / row_sums
            col_sums = P.sum(axis=0, keepdims=True)
            col_sums = np.maximum(col_sums, 1e-10)
            P = P / col_sums
            
        # Hard assignment
        # Find best permutation from P
        # This is an assignment problem: min sum P[i, p[i]] ? No, we want to maximize P[i, p[i]]
        # Or simply take argmax?
        # Better: Use Hungarian on -P to find the most likely permutation.
        
        # Hungarian algorithm is O(n^3). n=256. 256^3 ~ 16M. Fast enough.
        row_ind, col_ind = linear_sum_assignment(-P)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        
        cost = _qap_cost(a, b, p)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
            
        # Anneal temperature
        temp *= decay
        
        # If stuck, perturb
        if random.random() < 0.01:
            # Add noise
            P += np.random.normal(0, 0.1, P.shape)
            P = np.maximum(P, 1e-10)
            # Re-normalize
            row_sums = P.sum(axis=1, keepdims=True)
            P = P / np.maximum(row_sums, 1e-10)
            col_sums = P.sum(axis=0, keepdims=True)
            P = P / np.maximum(col_sums, 1e-10)

    return best_p, best_cost

def _local_search_2opt(a, b, a_sym, p, max_iter=50, time_limit=None, start_time=None):
    """
    Perform 2-opt local search to refine the solution.
    """
    n = len(a)
    current_cost = _qap_cost(a, b, p)
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
                
        # Simple 2-opt pass
        for i in range(n - 1):
            x = p[i]
            A_i = a_sym[x, :]
            B_i = b[x, :]
            
            best_j = -1
            best_delta = 0
            
            for j in range(i + 1, n):
                y = p[j]
                diff_a = A_i - a_sym[y, :]
                diff_b = b[y, :] - B_i
                
                temp_full = diff_a * diff_b[p]
                delta = np.sum(temp_full) - diff_a[i] * diff_b[x] - diff_a[j] * diff_b[y]
                
                if delta < best_delta:
                    best_delta = delta
                    best_j = j
            
            if best_j >= 0 and best_delta < -1e-9:
                p[i], p[best_j] = p[best_j], p[i]
                current_cost += best_delta
                improved = True
    
    return p, current_cost

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 118.0
    
    # Precompute symmetric a
    a_sym = a + a.T
    
    # Phase 1: Sinkhorn Relaxation
    # Run for a portion of the time
    sinkhorn_time = time_limit * 0.6
    
    best_p, best_cost = _solve_qap_sinkhorn(a, b, time_limit=sinkhorn_time, start_time=start_time)
    
    # If sinkhorn didn't find a solution (shouldn't happen), fallback to random
    if best_p is None:
        p = np.random.permutation(n).astype(np.int64)
        best_p = p
        best_cost = _qap_cost(a, b, p)
    
    # Phase 2: Local Search Polish
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 2.0:
        p = best_p.copy()
        final_p, final_cost = _local_search_2opt(a, b, a_sym, p, max_iter=100, time_limit=remaining_time - 1.0, start_time=start_time)
        
        if final_cost < best_cost:
            best_cost = final_cost
            best_p = final_p
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
