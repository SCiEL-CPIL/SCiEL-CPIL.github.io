"""
QAP Solver using Soft-Assignment Iteration (SAI) Continuous Relaxation.

Strategy:
1. Initialize with a rank-matching heuristic.
2. Convert the discrete permutation to a doubly stochastic matrix (assignment matrix).
3. Perform gradient descent on the logit matrix to minimize the continuous QAP cost:
   C(P) = sum_{i,j,k,l} A[i,j] B[k,l] P[i,k] P[j,l]
   This is equivalent to minimizing ||A - P^T B P||_F^2 in a relaxed sense.
4. Periodically extract the hard permutation from the current doubly stochastic matrix
   using the Hungarian algorithm (or row-max if fast) and update the best solution.
5. Use the extracted permutation to re-initialize the soft assignment (or continue from logits)
   to allow the continuous dynamics to refine the discrete structure.
6. Final polish with a few iterations of 2-opt on the best discrete solution found.
"""

import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a_sym, b, p, i, j):
    """
    Compute delta for swapping positions i and j in permutation p.
    a_sym[i, k] = a[i, k] + a[k, i]
    """
    x = p[i]
    y = p[j]
    
    diff_a = a_sym[i, :] - a_sym[j, :]
    diff_b = b[y, :] - b[x, :]
    
    temp = diff_a * diff_b[p]
    
    delta = np.sum(temp) - diff_a[i] * diff_b[x] - diff_a[j] * diff_b[y]
    
    return delta

def _local_search_2opt(a, b, a_sym, p, max_iter=50, time_limit=None, start_time=None):
    """
    Perform 2-opt local search to refine the solution.
    """
    n = len(p)
    current_cost = _qap_cost(a, b, p)
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
                
        for i in range(n):
            for j in range(i + 1, n):
                delta = _swap_delta(a_sym, b, p, i, j)
                if delta < -1e-9:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    # Restart inner loop on improvement to ensure convergence
                    # But for speed, we just continue the pass. 
                    # Standard 2-opt often restarts, but for large n, first-improvement 
                    # within a pass is acceptable for a "polish" step.
                    
    return p, current_cost

def _extract_permutation_from_ds(ds_matrix, a, b, p_init=None):
    """
    Extract a hard permutation from a doubly stochastic matrix.
    Uses the Hungarian algorithm on the negative of the matrix to maximize
    the sum of selected entries, which corresponds to the most "confident" assignment.
    However, a better heuristic for QAP is to solve the LAP relaxation of the QAP
    centered around the current soft assignment.
    
    For simplicity and speed, we first try the direct Hungarian on -DS.
    If that fails or is slow, we fall back to greedy row-max.
    """
    n = ds_matrix.shape[0]
    
    # Method 1: Hungarian on -DS (Maximize overlap with current soft state)
    # This finds the permutation closest to the current soft assignment in terms of mass.
    try:
        row_ind, col_ind = linear_sum_assignment(-ds_matrix)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p
    except Exception:
        # Fallback: Greedy row max
        p = np.empty(n, dtype=np.int64)
        used = np.zeros(n, dtype=bool)
        for i in range(n):
            # Find max in row i among unused columns
            row = ds_matrix[i].copy()
            row[used] = -np.inf
            j = np.argmax(row)
            p[i] = j
            used[j] = True
        return p

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0 # Safety margin
    
    # Precompute symmetric a for fast delta
    a_sym = a + a.T
    
    # 1. Initial Solution: Rank Matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()
    
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Initialize Doubly Stochastic Matrix (DSM)
    # Start with a sharp version of the initial permutation
    # We use a temperature parameter to control the "softness"
    # P[i, p[i]] = 1, else 0 is the hard limit.
    # We represent the DSM via logits: L = log(P).
    # To ensure row/col sums are 1, we use iterative scaling or gradient descent on L
    # with constraints. A common approach is to minimize the QAP cost on the manifold
    # of doubly stochastic matrices.
    
    # Let's use a simple gradient descent on the logit matrix Z.
    # P = softmax(Z) row-wise, then column-scale to 1.
    # Actually, Sinkhorn-Knopp is faster for projection, but we need gradient of the QAP cost.
    
    # Cost C(P) = Tr(A^T P^T B P) = sum_{i,j,k,l} A[i,j] B[k,l] P[i,k] P[j,l]
    # Gradient w.r.t P[i,k]:
    # dC/dP[i,k] = sum_{j,l} (A[i,j] + A[j,i]) B[k,l] P[j,l]
    #             = sum_j A_sym[i,j] (B P)^[k,l] ... wait.
    # Let M = B P. M[k,l] = sum_m B[k,m] P[m,l].
    # Then dC/dP[i,k] = sum_j A_sym[i,j] M[j,k]? No.
    # C = sum_{i,j,k,l} A[i,j] B[k,l] P[i,k] P[j,l]
    # Term involving P[i,k]:
    # 1. i is first index of P: sum_{j,l} A[i,j] B[k,l] P[j,l]
    # 2. j is first index of P (if we consider symmetry or just the sum):
    #    The sum is over all i,j. So P[i,k] appears in terms where i is the row of the first P
    #    and where i is the row of the second P (i.e., j=i).
    #    So dC/dP[i,k] = sum_{j,l} A[i,j] B[k,l] P[j,l] + sum_{j,l} A[j,i] B[k,l] P[j,l]
    #                   = sum_j (A[i,j] + A[j,i]) (sum_l B[k,l] P[j,l])
    #                   = sum_j A_sym[i,j] (B P)[k,j]  (Note: (BP)[k,j] = sum_l B[k,l] P[l,j]? No. P is row-stochastic. P[j,l] is facility j at loc l.
    #                   Let's be precise.
    #                   Let Q = P.
    #                   Term 1: sum_{j,l} A[i,j] B[k,l] Q[j,l]. Let R[k,j] = sum_l B[k,l] Q[j,l].
    #                   Then Term 1 = sum_j A[i,j] R[k,j].
    #                   Term 2: sum_{j,l} A[j,i] B[k,l] Q[j,l]. Let S[k,j] = sum_l B[k,l] Q[j,l] = R[k,j].
    #                   Then Term 2 = sum_j A[j,i] R[k,j].
    #                   So dC/dP[i,k] = sum_j (A[i,j] + A[j,i]) R[k,j] = (A_sym @ R.T)[i,k]?
    #                   R[k,j] = (B @ Q)[k,j]?
    #                   (B @ Q)[k,j] = sum_l B[k,l] Q[l,j].
    #                   My R was sum_l B[k,l] Q[j,l]. This is (B @ Q.T)[k,j].
    #                   So R = B @ Q.T.
    #                   dC/dP = A_sym @ R.T = A_sym @ (B @ Q.T).T = A_sym @ Q @ B.T.
    #                   Since B is symmetric, B.T = B.
    #                   Gradient G = A_sym @ Q @ B.
    
    # We want to minimize C(Q) subject to Q being doubly stochastic.
    # We can use projected gradient descent or mirror descent.
    # Mirror descent on the logit space is often more stable.
    # Update Z = Z - eta * G. Then Q = Sinkhorn(Z).
    
    # Initialize Q from p
    Q = np.zeros((n, n))
    Q[np.arange(n), p] = 1.0
    
    # Initialize Logits Z
    # Z = log(Q + epsilon)
    eps = 1e-10
    Z = np.log(Q + eps)
    
    # Precompute A_sym
    # G = A_sym @ Q @ B
    
    # Hyperparameters
    eta = 1.0 / (n * n * 0.1) # Learning rate, needs tuning
    # A safer learning rate for gradient descent on QAP is often small.
    # Let's use a fixed number of iterations or time-based.
    
    # We will alternate between:
    # 1. Gradient steps on Z (with Sinkhorn projection)
    # 2. Extracting permutation and evaluating
    # 3. Re-seeding Z if stuck
    
    num_grad_steps_per_extract = 50
    max_total_grad_steps = 2000 # Adjust based on time
    
    step_count = 0
    last_extract_time = start_time
    extract_interval = 1.0 # seconds
    
    while time.time() - start_time < time_limit:
        # Gradient Descent Phase
        for _ in range(num_grad_steps_per_extract):
            if time.time() - start_time > time_limit - 5.0:
                break
            
            # Compute Gradient
            # G = A_sym @ Q @ B
            G = a_sym @ (Q @ b)
            
            # Update Z
            Z -= eta * G
            
            # Project Z to Doubly Stochastic Q using Sinkhorn-Knopp
            # Sinkhorn: Alternate row and col normalization on exp(Z)
            # For stability, we use log-domain Sinkhorn or just exp with clipping
            # Simple Sinkhorn on exp(Z)
            M = np.exp(Z - np.max(Z)) # Numerical stability
            # Normalize rows
            row_sums = M.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            M = M / row_sums
            # Normalize cols
            col_sums = M.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1.0
            M = M / col_sums
            
            # Iterate a few times to converge
            for _ in range(20):
                row_sums = M.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                M = M / row_sums
                col_sums = M.sum(axis=0, keepdims=True)
                col_sums[col_sums == 0] = 1.0
                M = M / col_sums
            
            Q = M
            step_count += 1
            
            if step_count >= max_total_grad_steps:
                break
        
        # Extract and Evaluate Phase
        if time.time() - last_extract_time > extract_interval or step_count >= max_total_grad_steps:
            last_extract_time = time.time()
            
            # Extract permutation
            p_new = _extract_permutation_from_ds(Q, a, b, p)
            
            # Evaluate
            cost_new = _qap_cost(a, b, p_new)
            
            if cost_new < best_cost:
                best_cost = cost_new
                best_p = p_new.copy()
                p = p_new.copy()
                
                # Re-seed Q to the new hard permutation to sharpen the search
                # This helps escape flat regions in the continuous space
                Q = np.zeros((n, n))
                Q[np.arange(n), p] = 1.0
                Z = np.log(Q + eps)
                
                # Optional: Do a quick 2-opt polish on the new best
                # This is cheap if max_iter is small
                p_polished, cost_polished = _local_search_2opt(a, b, a_sym, p, max_iter=10, time_limit=time_limit-2.0, start_time=start_time)
                if cost_polished < best_cost:
                    best_cost = cost_polished
                    best_p = p_polished.copy()
                    p = p_polished.copy()
                    # Re-seed again
                    Q = np.zeros((n, n))
                    Q[np.arange(n), p] = 1.0
                    Z = np.log(Q + eps)
                    
        if step_count >= max_total_grad_steps:
            break

    # Final Polish
    if time.time() < time_limit - 5.0:
        p_final, cost_final = _local_search_2opt(a, b, a_sym, best_p, max_iter=50, time_limit=time_limit-2.0, start_time=start_time)
        if cost_final < best_cost:
            best_cost = cost_final
            best_p = p_final.copy()

    return best_p.tolist()

# EVOLVE-BLOCK-END
