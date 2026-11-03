"""
Continuous relaxation approach for QAPLIB wil100.
Strategy:
1. Represent the assignment as a doubly stochastic matrix X (n x n).
2. Optimize the relaxed objective f(X) = sum a[i,j] * b[k,l] * X[i,k] * X[j,l]
   via projected gradient descent with Barzilai-Borwein step size.
3. Project X onto the Birkhoff polytope (doubly stochastic) using the
   Sinkhorn-Knopp algorithm (iterative row/column normalization).
4. Extract a hard permutation from the relaxed solution via the Hungarian
   algorithm on the cost matrix derived from X.
5. Polish the result with a fast vectorized local search (2-swap) to
   escape any suboptimality of the relaxation.
"""
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _project_birkhoff(X, max_iter=200, tol=1e-7):
    """
    Project matrix X onto the Birkhoff polytope (doubly stochastic)
    using the Sinkhorn-Knopp algorithm.
    """
    n = X.shape[0]
    # Ensure non-negative
    X = np.maximum(X, 0)
    
    # If all zeros, return uniform matrix
    if X.sum() < 1e-12:
        return np.ones((n, n)) / n
    
    X = X / X.sum() * n  # scale so total sum is n (target row sum 1)
    
    for _ in range(max_iter):
        # Normalize rows
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
        X = X / row_sums
        
        # Normalize columns
        col_sums = X.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums < 1e-12, 1.0, col_sums)
        X = X / col_sums
        
        # Check convergence
        row_err = np.abs(X.sum(axis=1) - 1).max()
        col_err = np.abs(X.sum(axis=0) - 1).max()
        if max(row_err, col_err) < tol:
            break
    
    return X

def _extract_permutation(X):
    """
    Extract a permutation from a doubly stochastic matrix X.
    We want to minimize sum a[i,j] * b[p[i], p[j]].
    This is equivalent to finding a permutation P such that X approximates P.
    We can solve an assignment problem where the cost of assigning facility i to location k
    is based on how well it fits the current relaxation.
    
    A common approach: solve min sum_{i,k} C[i,k] * X[i,k] where C is derived from the gradient.
    But simpler: just find the permutation closest to X in the sense of the QAP objective.
    
    Actually, the standard way is to solve the linear assignment problem with costs 
    derived from the relaxed solution. However, a more direct approach is:
    
    The relaxed objective is f(X) = sum_{i,j,k,l} A[i,j] B[k,l] X[i,k] X[j,l].
    The gradient with respect to X[i,k] is:
    g[i,k] = sum_{j,l} (A[i,j] B[k,l] X[j,l] + A[j,i] B[l,k] X[j,l])  -- not quite.
    
    Let's use a simpler heuristic: 
    Compute a "cost matrix" C where C[i,k] represents the marginal cost of placing facility i at location k.
    C[i,k] = sum_j A[i,j] * (sum_l B[k,l] * X[j,l])  +  sum_l B[k,l] * (sum_j A[j,i] * X[j,l])
    
    Then solve the assignment problem min sum_{i,k} C[i,k] * Y[i,k] where Y is a permutation matrix.
    """
    n = X.shape[0]
    # Compute row sums of A and B
    # For efficiency, let's precompute some terms.
    
    # Term 1: For each i, k: sum_j A[i,j] * (B @ X)[k, j]
    # B @ X is n x n. Let M1 = B @ X. Then term1[i,k] = sum_j A[i,j] * M1[k,j] = (A @ M1.T)[i,k]
    M1 = b @ X  # b is global
    term1 = a @ M1.T  # a is global
    
    # Term 2: For each i, k: sum_l B[k,l] * (A.T @ X)[l, i]
    # A.T @ X is n x n. Let M2 = A.T @ X. Then term2[i,k] = sum_l B[k,l] * M2[l,i] = (B @ M2)[i,k]
    M2 = a.T @ X
    term2 = b @ M2
    
    C = term1 + term2
    
    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p

def _local_search(a, b, p, time_limit, max_iter=1000):
    """
    Fast vectorized local search using 2-swap moves.
    """
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    no_improve_count = 0
    
    while time.time() - start_time < time_limit and no_improve_count < 20:
        improved = False
        
        # Evaluate all swap moves vectorized
        # For each pair (i, j), compute delta.
        # This is O(n^3) if done naively, but we can vectorize.
        
        # Let's use a simpler approach: iterate over i, and for each i, compute deltas for all j > i.
        # But that's still O(n^2) per iteration.
        
        # For n=100, O(n^2) = 10000 swaps. Each swap delta is O(n) if computed naively.
        # Total O(n^3) = 1e6 operations per iteration. Very fast.
        
        best_delta = 0
        best_i = -1
        best_j = -1
        
        for i in range(n):
            for j in range(i + 1, n):
                pi = p[i]
                pj = p[j]
                
                # Compute delta for swapping i and j
                # delta = sum_{k != i,j} (a[i,k] - a[j,k]) * (b[pj, p[k]] - b[pi, p[k]])
                #       + sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k], pj] - b[p[k], pi])
                #       + a[i,i]*(b[pj,pj] - b[pi,pi]) + a[j,j]*(b[pi,pi] - b[pj,pj])
                #       + a[i,j]*(b[pj,pi] - b[pi,pj]) + a[j,i]*(b[pi,pj] - b[pj,pi])
                
                # Vectorized computation
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                mask[j] = False
                p_masked = p[mask]
                
                diff_a_1 = a[i, mask] - a[j, mask]
                diff_b_1 = b[pj, p_masked] - b[pi, p_masked]
                term1 = np.sum(diff_a_1 * diff_b_1)
                
                diff_a_2 = a[mask, i] - a[mask, j]
                diff_b_2 = b[p_masked, pj] - b[p_masked, pi]
                term2 = np.sum(diff_a_2 * diff_b_2)
                
                term3 = a[i, i] * (b[pj, pj] - b[pi, pi]) + a[j, j] * (b[pi, pi] - b[pj, pj])
                term4 = a[i, j] * (b[pj, pi] - b[pi, pj]) + a[j, i] * (b[pi, pj] - b[pj, pi])
                
                delta = term1 + term2 + term3 + term4
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_i == -1 or best_delta >= 0:
            no_improve_count += 1
            break
        
        # Apply the best move
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += best_delta
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0
    start_time = time.time()
    
    # 1. Initialize X as a random doubly stochastic matrix
    # Start with a uniform matrix
    X = np.ones((n, n)) / n
    
    # 2. Projected Gradient Descent with Barzilai-Borwein step size
    # Objective: f(X) = sum_{i,j,k,l} A[i,j] B[k,l] X[i,k] X[j,l]
    # Gradient: g[i,k] = 2 * sum_{j,l} A[i,j] B[k,l] X[j,l]  (symmetric case)
    # More precisely: g[i,k] = sum_{j,l} (A[i,j] B[k,l] + A[j,i] B[l,k]) X[j,l]
    
    # Precompute for faster gradient computation
    # Let M1 = A @ (B.T @ X.T).T = A @ (X @ B).T ? No.
    # Let's derive:
    # g[i,k] = sum_j sum_l A[i,j] B[k,l] X[j,l] + sum_j sum_l A[j,i] B[l,k] X[j,l]
    # Term 1: sum_j A[i,j] (sum_l B[k,l] X[j,l]) = sum_j A[i,j] (B @ X.T)[k, j] ? 
    # Let's use matrix multiplication:
    # Let Y = B @ X.T  (n x n). Then Y[k, j] = sum_l B[k,l] X.T[j,l] = sum_l B[k,l] X[l,j].
    # That's not quite right. We want sum_l B[k,l] X[j,l].
    # Let Z = X @ B.T. Then Z[j, k] = sum_l X[j,l] B.T[l,k] = sum_l X[j,l] B[k,l].
    # So sum_l B[k,l] X[j,l] = Z[j, k].
    # Term 1 = sum_j A[i,j] Z[j,k] = (A @ Z)[i,k].
    
    # Term 2: sum_j sum_l A[j,i] B[l,k] X[j,l]
    # Let W = B @ X. Then W[l, j] = sum_k' B[l,k'] X[k',j]. Not helpful.
    # Let's rewrite: sum_l B[l,k] (sum_j A[j,i] X[j,l])
    # Let V = A.T @ X. Then V[i, l] = sum_j A.T[i,j] X[j,l] = sum_j A[j,i] X[j,l].
    # So sum_j A[j,i] X[j,l] = V[i,l].
    # Term 2 = sum_l B[l,k] V[i,l] = (V @ B.T)[i,k].
    
    # So g[i,k] = (A @ Z)[i,k] + (V @ B.T)[i,k] where Z = X @ B.T and V = A.T @ X.
    
    # Let's verify with small example.
    # For now, implement the gradient computation.
    
    prev_grad = None
    prev_X = None
    prev_step = 1.0 / (n * n)
    
    max_iter = 200
    for it in range(max_iter):
        if time.time() - start_time > total_deadline - 30:
            break
        
        # Compute gradient
        Z = X @ b.T
        V = a.T @ X
        grad = a @ Z + V @ b.T
        
        # Barzilai-Borwein step size
        if prev_grad is not None:
            s = X - prev_X
            y = grad - prev_grad
            sy = np.sum(s * y)
            yy = np.sum(y * y)
            ss = np.sum(s * s)
            if yy > 1e-12:
                # Use BB step size
                step = sy / yy if sy > 0 else prev_step
                step = max(1e-6, min(step, 1.0))
            else:
                step = prev_step
        else:
            step = 1.0 / n
        
        # Gradient descent step
        X_new = X - step * grad
        
        # Project onto Birkhoff polytope
        X_new = _project_birkhoff(X_new)
        
        # Update
        prev_X = X
        prev_grad = grad
        X = X_new
        prev_step = step
    
    # 3. Extract permutation from X
    p = _extract_permutation(X)
    
    # 4. Polish with local search
    time_remaining = total_deadline - (time.time() - start_time)
    if time_remaining > 10:
        p, cost = _local_search(a, b, p, time_remaining - 5)
    
    return p.tolist()
