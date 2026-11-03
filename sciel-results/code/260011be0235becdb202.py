"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment
import time


def _qap_cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _rank_matching_start(a, b):
    """Rank-matching heuristic: heavy flow rows to central location rows."""
    n = len(a)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _local_search_2opt(a, b, p, time_limit):
    """
    2-opt local search: try all pairs of swaps and keep the best improving one.
    Uses incremental cost updates for speed.
    """
    n = len(a)
    p = p.copy()
    best_cost = _qap_cost(a, b, p)
    start_time = time.time()
    
    # Precompute row sums of a for faster delta computation
    a_row_sums = a.sum(axis=1)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        best_delta = 0
        best_i, best_j = -1, -1
        
        # For 2-opt, swapping positions i and j in p:
        # delta = sum_k [a[i,k]*(b[p[j],p[k]] - b[p[i],p[k]]) + a[j,k]*(b[p[i],p[k]] - b[p[j],p[k]]) 
        #        + a[k,i]*(b[p[k],p[j]] - b[p[k],p[i]]) + a[k,j]*(b[p[k],p[i]] - b[p[k],p[j]])
        # Simplified: delta = (a[i,j]+a[j,i])*(b[p[i],p[j]]-b[p[j],p[i]]) 
        #                + sum_{k != i,j} (a[i,k]-a[j,k])*(b[p[i],p[k]]-b[p[j],p[k]])
        #                + sum_{k != i,j} (a[k,i]-a[k,j])*(b[p[k],p[i]]-b[p[k],p[j]])
        
        # This is O(n) per pair, O(n^3) total per pass. For n=256, that's ~16M ops per pass.
        # We'll do a smarter version: iterate over i, and for each i find best j.
        
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            pi = p[i]
            for j in range(i + 1, n):
                pj = p[j]
                # Compute delta for swapping i and j
                # delta = (a[i,j] + a[j,i]) * (b[pi,pj] - b[pj,pi])
                #       + sum_{k != i,j} (a[i,k] - a[j,k]) * (b[pi,p[k]] - b[pj,p[k]])
                #       + sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k],pi] - b[p[k],pj])
                
                # Vectorized computation
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                mask[j] = False
                
                delta = (a[i, j] + a[j, i]) * (b[pi, pj] - b[pj, pi])
                delta += np.sum((a[i, mask] - a[j, mask]) * (b[pi, p[mask]] - b[pj, p[mask]]))
                delta += np.sum((a[mask, i] - a[mask, j]) * (b[p[mask], pi] - b[p[mask], pj]))
                
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_j = i, j
        
        if best_i >= 0 and best_delta < -1e-9:
            p[best_i], p[best_j] = p[best_j], p[best_i]
            best_cost += best_delta
            improved = True
    
    return p, best_cost


def _frank_wolfe_qap(a, b, max_iter, time_limit):
    """
    Frank-Wolfe (conditional gradient) on the Birkhoff polytope for QAP.
    The objective is f(X) = sum_{i,j} a[i,j] * (X^T B X)[i,j] where X is doubly stochastic.
    Gradient: G = A * (X^T B X) + (A * (X^T B X))^T  ... actually need to be careful.
    
    For QAP with cost matrix A (flow) and B (distance), and assignment matrix X (doubly stochastic),
    the relaxed objective is: F(X) = sum_{i,j,k,l} A[i,j] * B[k,l] * X[i,k] * X[j,l]
    
    Gradient w.r.t. X[i,k]:
    dF/dX[i,k] = sum_{j,l} A[i,j] * B[k,l] * X[j,l] + sum_{j,l} A[j,i] * B[l,k] * X[j,l]
               = (A * (X^T B X))[i,k] + (A^T * (X B X^T))[i,k]
    
    Wait, let me re-derive:
    F(X) = sum_{i,j,k,l} A[i,j] B[k,l] X[i,k] X[j,l]
    = sum_{i,k} X[i,k] * sum_{j,l} A[i,j] B[k,l] X[j,l]
    
    dF/dX[i,k] = sum_{j,l} A[i,j] B[k,l] X[j,l] + sum_{j,l} A[j,i] B[l,k] X[j,l]
    
    Let M = X^T B X, so M[j,l] = sum_k X[k,j] B[k,l] ... no.
    M[j,l] = (X^T B X)[j,l] = sum_{k1,k2} X[k1,j] B[k1,k2] X[k2,l]
    
    That's not quite right either. Let me use matrix notation:
    F(X) = sum_{i,j} A[i,j] * (X^T B X)[i,j]  -- NO, that's not right.
    
    Actually: sum_{i,j,k,l} A[i,j] B[k,l] X[i,k] X[j,l] = trace(A^T * (X^T B X))
    
    Let C = X^T B X. Then C[i,j] = sum_{k,l} X[k,i] B[k,l] X[l,j]
    F = sum_{i,j} A[i,j] C[i,j] = trace(A^T C)
    
    dF/dX[i,k]:
    C[i,j] = sum_{k,l} X[k,i] B[k,l] X[l,j]
    dC[i,j]/dX[i,k] = sum_l B[k,l] X[l,j] (from first X) + sum_k X[k,i] B[k,i]... 
    No, X[i,k] appears in C[i,j] when the first index is i: X[k,i] with k=i gives X[i,i]... 
    
    Let me be more careful. C = X^T B X.
    C[p,q] = sum_{m,n} X[m,p] B[m,n] X[n,q]
    
    dF/dX[i,k] = sum_{p,q} A[p,q] * dC[p,q]/dX[i,k]
    
    dC[p,q]/dX[i,k] = sum_n B[i,n] X[n,q] * delta(p,i) + sum_m X[m,p] B[m,i] * delta(q,k)
                     = B[i,q'] ... 
    
    dC[p,q]/dX[i,k] = (B X)[i,q] if p==i, plus X^T B [p,i] if q==k
    = delta(p,i) * (B @ X)[i,q] + delta(q,k) * (X.T @ B)[p,i]
    
    So dF/dX[i,k] = sum_q A[i,q] * (B@X)[i,q] + sum_p A[p,k] * (X.T@B)[p,i]
                  = (A * (B@X))[i,:] @ ... 
    
    Let me just compute: 
    G[i,k] = sum_q A[i,q] * (B @ X)[i,q] + sum_p A[p,k] * (X.T @ B)[p,i]
    
    But (B@X)[i,q] = sum_m B[i,m] X[m,q]
    And (X.T@B)[p,i] = sum_m X[m,p] B[m,i]
    
    So G[i,k] = sum_q A[i,q] * sum_m B[i,m] X[m,q] + sum_p A[p,k] * sum_m X[m,p] B[m,i]
              = sum_m B[i,m] * sum_q A[i,q] X[m,q] + sum_m B[m,i] * sum_p A[p,k] X[m,p]
              = sum_m B[i,m] * (A @ X)[i,m] + sum_m B[m,i] * (X.T @ A)[m,k]
    
    So G = (B * (A @ X).T).T ... let me just compute numerically:
    G = (A @ X).T @ B.T + B @ (A @ X)  ... I'll just compute it directly in code.
    
    For Frank-Wolfe:
    1. Compute gradient G
    2. Solve linear assignment: min sum_{i,k} G[i,k] X[i,k] over permutation -> gives vertex V
    3. Step: X_new = (1-t)*X + t*V
    4. Repeat
    """
    n = len(a)
    start_time = time.time()
    
    # Initialize with rank matching, convert to doubly stochastic (it's already a permutation)
    p_init = _rank_matching_start(a, b)
    X = np.zeros((n, n))
    X[np.arange(n), p_init] = 1.0
    
    best_p = p_init.copy()
    best_cost = _qap_cost(a, b, p_init)
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute gradient
        # G[i,k] = sum_q A[i,q] * (B@X)[i,q] + sum_p A[p,k] * (X.T@B)[p,i]
        BX = b @ X
        XTB = X.T @ b
        AXT = a @ X  # (n,n), AXT[i,m] = sum_q A[i,q] X[q,m]
        
        # G[i,k] = sum_m B[i,m] * AXT[i,m] + sum_m XTB[m,i] * AXT.T[m,k]
        # First term: (B * AXT).sum(axis=1) broadcast over k? No.
        # G[i,k] = sum_m B[i,m] * AXT[i,m] + sum_m XTB[m,i] * AXT.T[m,k]
        # First term is independent of k: sum_m B[i,m] AXT[i,m]
        # Second term: (XTB @ AXT.T)[i,k]
        
        term1 = (b * AXT).sum(axis=1)  # shape (n,)
        term2 = XTB @ AXT.T  # shape (n,n)
        G = term1[:, np.newaxis] + term2
        
        # Solve linear assignment to minimize sum G[i,k] X[i,k]
        row_ind, col_ind = linear_sum_assignment(G)
        V = np.zeros((n, n))
        V[row_ind, col_ind] = 1.0
        
        # Frank-Wolfe step with adaptive step size
        # Use backtracking or fixed decay
        t = 2.0 / (it + 2)
        
        X_new = (1 - t) * X + t * V
        
        # Check convergence
        if np.max(np.abs(X_new - X)) < 1e-10:
            break
        
        X = X_new
        
        # Extract permutation from X (take argmax in each row)
        p_fw = np.argmax(X, axis=1)
        # Ensure it's a valid permutation (should be close to one)
        cost_fw = _qap_cost(a, b, p_fw)
        if cost_fw < best_cost:
            best_cost = cost_fw
            best_p = p_fw.copy()
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    total_deadline = time.time() + 145  # Leave 5s safety margin
    
    # Phase 1: Frank-Wolfe to get a good doubly stochastic solution
    fw_time_limit = min(60, (total_deadline - time.time()) * 0.5)
    p_fw, cost_fw = _frank_wolfe_qap(a, b, max_iter=200, time_limit=fw_time_limit)
    
    # Phase 2: Local search 2-opt starting from the best of rank-matching and FW
    p_rank = _rank_matching_start(a, b)
    cost_rank = _qap_cost(a, b, p_rank)
    
    if cost_fw < cost_rank:
        p_best = p_fw.copy()
        cost_best = cost_fw
    else:
        p_best = p_rank.copy()
        cost_best = cost_rank
    
    remaining = total_deadline - time.time()
    if remaining > 5:
        p_best, cost_best = _local_search_2opt(a, b, p_best, time_limit=remaining - 5)
    
    # Final cost check
    final_cost = _qap_cost(a, b, p_best)
    return p_best.tolist()
# EVOLVE-BLOCK-END
