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
    """
    n = len(a)
    start_time = time.time()
    
    # Initialize with rank matching
    p_init = _rank_matching_start(a, b)
    X = np.zeros((n, n))
    X[np.arange(n), p_init] = 1.0
    
    best_p = p_init.copy()
    best_cost = _qap_cost(a, b, p_init)
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Compute gradient G
        # F(X) = trace(A^T (X^T B X))
        # Gradient G = (B @ (A @ X)) + (B @ (A @ X)).T is NOT correct.
        # Correct gradient derivation:
        # F(X) = sum_{i,j,k,l} A[i,j] B[k,l] X[i,k] X[j,l]
        # dF/dX[i,k] = sum_{j,l} A[i,j] B[k,l] X[j,l] + sum_{j,l} A[j,i] B[l,k] X[j,l]
        # Let Y = A @ X. Y[i,m] = sum_j A[i,j] X[j,m].
        # Term 1: sum_l B[k,l] (sum_j A[i,j] X[j,l]) = sum_l B[k,l] Y[i,l] = (B.T @ Y)[k,i]? No.
        # (B @ Y.T)[i,k] = sum_l B[i,l] Y.T[l,k] = sum_l B[i,l] Y[k,l]. This is not it.
        
        # Let's use the explicit formula:
        # G[i,k] = sum_{j,l} A[i,j] B[k,l] X[j,l] + sum_{j,l} A[j,i] B[l,k] X[j,l]
        
        # Vectorized:
        # Term 1: sum_j A[i,j] * (sum_l B[k,l] X[j,l])
        # Let M1 = B @ X. M1[k,l] = sum_m B[k,m] X[m,l].
        # Then sum_l B[k,l] X[j,l] is not M1.
        # M1[k,j] = sum_l B[k,l] X[l,j].
        # So sum_l B[k,l] X[j,l] = M1[k,j] if X was symmetric? No.
        # X[j,l] vs X[l,j].
        
        # Let's compute G directly using matrix multiplications.
        # G1[i,k] = sum_j A[i,j] (B @ X)[k,j] ?
        # (B @ X)[k,j] = sum_l B[k,l] X[l,j].
        # We need sum_l B[k,l] X[j,l].
        # Note that (B @ X)[k,j] uses X[l,j]. We want X[j,l].
        # So we need (B @ X.T)? No.
        # Let Z = B @ X. Z[k,l] = sum_m B[k,m] X[m,l].
        # We want sum_l Z[k,l] X[j,l]? No.
        
        # Let's stick to the previous derivation which was:
        # G = term1[:, np.newaxis] + term2
        # term1 = (b * (a @ X)).sum(axis=1)
        # term2 = (X.T @ b) @ (a @ X).T
        
        AX = a @ X
        BX = b @ X
        XTB = X.T @ b
        
        # G[i,k] = sum_m B[i,m] AX[i,m] + sum_m XTB[m,i] AX[m,k]
        # First part: sum_m B[i,m] AX[i,m] -> (B * AX).sum(axis=1)
        # This is a vector of shape (n,). It is the same for all k?
        # Wait.
        # dF/dX[i,k] = sum_{j,l} A[i,j] B[k,l] X[j,l] + sum_{j,l} A[j,i] B[l,k] X[j,l]
        
        # Part 1: S1[i,k] = sum_{j,l} A[i,j] B[k,l] X[j,l]
        # Let W = A @ X. W[i,l] = sum_j A[i,j] X[j,l].
        # Then S1[i,k] = sum_l B[k,l] W[i,l] = (W @ B.T)[i,k].
        
        # Part 2: S2[i,k] = sum_{j,l} A[j,i] B[l,k] X[j,l]
        # Let V = X.T @ A.T. V[l,j] = sum_m X[m,l] A[m,j]. No.
        # V[l,i] = sum_j X[j,l] A[j,i] = (X.T @ A)[l,i].
        # Then S2[i,k] = sum_l B[l,k] V[l,i] = (B.T @ V)[k,i]? No.
        # S2[i,k] = sum_l V[l,i] B[l,k] = (V @ B)[i,k]?
        # V[l,i] is element (l,i) of (X.T @ A).
        # Let M2 = X.T @ A. M2[l,i] = sum_j X[j,l] A[j,i].
        # S2[i,k] = sum_l M2[l,i] B[l,k] = (M2.T @ B)[i,k]?
        # M2.T[i,l] = M2[l,i].
        # (M2.T @ B)[i,k] = sum_l M2.T[i,l] B[l,k] = sum_l M2[l,i] B[l,k]. Yes.
        
        S1 = AX @ b.T  # (n,n)
        M2 = X.T @ a   # (n,n)
        S2 = M2.T @ b  # (n,n)
        
        G = S1 + S2
        
        # Solve linear assignment to minimize sum G[i,k] V[i,k]
        row_ind, col_ind = linear_sum_assignment(G)
        V = np.zeros((n, n))
        V[row_ind, col_ind] = 1.0
        
        # Frank-Wolfe step
        t = 2.0 / (it + 2)
        X_new = (1 - t) * X + t * V
        
        if np.max(np.abs(X_new - X)) < 1e-10:
            break
            
        X = X_new
        
        # Extract valid permutation from X using linear_sum_assignment on -X
        # We want to maximize sum X[i,k] P[i,k], which is minimizing sum -X[i,k] P[i,k]
        p_rows, p_cols = linear_sum_assignment(-X)
        p_fw = p_cols.copy()
        
        cost_fw = _qap_cost(a, b, p_fw)
        if cost_fw < best_cost:
            best_cost = cost_fw
            best_p = p_fw.copy()
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    # Use a strict deadline to ensure we return a valid permutation
    total_deadline = time.time() + 140  # Leave 10s safety margin
    
    # Phase 1: Frank-Wolfe
    # Allocate 50% of the time to FW, but cap it at 60s
    fw_time_limit = min(60, (total_deadline - time.time()) * 0.5)
    p_fw, cost_fw = _frank_wolfe_qap(a, b, max_iter=100, time_limit=fw_time_limit)
    
    # Phase 2: Rank Matching
    p_rank = _rank_matching_start(a, b)
    cost_rank = _qap_cost(a, b, p_rank)
    
    # Choose best starting point
    if cost_fw < cost_rank:
        p_best = p_fw.copy()
        cost_best = cost_fw
    else:
        p_best = p_rank.copy()
        cost_best = cost_rank
    
    # Phase 3: Local Search
    remaining = total_deadline - time.time()
    if remaining > 10:
        p_best, cost_best = _local_search_2opt(a, b, p_best, time_limit=remaining - 10)
    
    # Ensure we return a valid permutation
    # p_best is already a permutation from FW or Rank Matching or Local Search
    return p_best.tolist()
# EVOLVE-BLOCK-END
