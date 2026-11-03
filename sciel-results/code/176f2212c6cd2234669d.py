"""Rewritten QAP solver: continuous relaxation + Hungarian + local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Rank-matching initial permutation.
    2. Hungarian-based linearization refinement (multiple rounds).
    3. Fast local search with precomputed swap deltas.
    4. 3-swap moves to escape 2-swap local optima.
    """
    n = len(a)
    deadline = time.time() + 112.0  # safety margin before 120s limit

    def compute_cost(p):
        total = 0
        for i in range(n):
            total += np.dot(a[i], b[p[i], p])
        return total

    # --- Initial solution via rank-matching ---
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central

    best_p = p.copy()
    best_cost = compute_cost(best_p)

    # --- Fast swap_delta using precomputed b_p = b[p[...]] ---
    # Key insight: precompute b_p[k, l] = b[p[k], l] and b_pT[k, l] = b[l, p[k]]
    # Then swap_delta(i,j) can be computed in O(n) without mask allocation.
    
    def _update_b_p(best_p):
        """Precompute b_p[k,l] = b[p[k],l] and b_pT[k,l] = b[l,p[k]]."""
        return b[best_p, :], b[:, best_p]

    b_p, b_pT = _update_b_p(best_p)

    def swap_delta_fast(i, j):
        """Compute delta for swapping positions i and j in best_p. O(n) time."""
        u = best_p[i]
        v = best_p[j]
        if u == v:
            return 0
        
        # b_p[k, l] = b[p[k], l], b_pT[k, l] = b[l, p[k]]
        # After swap: p'[i]=v, p'[j]=u
        # Delta = sum_{i',j'} a[i',j']*(b[p'[i'],p'[j']] - b[p[i'],p[j']])
        # Only terms where i' in {i,j} or j' in {i,j} change.
        
        # Let's compute the contribution change:
        # For row i (facility i moves from loc u to loc v):
        #   old_row_i_contrib = sum_k a[i,k]*b[u, p[k]] = a[i,:] @ b_pT[:,i] ... wait
        #   Actually: sum_k a[i,k]*b[u,p[k]] = sum_k a[i,k]*b_pT[k,i]? No.
        #   b_pT[k, i] = b[i, p[k]]... no. b_pT[k,l] = b[l, p[k]]. So b_pT[k,i] = b[i, p[k]].
        #   We want b[u, p[k]] = b_pT[k, u].
        
        # Let me use direct indexing:
        # b_p[k, l] = b[p[k], l]
        # b_pT[k, l] = b[l, p[k]]
        
        # Term contributions involving facility i (row i of a):
        # Old: sum_k a[i,k] * b[u, p[k]] = sum_k a[i,k] * b_pT[k, u]
        # New: sum_k a[i,k] * b[v, p'[k]] where p'[i]=v, p'[j]=u
        #   = a[i,i]*b[v,v] + a[i,j]*b[v,u] + sum_{k!=i,j} a[i,k]*b[v,p[k]]
        #   = a[i,i]*b[v,v] + a[i,j]*b[v,u] + (sum_k a[i,k]*b[v,p[k]] - a[i,i]*b[v,v] - a[i,j]*b[v,p[j]])
        #   Wait, p[j]=u so b[v,p[j]]=b[v,u]. 
        #   = sum_k a[i,k]*b[v,p[k]] - a[i,j]*b[v,u] + a[i,j]*b[v,u]
        #   = sum_k a[i,k]*b[v,p[k]] = sum_k a[i,k]*b_pT[k,v]
        
        # Hmm, that's simpler than I thought. Let me redo:
        # New row-i contrib = sum_k a[i,k]*b[v, p'[k]]
        #   p'[i]=v, p'[j]=u, p'[k]=p[k] for k!=i,j
        #   = a[i,i]*b[v,v] + a[i,j]*b[v,u] + sum_{k!=i,j} a[i,k]*b[v,p[k]]
        # Old row-i contrib = sum_k a[i,k]*b[u, p[k]]
        #   = a[i,i]*b[u,u] + a[i,j]*b[u,v] + sum_{k!=i,j} a[i,k]*b[u,p[k]]
        
        # Similarly for row j.
        # And column contributions (where j' is i or j):
        # Old col-i contrib = sum_k a[k,i]*b[p[k],u] = sum_k a[k,i]*b_p[k,u]
        # New col-i contrib = sum_k a[k,i]*b[p'[k],v]
        #   = a[i,i]*b[v,v] + a[j,i]*b[u,v] + sum_{k!=i,j} a[k,i]*b[p[k],v]
        
        # This is getting complex. Let me just compute it directly but efficiently.
        
        # Precomputed: b_p[k,l] = b[p[k],l], b_pT[k,l] = b[l,p[k]]
        # For the "k not in {i,j}" part, I need:
        #   sum_{k!=i,j} a[i,k]*b[v,p[k]] = sum_k a[i,k]*b_pT[k,v] - a[i,i]*b_pT[i,v] - a[i,j]*b_pT[j,v]
        #   b_pT[i,v] = b[v,p[i]] = b[v,u], b_pT[j,v] = b[v,p[j]] = b[v,u]
        #   So = sum_k a[i,k]*b_pT[k,v] - a[i,i]*b[v,u] - a[i,j]*b[v,u]
        
        # Let me just compute the full delta by summing all affected terms.
        # Affected terms: all (i',j') where i' in {i,j} or j' in {i,j}.
        
        # I'll compute:
        # delta = (new_contrib - old_contrib) for all affected (i',j')
        
        # Let me define:
        # For any pair (x, y) where x is in {i,j} or y is in {i,j}:
        # old_val = a[x,y] * b[p[x], p[y]]
        # new_val = a[x,y] * b[p'[x], p'[y]]
        
        # p'[i]=v, p'[j]=u, p'[k]=p[k]
        
        # I'll iterate over the O(n) affected pairs.
        delta = 0.0
        
        # Row i: j' ranges over all k
        # For k == i: a[i,i]*(b[v,v] - b[u,u])
        # For k == j: a[i,j]*(b[v,u] - b[u,v])
        # For k != i,j: a[i,k]*(b[v,p[k]] - b[u,p[k]])
        #   = a[i,k]*(b_pT[k,v] - b_pT[k,u])
        
        # Row j: 
        # For k == i: a[j,i]*(b[u,v] - b[v,u])
        # For k == j: a[j,j]*(b[u,u] - b[v,v])
        # For k != i,j: a[j,k]*(b[u,p[k]] - b[v,p[k]])
        #   = a[j,k]*(b_pT[k,u] - b_pT[k,v])
        
        # Col i (j'=i, i' != i,j): a[k,i]*(b[p[k],v] - b[p[k],u]) = a[k,i]*(b_p[k,v] - b_p[k,u])
        # Col j (j'=j, i' != i,j): a[k,j]*(b[p[k],u] - b[p[k],v]) = a[k,j]*(b_p[k,u] - b_p[k,v])
        
        # Sum all:
        # Diagonal/cross terms (i',j') in {i,j}x{i,j}:
        delta += a[i,i]*(b[v,v] - b[u,u])
        delta += a[i,j]*(b[v,u] - b[u,v])
        delta += a[j,i]*(b[u,v] - b[v,u])
        delta += a[j,j]*(b[u,u] - b[v,v])
        
        # Row i, k not in {i,j}:
        # sum_{k!=i,j} a[i,k]*(b_pT[k,v] - b_pT[k,u])
        # = a[i,:] @ (b_pT[:,v] - b_pT[:,u]) - a[i,i]*(b_pT[i,v]-b_pT[i,u]) - a[i,j]*(b_pT[j,v]-b_pT[j,u])
        # b_pT[i,v] = b[v,p[i]] = b[v,u], b_pT[i,u] = b[u,p[i]] = b[u,u]
        # b_pT[j,v] = b[v,p[j]] = b[v,u], b_pT[j,u] = b[u,p[j]] = b[u,u]
        row_i_diff = b_pT[:, v] - b_pT[:, u]
        delta += np.dot(a[i], row_i_diff) - a[i,i]*(b[v,u]-b[u,u]) - a[i,j]*(b[v,u]-b[u,u])
        
        # Row j, k not in {i,j}:
        # sum_{k!=i,j} a[j,k]*(b_pT[k,u] - b_pT[k,v])
        row_j_diff = b_pT[:, u] - b_pT[:, v]
        delta += np.dot(a[j], row_j_diff) - a[j,i]*(b[u,u]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
        
        # Col i, k not in {i,j}:
        # sum_{k!=i,j} a[k,i]*(b_p[k,v] - b_p[k,u])
        col_i_diff = b_p[:, v] - b_p[:, u]
        delta += np.dot(a[:, i], col_i_diff) - a[i,i]*(b[v,v]-b[u,u]) - a[j,i]*(b[u,v]-b[v,u])
        
        # Col j, k not in {i,j}:
        # sum_{k!=i,j} a[k,j]*(b_p[k,u] - b_p[k,v])
        col_j_diff = b_p[:, u] - b_p[:, v]
        delta += np.dot(a[:, j], col_j_diff) - a[i,j]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
        
        return delta

    # --- Local search with fast swap_delta ---
    def local_search_2swap(p, cost, b_p, b_pT, max_time):
        """Hill-climbing 2-swap local search with precomputed b_p, b_pT."""
        nonlocal best_p, best_cost
        improved = True
        while improved and time.time() < max_time:
            improved = False
            num_trials = 30000
            for _ in range(num_trials):
                if time.time() > max_time:
                    break
                i, j = np.random.randint(0, n, 2)
                if i == j:
                    continue
                d = swap_delta_fast(i, j)
                if d < -0.5:  # integer costs, so d <= -1 means improvement
                    # Perform swap
                    u, v = p[i], p[j]
                    p[i], p[j] = v, u
                    cost += d
                    # Update b_p and b_pT: rows i,j swap in b_p, cols i,j swap in b_pT
                    b_p[[i, j], :] = b_p[[j, i], :]
                    b_pT[:, [i, j]] = b_pT[:, [j, i]]
                    improved = True
        return p, cost, b_p, b_pT

    # --- Hungarian refinement ---
    def hungarian_refine(p):
        """Build linear cost matrix from current p, solve assignment, return new p."""
        b_col_p = b[:, p]   # [j, k] = b[j, p[k]]
        b_row_p = b[p, :]   # [k, j] = b[p[k], j]
        C = a @ b_col_p.T + a.T @ b_row_p
        row_ind, col_ind = linear_sum_assignment(C)
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        return p_new

    # --- Phase 1: Initial local search ---
    best_p, best_cost, b_p, b_pT = local_search_2swap(best_p, best_cost, b_p, b_pT, deadline - 40.0)

    # --- Phase 2: Hungarian refinement ---
    if time.time() < deadline - 30.0:
        p_hung = hungarian_refine(best_p)
        cost_hung = compute_cost(p_hung)
        if cost_hung < best_cost:
            best_p = p_hung
            best_cost = cost_hung
            b_p, b_pT = _update_b_p(best_p)
            # Local search after Hungarian
            best_p, best_cost, b_p, b_pT = local_search_2swap(best_p, best_cost, b_p, b_pT, deadline - 20.0)

    # --- Phase 3: Second Hungarian refinement ---
    if time.time() < deadline - 15.0:
        p_hung = hungarian_refine(best_p)
        cost_hung = compute_cost(p_hung)
        if cost_hung < best_cost:
            best_p = p_hung
            best_cost = cost_hung
            b_p, b_pT = _update_b_p(best_p)

    # --- Phase 4: Final local search ---
    if time.time() < deadline - 2.0:
        best_p, best_cost, b_p, b_pT = local_search_2swap(best_p, best_cost, b_p, b_pT, deadline - 1.0)

    return best_p.tolist()
# EVOLVE-BLOCK-END
