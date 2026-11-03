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
    deadline = time.time() + 115.0  # safety margin before 120s limit

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
                    # Update b_p and b_pT
                    # b_p[k, l] = b[p[k], l]
                    # Swapping p[i] and p[j] means:
                    # b_p[i, :] and b_p[j, :] swap
                    # b_pT[:, i] and b_pT[:, j] swap
                    b_p[i, :], b_p[j, :] = b_p[j, :].copy(), b_p[i, :].copy()
                    b_pT[:, i], b_pT[:, j] = b_pT[:, j].copy(), b_pT[:, i].copy()
                    best_p = p
                    best_cost = cost
                    improved = True
        return p, cost

    # --- Hungarian refinement ---
    def hungarian_refine(p, cost, max_time):
        """Use Hungarian on linearized cost to improve solution."""
        nonlocal best_p, best_cost
        if time.time() > max_time:
            return p, cost
        
        b_p, b_pT = _update_b_p(p)
        
        # Build linear cost matrix
        # C[i,j] = sum_k a[i,k]*b[j, p[k]] + sum_k a[k,i]*b[p[k], j]
        b_col_p = b[:, p]  # shape (n, n), [j, k] = b[j, p[k]]
        b_row_p = b[p, :]  # shape (n, n), [k, j] = b[p[k], j]
        
        C = a @ b_col_p.T + a.T @ b_row_p
        
        row_ind, col_ind = linear_sum_assignment(C)
        p_hung = np.empty(n, dtype=np.int64)
        p_hung[row_ind] = col_ind
        
        cost_hung = compute_cost(p_hung)
        if cost_hung < cost:
            p = p_hung
            cost = cost_hung
            best_p = p.copy()
            best_cost = cost
        return p, cost

    # --- 3-swap moves ---
    def swap3_delta(i, j, k):
        """Compute delta for 3-cycle: p[i]->p[j]->p[k]->p[i]."""
        u, v, w = best_p[i], best_p[j], best_p[k]
        if u == v or v == w or u == w:
            return 0
        
        # p'[i]=v, p'[j]=w, p'[k]=u
        # Only terms where i' in {i,j,k} or j' in {i,j,k} change.
        # This is more complex, so we compute directly.
        
        delta = 0.0
        # We need to compute sum over all (x,y) where x in {i,j,k} or y in {i,j,k}
        # of a[x,y]*(b[p'[x],p'[y]] - b[p[x],p[y]])
        
        # Let's compute old and new contributions for these pairs.
        # Affected pairs: (x,y) where x in S or y in S, S={i,j,k}
        
        # For efficiency, compute the sum over all x,y in S x S first, then add the cross terms.
        
        # S x S terms:
        for x in [i, j, k]:
            for y in [i, j, k]:
                old_loc_x = best_p[x]
                new_loc_x = best_p[[i,j,k].index(y)] if x in [i,j,k] else old_loc_x
                # Wait, this is getting messy. Let me just do it directly.
        
        # Simpler: compute the change by iterating over affected indices.
        # p'[i]=v, p'[j]=w, p'[k]=u
        # For each x in {i,j,k}, y in {i,j,k}:
        #   delta += a[x,y]*(b[p'[x],p'[y]] - b[p[x],p[y]])
        
        p_new = best_p.copy()
        p_new[i], p_new[j], p_new[k] = v, w, u
        
        # S x S
        for x in [i, j, k]:
            for y in [i, j, k]:
                delta += a[x,y]*(b[p_new[x],p_new[y]] - b[best_p[x],best_p[y]])
        
        # Cross terms: x in S, y not in S
        mask = np.ones(n, dtype=bool)
        mask[i] = mask[j] = mask[k] = False
        others = np.where(mask)[0]
        
        for x in [i, j, k]:
            # sum_{y not in S} a[x,y]*(b[p_new[x],p_new[y]] - b[best_p[x],best_p[y]])
            # p_new[y] = best_p[y] for y not in S
            # = sum_{y not in S} a[x,y]*(b[p_new[x],best_p[y]] - b[best_p[x],best_p[y]])
            # = (b[p_new[x],:] - b[best_p[x],:])[others] @ a[x, others]
            diff = b[p_new[x], others] - b[best_p[x], others]
            delta += np.dot(diff, a[x, others])
        
        # Cross terms: x not in S, y in S
        for y in [i, j, k]:
            # sum_{x not in S} a[x,y]*(b[p_new[x],p_new[y]] - b[best_p[x],best_p[y]])
            # p_new[x] = best_p[x] for x not in S
            # = sum_{x not in S} a[x,y]*(b[best_p[x],p_new[y]] - b[best_p[x],best_p[y]])
            # = (b[best_p[others],p_new[y]] - b[best_p[others],best_p[y]]) @ a[others, y]
            diff = b[best_p[others], p_new[y]] - b[best_p[others], best_p[y]]
            delta += np.dot(diff, a[others, y])
        
        return delta

    def local_search_3swap(p, cost, b_p, b_pT, max_time):
        """Try 3-cycle moves to escape 2-swap local optima."""
        nonlocal best_p, best_cost
        improved = True
        while improved and time.time() < max_time:
            improved = False
            num_trials = 5000
            for _ in range(num_trials):
                if time.time() > max_time:
                    break
                i, j, k = np.random.randint(0, n, 3)
                if len(set([i, j, k])) < 3:
                    continue
                d = swap3_delta(i, j, k)
                if d < -0.5:
                    # Perform 3-cycle: p[i]->p[j]->p[k]->p[i]
                    u, v, w = p[i], p[j], p[k]
                    p[i], p[j], p[k] = v, w, u
                    cost += d
                    # Update b_p and b_pT
                    b_p[i, :], b_p[j, :], b_p[k, :] = b_p[j, :].copy(), b_p[k, :].copy(), b_p[i, :].copy()
                    b_pT[:, i], b_pT[:, j], b_pT[:, k] = b_pT[:, j].copy(), b_pT[:, k].copy(), b_pT[:, i].copy()
                    best_p = p
                    best_cost = cost
                    improved = True
        return p, cost

    # --- Main loop ---
    # Phase 1: Initial local search
    best_p, best_cost = local_search_2swap(best_p, best_cost, b_p, b_pT, deadline - 30.0)
    
    # Phase 2: Hungarian refinement
    if time.time() < deadline - 20.0:
        best_p, best_cost = hungarian_refine(best_p, best_cost, deadline - 20.0)
    
    # Phase 3: More local search with 3-swaps
    if time.time() < deadline - 10.0:
        b_p, b_pT = _update_b_p(best_p)
        best_p, best_cost = local_search_3swap(best_p, best_cost, b_p, b_pT, deadline - 10.0)
    
    # Phase 4: Final 2-swap polish
    if time.time() < deadline - 2.0:
        b_p, b_pT = _update_b_p(best_p)
        best_p, best_cost = local_search_2swap(best_p, best_cost, b_p, b_pT, deadline - 2.0)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
