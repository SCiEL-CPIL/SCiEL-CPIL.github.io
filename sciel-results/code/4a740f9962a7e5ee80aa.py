"""Rewritten QAP solver: spectral init + linearized greedy + Hungarian + incremental local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Spectral initialization using eigenvectors of QAP-related matrices.
    2. Linearized greedy construction as alternative initialization.
    3. Hungarian-based linearization refinement.
    4. Fast local search with O(n) incremental updates to precomputed dot products.
    """
    n = len(a)
    deadline = time.time() + 115.0  # safety margin before 120s limit
    
    def compute_cost(p):
        """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
        total = 0
        for i in range(n):
            total += np.dot(a[i], b[p[i], p])
        return total
    
    # --- Spectral initialization ---
    def spectral_init(method='flow'):
        """Generate initial permutation using spectral methods."""
        if method == 'flow':
            a_centered = a - a.mean()
            try:
                eigvals, eigvecs = np.linalg.eigh(a_centered)
                idx = np.argsort(-np.abs(eigvals))
                spec_coords = eigvecs[:, idx[0]]
                fac_rank = np.argsort(spec_coords)
                loc_rank = np.argsort(b.sum(axis=1))
                p = np.empty(n, dtype=np.int64)
                p[fac_rank] = loc_rank
            except:
                heavy = np.argsort(-a.sum(axis=1), kind="stable")
                central = np.argsort(b.sum(axis=1), kind="stable")
                p = np.empty(n, dtype=np.int64)
                p[heavy] = central
            return p
        elif method == 'distance':
            b_centered = b - b.mean()
            try:
                eigvals, eigvecs = np.linalg.eigh(b_centered)
                idx = np.argsort(-np.abs(eigvals))
                spec_coords = eigvecs[:, idx[0]]
                loc_rank = np.argsort(spec_coords)
                fac_rank = np.argsort(a.sum(axis=1))
                p = np.empty(n, dtype=np.int64)
                p[fac_rank] = loc_rank
            except:
                heavy = np.argsort(-a.sum(axis=1), kind="stable")
                central = np.argsort(b.sum(axis=1), kind="stable")
                p = np.empty(n, dtype=np.int64)
                p[heavy] = central
            return p
        elif method == 'combined':
            a_centered = a - a.mean()
            b_centered = b - b.mean()
            try:
                eigvals_a, eigvecs_a = np.linalg.eigh(a_centered)
                eigvals_b, eigvecs_b = np.linalg.eigh(b_centered)
                idx_a = np.argsort(-np.abs(eigvals_a))
                idx_b = np.argsort(-np.abs(eigvals_b))
                spec_a = eigvecs_a[:, idx_a[0]]
                spec_b = eigvecs_b[:, idx_b[0]]
                fac_rank = np.argsort(spec_a)
                loc_rank = np.argsort(spec_b)
                p = np.empty(n, dtype=np.int64)
                p[fac_rank] = loc_rank
            except:
                heavy = np.argsort(-a.sum(axis=1), kind="stable")
                central = np.argsort(b.sum(axis=1), kind="stable")
                p = np.empty(n, dtype=np.int64)
                p[heavy] = central
            return p
        else:
            heavy = np.argsort(-a.sum(axis=1), kind="stable")
            central = np.argsort(b.sum(axis=1), kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
            return p
    
    # --- Linearized Greedy Construction ---
    def greedy_init():
        """Build permutation sequentially using dynamic marginal cost matrix."""
        p = np.empty(n, dtype=np.int64)
        assigned_loc = np.zeros(n, dtype=bool)  # which locations are taken
        assigned_fac = np.zeros(n, dtype=bool)  # which facilities are placed
        
        # Precompute row sums of a for quick marginal cost estimation
        a_row_sums = a.sum(axis=1)
        a_col_sums = a.sum(axis=0)
        
        for step in range(n):
            best_fac = -1
            best_loc = -1
            best_cost = np.inf
            
            for i in range(n):
                if assigned_fac[i]:
                    continue
                for j in range(n):
                    if assigned_loc[j]:
                        continue
                    # Marginal cost of placing facility i at location j given current partial assignment
                    # Approximation: a[i,:] dot b[j, assigned_locs] + a[:,i] dot b[assigned_locs, j]
                    # For speed, use precomputed partial sums
                    # Full cost contribution of (i,j) to total:
                    # sum_k a[i,k]*b[j,p[k]] + sum_k a[k,i]*b[p[k],j] + a[i,i]*b[j,j]
                    # For unassigned k, we don't know p[k] yet, so use average as proxy
                    # Better: just use the terms with already-assigned facilities
                    partial = 0.0
                    # Terms with already-assigned facilities
                    for k in range(n):
                        if assigned_fac[k]:
                            pk = p[k]
                            partial += a[i, k] * b[j, pk]
                            partial += a[k, i] * b[pk, j]
                    partial += a[i, i] * b[j, j]
                    
                    if partial < best_cost:
                        best_cost = partial
                        best_fac = i
                        best_loc = j
            
            if best_fac >= 0:
                p[best_fac] = best_loc
                assigned_fac[best_fac] = True
                assigned_loc[best_loc] = True
        
        return p
    
    # --- Local search with incremental O(n) updates ---
    def local_search(p, cost, time_limit):
        """Perform local search with O(n) incremental updates to dot_row/dot_col."""
        p = p.copy()
        
        # Initial precomputation: O(n^3)
        b_col_p = b[:, p]   # [l, k] = b[l, p[k]]
        b_row_p = b[p, :]   # [k, l] = b[p[k], l]
        dot_row = a @ b_col_p.T   # [i, l] = sum_k a[i,k]*b[l, p[k]]
        dot_col = a.T @ b_row_p   # [i, l] = sum_k a[k,i]*b[p[k], l]
        
        # Precompute constants for delta formula
        b_diag = np.diag(b)
        
        swap_count = 0
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            num_trials = 50000  # Can afford more since each trial is O(1)
            for _ in range(num_trials):
                if time.time() > time_limit:
                    break
                i, j = np.random.randint(0, n, 2)
                if i == j:
                    continue
                
                u = p[i]
                v = p[j]
                if u == v:
                    continue
                
                # Compute delta using precomputed dot products - O(1)
                delta = 0.0
                delta += a[i, i] * (b[v, v] - b[u, u])
                delta += a[i, j] * (b[v, u] - b[u, v])
                delta += a[j, i] * (b[u, v] - b[v, u])
                delta += a[j, j] * (b[u, u] - b[v, v])
                
                delta += (dot_row[i, v] - dot_row[i, u]) - a[i,i]*(b[v,v]-b[u,u]) - a[i,j]*(b[v,u]-b[u,v])
                delta += (dot_row[j, u] - dot_row[j, v]) - a[j,i]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
                delta += (dot_col[i, v] - dot_col[i, u]) - a[i,i]*(b[v,v]-b[u,u]) - a[j,i]*(b[v,u]-b[u,v])
                delta += (dot_col[j, u] - dot_col[j, v]) - a[i,j]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
                
                if delta < -1e-9:
                    # Accept the swap
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
                    swap_count += 1
                    
                    # Incremental O(n) update of dot_row and dot_col
                    # After swapping p[i]=u and p[j]=v:
                    # For dot_row[k, l] = sum_t a[k,t]*b[l, p[t]]:
                    #   Only p[i] and p[j] changed, so only terms t=i and t=j are affected
                    #   dot_row[k, l] changes by: a[k,i]*(b[l,v]-b[l,u]) + a[k,j]*(b[l,u]-b[l,v])
                    #   = (a[k,i]-a[k,j])*(b[l,v]-b[l,u])
                    # For dot_col[k, l] = sum_t a[t,k]*b[p[t], l]:
                    #   dot_col[k, l] changes by: (a[i,k]-a[j,k])*(b[v,l]-b[u,l])
                    
                    # Update dot_row: for all k, dot_row[k, :] += (a[k,i]-a[k,j])*(b[:,v]-b[:,u])
                    # (a[k,i]-a[k,j]) is vector of length n
                    # (b[:,v]-b[:,u]) is vector of length n
                    # dot_row[k, l] += (a[k,i]-a[k,j]) * (b[l,v]-b[l,u])
                    # This is an outer product update: dot_row += outer(a[:,i]-a[:,j], b[:,v]-b[:,u])
                    diff_a = a[:, i] - a[:, j]  # shape (n,)
                    diff_b_row = b[:, v] - b[:, u]  # shape (n,)
                    dot_row += np.outer(diff_a, diff_b_row)
                    
                    # Update dot_col: dot_col[k, l] += (a[i,k]-a[j,k]) * (b[v,l]-b[u,l])
                    # (a[i,k]-a[j,k]) = (a[i,:]-a[j,:])[k], shape (n,)
                    # (b[v,l]-b[u,l]) = (b[v,:]-b[u,:])[l], shape (n,)
                    diff_a_col = a[i, :] - a[j, :]  # shape (n,)
                    diff_b_col = b[v, :] - b[u, :]  # shape (n,)
                    dot_col += np.outer(diff_a_col, diff_b_col)
        
        return p, cost
    
    # --- Try multiple initial solutions and keep the best ---
    init_methods = ['flow', 'distance', 'combined', 'default']
    best_p = None
    best_cost = np.inf
    
    for method in init_methods:
        if time.time() > deadline - 60.0:
            break
        p_init = spectral_init(method)
        cost_init = compute_cost(p_init)
        if cost_init < best_cost:
            best_cost = cost_init
            best_p = p_init.copy()
    
    # Also try greedy construction
    if time.time() < deadline - 60.0:
        p_greedy = greedy_init()
        cost_greedy = compute_cost(p_greedy)
        if cost_greedy < best_cost:
            best_cost = cost_greedy
            best_p = p_greedy.copy()
    
    # --- Main local search ---
    time_limit = deadline - 25.0
    best_p, best_cost = local_search(best_p, best_cost, time_limit)
    
    # --- Hungarian-based refinement ---
    if time.time() < deadline - 15.0:
        b_col_p = b[:, best_p]
        b_row_p = b[best_p, :]
        C = a @ b_col_p.T + a.T @ b_row_p
        row_ind, col_ind = linear_sum_assignment(C)
        p_hung = np.empty(n, dtype=np.int64)
        p_hung[row_ind] = col_ind
        cost_hung = compute_cost(p_hung)
        if cost_hung < best_cost:
            best_p = p_hung
            best_cost = cost_hung
    
    # --- Final local search pass ---
    if time.time() < deadline - 5.0:
        time_limit = deadline - 3.0
        best_p, best_cost = local_search(best_p, best_cost, time_limit)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
