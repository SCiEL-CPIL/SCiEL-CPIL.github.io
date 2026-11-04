"""QAP solver: spectral initialization + iterative linearized assignment + fast incremental local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Multiple spectral/rank initializations.
    2. Iterative linearized assignment (Hungarian) to escape local minima.
    3. Fast local search with O(1) delta computation via precomputed dot products.
    4. Final high-precision polish.
    """
    n = len(a)
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin before 120s limit (evaluator kills at 150s)
    
    # Ensure a and b are float64 for precision
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    def compute_cost(p):
        """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
        # Vectorized: sum_i a[i, :] @ b[p[i], p]
        # b[p, p] is the permuted matrix
        return np.sum(a * b[np.ix_(p, p)])
    
    # --- Precompute matrix properties for initialization ---
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
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
                loc_rank = np.argsort(b_sums)
                p = np.empty(n, dtype=np.int64)
                p[fac_rank] = loc_rank
            except:
                heavy = np.argsort(-a_sums, kind="stable")
                central = np.argsort(b_sums, kind="stable")
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
                fac_rank = np.argsort(a_sums)
                p = np.empty(n, dtype=np.int64)
                p[fac_rank] = loc_rank
            except:
                heavy = np.argsort(-a_sums, kind="stable")
                central = np.argsort(b_sums, kind="stable")
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
                heavy = np.argsort(-a_sums, kind="stable")
                central = np.argsort(b_sums, kind="stable")
                p = np.empty(n, dtype=np.int64)
                p[heavy] = central
            return p
        else:
            # Default: rank matching
            heavy = np.argsort(-a_sums, kind="stable")
            central = np.argsort(b_sums, kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
            return p
    
    # --- Precomputation for fast local search ---
    # dot_row[i, l] = sum_k a[i,k] * b[l, p[k]]
    # dot_col[i, l] = sum_k a[k,i] * b[p[k], l]
    
    def recompute_precomp(p):
        """Recompute dot_row and dot_col for current permutation p."""
        # b[:, p] has shape (n, n): [l, k] = b[l, p[k]]
        b_col_p = b[:, p]
        # b[p, :] has shape (n, n): [k, l] = b[p[k], l]
        b_row_p = b[p, :]
        
        # dot_row[i, l] = sum_k a[i,k] * b[l, p[k]] = (a @ b_col_p.T)[i, l]
        dot_row = a @ b_col_p.T
        
        # dot_col[i, l] = sum_k a[k,i] * b[p[k], l] = (a.T @ b_row_p)[i, l]
        dot_col = a.T @ b_row_p
        
        return dot_row, dot_col
    
    def local_search(p, cost, time_limit, max_passes=100):
        """Perform local search starting from p with initial cost.
        
        Uses precomputed dot products for O(1) delta computation per swap.
        """
        p = p.copy()
        dot_row, dot_col = recompute_precomp(p)
        
        # Precompute b differences for common pairs? No, just compute on the fly.
        # Precompute a[i,i], etc? They are constant.
        
        for _pass in range(max_passes):
            if time.time() > time_limit:
                break
            
            improved = False
            # Systematic sweep is better for local search than random for convergence
            # But O(n^2) sweep with O(1) delta is fast enough for n=150 (22500 pairs)
            # Each pass is ~22500 * (few flops) which is very fast.
            
            for i in range(n):
                if time.time() > time_limit:
                    return p, cost
                for j in range(i + 1, n):
                    if time.time() > time_limit:
                        return p, cost
                    
                    u = p[i]
                    v = p[j]
                    if u == v:
                        continue
                    
                    # Compute delta using precomputed dot products
                    # Terms 1-4 (diagonal and cross terms involving i,j directly)
                    # delta = sum_{k,l} (a[i,k]*b[v,p[k]] + a[j,k]*b[u,p[k]] + a[k,i]*b[p[k],v] + a[k,j]*b[p[k],u]) 
                    #       - (a[i,k]*b[u,p[k]] + a[j,k]*b[v,p[k]] + a[k,i]*b[p[k],u] + a[k,j]*b[p[k],v])
                    #       - (terms where k=i or k=j are handled separately in the dot products)
                    
                    # Let's use the derived formula:
                    # delta = (dot_row[i, v] - dot_row[i, u]) + (dot_row[j, u] - dot_row[j, v])
                    #       + (dot_col[i, v] - dot_col[i, u]) + (dot_col[j, u] - dot_col[j, v])
                    #       - 2 * (a[i,i]*(b[v,v]-b[u,u]) + a[i,j]*(b[v,u]-b[u,v]) + a[j,i]*(b[u,v]-b[v,u]) + a[j,j]*(b[u,u]-b[v,v]))
                    # Wait, let's re-derive carefully.
                    
                    # Cost(p) = sum_{x,y} a[x,y] b[p[x], p[y]]
                    # Swap i and j in p. Let u=p[i], v=p[j].
                    # New cost = Cost(p) - (terms involving i or j) + (new terms involving i or j)
                    
                    # Terms involving i or j in old cost:
                    # 1. x=i, y=i: a[i,i] b[u,u]
                    # 2. x=i, y=j: a[i,j] b[u,v]
                    # 3. x=j, y=i: a[j,i] b[v,u]
                    # 4. x=j, y=j: a[j,j] b[v,v]
                    # 5. x=k (k!=i,j), y=i: a[k,i] b[p[k], u]
                    # 6. x=i, y=k (k!=i,j): a[i,k] b[u, p[k]]
                    # 7. x=k (k!=i,j), y=j: a[k,j] b[p[k], v]
                    # 8. x=j, y=k (k!=i,j): a[j,k] b[v, p[k]]
                    
                    # New terms (swap u and v):
                    # 1'. x=i, y=i: a[i,i] b[v,v]
                    # 2'. x=i, y=j: a[i,j] b[v,u]
                    # 3'. x=j, y=i: a[j,i] b[u,v]
                    # 4'. x=j, y=j: a[j,j] b[u,u]
                    # 5'. x=k, y=i: a[k,i] b[p[k], v]
                    # 6'. x=i, y=k: a[i,k] b[v, p[k]]
                    # 7'. x=k, y=j: a[k,j] b[p[k], u]
                    # 8'. x=j, y=k: a[j,k] b[u, p[k]]
                    
                    # Delta = (1'-1) + (2'-2) + (3'-3) + (4'-4) + (5'-5) + (6'-6) + (7'-7) + (8'-8)
                    
                    # 1'-1 = a[i,i] (b[v,v] - b[u,u])
                    # 2'-2 = a[i,j] (b[v,u] - b[u,v])
                    # 3'-3 = a[j,i] (b[u,v] - b[v,u])
                    # 4'-4 = a[j,j] (b[u,u] - b[v,v])
                    
                    # 5'-5 = sum_{k!=i,j} a[k,i] (b[p[k], v] - b[p[k], u])
                    # 6'-6 = sum_{k!=i,j} a[i,k] (b[v, p[k]] - b[u, p[k]])
                    # 7'-7 = sum_{k!=i,j} a[k,j] (b[p[k], u] - b[p[k], v])
                    # 8'-8 = sum_{k!=i,j} a[j,k] (b[u, p[k]] - b[v, p[k]])
                    
                    # Using dot_row and dot_col:
                    # dot_row[i, l] = sum_k a[i,k] b[l, p[k]]
                    # dot_col[i, l] = sum_k a[k,i] b[p[k], l]
                    
                    # 6' - 6 = sum_k a[i,k] b[v, p[k]] - sum_k a[i,k] b[u, p[k]] - (k=i term) - (k=j term)
                    # = dot_row[i, v] - dot_row[i, u] - a[i,i]b[v,u] - a[i,j]b[v,v] + a[i,i]b[u,u] + a[i,j]b[u,v]
                    # = dot_row[i, v] - dot_row[i, u] - a[i,i](b[v,u] - b[u,u]) - a[i,j](b[v,v] - b[u,v])
                    
                    # 8' - 8 = sum_k a[j,k] b[u, p[k]] - sum_k a[j,k] b[v, p[k]] - (k=i term) - (k=j term)
                    # = dot_row[j, u] - dot_row[j, v] - a[j,i]b[u,v] - a[j,j]b[u,u] + a[j,i]b[v,u] + a[j,j]b[v,v]
                    # = dot_row[j, u] - dot_row[j, v] - a[j,i](b[u,v] - b[v,u]) - a[j,j](b[u,u] - b[v,v])
                    
                    # 5' - 5 = sum_k a[k,i] b[p[k], v] - sum_k a[k,i] b[p[k], u] - (k=i term) - (k=j term)
                    # = dot_col[i, v] - dot_col[i, u] - a[i,i]b[v,v] - a[j,i]b[p[j],v] + a[i,i]b[u,u] + a[j,i]b[p[j],u]
                    # Note: p[j] is v. So b[p[j],v] = b[v,v]. b[p[j],u] = b[v,u].
                    # = dot_col[i, v] - dot_col[i, u] - a[i,i](b[v,v] - b[u,u]) - a[j,i](b[v,v] - b[v,u])
                    # Wait, k=j term in sum_k a[k,i] b[p[k], l] is a[j,i] b[p[j], l] = a[j,i] b[v, l].
                    # So subtract a[j,i] b[v,v] for l=v and a[j,i] b[v,u] for l=u.
                    # = dot_col[i, v] - dot_col[i, u] - a[i,i](b[v,v] - b[u,u]) - a[j,i](b[v,v] - b[v,u])
                    
                    # 7' - 7 = sum_k a[k,j] b[p[k], u] - sum_k a[k,j] b[p[k], v] - (k=i term) - (k=j term)
                    # = dot_col[j, u] - dot_col[j, v] - a[i,j]b[p[i],u] - a[j,j]b[p[j],u] + a[i,j]b[p[i],v] + a[j,j]b[p[j],v]
                    # p[i]=u, p[j]=v.
                    # = dot_col[j, u] - dot_col[j, v] - a[i,j](b[u,u] - b[u,v]) - a[j,j](b[u,u] - b[v,v])
                    
                    # This is getting complicated to combine. Let's just sum the 8 parts directly.
                    
                    d1 = a[i, i] * (b[v, v] - b[u, u])
                    d2 = a[i, j] * (b[v, u] - b[u, v])
                    d3 = a[j, i] * (b[u, v] - b[v, u])
                    d4 = a[j, j] * (b[u, u] - b[v, v])
                    
                    # 6-6 part
                    d6 = (dot_row[i, v] - dot_row[i, u]) - a[i,i]*(b[v,u] - b[u,u]) - a[i,j]*(b[v,v] - b[u,v])
                    # 8-8 part
                    d8 = (dot_row[j, u] - dot_row[j, v]) - a[j,i]*(b[u,v] - b[v,u]) - a[j,j]*(b[u,u] - b[v,v])
                    # 5-5 part
                    d5 = (dot_col[i, v] - dot_col[i, u]) - a[i,i]*(b[v,v] - b[u,u]) - a[j,i]*(b[v,v] - b[v,u])
                    # 7-7 part
                    d7 = (dot_col[j, u] - dot_col[j, v]) - a[i,j]*(b[u,u] - b[u,v]) - a[j,j]*(b[u,u] - b[v,v])
                    
                    delta = d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8
                    
                    if delta < -1e-9:
                        # Accept swap
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        
                        # Incremental update of dot_row and dot_col
                        # dot_row[k, l] = sum_m a[k,m] b[l, p[m]]
                        # When we swap p[i] and p[j], the term for m=i changes from b[l, u] to b[l, v]
                        # and m=j changes from b[l, v] to b[l, u].
                        # Delta in dot_row[k, l] = a[k,i](b[l, v] - b[l, u]) + a[k,j](b[l, u] - b[l, v])
                        # = (a[k,i] - a[k,j]) (b[l, v] - b[l, u])
                        
                        diff_a_col = a[:, i] - a[:, j]  # shape (n,)
                        diff_b_row = b[v, :] - b[u, :]  # shape (n,)
                        dot_row += np.outer(diff_a_col, diff_b_row)
                        
                        # dot_col[k, l] = sum_m a[m,k] b[p[m], l]
                        # Delta in dot_col[k, l] = a[i,k](b[v, l] - b[u, l]) + a[j,k](b[u, l] - b[v, l])
                        # = (a[i,k] - a[j,k]) (b[v, l] - b[u, l])
                        
                        diff_a_row = a[i, :] - a[j, :]  # shape (n,)
                        # diff_b_row is already b[v, :] - b[u, :]
                        dot_col += np.outer(diff_a_row, diff_b_row)
            
            if not improved:
                break
        
        return p, cost
    
    # --- Main Loop ---
    best_p = None
    best_cost = np.inf
    
    # 1. Try multiple initializations
    init_methods = ['flow', 'distance', 'combined', 'default']
    for method in init_methods:
        if time.time() > deadline - 60.0:
            break
        p_init = spectral_init(method)
        cost_init = compute_cost(p_init)
        if cost_init < best_cost:
            best_cost = cost_init
            best_p = p_init.copy()
    
    # 2. Iterative Linearized Assignment + Local Search
    # Alternate between Hungarian assignment and local search
    # This helps escape local minima
    
    while time.time() < deadline - 10.0:
        # Step 1: Linearized Assignment
        # C[i, j] = sum_k a[i, k] * b[j, p[k]]
        # C = a @ b[:, p].T
        b_col_p = b[:, best_p]
        C = a @ b_col_p.T
        row_ind, col_ind = linear_sum_assignment(C)
        p_new = col_ind
        
        cost_new = compute_cost(p_new)
        
        if cost_new < best_cost:
            best_p = p_new
            best_cost = cost_new
        
        # Step 2: Local Search on the new solution
        if time.time() < deadline - 5.0:
            p_ls, cost_ls = local_search(p_new, cost_new, deadline - 5.0)
            if cost_ls < best_cost:
                best_p = p_ls
                best_cost = cost_ls
        else:
            break
    
    # 3. Final Polish
    if time.time() < deadline - 2.0:
        p_final, cost_final = local_search(best_p, best_cost, deadline - 1.0)
        if cost_final < best_cost:
            best_p = p_final
            best_cost = cost_final
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
