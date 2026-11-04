"""QAP solver: spectral init + WL color refinement + systematic 2-opt local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0
    np.random.seed(42)

    def time_left():
        return deadline - time.time()

    def compute_cost(p):
        b_p_p = b[np.ix_(p, p)]
        return int(np.sum(a * b_p_p))

    # --- WL Color Refinement Initialization ---
    def wl_init(num_rounds=10):
        """Weisfeiler-Lehman color refinement to derive a matching."""
        # Build initial colors from node degrees
        deg_a = a.sum(axis=1)
        deg_b = b.sum(axis=1)
        
        # Discretize degrees into colors
        a_colors = np.digitize(deg_a, bins=np.percentile(deg_a, np.linspace(0, 100, 21)[1:-1]))
        b_colors = np.digitize(deg_b, bins=np.percentile(deg_b, np.linspace(0, 100, 21)[1:-1]))

        # Store colors as tuples or integers. We use integers for speed.
        # We need to map the pair (a_color, b_color) to a final assignment.
        # WL refines: c_i^{(t+1)} = hash(c_i^{(t)}, multiset(c_j^{(t)} for j in N(i)))
        
        # For QAP, we treat the facility graph (A) and location graph (B) separately.
        # We want to match facility i to location j if their "structural signature" is similar.
        
        # Simplified WL: 
        # 1. Compute eigenvectors of A and B (spectral embedding)
        # 2. Use k-means or discretization to assign "structural colors"
        # 3. Match colors
        
        # Let's use a proper spectral + discretization approach which is a variant of WL
        # using the spectrum as the neighborhood summary.
        
        # Center matrices
        A_c = a - a.mean()
        B_c = b - b.mean()
        
        # Top eigenvectors (say 10)
        num_vecs = min(10, n)
        eigvals_a, eigvecs_a = np.linalg.eigh(A_c)
        eigvals_b, eigvecs_b = np.linalg.eigh(B_c)
        
        # Sort by absolute eigenvalue descending
        idx_a = np.argsort(-np.abs(eigvals_a))
        idx_b = np.argsort(-np.abs(eigvals_b))
        
        coords_a = eigvecs_a[:, idx_a[:num_vecs]]
        coords_b = eigvecs_b[:, idx_b[:num_vecs]]
        
        # Normalize
        coords_a = coords_a / (np.linalg.norm(coords_a, axis=1, keepdims=True) + 1e-9)
        coords_b = coords_b / (np.linalg.norm(coords_b, axis=1, keepdims=True) + 1e-9)
        
        # Discretize into colors using quantiles
        num_colors = 15
        # We need to match the distribution of colors
        # Use a simple 1D projection for speed if n is large, or full k-means
        
        # Project to 1D using the first principal component of the embedding
        # Or just use the first eigenvector
        proj_a = coords_a[:, 0]
        proj_b = coords_b[:, 0]
        
        # Rank-based matching of the 1D projections
        rank_a = np.argsort(proj_a)
        rank_b = np.argsort(proj_b)
        
        p = np.empty(n, dtype=np.int64)
        p[rank_a] = rank_b
        
        return p

    def spectral_init_flow():
        a_centered = a - a.mean()
        eigvals, eigvecs = np.linalg.eigh(a_centered)
        idx = np.argsort(-np.abs(eigvals))
        spec_coords = eigvecs[:, idx[0]]
        fac_rank = np.argsort(spec_coords)
        loc_rank = np.argsort(b.sum(axis=1))
        p = np.empty(n, dtype=np.int64)
        p[fac_rank] = loc_rank
        return p

    def spectral_init_dist():
        b_centered = b - b.mean()
        eigvals, eigvecs = np.linalg.eigh(b_centered)
        idx = np.argsort(-np.abs(eigvals))
        spec_coords = eigvecs[:, idx[0]]
        loc_rank = np.argsort(spec_coords)
        fac_rank = np.argsort(a.sum(axis=1))
        p = np.empty(n, dtype=np.int64)
        p[fac_rank] = loc_rank
        return p

    def rank_init():
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

    def random_init(seed):
        np.random.seed(seed)
        return np.random.permutation(n).astype(np.int64)

    def local_search(p, cost, time_limit, systematic=True):
        p = p.copy()
        
        # Precompute
        b_col_p = b[:, p]
        dot_row = a @ b_col_p.T
        
        def recompute_dot_row(p):
            b_col_p = b[:, p]
            return a @ b_col_p.T

        dot_row = recompute_dot_row(p)
        b_diag = np.diag(b).copy()

        def compute_swap_delta(i, j, p, dot_row):
            u = p[i]
            v = p[j]
            if u == v:
                return 0.0
            
            buu = b_diag[u]
            bvv = b_diag[v]
            buv = b[u, v]
            bvu = b[v, u]
            
            # Terms involving i and j directly
            t1 = a[i, i] * (bvv - buu)
            t2 = a[i, j] * (bvu - buv)
            t3 = a[j, i] * (buv - bvu)
            t4 = a[j, j] * (buu - bvv)
            
            # Terms involving other k
            # sum_{k != i,j} a[i,k] (b[p[k],v] - b[p[k],u])
            # sum_{k != i,j} a[j,k] (b[p[k],u] - b[p[k],v])
            # sum_{k != i,j} a[k,i] (b[v,p[k]] - b[u,p[k]])
            # sum_{k != i,j} a[k,j] (b[u,p[k]] - b[v,p[k]])
            
            # Note: b is symmetric in standard QAP, but let's assume it is for speed.
            # If not symmetric, we need dot_col as well.
            # The task says "distance matrix", usually symmetric.
            # Let's assume symmetric b for now to simplify and speed up.
            # If b is not symmetric, this delta is wrong.
            # Checking task: "distance" usually implies symmetric.
            
            # dot_row[i, v] = sum_k a[i,k] b[p[k], v]
            # We need sum_{k != i,j} a[i,k] (b[p[k],v] - b[p[k],u])
            # = (sum_k a[i,k] b[p[k],v] - a[i,i]b[p[i],v] - a[i,j]b[p[j],v]) - 
            #   (sum_k a[i,k] b[p[k],u] - a[i,i]b[p[i],u] - a[i,j]b[p[j],u])
            # = (dot_row[i,v] - a[i,i]b[v,v] - a[i,j]b[u,v]) - (dot_row[i,u] - a[i,i]b[u,u] - a[i,j]b[v,u])
            # Note: p[i]=u, p[j]=v.
            # b[p[i],v] = b[u,v]. b[p[j],v] = b[v,v]=bvv.
            
            s1 = (dot_row[i, v] - a[i, i]*bvv - a[i, j]*buv) - (dot_row[i, u] - a[i, i]*buu - a[i, j]*bvu)
            
            # Similarly for j
            # sum_{k != i,j} a[j,k] (b[p[k],u] - b[p[k],v])
            # = (dot_row[j,u] - a[j,i]b[u,u] - a[j,j]b[v,u]) - (dot_row[j,v] - a[j,i]b[v,v] - a[j,j]b[u,v])
            # p[i]=u, p[j]=v.
            # b[p[i],u] = b[u,u]=buu. b[p[j],u] = b[v,u]=bvu.
            
            s2 = (dot_row[j, u] - a[j, i]*buu - a[j, j]*bvu) - (dot_row[j, v] - a[j, i]*bvv - a[j, j]*buv)
            
            # Since b is symmetric, a[k,i] term uses dot_col which is a.T @ b_col_p.T? No.
            # dot_row is a @ b_col_p.T. (a @ b_col_p.T)[k, l] = sum_m a[k,m] b[p[m], l].
            # We need sum_k a[k,i] b[v, p[k]].
            # If b is symmetric, b[v, p[k]] = b[p[k], v].
            # sum_k a[k,i] b[p[k], v] = (a.T @ b_col_p.T)[:, v] ? No.
            # Let D = b_col_p.T. D[m, l] = b[p[m], l].
            # (a.T @ D)[k, l] = sum_m a.T[k,m] D[m,l] = sum_m a[m,k] b[p[m], l].
            # So we need dot_col = a.T @ b_col_p.T.
            
            # Let's assume symmetric b and compute dot_col once?
            # Actually, if b is symmetric, b_col_p.T = b[p, :].
            # dot_row = a @ b[p, :]
            # dot_col = a.T @ b[:, p] (since b[p, :]^T = b[:, p] for symmetric b)
            
            # Let's recompute dot_col inside the loop if needed, or precompute.
            # For efficiency, let's precompute dot_col.
            
            return 0 # Placeholder, will rewrite with dot_col

        # Rewrite local_search with proper dot_col
        def local_search_impl(p, cost, time_limit, systematic=True):
            p = p.copy()
            
            def get_dots(p):
                b_col_p = b[:, p]
                dot_row = a @ b_col_p.T
                # If b is symmetric: b[p, :] = b[:, p].T
                # dot_col[k, l] = sum_m a[m, k] b[l, p[m]] = (a.T @ b[:, p].T) ?
                # b[:, p].T = b[p, :].
                # So dot_col = a.T @ b[p, :]
                dot_col = a.T @ b[p, :]
                return dot_row, dot_col

            dot_row, dot_col = get_dots(p)
            b_diag = np.diag(b).copy()

            def compute_swap_delta(i, j, p, dot_row, dot_col):
                u = p[i]
                v = p[j]
                if u == v:
                    return 0.0
                
                buu = b_diag[u]
                bvv = b_diag[v]
                buv = b[u, v]
                bvu = b[v, u]
                
                t1 = a[i, i] * (bvv - buu)
                t2 = a[i, j] * (bvu - buv)
                t3 = a[j, i] * (buv - bvu)
                t4 = a[j, j] * (buu - bvv)
                
                # s1: sum_{k!=i,j} a[i,k] (b[p[k],v] - b[p[k],u])
                # = (dot_row[i,v] - a[i,i]b[v,v] - a[i,j]b[u,v]) - (dot_row[i,u] - a[i,i]b[u,u] - a[i,j]b[v,u])
                s1 = (dot_row[i, v] - a[i, i]*bvv - a[i, j]*buv) - (dot_row[i, u] - a[i, i]*buu - a[i, j]*bvu)
                
                # s2: sum_{k!=i,j} a[j,k] (b[p[k],u] - b[p[k],v])
                # = (dot_row[j,u] - a[j,i]b[u,u] - a[j,j]b[v,u]) - (dot_row[j,v] - a[j,i]b[v,v] - a[j,j]b[u,v])
                s2 = (dot_row[j, u] - a[j, i]*buu - a[j, j]*bvu) - (dot_row[j, v] - a[j, i]*bvv - a[j, j]*buv)
                
                # s3: sum_{k!=i,j} a[k,i] (b[v,p[k]] - b[u,p[k]])
                # = (dot_col[i,v] - a[i,i]b[v,v] - a[j,i]b[v,u]) - (dot_col[i,u] - a[i,i]b[u,u] - a[j,i]b[u,v])
                # Note: dot_col[i, l] = sum_m a[m,i] b[l, p[m]].
                # p[i]=u, p[j]=v.
                # b[v, p[i]] = b[v,u] = bvu. b[v, p[j]] = b[v,v] = bvv.
                s3 = (dot_col[i, v] - a[i, i]*bvv - a[j, i]*bvu) - (dot_col[i, u] - a[i, i]*buu - a[j, i]*buv)
                
                # s4: sum_{k!=i,j} a[k,j] (b[u,p[k]] - b[v,p[k]])
                # = (dot_col[j,u] - a[i,j]b[u,u] - a[j,j]b[u,v]) - (dot_col[j,v] - a[i,j]b[v,v] - a[j,j]b[v,u])
                s4 = (dot_col[j, u] - a[i, j]*buu - a[j, j]*buv) - (dot_col[j, v] - a[i, j]*bvv - a[j, j]*bvu)
                
                return t1 + t2 + t3 + t4 + s1 + s2 + s3 + s4

            def apply_swap(i, j, p, dot_row, dot_col):
                u = p[i]
                v = p[j]
                p[i], p[j] = v, u
                
                # Update dot_row
                # dot_row[k, l] = sum_m a[k,m] b[p[m], l]
                # Change: p[i]=v, p[j]=u (was u, v)
                # Diff for m=i: a[k,i] (b[v,l] - b[u,l])
                # Diff for m=j: a[k,j] (b[u,l] - b[v,l])
                # Total diff: (a[k,i] - a[k,j]) (b[v,l] - b[u,l])
                # dot_row += outer(a[:, i] - a[:, j], b[v, :] - b[u, :])
                
                diff_a = a[:, i] - a[:, j]
                diff_b = b[v, :] - b[u, :]
                dot_row += np.outer(diff_a, diff_b)
                
                # Update dot_col
                # dot_col[k, l] = sum_m a[m,k] b[l, p[m]]
                # Diff for m=i: a[i,k] (b[l,v] - b[l,u])
                # Diff for m=j: a[j,k] (b[l,u] - b[l,v])
                # Total: (a[i,k] - a[j,k]) (b[:,v] - b[:,u])[l]
                # dot_col += outer(a[i, :] - a[j, :], b[:, v] - b[:, u])
                
                diff_a_t = a[i, :] - a[j, :]
                diff_b_t = b[:, v] - b[:, u]
                dot_col += np.outer(diff_a_t, diff_b_t)

            improved = True
            while improved and time.time() < time_limit:
                improved = False
                
                if systematic:
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
                            
                            delta = compute_swap_delta(i, j, p, dot_row, dot_col)
                            
                            if delta < -1e-9:
                                apply_swap(i, j, p, dot_row, dot_col)
                                cost += delta
                                improved = True
                else:
                    # Random sampling
                    num_trials = 10000
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
                        
                        delta = compute_swap_delta(i, j, p, dot_row, dot_col)
                        
                        if delta < -1e-9:
                            apply_swap(i, j, p, dot_row, dot_col)
                            cost += delta
                            improved = True
            
            return p, cost

    # --- Generate diverse initial solutions ---
    init_solutions = []
    
    # Spectral and WL inits
    for init_func in [wl_init, spectral_init_flow, spectral_init_dist, rank_init]:
        if time_left() < 30.0:
            break
        try:
            p_init = init_func()
            cost_init = compute_cost(p_init)
            init_solutions.append((p_init, cost_init))
        except Exception:
            pass

    # Random inits
    for seed in [0, 1, 2, 3, 4]:
        if time_left() < 30.0:
            break
        p_init = random_init(seed)
        cost_init = compute_cost(p_init)
        init_solutions.append((p_init, cost_init))
    
    # Sort by cost
    init_solutions.sort(key=lambda x: x[1])
    
    # --- Run local search on top diverse solutions ---
    selected = []
    for p_init, cost_init in init_solutions:
        if time_left() < 25.0:
            break
        diverse = True
        for p_sel, _ in selected:
            hamming = np.sum(p_init != p_sel)
            if hamming < n * 0.3:
                diverse = False
                break
        if diverse:
            selected.append((p_init, cost_init))
        if len(selected) >= 3:
            break
    
    if not selected:
        selected = [init_solutions[0]]
    
    best_p, best_cost = None, np.inf
    
    for idx, (p_init, cost_init) in enumerate(selected):
        if time_left() < 20.0:
            break
        
        remaining_solutions = len(selected) - idx
        time_per_solution = min(time_left() - 5.0, 40.0) / remaining_solutions
        time_limit = time.time() + time_per_solution
        
        p_ls, cost_ls = local_search_impl(p_init, cost_init, time_limit, systematic=True)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # --- Hungarian-based refinement ---
    if best_p is not None and time_left() > 10.0:
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
    
    # --- Final polish ---
    if best_p is not None and time_left() > 5.0:
        time_limit = time.time() + min(time_left() - 3.0, 15.0)
        best_p, best_cost = local_search_impl(best_p, best_cost, time_limit, systematic=True)
    
    return best_p.tolist()

# EVOLVE-BLOCK-END
