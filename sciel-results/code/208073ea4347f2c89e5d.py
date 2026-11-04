"""QAP solver: spectral init + fast incremental local search + Hungarian refinement + SA escape."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Spectral initialization using eigenvectors of QAP-related matrices.
    2. Fast local search with O(n) incremental updates per swap (vectorized delta).
    3. Hungarian-based linearization refinement.
    4. Simulated annealing escape for local minima.
    5. Multi-start with diverse initial solutions.
    """
    n = len(a)
    deadline = time.time() + 115.0  # safety margin before 120s limit
    np.random.seed(42)
    
    # Precompute constants that don't change
    b_diag = np.diag(b).astype(np.float64)
    
    def compute_cost(p):
        """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
        return np.sum(a * b[np.ix_(p, p)])
    
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
        elif method == 'multi_eig':
            # Use multiple eigenvectors for richer embedding
            a_centered = a - a.mean()
            b_centered = b - b.mean()
            try:
                eigvals_a, eigvecs_a = np.linalg.eigh(a_centered)
                eigvals_b, eigvecs_b = np.linalg.eigh(b_centered)
                idx_a = np.argsort(-np.abs(eigvals_a))[:3]
                idx_b = np.argsort(-np.abs(eigvals_b))[:3]
                # Project to 1D using weighted sum of top eigenvectors
                w = np.array([1.0, 0.5, 0.25])
                spec_a = eigvecs_a[:, idx_a[0]] * w[0] + eigvecs_a[:, idx_a[1]] * w[1] + eigvecs_a[:, idx_a[2]] * w[2]
                spec_b = eigvecs_b[:, idx_b[0]] * w[0] + eigvecs_b[:, idx_b[1]] * w[1] + eigvecs_b[:, idx_b[2]] * w[2]
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
    
    # --- Fast local search with vectorized delta computation ---
    def local_search(p, cost, time_limit, max_iter=None):
        """Perform local search with fast vectorized delta updates.
        
        Uses the formula:
        delta(i,j) = (a[i,:]-a[j,:]) @ (b[p[i],p]-b[p[j],p]) 
                    + (a[:,i]-a[:,j]) @ (b[p,p[i]]-b[p,p[j]])
                    - (a[i,i]-a[j,j])*(b[p[i],p[i]]-b[p[j],p[j]])
                    - (a[i,j]-a[j,i])*(b[p[i],p[j]]-b[p[j],p[i]])
                    - (a[i,i]-a[j,i])*(b[p[j],p[i]]-b[p[i],p[i]])
                    - (a[i,j]-a[j,j])*(b[p[j],p[j]]-b[p[i],p[j]])
        """
        p = p.copy()
        
        def recompute_precomp(p):
            b_col_p = b[:, p]  # (n, n): [l, k] = b[l, p[k]]
            b_row_p = b[p, :]  # (n, n): [k, l] = b[p[k], l]
            dot_row = a @ b_col_p.T
            dot_col = a.T @ b_row_p
            return dot_row, dot_col
        
        dot_row, dot_col = recompute_precomp(p)
        
        improved = True
        iters = 0
        while improved and time.time() < time_limit:
            improved = False
            
            if max_iter is not None:
                num_trials = max_iter
            else:
                num_trials = 30000
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
                
                # Vectorized delta computation
                # delta = sum_k (a[i,k]-a[j,k])*(b[v,p[k]]-b[u,p[k]])
                #       + sum_k (a[k,i]-a[k,j])*(b[p[k],v]-b[p[k],u])
                #       - corrections for k=i and k=j terms
                
                diff_a_row = a[i, :] - a[j, :]  # (n,)
                diff_b_row = b[v, p] - b[u, p]  # (n,)
                
                diff_a_col = a[:, i] - a[:, j]  # (n,)
                diff_b_col = b[p, v] - b[p, u]  # (n,)
                
                # Full dot products
                term_row = np.dot(diff_a_row, diff_b_row)
                term_col = np.dot(diff_a_col, diff_b_col)
                
                # Subtract k=i terms:
                # From term_row: (a[i,i]-a[j,i]) * (b[v,u] - b[u,u])
                # From term_col: (a[i,i]-a[i,j]) * (b[u,v] - b[v,v])
                # Subtract k=j terms:
                # From term_row: (a[i,j]-a[j,j]) * (b[v,v] - b[u,v])
                # From term_col: (a[j,i]-a[j,j]) * (b[v,u] - b[u,v])
                
                buu = b_diag[u]
                bvv = b_diag[v]
                buv = b[u, v]
                bvu = b[v, u]
                
                corr_ri = (a[i, i] - a[j, i]) * (bvu - buu)
                corr_rj = (a[i, j] - a[j, j]) * (bvv - buv)
                corr_ci = (a[i, i] - a[i, j]) * (buv - bvv)
                corr_cj = (a[j, i] - a[j, j]) * (bvu - buv)
                
                delta = term_row + term_col - corr_ri - corr_rj - corr_ci - corr_cj
                
                if delta < -1e-9:
                    # Accept the swap
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
                    iters += 1
                    
                    # Incremental update of dot_row and dot_col in O(n^2)
                    diff_a = a[:, i] - a[:, j]
                    diff_b = b[:, v] - b[:, u]
                    dot_row += np.outer(diff_a, diff_b)
                    
                    diff_a2 = a[i, :] - a[j, :]
                    diff_b2 = b[v, :] - b[u, :]
                    dot_col += np.outer(diff_a2, diff_b2)
        
        return p, cost
    
    # --- Simulated Annealing escape ---
    def sa_escape(p, cost, time_limit, T_init=100.0, T_min=0.5, alpha=0.9995):
        """Simulated annealing to escape local minima."""
        p = p.copy()
        T = T_init
        
        while T > T_min and time.time() < time_limit:
            i, j = np.random.randint(0, n, 2)
            if i == j:
                continue
            
            u = p[i]
            v = p[j]
            if u == v:
                continue
            
            # Compute delta
            diff_a_row = a[i, :] - a[j, :]
            diff_b_row = b[v, p] - b[u, p]
            diff_a_col = a[:, i] - a[:, j]
            diff_b_col = b[p, v] - b[p, u]
            
            term_row = np.dot(diff_a_row, diff_b_row)
            term_col = np.dot(diff_a_col, diff_b_col)
            
            buu = b_diag[u]
            bvv = b_diag[v]
            buv = b[u, v]
            bvu = b[v, u]
            
            corr_ri = (a[i, i] - a[j, i]) * (bvu - buu)
            corr_rj = (a[i, j] - a[j, j]) * (bvv - buv)
            corr_ci = (a[i, i] - a[i, j]) * (buv - bvv)
            corr_cj = (a[j, i] - a[j, j]) * (bvu - buv)
            
            delta = term_row + term_col - corr_ri - corr_rj - corr_ci - corr_cj
            
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                p[i], p[j] = p[j], p[i]
                cost += delta
            
            T *= alpha
        
        return p, cost
    
    # --- Hungarian-based refinement ---
    def hungarian_refine(p, cost):
        """Use Hungarian algorithm on linearized cost matrix."""
        b_col_p = b[:, p]
        b_row_p = b[p, :]
        C = a @ b_col_p.T + a.T @ b_row_p
        row_ind, col_ind = linear_sum_assignment(C)
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        cost_new = compute_cost(p_new)
        return p_new, cost_new
    
    # --- Try multiple initial solutions and keep the best ---
    init_methods = ['flow', 'distance', 'combined', 'multi_eig', 'default']
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
    
    # Also try a random initialization
    p_rand = np.random.permutation(n)
    cost_rand = compute_cost(p_rand)
    if cost_rand < best_cost:
        best_cost = cost_rand
        best_p = p_rand.copy()
    
    # --- Run local search on best initial solution ---
    time_limit = deadline - 45.0
    best_p, best_cost = local_search(best_p, best_cost, time_limit)
    
    # --- Hungarian-based refinement ---
    if time.time() < deadline - 35.0:
        p_hung, cost_hung = hungarian_refine(best_p, best_cost)
        if cost_hung < best_cost:
            best_p = p_hung
            best_cost = cost_hung
    
    # --- Second local search pass ---
    if time.time() < deadline - 25.0:
        time_limit2 = deadline - 20.0
        best_p, best_cost = local_search(best_p, best_cost, time_limit2)
    
    # --- SA escape to break local minima ---
    if time.time() < deadline - 15.0:
        time_limit3 = deadline - 12.0
        p_sa, cost_sa = sa_escape(best_p, best_cost, time_limit3, T_init=50.0, T_min=0.5, alpha=0.9995)
        if cost_sa < best_cost:
            best_p = p_sa
            best_cost = cost_sa
    
    # --- Polish after SA with local search ---
    if time.time() < deadline - 10.0:
        time_limit4 = deadline - 8.0
        best_p, best_cost = local_search(best_p, best_cost, time_limit4)
    
    # --- Another Hungarian refinement if time allows ---
    if time.time() < deadline - 8.0:
        p_hung2, cost_hung2 = hungarian_refine(best_p, best_cost)
        if cost_hung2 < best_cost:
            best_p = p_hung2
            best_cost = cost_hung2
    
    # --- Final local search ---
    if time.time() < deadline - 5.0:
        time_limit5 = deadline - 1.0
        best_p, best_cost = local_search(best_p, best_cost, time_limit5)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
