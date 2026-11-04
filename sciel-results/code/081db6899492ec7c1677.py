"""QAP solver: spectral initialization + linearized assignment + fast local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Multiple initializations (spectral + rank matching).
    2. Iterative linearized assignment (alternating between row/col linearization).
    3. Fast local search with precomputed dot products and incremental updates.
    4. Final refinement pass.
    """
    n = len(a)
    deadline = time.time() + 115.0  # safety margin before 120s limit
    
    def compute_cost(p):
        """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
        return np.sum(a * b[np.ix_(p, p)])
    
    # --- Initialization methods ---
    def init_rank_matching():
        """Rank matching: heavy facilities to central locations."""
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p
    
    def init_spectral_flow():
        """Spectral initialization using flow matrix eigenvectors."""
        try:
            a_centered = a - a.mean()
            eigvals, eigvecs = np.linalg.eigh(a_centered)
            idx = np.argsort(-np.abs(eigvals))
            spec_coords = eigvecs[:, idx[0]]
            fac_rank = np.argsort(spec_coords)
            loc_rank = np.argsort(b.sum(axis=1))
            p = np.empty(n, dtype=np.int64)
            p[fac_rank] = loc_rank
        except:
            p = init_rank_matching()
        return p
    
    def init_spectral_dist():
        """Spectral initialization using distance matrix eigenvectors."""
        try:
            b_centered = b - b.mean()
            eigvals, eigvecs = np.linalg.eigh(b_centered)
            idx = np.argsort(-np.abs(eigvals))
            spec_coords = eigvecs[:, idx[0]]
            loc_rank = np.argsort(spec_coords)
            fac_rank = np.argsort(a.sum(axis=1))
            p = np.empty(n, dtype=np.int64)
            p[fac_rank] = loc_rank
        except:
            p = init_rank_matching()
        return p
    
    def init_mds():
        """MDS-based initialization using Procrustes alignment."""
        try:
            def mds_embed(M, k=10):
                n_m = M.shape[0]
                J = np.eye(n_m) - np.ones((n_m, n_m)) / n_m
                B = -0.5 * J @ M @ J
                w, V = np.linalg.eigh(B)
                idx = np.argsort(-w)[:k]
                w = w[idx]
                V = V[:, idx]
                X = V * np.sqrt(np.maximum(w, 0))
                return X
            
            X_a = mds_embed(a, 10)
            X_b = mds_embed(b, 10)
            
            # Procrustes alignment
            XX = np.sum(X_a * X_a, axis=1)[:, None]
            YY = np.sum(X_b * X_b, axis=1)[None, :]
            C = XX - 2.0 * (X_a @ X_b.T) + YY
            row_ind, col_ind = linear_sum_assignment(C)
            p = np.empty(n, dtype=np.int64)
            p[row_ind] = col_ind
        except:
            p = init_rank_matching()
        return p
    
    # --- Fast local search with precomputed dot products ---
    def local_search(p, cost, time_limit, max_trials_per_iter=50000):
        """Perform local search starting from p with initial cost."""
        p = p.copy()
        
        def recompute_precomp(p):
            b_col_p = b[:, p]  # (n, n)
            b_row_p = b[p, :]  # (n, n)
            dot_row = a @ b_col_p.T  # (n, n)
            dot_col = a.T @ b_row_p  # (n, n)
            return dot_row, dot_col
        
        dot_row, dot_col = recompute_precomp(p)
        
        improved = True
        iteration = 0
        while improved and time.time() < time_limit:
            improved = False
            iteration += 1
            
            # Try random swaps
            for _ in range(max_trials_per_iter):
                if time.time() > time_limit:
                    break
                i, j = np.random.randint(0, n, 2)
                if i == j:
                    continue
                
                u = p[i]
                v = p[j]
                if u == v:
                    continue
                
                # Compute delta using precomputed dot products
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
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
                    
                    # Incremental update
                    D_b = b[v, :] - b[u, :]
                    diff_a_col = a[:, i] - a[:, j]
                    dot_row += np.outer(diff_a_col, D_b)
                    diff_a_row = a[i, :] - a[j, :]
                    dot_col += np.outer(diff_a_row, D_b)
                    
                    # Periodic re-computation to correct floating point drift
                    if _ % 5000 == 0:
                        dot_row, dot_col = recompute_precomp(p)
        
        return p, cost
    
    # --- Linearized assignment ---
    def linearized_assign_row(p):
        """Solve linearized assignment minimizing sum_i sum_k a[i,k]*b[j,p[k]] for location j."""
        b_col_p = b[:, p]  # (n, n), [j, k] = b[j, p[k]]
        C = a @ b_col_p.T  # (n, n), C[i, j] = sum_k a[i,k] * b[j, p[k]]
        row_ind, col_ind = linear_sum_assignment(C)
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        return p_new
    
    def linearized_assign_col(p):
        """Solve linearized assignment minimizing sum_i sum_k a[k,i]*b[p[k],j] for location j."""
        b_row_p = b[p, :]  # (n, n), [k, j] = b[p[k], j]
        C = a.T @ b_row_p  # (n, n), C[i, j] = sum_k a[k,i] * b[p[k], j]
        row_ind, col_ind = linear_sum_assignment(C)
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind
        return p_new
    
    # --- Main algorithm ---
    # Generate multiple initial solutions
    init_methods = [init_rank_matching, init_spectral_flow, init_spectral_dist, init_mds]
    best_p = None
    best_cost = np.inf
    
    for method in init_methods:
        if time.time() > deadline - 60.0:
            break
        p_init = method()
        cost_init = compute_cost(p_init)
        if cost_init < best_cost:
            best_cost = cost_init
            best_p = p_init.copy()
    
    # Iterative refinement: alternate between linearized assignment and local search
    # Phase 1: Linearized assignment iterations
    time_for_lin = deadline - 40.0
    while time.time() < time_for_lin:
        # Row linearization
        p_new = linearized_assign_row(best_p)
        cost_new = compute_cost(p_new)
        if cost_new < best_cost:
            best_p = p_new
            best_cost = cost_new
            # Quick local search after improvement
            if time.time() < time_for_lin - 5.0:
                best_p, best_cost = local_search(best_p, best_cost, time.time() + 5.0, max_trials_per_iter=20000)
        
        # Column linearization
        p_new = linearized_assign_col(best_p)
        cost_new = compute_cost(p_new)
        if cost_new < best_cost:
            best_p = p_new
            best_cost = cost_new
            if time.time() < time_for_lin - 5.0:
                best_p, best_cost = local_search(best_p, best_cost, time.time() + 5.0, max_trials_per_iter=20000)
        
        # Check convergence
        p_new2 = linearized_assign_row(best_p)
        if np.array_equal(p_new2, best_p):
            break
    
    # Phase 2: Extended local search
    time_limit_ls = deadline - 10.0
    best_p, best_cost = local_search(best_p, best_cost, time_limit_ls, max_trials_per_iter=100000)
    
    # Phase 3: Final linearized assignment + local search
    if time.time() < deadline - 8.0:
        p_new = linearized_assign_row(best_p)
        cost_new = compute_cost(p_new)
        if cost_new < best_cost:
            best_p = p_new
            best_cost = cost_new
        p_new = linearized_assign_col(best_p)
        cost_new = compute_cost(p_new)
        if cost_new < best_cost:
            best_p = p_new
            best_cost = cost_new
        best_p, best_cost = local_search(best_p, best_cost, deadline - 3.0, max_trials_per_iter=50000)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
