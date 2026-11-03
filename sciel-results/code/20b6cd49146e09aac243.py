"""Rewritten QAP solver: spectral init + Hungarian + fast incremental local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    """Solve QAP: assign n facilities to n locations minimizing sum a[i,j]*b[p[i],p[j]].
    
    Strategy:
    1. Spectral initialization using eigenvectors of QAP-related matrices.
    2. Hungarian-based linearization refinement.
    3. Fast local search with O(n) incremental updates per swap.
    4. Multi-start with diverse initial solutions.
    """
    n = len(a)
    deadline = time.time() + 140.0  # safety margin before 150s limit
    np.random.seed(42)
    
    def compute_cost(p):
        """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]"""
        total = 0
        for i in range(n):
            total += np.dot(a[i], b[p[i], p])
        return total
    
    # --- Spectral initialization ---
    # Use eigenvectors of the QAP-related matrices to generate high-quality initial solutions
    # The QAP can be related to eigenvalue problems. We use multiple spectral approaches:
    
    def spectral_init(method='flow'):
        """Generate initial permutation using spectral methods."""
        if method == 'flow':
            # Use eigenvectors of the flow matrix a
            # Center a to remove mean
            a_centered = a - a.mean()
            # Get top eigenvectors
            try:
                eigvals, eigvecs = np.linalg.eigh(a_centered)
                # Sort by eigenvalue magnitude (descending)
                idx = np.argsort(-np.abs(eigvals))
                eigvecs_sorted = eigvecs[:, idx[:min(50, n)]]
                # Use the top eigenvector to create a ranking
                # Map facility indices to their "spectral coordinate"
                spec_coords = eigvecs_sorted[:, 0]
                # Rank facilities by spectral coordinate
                fac_rank = np.argsort(spec_coords)
                # Rank locations by their row sum of b (centrality)
                loc_rank = np.argsort(b.sum(axis=1))
                # Assign: facility with rank r gets location with rank r
                p = np.empty(n, dtype=np.int64)
                p[fac_rank] = loc_rank
            except:
                # Fallback to rank matching
                heavy = np.argsort(-a.sum(axis=1), kind="stable")
                central = np.argsort(b.sum(axis=1), kind="stable")
                p = np.empty(n, dtype=np.int64)
                p[heavy] = central
            return p
        elif method == 'distance':
            # Use eigenvectors of the distance matrix b
            b_centered = b - b.mean()
            try:
                eigvals, eigvecs = np.linalg.eigh(b_centered)
                idx = np.argsort(-np.abs(eigvals))
                eigvecs_sorted = eigvecs[:, idx[:min(50, n)]]
                spec_coords = eigvecs_sorted[:, 0]
                loc_rank = np.argsort(spec_coords)
                # Rank facilities by their row sum of a (flow intensity)
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
            # Combine flow and distance spectral info
            a_centered = a - a.mean()
            b_centered = b - b.mean()
            try:
                eigvals_a, eigvecs_a = np.linalg.eigh(a_centered)
                eigvals_b, eigvecs_b = np.linalg.eigh(b_centered)
                idx_a = np.argsort(-np.abs(eigvals_a))
                idx_b = np.argsort(-np.abs(eigvals_b))
                # Use top eigenvectors from both
                spec_a = eigvecs_a[:, idx_a[0]]
                spec_b = eigvecs_b[:, idx_b[0]]
                # Combine: use flow spectral for facilities, distance spectral for locations
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
            # Default: rank matching
            heavy = np.argsort(-a.sum(axis=1), kind="stable")
            central = np.argsort(b.sum(axis=1), kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
            return p
    
    # --- Optimized swap delta computation with incremental updates ---
    
    def compute_delta(i, j, p, dot_row, dot_col):
        """Compute swap delta using precomputed dot products."""
        u = p[i]
        v = p[j]
        if u == v:
            return 0.0
        
        # Terms 1-4 (diagonal and cross terms)
        delta = 0.0
        delta += a[i, i] * (b[v, v] - b[u, u])
        delta += a[i, j] * (b[v, u] - b[u, v])
        delta += a[j, i] * (b[u, v] - b[v, u])
        delta += a[j, j] * (b[u, u] - b[v, v])
        
        # Terms 5-8 using precomputed dot products
        delta += (dot_row[i, v] - dot_row[i, u]) - a[i,i]*(b[v,v]-b[u,u]) - a[i,j]*(b[v,u]-b[u,v])
        delta += (dot_row[j, u] - dot_row[j, v]) - a[j,i]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
        delta += (dot_col[i, v] - dot_col[i, u]) - a[i,i]*(b[v,v]-b[u,u]) - a[j,i]*(b[v,u]-b[u,v])
        delta += (dot_col[j, u] - dot_col[j, v]) - a[i,j]*(b[u,v]-b[v,u]) - a[j,j]*(b[u,u]-b[v,v])
        
        return delta
    
    def recompute_precomp(p):
        """Recompute dot_row and dot_col from scratch."""
        b_col_p = b[:, p]
        b_row_p = b[p, :]
        dot_row = a @ b_col_p.T
        dot_col = a.T @ b_row_p
        return dot_row, dot_col
    
    def incremental_update(p, i, j, dot_row, dot_col):
        """Update dot_row and dot_col after swapping p[i] and p[j]. O(n) per update."""
        u = p[i]
        v = p[j]
        if u == v:
            return dot_row, dot_col
        
        # After swap: p[i]=v, p[j]=u
        # dot_row[k, l] = sum_m a[k,m] * b[l, p[m]]
        # Only p[i] and p[j] changed, so only b[l, p[i]] and b[l, p[j]] terms change.
        # For each k:
        #   dot_row[k, l] changes by a[k,i]*(b[l,v]-b[l,u]) + a[k,j]*(b[l,u]-b[l,v])
        #   = (a[k,i]-a[k,j]) * (b[l,v]-b[l,u])
        
        # Vectorized: for all k and all l:
        # delta_dot_row[k, l] = (a[k,i] - a[k,j]) * (b[l,v] - b[l,u])
        # This is outer product: (a[:,i]-a[:,j]) * (b[:,v]-b[:,u]).T
        
        diff_a = a[:, i] - a[:, j]  # shape (n,)
        diff_b = b[:, v] - b[:, u]  # shape (n,)
        delta_row = np.outer(diff_a, diff_b)  # shape (n, n): [k, l]
        dot_row += delta_row
        
        # dot_col[k, l] = sum_m a[m,k] * b[p[m], l]
        # Only p[i] and p[j] changed:
        #   dot_col[k, l] changes by a[i,k]*(b[v,l]-b[u,l]) + a[j,k]*(b[u,l]-b[v,l])
        #   = (a[i,k]-a[j,k]) * (b[v,l]-b[u,l])
        
        # For all k and all l:
        # delta_dot_col[k, l] = (a[i,k]-a[j,k]) * (b[v,l]-b[u,l])
        # = (a[i,:]-a[j,:]) * (b[v,:]-b[u,:]).T
        
        diff_a2 = a[i, :] - a[j, :]  # shape (n,)
        diff_b2 = b[v, :] - b[u, :]  # shape (n,)
        delta_col = np.outer(diff_a2, diff_b2)  # shape (n, n): [k, l]
        dot_col += delta_col
        
        return dot_row, dot_col
    
    # --- Try multiple initial solutions and keep the best ---
    init_methods = ['flow', 'distance', 'combined', 'default']
    best_p = None
    best_cost = np.inf
    
    for method in init_methods:
        if time.time() > deadline - 30.0:
            break
        p_init = spectral_init(method)
        cost_init = compute_cost(p_init)
        if cost_init < best_cost:
            best_cost = cost_init
            best_p = p_init.copy()
    
    # --- Local search with incremental dot product updates ---
    def local_search(p, cost, time_limit):
        """Perform local search starting from p with initial cost."""
        p = p.copy()
        
        dot_row, dot_col = recompute_precomp(p)
        
        improved = True
        no_improve_count = 0
        while improved and time.time() < time_limit:
            improved = False
            
            # Random sampling of pairs
            num_trials = 20000
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
                
                delta = compute_delta(i, j, p, dot_row, dot_col)
                
                if delta < -1e-9:
                    # Accept the swap
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
                    no_improve_count = 0
                    
                    # O(n^2) incremental update (but much faster than O(n^3) recompute)
                    dot_row, dot_col = incremental_update(p, i, j, dot_row, dot_col)
            if not improved:
                no_improve_count += 1
                if no_improve_count >= 3:
                    break
        
        return p, cost
    
    # Run local search
    time_limit = deadline - 30.0
    best_p, best_cost = local_search(best_p, best_cost, time_limit)
    
    # --- Hungarian-based refinement (multiple rounds) ---
    for round_num in range(3):
        if time.time() > deadline - 15.0:
            break
        # Build linear cost matrix
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
        else:
            break
    
    # --- Final local search pass with exhaustive 2-opt polish ---
    if time.time() < deadline - 10.0:
        time_limit = deadline - 5.0
        best_p, best_cost = local_search(best_p, best_cost, time_limit)
    
    # --- Final exhaustive 2-opt polish ---
    if time.time() < deadline - 3.0:
        p = best_p.copy()
        cost = best_cost
        dot_row, dot_col = recompute_precomp(p)
        
        # Exhaustive scan for any improving swap
        changed = True
        while changed and time.time() < deadline - 3.0:
            changed = False
            for i in range(n):
                if time.time() > deadline - 3.0:
                    break
                for j in range(i+1, n):
                    if time.time() > deadline - 3.0:
                        break
                    u = p[i]
                    v = p[j]
                    if u == v:
                        continue
                    delta = compute_delta(i, j, p, dot_row, dot_col)
                    if delta < -1e-9:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        changed = True
                        dot_row, dot_col = incremental_update(p, i, j, dot_row, dot_col)
        
        if cost < best_cost:
            best_p = p
            best_cost = cost
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
