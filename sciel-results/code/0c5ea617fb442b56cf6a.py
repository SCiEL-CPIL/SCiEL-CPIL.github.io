"""QAP tai256c solver: iterated local search with spectral embedding, Hungarian re-optimization, and tabu diversification."""
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin under 150s limit

    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    def fast_swap_delta(p, u, v):
        pu = p[u]
        pv = p[v]

        d_uv = b[pv, :] - b[pu, :]
        d_vu = b[:, pv] - b[:, pu]

        s1 = np.dot(a[u, :] - a[v, :], d_uv[p])
        s2 = np.dot(a[:, u] - a[:, v], d_vu[p])

        s1 -= (a[u, u] - a[v, u]) * d_uv[pu]
        s1 -= (a[u, v] - a[v, v]) * d_uv[pv]

        s2 -= (a[u, u] - a[u, v]) * d_vu[pu]
        s2 -= (a[v, u] - a[v, v]) * d_vu[pv]

        cross = (a[u, v] - a[v, u]) * (b[pv, pu] - b[pu, pv])

        return s1 + s2 + cross

    def local_search(p, cost, max_passes=20):
        """Optimized local search with precomputed B_p for faster delta computation."""
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            
            for i in range(n):
                if time.time() > deadline:
                    return p, cost
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        return p, cost
                    
                    delta = fast_swap_delta(p, i, j)
                    
                    if delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += delta
                        improved = True
                        break
                if improved:
                    break
        return p, cost

    def hungarian_reopt(p):
        """Use Hungarian algorithm on a linearized cost matrix to escape local optima."""
        B_p = b[np.ix_(np.arange(n), p)]
        M1 = a @ B_p.T  # M1[i,j] = sum_k a[i,k]*b[j,p[k]]
        
        Bp2 = b[np.ix_(p, np.arange(n))]
        M2 = a.T @ Bp2  # M2[i,j] = sum_k a[k,i]*b[p[k],j]
        
        C = M1 + M2
        
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        return new_p

    def spectral_embedding_init():
        """Generate initial permutation using spectral embedding approach."""
        # Construct a linearized cost matrix for spectral embedding
        # M[i,j] = sum_k a[i,k] * b[j,k] + sum_k a[k,i] * b[k,j]
        # This captures the interaction between facilities and locations
        
        # First term: M1[i,j] = sum_k a[i,k] * b[j,k]
        M1 = a @ b.T  # M1[i,j] = sum_k a[i,k]*b[j,k]
        
        # Second term: M2[i,j] = sum_k a[k,i] * b[k,j]
        M2 = a.T @ b  # M2[i,j] = sum_k a[k,i]*b[k,j]
        
        M = M1 + M2
        
        # Get leading eigenvectors
        # For a 256x256 matrix, we can use scipy's sparse eigensolver or dense
        try:
            from scipy.sparse.linalg import eigsh
            # Use dense eigendecomposition for reliability
            eigenvalues, eigenvectors = np.linalg.eigh(M)
            # Take the top few eigenvectors (largest eigenvalues)
            # Sort by eigenvalue in descending order
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]
            
            # Use the top 2-4 eigenvectors to create a 2D/3D/4D embedding
            # We'll use 2D for simplicity
            embed = eigenvectors[:, :2]
            
            # Sort facilities by their angle in the 2D embedding
            angles = np.arctan2(embed[:, 1], embed[:, 0])
            facility_order = np.argsort(angles)
            
            # Sort locations by their angle in a similar embedding
            # For locations, we can use the same approach on M.T or just use the same ordering
            # Actually, let's use the row sums of b to rank locations
            b_row_sums = b.sum(axis=1)
            location_order = np.argsort(b_row_sums)
            
            p = np.empty(n, dtype=np.int64)
            p[facility_order] = location_order
            
            return p
            
        except Exception:
            # Fallback to rank matching if spectral fails
            a_row_sums = a.sum(axis=1)
            b_col_sums = b.sum(axis=1)
            heavy = np.argsort(-a_row_sums, kind="stable")
            central = np.argsort(b_col_sums, kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
            return p

    def tabu_kick(p, cost, n_kicks=5):
        """Apply block-reversal kicks with tabu for stronger diversification."""
        best_p = p.copy()
        best_cost = cost
        tabu_tenure = 8
        tabu_list = {}
        iteration = 0
        
        for _ in range(n_kicks):
            if time.time() > deadline - 5:
                break
            
            # Block reversal: pick a random segment and reverse it
            i, j = np.random.choice(n, 2, replace=False)
            if i > j:
                i, j = j, i
            key = (i, j)
            
            # Check tabu
            is_tabu = key in tabu_list and tabu_list[key] > iteration
            
            # Apply block reversal
            p[i:j+1] = p[i:j+1][::-1]
            new_cost = compute_cost(p)
            
            # Accept if not tabu, or if it improves global best (aspiration)
            if not is_tabu or (new_cost < best_cost):
                cost = new_cost
                tabu_list[key] = iteration + tabu_tenure
                
                if cost < best_cost:
                    best_cost = cost
                    best_p = p.copy()
            else:
                # Undo the reversal
                p[i:j+1] = p[i:j+1][::-1]
            
            iteration += 1
        
        return best_p, best_cost

    # Initial solution: use spectral embedding for a globally informed start
    p = spectral_embedding_init()
    
    best_p = p.copy()
    best_cost = compute_cost(best_p)
    current_p = p.copy()
    current_cost = best_cost

    # First local search
    current_p, current_cost = local_search(current_p, current_cost)
    if current_cost < best_cost:
        best_p = current_p.copy()
        best_cost = current_cost

    # Also try rank-matching as a second initial solution
    a_row_sums = a.sum(axis=1)
    b_col_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_col_sums, kind="stable")
    p2 = np.empty(n, dtype=np.int64)
    p2[heavy] = central
    
    p2, cost2 = local_search(p2.copy(), compute_cost(p2))
    if cost2 < best_cost:
        best_p = p2.copy()
        best_cost = cost2

    # Iterated Local Search with periodic Hungarian re-optimization and tabu kicks
    iteration = 0
    while time.time() < deadline - 15:  # Leave 15s buffer for final polish
        iteration += 1

        # Perturb: use block reversal for stronger diversification
        p = best_p.copy()
        
        # Random block reversal (length 3-10)
        block_len = np.random.randint(3, 11)
        start = np.random.randint(0, n - block_len)
        p[start:start+block_len] = p[start:start+block_len][::-1]
        
        # Tabu kick for additional diversification
        p, cost = tabu_kick(p, compute_cost(p), n_kicks=2)

        # Local search from perturbed
        p, cost = local_search(p, cost)

        # Every few iterations, try Hungarian re-optimization
        if iteration % 4 == 0 and time.time() < deadline - 30:
            h_p = hungarian_reopt(p)
            h_cost = compute_cost(h_p)
            h_p, h_cost = local_search(h_p, h_cost)
            if h_cost < cost:
                p, cost = h_p, h_cost

        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost

    # Final high-precision polish: multiple intensive local search passes
    if time.time() < deadline - 3:
        final_p = best_p.copy()
        final_cost = best_cost
        
        # Run local search with more passes for final polish
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            final_p, final_cost = local_search(final_p, final_cost, max_passes=30)
            if final_cost < best_cost:
                best_p = final_p.copy()
                best_cost = final_cost
    
    return best_p.tolist()
