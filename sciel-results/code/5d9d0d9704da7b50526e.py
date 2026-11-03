"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin before 120s kill

    def compute_cost(p):
        bp = b[np.ix_(p, p)]
        return int(np.sum(a * bp))

    # Precompute interaction matrix for fast swap delta
    # M[i,j] = a[i,j] + a[j,i] (symmetric flow)
    M = a + a.T

    # Precompute all pair indices for SA
    pair_indices = np.array([(i, j) for i in range(n) for j in range(i + 1, n)])
    n_pairs = len(pair_indices)

    def compute_delta_swap(p, i, j):
        """Compute cost delta for swapping facilities i and j in permutation p."""
        u = p[i]
        v = p[j]
        # delta = sum_{k!=i,j} (M[i,k]-M[j,k])*(b[v,p[k]]-b[u,p[k]])
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        idxs = np.where(mask)[0]
        
        pm = p[idxs]
        diff_m = M[i, idxs] - M[j, idxs]
        diff_b = b[v, pm] - b[u, pm]
        
        return float(np.dot(diff_m, diff_b))

    def compute_delta_3exchange(p, i, j, k):
        """Compute cost delta for 3-exchange: p[i], p[j], p[k] -> p[j], p[k], p[i]
        This is a cyclic shift of positions i, j, k."""
        old_i, old_j, old_k = p[i], p[j], p[k]
        new_i, new_j, new_k = old_j, old_k, old_i
        
        # For each position, compute the change in interaction with all other positions
        total_delta = 0.0
        
        # Position i: was old_i, now new_i
        # Interactions with all k' != i, j, k
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        mask[k] = False
        idxs = np.where(mask)[0]
        
        # Delta from position i's change
        for pos in idxs:
            old_val = M[old_i, pos] * b[old_i, p[pos]] + M[pos, old_i] * b[p[pos], old_i]
            new_val = M[new_i, pos] * b[new_i, p[pos]] + M[pos, new_i] * b[p[pos], new_i]
            total_delta += new_val - old_val
        
        # Interactions between i, j, k (pairwise)
        # Old: (old_i, old_j), (old_i, old_k), (old_j, old_k)
        # New: (new_i, new_j), (new_i, new_k), (new_j, new_k)
        # = (old_j, old_k), (old_j, old_i), (old_k, old_i)
        
        old_ij = M[old_i, old_j] * b[old_i, old_j] + M[old_j, old_i] * b[old_j, old_i]
        old_ik = M[old_i, old_k] * b[old_i, old_k] + M[old_k, old_i] * b[old_k, old_i]
        old_jk = M[old_j, old_k] * b[old_j, old_k] + M[old_k, old_j] * b[old_k, old_j]
        
        new_ij = M[new_i, new_j] * b[new_i, new_j] + M[new_j, new_i] * b[new_j, new_i]
        new_ik = M[new_i, new_k] * b[new_i, new_k] + M[new_k, new_i] * b[new_k, new_i]
        new_jk = M[new_j, new_k] * b[new_j, new_k] + M[new_k, new_j] * b[new_k, new_j]
        
        total_delta += (new_ij + new_ik + new_jk) - (old_ij + old_ik + old_jk)
        
        return total_delta

    def local_search_2swap(p, cur_cost, max_rounds=3):
        """Fast local search using 2-swap moves."""
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    d = compute_delta_swap(p, i, j)
                    if d < 0:
                        p[i], p[j] = p[j], p[i]
                        cur_cost += d
                        improved = True
            if not improved:
                break
        return p, cur_cost

    def vnd_local_search(p, cur_cost, max_rounds=2):
        """Variable Neighborhood Descent: alternate between 2-swap and 3-exchange neighborhoods."""
        for rnd in range(max_rounds):
            if time.time() > deadline:
                break
            
            # Phase 1: 2-swap local search
            p, cur_cost = local_search_2swap(p, cur_cost, max_rounds=1)
            
            # Phase 2: 3-exchange local search
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    for k in range(j + 1, n):
                        if time.time() > deadline:
                            break
                        d = compute_delta_3exchange(p, i, j, k)
                        if d < 0:
                            # Cyclic shift: p[i], p[j], p[k] -> p[j], p[k], p[i]
                            p[i], p[j], p[k] = p[j], p[k], p[i]
                            cur_cost += d
                            improved = True
            if not improved:
                break
        return p, cur_cost

    # Spectral-projection based surrogate for Hungarian
    # Use top-k principal components to capture non-linear structure
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    # Center the matrices
    a_mean = a_sym.mean()
    b_mean = b_sym.mean()
    a_c = a_sym - a_mean
    b_c = b_sym - b_mean
    
    # SVD for top components
    try:
        U_a, S_a, Vt_a = np.linalg.svd(a_c, full_matrices=False)
        U_b, S_b, Vt_b = np.linalg.svd(b_c, full_matrices=False)
    except np.linalg.LinAlgError:
        U_a, S_a, Vt_a = np.linalg.svd(a_sym, full_matrices=False)
        U_b, S_b, Vt_b = np.linalg.svd(b_sym, full_matrices=False)
    
    # Use top-k components (k=5 is a reasonable choice)
    k = min(5, n)
    # Project facilities onto top-k left singular vectors of flow
    # Project locations onto top-k left singular vectors of distance
    # Surrogate cost: negative correlation between projections
    proj_a = U_a[:, :k] @ np.diag(S_a[:k])  # facility embeddings
    proj_b = U_b[:, :k] @ np.diag(S_b[:k])  # location embeddings
    
    # We want high-flow facilities (high proj_a norm) with low-distance locations
    # Use negative dot product as surrogate
    surrogate_cost = -proj_a @ proj_b.T
    
    # Also add row-sum based term for stability
    flow_cent = a.sum(axis=1)
    dist_cent = b.sum(axis=1)
    # Normalize
    flow_cent_norm = flow_cent / (np.max(flow_cent) + 1e-10)
    dist_cent_norm = dist_cent / (np.max(dist_cent) + 1e-10)
    surrogate_cost += 0.1 * np.outer(flow_cent_norm, dist_cent_norm)
    
    row_ind, col_ind = linear_sum_assignment(surrogate_cost)
    p0 = np.empty(n, dtype=np.int64)
    p0[row_ind] = col_ind
    
    # Use VND for initial local search to get a stronger starting point
    p0, init_cost = vnd_local_search(p0, compute_cost(p0), max_rounds=3)
    
    # Iterated Local Search with multiple starts
    np.random.seed(42)
    
    best_p = p0.copy()
    best_cost = init_cost
    
    # Multiple starts with perturbation
    n_starts = 5
    
    for start in range(n_starts):
        if time.time() > deadline:
            break
        
        if start == 0:
            p = p0.copy()
            cur_cost = best_cost
        else:
            # Perturb the best solution
            p = best_p.copy()
            cur_cost = best_cost
            # Apply random swaps
            n_swaps = max(5, n // 5)
            for _ in range(n_swaps):
                i, j = np.random.choice(n, 2, replace=False)
                p[i], p[j] = p[j], p[i]
            cur_cost = compute_cost(p)
        
        # Use VND for local search
        p, cur_cost = vnd_local_search(p, cur_cost, max_rounds=2)
        
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_p = p.copy()
        
        # Simulated Annealing from this local optimum
        if time.time() > deadline:
            break
        
        T_start = max(best_cost * 0.005, 500.0)
        T_end = max(best_cost * 0.00005, 0.5)
        cooling_rate = 0.9995
        max_sa_iters = 30000
        
        p = best_p.copy()
        cur_cost = best_cost
        
        sa_iter = 0
        while time.time() < deadline and sa_iter < max_sa_iters:
            idx = np.random.randint(0, n_pairs)
            i, j = pair_indices[idx]
            
            delta = compute_delta_swap(p, i, j)
            
            if delta < 0:
                p[i], p[j] = p[j], p[i]
                cur_cost += delta
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_p = p.copy()
            else:
                if np.random.random() < np.exp(-delta / T_start):
                    p[i], p[j] = p[j], p[i]
                    cur_cost += delta
            
            T_start = max(T_start * cooling_rate, T_end)
            sa_iter += 1
            
            # Periodic VND local search
            if sa_iter % 1000 == 0 and time.time() < deadline:
                p, cur_cost = vnd_local_search(p, cur_cost, max_rounds=1)
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_p = p.copy()
    
    # Final polish with deeper VND
    if time.time() < deadline:
        p = best_p.copy()
        cur_cost = best_cost
        p, cur_cost = vnd_local_search(p, cur_cost, max_rounds=5)
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_p = p.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
