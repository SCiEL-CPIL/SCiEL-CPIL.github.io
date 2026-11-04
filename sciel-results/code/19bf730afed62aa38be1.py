"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import scipy.optimize as opt

def _compute_cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    n = len(p)
    D = b[np.ix_(p, p)]
    return int(np.sum(a * D))

def _local_search(a, b, p, time_limit, start_time):
    """
    Optimized 2-opt local search for QAP.
    Uses vectorized delta calculation for all j > i at each step i.
    """
    n = len(p)
    p = p.copy()
    # D[i, j] = b[p[i], p[j]]
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    # Precompute symmetric A for delta calculation
    A_sym = a + a.T
    A_diag = np.diag(a)
    
    while time.time() - start_time < time_limit:
        improved = False
        
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return p, current_cost
            
            # Indices j > i
            js = np.arange(i + 1, n)
            
            # Vectorized delta calculation
            diff_A = A_sym[i] - A_sym[js]
            diff_D = D[js] - D[i]
            
            S = np.sum(diff_A * diff_D, axis=1)
            
            # Correction terms for the diagonal and i-j interactions
            coeff_i = A_sym[i, i] - A_sym[js, i]
            term_i = coeff_i * (D[js, i] - D[i, i])
            
            coeff_j = A_sym[i, js] - A_diag[js]
            term_j = coeff_j * (D[js, js] - D[i, js])
            
            diag_term = (A_diag[i] - A_diag[js]) * (D[js, js] - D[i, i])
            
            cross_term = (a[i, js] - a[js, i]) * (D[js, i] - D[i, js])
            
            deltas = S - term_i - term_j + diag_term + cross_term
            
            min_idx = np.argmin(deltas)
            min_delta = deltas[min_idx]
            
            if min_delta < 0:
                j = js[min_idx]
                # Perform swap
                p[i], p[j] = p[j], p[i]
                
                # Update D
                D[i], D[j] = D[j].copy(), D[i].copy()
                D[:, i], D[:, j] = D[:, j].copy(), D[:, i].copy()
                
                current_cost += int(min_delta)
                improved = True
        
        if not improved:
            break
    
    return p, current_cost

def _hungarian_init(a, b):
    """
    Generate an initial assignment using the Hungarian algorithm on a surrogate cost matrix.
    """
    n = len(a)
    cost_matrix = a @ b + a.T @ b.T
    row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=int)
    p[row_ind] = col_ind
    return p

def _spectral_init(a, b):
    """
    Phase-field-inspired initialization using MDS embeddings.
    """
    n = len(a)
    
    def mds(dissim, n_dim=2):
        n = len(dissim)
        J = np.eye(n) - np.ones((n, n)) / n
        M = -0.5 * J @ (dissim ** 2) @ J
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        V = eigenvectors[:, :n_dim]
        D = np.sqrt(np.maximum(eigenvalues[:n_dim], 0))
        coords = V * D
        return coords
    
    try:
        loc_coords = mds(b, n_dim=2)
        fac_coords = mds(-a, n_dim=2)
        
        cost_matrix = np.sum((fac_coords[:, np.newaxis, :] - loc_coords[np.newaxis, :, :]) ** 2, axis=2)
        
        row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=int)
        p[row_ind] = col_ind
        return p
    except Exception:
        return np.random.permutation(n)

def _greedy_init(a, b, n):
    """
    Greedy constructive heuristic.
    """
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    
    fac_order = np.argsort(flow_sums)
    loc_order = np.argsort(dist_sums)
    
    p = np.zeros(n, dtype=int)
    used_locs = set()
    
    for fac in fac_order:
        candidates = [loc for loc in loc_order if loc not in used_locs]
        top_k = min(5, len(candidates))
        chosen = random.choice(candidates[:top_k])
        p[fac] = chosen
        used_locs.add(chosen)
    
    return p

def _perturb(p, k):
    """Randomly swap k pairs of positions."""
    p = p.copy()
    n = len(p)
    for _ in range(k):
        i, j = random.sample(range(n), 2)
        p[i], p[j] = p[j], p[i]
    return p

def _block_reverse_perturb(p, L):
    """Reverse a random block of length L."""
    p = p.copy()
    n = len(p)
    L = min(L, n)
    i = random.randint(0, n - L)
    p[i:i+L] = p[i:i+L][::-1]
    return p

def _mcmc_search(a, b, p_init, time_limit, start_time):
    """
    MCMC with simulated annealing using block-reversal moves.
    Explores large regions of permutation space to escape local optima.
    """
    n = len(p_init)
    p = p_init.copy()
    D = b[np.ix_(p, p)].copy()
    current_cost = int(np.sum(a * D))
    
    A_sym = a + a.T
    A_diag = np.diag(a)
    
    # Simulated annealing parameters
    T_start = 500.0
    T_end = 1.0
    cooling_rate = 0.995
    
    T = T_start
    best_p = p.copy()
    best_cost = current_cost
    
    # Precompute for delta calculation
    while time.time() - start_time < time_limit:
        # Choose a random block reversal
        L = random.randint(3, min(20, n))
        i = random.randint(0, n - L)
        
        # Compute delta cost for reversing block [i, i+L)
        # Internal pairs remain unchanged, only boundary pairs change
        # Pairs affected: (k, m) where k in [0, i-1] and m in [i, i+L-1]
        # and k in [i+L, n-1] and m in [i, i+L-1]
        # Also pairs where one is in the block and the other is outside
        
        # More precisely: for each position k in the block, its new position is i + (i+L-1-k)
        # So position k maps to position i + L - 1 - (k - i) = 2*i + L - 1 - k
        
        # Let's compute the delta by tracking which facilities move where
        # Old: position k has facility p[k]
        # New: position k has facility p[2*i + L - 1 - k] for k in [i, i+L-1]
        
        # Delta = sum over all pairs (u,v) of a[u,v] * (new_b[u,v] - old_b[u,v])
        # Only pairs where at least one endpoint is in the block change
        
        # For efficiency, compute delta as:
        # For each facility in the block, compute its new distance to all facilities outside
        # and compare with old distance
        
        block_indices = np.arange(i, i + L)
        # New positions for block facilities
        new_positions = 2 * i + L - 1 - block_indices
        
        # Facilities in block
        fac_block = p[block_indices]
        
        # Old distances from block facilities to all facilities
        old_dist_block = D[block_indices, :]  # shape (L, n)
        
        # New distances: facility at old position k goes to new position new_pos
        # Its new distance to facility at position m is b[p[new_pos], p[m]] = D[new_pos, m]
        new_dist_block = D[new_positions, :]  # shape (L, n)
        
        # Delta for pairs where one endpoint is in block
        # For each block position k, old contribution: sum_m a[p[k], p[m]] * D[k, m]
        # New contribution: sum_m a[p[k], p[m]] * D[new_pos(k), m]
        # But we need to be careful about double counting
        
        # Actually, let's use a simpler approach: compute the cost change
        # by considering the full cost difference
        
        # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # After reversal, positions in [i, i+L) have facilities in reverse order
        
        # Delta = sum over affected pairs
        # Affected pairs: (k, m) where k in [i, i+L) or m in [i, i+L)
        
        # Let's compute this directly
        delta = 0
        
        # Pairs where both are in block: internal structure reverses
        # sum_{k,m in block} a[p[k], p[m]] * (D[new_pos(k), new_pos(m)] - D[k, m])
        # But D[new_pos(k), new_pos(m)] = D[2*i+L-1-k, 2*i+L-1-m]
        # And since the reversal is an involution, D[new_pos(k), new_pos(m)] = D[k, m] for internal pairs?
        # No, that's not right. D[k,m] = b[p[k], p[m]], and after reversal,
        # position k has facility p[new_pos(k)], so D_new[k,m] = b[p[new_pos(k)], p[new_pos(m)]]
        # For internal pairs, the set of facility pairs is the same, just permuted
        # So internal contribution is unchanged!
        
        # Only boundary pairs change: one endpoint in block, one outside
        
        # For k in block, m outside block:
        # Old: a[p[k], p[m]] * D[k, m]
        # New: a[p[k], p[m]] * D[new_pos(k), m]
        # Delta_k = sum_{m outside} a[p[k], p[m]] * (D[new_pos(k), m] - D[k, m])
        
        outside = np.setdiff1d(np.arange(n), block_indices)
        
        if len(outside) > 0:
            # For each block position k, compute delta with all outside positions
            # a[p[k], p[outside]] * (D[new_pos(k), outside] - D[k, outside])
            
            a_block_outside = a[np.ix_(fac_block, p[outside])]  # shape (L, n-L)
            D_old_block_outside = D[np.ix_(block_indices, outside)]  # shape (L, n-L)
            D_new_block_outside = D[np.ix_(new_positions, outside)]  # shape (L, n-L)
            
            delta_boundary = np.sum(a_block_outside * (D_new_block_outside - D_old_block_outside))
            delta += delta_boundary
        
        # Also need to account for the fact that a is not symmetric
        # The cost is sum_{i,j} a[i,j] * b[p[i], p[j]], which counts each ordered pair
        # Our delta computation above handles this correctly since we're summing over all ordered pairs
        
        # Acceptance criterion
        if delta < 0:
            # Always accept improving moves
            p[block_indices] = p[block_indices][::-1]
            # Update D
            D[block_indices, :] = D[new_positions, :].copy()
            D[:, block_indices] = D[:, new_positions].copy()
            current_cost += int(delta)
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            # Metropolis acceptance
            if T > T_end:
                acceptance_prob = np.exp(-delta / T)
                if random.random() < acceptance_prob:
                    p[block_indices] = p[block_indices][::-1]
                    D[block_indices, :] = D[new_positions, :].copy()
                    D[:, block_indices] = D[:, new_positions].copy()
                    current_cost += int(delta)
        
        # Cool down
        T = max(T * cooling_rate, T_end)
        
        # If temperature has reached the end, do greedy local search
        if T <= T_end * 1.01:
            # Switch to local search
            p_ls, cost_ls = _local_search(a, b, p, time_limit - (time.time() - start_time), start_time)
            return p_ls, cost_ls
    
    return best_p, best_cost

def _polish_pass(a, b, p, time_limit, start_time):
    """
    High-precision polish: run local search with multiple restarts from 
    small perturbations of the current solution to escape shallow local optima.
    """
    n = len(p)
    best_p = p.copy()
    best_cost = _compute_cost(a, b, p)
    
    # Try a few small perturbations and local search
    for k in [1, 2, 3, 5]:
        if time.time() - start_time > time_limit:
            break
        for trial in range(3):
            if time.time() - start_time > time_limit:
                break
            p_pert = p.copy()
            for _ in range(k):
                i, j = random.sample(range(n), 2)
                p_pert[i], p_pert[j] = p_pert[j], p_pert[i]
            
            remaining = time_limit - (time.time() - start_time)
            if remaining < 1:
                break
            ls_time = min(remaining * 0.3, 10)
            p_ls, cost_ls = _local_search(a, b, p_pert, ls_time, start_time)
            if cost_ls < best_cost:
                best_cost = cost_ls
                best_p = p_ls.copy()
    
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 145  # leave 5s buffer for the 150s limit
    
    best_p = None
    best_cost = np.inf
    
    # Phase 1: Multiple initializations
    
    # 1. Hungarian Initialization
    p_hung = _hungarian_init(a, b)
    p_ls, cost_ls = _local_search(a, b, p_hung, time_budget * 0.15, start_time)
    if cost_ls < best_cost:
        best_cost = cost_ls
        best_p = p_ls.copy()
    
    # 2. Spectral Initialization
    if time.time() - start_time < time_budget * 0.25:
        p_spec = _spectral_init(a, b)
        p_ls, cost_ls = _local_search(a, b, p_spec, time_budget * 0.1, start_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            
    # 3. Greedy Initialization
    if time.time() - start_time < time_budget * 0.35:
        p_greedy = _greedy_init(a, b, n)
        p_ls, cost_ls = _local_search(a, b, p_greedy, time_budget * 0.1, start_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # 4. Random Initials
    num_randoms = 3
    for _ in range(num_randoms):
        if time.time() - start_time > time_budget * 0.45:
            break
        p_init = np.random.permutation(n)
        p_ls, cost_ls = _local_search(a, b, p_init, time_budget * 0.08, start_time)
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
    
    # Phase 2: ILS with MCMC
    elapsed = time.time() - start_time
    remaining_time = time_budget - elapsed
    
    if remaining_time < 5:
        return best_p.tolist()
        
    consecutive_no_improve = 0
    base_perturb = 3
    
    while time.time() - start_time < time_budget - 5:
        if time.time() - start_time > time_budget - 5:
            break
            
        # Adaptive perturbation: mix of swaps and block reversals
        if consecutive_no_improve > 8:
            # Use larger perturbations including block reversals
            if random.random() < 0.5:
                L = random.randint(5, min(30, n))
                p_pert = _block_reverse_perturb(best_p, L)
            else:
                perturb_k = min(20, base_perturb + consecutive_no_improve // 2)
                p_pert = _perturb(best_p, perturb_k)
        elif consecutive_no_improve > 4:
            if random.random() < 0.3:
                L = random.randint(3, min(15, n))
                p_pert = _block_reverse_perturb(best_p, L)
            else:
                p_pert = _perturb(best_p, base_perturb + 2)
        else:
            p_pert = _perturb(best_p, base_perturb)
        
        duration = (time_budget - 5) - (time.time() - start_time)
        if duration < 1:
            break
            
        p_ls, cost_ls = _local_search(a, b, p_pert, duration, start_time)
        
        if cost_ls < best_cost:
            best_cost = cost_ls
            best_p = p_ls.copy()
            consecutive_no_improve = 0
        else:
            consecutive_no_improve += 1
            
        # If stuck, try MCMC
        if consecutive_no_improve > 10 and time.time() - start_time < time_budget - 10:
            mcmc_time = min(time_budget - 10 - (time.time() - start_time), 15)
            if mcmc_time > 3:
                p_mcmc, cost_mcmc = _mcmc_search(a, b, best_p, mcmc_time, start_time)
                if cost_mcmc < best_cost:
                    best_cost = cost_mcmc
                    best_p = p_mcmc.copy()
                    consecutive_no_improve = 0
    
    # Phase 3: Final polish
    elapsed = time.time() - start_time
    remaining_time = time_budget - elapsed
    
    if remaining_time > 5:
        polish_time = min(remaining_time - 2, 20)
        best_p, best_cost = _polish_pass(a, b, best_p, polish_time, start_time)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
