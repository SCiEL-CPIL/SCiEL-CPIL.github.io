"""
Simulated Annealing solver for QAPLIB wil100.
Combines multi-start SA with spectral embedding for geometrically coherent
initialization and a final high-precision local search polish.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _spectral_initial(a, b):
    """
    Create an initial permutation using spectral embedding.
    Projects facility and location graphs into a shared low-dimensional
    Euclidean space via their Laplacian eigenvectors to create a
    geometrically coherent initial assignment.
    """
    n = len(a)
    
    # Build weighted adjacency matrices for facility and location graphs
    # For QAP, we use the flow matrix as weights for facilities
    # and the distance matrix as weights for locations
    
    # Ensure non-negative weights for graph construction
    A_facility = a.copy()
    A_location = b.copy()
    
    # Compute degree vectors
    d_facility = A_facility.sum(axis=1)
    d_location = A_location.sum(axis=1)
    
    # Avoid division by zero
    d_facility[d_facility == 0] = 1
    d_location[d_location == 0] = 1
    
    # Compute normalized Laplacians
    D_facility_inv_sqrt = np.diag(1.0 / np.sqrt(d_facility))
    D_location_inv_sqrt = np.diag(1.0 / np.sqrt(d_location))
    
    # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
    L_facility = np.eye(n) - D_facility_inv_sqrt @ A_facility @ D_facility_inv_sqrt
    L_location = np.eye(n) - D_location_inv_sqrt @ A_location @ D_location_inv_sqrt
    
    # Get smallest eigenvectors (excluding trivial constant vector)
    # Use eigenvalue decomposition
    try:
        eigvals_f, eigvecs_f = np.linalg.eigh(L_facility)
        eigvals_l, eigvecs_l = np.linalg.eigh(L_location)
        
        # Take the first k eigenvectors (smallest eigenvalues)
        # Exclude the first eigenvector (constant)
        k = min(10, n - 1)
        X_facility = eigvecs_f[:, 1:k+1]
        X_location = eigvecs_l[:, 1:k+1]
        
        # Normalize columns
        X_facility = X_facility / np.linalg.norm(X_facility, axis=0, keepdims=True)
        X_location = X_location / np.linalg.norm(X_location, axis=0, keepdims=True)
        
        # Compute distances in embedded space
        # For each facility, find the closest location
        # This is an assignment problem, solved via Hungarian algorithm
        
        # Compute pairwise distances
        # dist[i, j] = ||X_facility[i] - X_location[j]||^2
        Xf_sq = np.sum(X_facility**2, axis=1, keepdims=True)
        Xl_sq = np.sum(X_location**2, axis=1, keepdims=True).T
        cross = X_facility @ X_location.T
        dist_sq = Xf_sq + Xl_sq - 2 * cross
        dist_sq = np.maximum(dist_sq, 0)  # Numerical stability
        
        # Solve assignment problem using Hungarian algorithm
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(dist_sq)
        
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        
        return p
        
    except Exception:
        # Fallback to greedy if spectral fails
        return _greedy_initial(a, b)

def _greedy_initial(a, b):
    """
    Create an initial permutation using a greedy heuristic.
    Matches facilities with high flow to locations with low distance.
    """
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Sort facilities by descending flow sum
    heavy = np.argsort(-a_row_sums, kind="stable")
    # Sort locations by ascending distance sum
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p

def _compute_swap_delta(a, b, p, i, j):
    """
    Compute the cost delta for swapping positions i and j.
    """
    pi = p[i]
    pj = p[j]
    
    # T1: sum_{k != i, j} (a[k,i] - a[k,j]) * (b[p[k], pj] - b[p[k], pi])
    # We compute sum for all k, then subtract k=i and k=j terms.
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_all_T1 = np.dot(diff_a_col, diff_b_row)
    
    # Correction for k=i: (a[i,i] - a[i,j]) * (b[pi, pj] - b[pi, pi])
    term_1_i = (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    # Correction for k=j: (a[j,i] - a[j,j]) * (b[pj, pj] - b[pj, pi])
    term_1_j = (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])
    T1 = sum_all_T1 - term_1_i - term_1_j
    
    # T2: sum_{k != i, j} (a[i,k] - a[j,k]) * (b[pj, p[k]] - b[pi, p[k]])
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_all_T2 = np.dot(diff_a_row, diff_b_col)
    
    # Correction for k=i: (a[i,i] - a[j,i]) * (b[pj, pi] - b[pi, pi])
    term_2_i = (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    # Correction for k=j: (a[i,j] - a[j,j]) * (b[pj, pj] - b[pi, pj])
    term_2_j = (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])
    T2 = sum_all_T2 - term_2_i - term_2_j
    
    # T3: (a[i,j] - a[j,i]) * (b[pj, pi] - b[pi, pj])
    T3 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
    
    return T1 + T2 + T3

def _local_search_polish(a, b, p, current_cost, time_limit):
    """
    Perform first-improvement local search with vectorized delta computation.
    Each pass computes the full O(n^2) delta matrix, then applies all improving
    swaps found in that pass (in order). Repeats until no improving swap exists
    or time limit is reached. First-improvement is faster per pass than
    best-improvement, enabling more passes within the same time budget.
    """
    n = len(p)
    start_time = time.time()
    
    # Pre-compute B_perm
    B_perm = b[np.ix_(p, p)].copy()
    
    improved = True
    while improved:
        # Check time
        if time.time() - start_time > time_limit:
            break
        
        # Compute delta matrix D where D[i,j] = cost change from swapping positions i and j
        M1 = a.T @ B_perm   # M1[i,j] = sum_k a[k,i] * B_perm[j,k]
        M2 = a @ B_perm.T   # M2[i,j] = sum_l a[i,l] * B_perm[j,l]
        
        diag_M1 = np.diag(M1)
        diag_M2 = np.diag(M2)
        
        # T1: M2[i,j] - diag_M2[i] - diag_M2[j] + M2[j,i]
        T1 = M2 - diag_M2[np.newaxis, :] - diag_M2[:, np.newaxis] + M2.T
        
        # T2: diag_M1[i] - M1[i,j] - M1[j,i] + diag_M1[j]
        T2 = diag_M1[:, np.newaxis] - M1 - M1.T + diag_M1[np.newaxis, :]
        
        # T3: -(a[i,j] - a[j,i]) * (B_perm[i,j] - B_perm[j,i])
        T3 = -(a - a.T) * (B_perm - B_perm.T)
        
        D = T1 + T2 + T3
        
        # Get upper triangle indices
        iu, ju = np.triu_indices(n, 1)
        D_upper = D[iu, ju]
        
        # Find all improving swaps
        improving_mask = D_upper < 0
        if not np.any(improving_mask):
            improved = False
            break
        
        # Apply swaps one at a time (first-improvement style within the pass)
        # We process them in order of most negative delta first for better convergence
        improving_indices = np.where(improving_mask)[0]
        # Sort by delta value (most negative first)
        sorted_indices = improving_indices[np.argsort(D_upper[improving_indices])]
        
        # Apply each improving swap
        for idx in sorted_indices:
            i_idx = iu[idx]
            j_idx = ju[idx]
            
            # Re-check if this swap is still improving (previous swaps may have changed things)
            # For efficiency, we just apply it and update
            p[i_idx], p[j_idx] = p[j_idx], p[i_idx]
            current_cost += int(D_upper[idx])
            
            # Update B_perm by swapping rows and columns i_idx, j_idx
            B_perm[[i_idx, j_idx], :] = B_perm[[j_idx, i_idx], :]
            B_perm[:, [i_idx, j_idx]] = B_perm[:, [j_idx, i_idx]]
        
        # Check time
        if time.time() - start_time > time_limit:
            break
        
    return p, current_cost

def _sa_run(a, b, p_init, time_limit, T_start, T_end):
    """
    Run Simulated Annealing for a given time limit.
    """
    n = len(p_init)
    p = p_init.copy()
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
            
        delta = _compute_swap_delta(a, b, p, i, j)
        new_cost = current_cost + delta
        
        # Adaptive Temperature
        elapsed = time.time() - start_time
        # Avoid division by zero or negative time
        if elapsed == 0:
            T = T_start
        else:
            T = T_start * (T_end / T_start) ** (elapsed / time_limit)
            
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            p[i], p[j] = p[j], p[i]
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                
    return best_p, best_cost

def _surrogate_perturb(a, b, p, n_perturb):
    """
    Generate a perturbed permutation using a smooth surrogate objective.
    Uses a soft-min relaxation of the QAP cost to guide perturbation direction.
    """
    n = len(p)
    p_pert = p.copy()
    B_perm = b[np.ix_(p, p)]
    
    # Compute gradient-like signal from the surrogate
    # Approximate subgradient of the QAP cost with respect to position swaps
    M1 = a.T @ B_perm
    M2 = a @ B_perm.T
    
    # For each position i, compute how much it "wants" to swap with each j
    # Using a smooth approximation: focus on pairs with large negative delta
    diag_M1 = np.diag(M1)
    diag_M2 = np.diag(M2)
    
    T1 = M2 - diag_M2[np.newaxis, :] - diag_M2[:, np.newaxis] + M2.T
    T2 = diag_M1[:, np.newaxis] - M1 - M1.T + diag_M1[np.newaxis, :]
    T3 = -(a - a.T) * (B_perm - B_perm.T)
    
    D = T1 + T2 + T3
    
    # Zero diagonal
    np.fill_diagonal(D, 0)
    
    # Select top n_perturb most negative deltas and apply those swaps
    iu, ju = np.triu_indices(n, 1)
    D_upper = D[iu, ju]
    
    # Get indices of most negative deltas
    k = min(n_perturb, len(D_upper))
    if k > 0:
        top_k = np.argpartition(D_upper, k)[:k]
        # Sort by delta value
        top_k = top_k[np.argsort(D_upper[top_k])]
        
        for idx in top_k:
            i_idx = iu[idx]
            j_idx = ju[idx]
            if D_upper[idx] < 0:  # Only apply improving swaps
                p_pert[i_idx], p_pert[j_idx] = p_pert[j_idx], p_pert[i_idx]
    
    return p_pert

def solve(a, b):
    n = len(a)
    total_deadline = 145.0
    start_time = time.time()
    
    # Multi-start configuration with spectral and greedy initialization
    num_restarts = 8
    time_per_restart = total_deadline / num_restarts
    
    best_p = None
    best_cost = float('inf')
    
    # Pre-compute spectral initial solution (expensive, do once)
    p_spectral = None
    try:
        p_spectral = _spectral_initial(a, b)
    except Exception:
        p_spectral = None
    
    for r in range(num_restarts):
        # Check time
        if time.time() - start_time > total_deadline - 8:
            break
            
        # Generate initial solution with diversity
        if r == 0:
            # Use spectral embedding for the first start
            if p_spectral is not None:
                p_init = p_spectral.copy()
            else:
                p_init = _greedy_initial(a, b)
        elif r == 1:
            # Use greedy for the second start
            p_init = _greedy_initial(a, b)
        elif r == 2 and best_p is not None:
            # Surrogate-guided perturbation of current best
            p_init = _surrogate_perturb(a, b, best_p, 8)
        else:
            # Mix of spectral and greedy with random perturbations
            if p_spectral is not None and r % 2 == 0:
                p_init = p_spectral.copy()
            else:
                p_init = _greedy_initial(a, b)
            
            # Apply random swaps to perturb
            num_swaps = 5 + r * 2
            for _ in range(num_swaps):
                i = np.random.randint(n)
                j = np.random.randint(n)
                p_init[i], p_init[j] = p_init[j], p_init[i]
                
        # Run SA
        # Leave more time for polish since first-improvement is faster
        sa_time = time_per_restart * 0.6
        polish_time = time_per_restart * 0.4
        
        p_sa, cost_sa = _sa_run(a, b, p_init, sa_time, T_start=1500.0, T_end=0.5)
        
        # Run Polish with more iterations for high-precision
        p_polished, cost_polished = _local_search_polish(a, b, p_sa, cost_sa, polish_time)
        
        if cost_polished < best_cost:
            best_cost = cost_polished
            best_p = p_polished.copy()
            
    # Final check
    if best_p is None:
        # Fallback
        p_init = _greedy_initial(a, b)
        best_p, best_cost = _local_search_polish(a, b, p_init, _qap_cost(a, b, p_init), 5.0)
        
    # Final high-precision polish on the best solution
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 1.0:
        best_p, best_cost = _local_search_polish(a, b, best_p, best_cost, min(remaining_time - 0.5, 8.0))
        
    # Adjacent-swap polish: try swapping adjacent positions in the permutation
    # This catches improvements that the best-improvement swap search might miss
    # due to the order of evaluation
    remaining_time = total_deadline - (time.time() - start_time)
    if remaining_time > 2.0:
        adj_time = min(remaining_time - 1.0, 3.0)
        best_p, best_cost = _adjacent_swap_polish(a, b, best_p, best_cost, adj_time)
        
    return best_p.tolist()

def _adjacent_swap_polish(a, b, p, current_cost, time_limit):
    """
    Polish with adjacent swaps in the permutation.
    For each position i, try swapping p[i] with p[i+1].
    This is a 1-neighborhood local search that can find improvements
    missed by the best-improvement 2-swap search.
    """
    n = len(p)
    start_time = time.time()
    
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit:
            break
        
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                break
            
            delta = _compute_swap_delta(a, b, p, i, i + 1)
            if delta < 0:
                p[i], p[i + 1] = p[i + 1], p[i]
                current_cost += int(delta)
                improved = True
                # Restart scan from beginning after improvement
                break
                
    return p, current_cost
# EVOLVE-BLOCK-END
