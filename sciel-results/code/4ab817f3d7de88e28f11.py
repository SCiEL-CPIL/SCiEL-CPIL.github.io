<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import eigh


def _spectral_init(a, b):
    """
    Spectral initialization: align high-flow facilities with low-distance locations.
    
    We compute the principal eigenvectors of the flow matrix A and distance matrix B.
    Facilities with high values in the principal eigenvector of A (high "flow centrality")
    should be matched with locations having low values in the principal eigenvector of B
    (low "distance centrality"), to minimize the sum of flow*distance products.
    """
    n = a.shape[0]
    
    # Symmetrize matrices for eigenvalue decomposition
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    # Get the principal (largest magnitude) eigenvectors
    # For A: we want facilities that are "central" in the flow network
    # For B: we want locations that are "peripheral" (low distance centrality)
    
    try:
        # Compute eigenvalues and eigenvectors of A_sym
        # We want the eigenvector corresponding to the largest eigenvalue
        eigvals_a, eigvecs_a = eigh(a_sym)
        # Largest eigenvalue is the last one in ascending order from eigh
        v_a = eigvecs_a[:, -1]
        
        # Compute eigenvalues and eigenvectors of B_sym
        eigvals_b, eigvecs_b = eigh(b_sym)
        # Smallest eigenvalue corresponds to the most "peripheral" structure
        # Actually, we want to align high flow with low distance.
        # The principal eigenvector of B gives distance centrality.
        # We want to assign high-flow facilities to low-distance-centrality locations.
        v_b = eigvecs_b[:, -1]  # Principal eigenvector of B
        
        # Rank facilities by their value in v_a (higher = more central in flow)
        # Rank locations by their value in v_b (lower = more peripheral in distance)
        facility_rank = np.argsort(-v_a)  # Descending order of flow centrality
        location_rank = np.argsort(v_b)   # Ascending order of distance centrality
        
        p = np.empty(n, dtype=np.int64)
        p[facility_rank] = location_rank
        
    except Exception:
        # Fallback to rank-matching if spectral fails
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
    
    return p


def _compute_cost(a, b, p):
    """Compute the QAP cost for a given permutation p."""
    n = len(p)
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # Vectorized: create the permuted distance matrix
    b_perm = b[np.ix_(p, p)]
    return np.sum(a * b_perm)


def _neighborhood_2opt(a, b, p, max_swaps=None):
    """
    Generate and evaluate 2-swap (swap two facilities) neighbors.
    Returns the best neighbor permutation and its cost improvement.
    """
    n = len(p)
    current_cost = _compute_cost(a, b, p)
    
    best_cost = current_cost
    best_p = p.copy()
    
    if max_swaps is None:
        max_swaps = n * (n - 1) // 2
    
    # Evaluate all pairs of swaps
    # For efficiency, we can compute the delta cost for swapping positions i and j
    # But for simplicity and correctness, let's do a direct approach with some optimization
    
    # Precompute row sums and column sums for faster delta computation
    # Actually, let's just do a simple loop over a subset of swaps if n is large
    # For n=100, there are 4950 pairs, which is manageable
    
    p_list = p.tolist()
    
    for i in range(n):
        for j in range(i + 1, n):
            # Swap positions i and j
            p_list[i], p_list[j] = p_list[j], p_list[i]
            
            # Compute new cost
            new_cost = _compute_cost(a, b, np.array(p_list, dtype=np.int64))
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_p = np.array(p_list, dtype=np.int64)
            
            # Swap back
            p_list[i], p_list[j] = p_list[j], p_list[i]
    
    return best_p, best_cost


def _iterated_local_search(a, b, p_init, max_iterations=50, time_budget=100):
    """
    Iterated Local Search:
    1. Start from a good initial solution (spectral init)
    2. Perform local search (2-swap) to reach a local optimum
    3. Apply a perturbation
    4. Repeat local search
    5. Keep the best solution found
    """
    import time
    
    start_time = time.time()
    n = len(p_init)
    
    # Start from spectral initialization
    p_best = p_init.copy()
    cost_best = _compute_cost(a, b, p_best)
    
    p_current = p_best.copy()
    
    for iteration in range(max_iterations):
        # Check time budget
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        # Local search: 2-swap
        p_current, cost_current = _neighborhood_2opt(a, b, p_current)
        
        # Update best
        if cost_current < cost_best:
            cost_best = cost_current
            p_best = p_current.copy()
        
        # Perturbation: swap a random pair of elements
        # The strength of perturbation can increase with iterations
        num_swaps = min(1 + iteration // 10, 5)
        for _ in range(num_swaps):
            i, j = np.random.choice(n, size=2, replace=False)
            p_current[i], p_current[j] = p_current[j], p_current[i]
        
        # Check time budget again
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
    
    return p_best, cost_best


def solve(a, b):
    n = a.shape[0]
    
    # Spectral initialization
    p_spectral = _spectral_init(a, b)
    
    # Iterated Local Search
    p_best, cost_best = _iterated_local_search(a, b, p_spectral, max_iterations=30, time_budget=110)
    
    return p_best.tolist()
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import eigh


def _spectral_init(a, b, seed=0):
    """
    Spectral initialization using geometric alignment.
    
    Computes eigenvectors of flow matrix A and distance matrix B.
    Aligns high-flow facilities with low-distance locations to minimize cost.
    """
    n = a.shape[0]
    rng = np.random.RandomState(seed)
    
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    try:
        eigvals_a, eigvecs_a = eigh(a_sym)
        eigvals_b, eigvecs_b = eigh(b_sym)
        
        # Use principal eigenvector (largest eigenvalue)
        v_a = eigvecs_a[:, -1]
        v_b = eigvecs_b[:, -1]
        
        # Add small random perturbation to break ties
        v_a_perturbed = v_a + rng.randn(n) * 1e-8
        v_b_perturbed = v_b + rng.randn(n) * 1e-8
        
        # Rank facilities by flow centrality (descending)
        # Rank locations by distance centrality (ascending - peripheral first)
        facility_rank = np.argsort(-v_a_perturbed, kind='stable')
        location_rank = np.argsort(v_b_perturbed, kind='stable')
        
        p = np.empty(n, dtype=np.int64)
        p[facility_rank] = location_rank
        
    except Exception:
        # Fallback: rank by row sums
        heavy = np.argsort(-a.sum(axis=1), kind='stable')
        central = np.argsort(b.sum(axis=1), kind='stable')
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
    
    return p


def _compute_cost(a, b, p):
    """Compute the QAP cost for a given permutation p."""
    b_perm = b[np.ix_(p, p)]
    return int(np.sum(a * b_perm))


def _compute_delta_swap(a, b, p, i, j):
    """
    Compute the change in cost when swapping positions i and j in permutation p.
    
    When we swap p[i] and p[j], the cost changes by:
    - Removing contributions involving old p[i] and p[j]
    - Adding contributions involving new p[i] and p[j]
    
    For QAP cost = sum_{k,l} a[k,l] * b[p[k], p[l]]:
    
    The terms affected are those where k or l equals i or j.
    """
    pi = p[i]
    pj = p[j]
    
    # Terms involving position i: sum_l a[i,l]*b[pi,p[l]] + sum_k a[k,i]*b[p[k],pi] - a[i,i]*b[pi,pi]
    # Terms involving position j: sum_l a[j,l]*b[pj,p[l]] + sum_k a[k,j]*b[p[k],pj] - a[j,j]*b[pj,pj]
    # Terms involving both i and j: a[i,j]*b[pi,pj] + a[j,i]*b[pj,pi]
    
    # Old cost contribution from pairs (i,*) and (*,i) and (j,*) and (*,j)
    # But we need to be careful not to double count (i,j) and (j,i)
    
    # Let's compute the delta directly:
    # New cost - Old cost
    
    # For each pair (k,l) where k or l is in {i,j}, compute old and new contribution
    
    # Vectorized approach:
    # old_contrib = sum over all (k,l) of a[k,l] * b[p_old[k], p_old[l]]
    # new_contrib = sum over all (k,l) of a[k,l] * b[p_new[k], p_new[l]]
    # where p_new differs from p_old only at positions i and j
    
    # The difference is:
    # For k=i, l!=i,j: a[i,l] * (b[pi, p[l]] - b[pj, p[l]])  [pi and pj swapped]
    # For k=j, l!=i,j: a[j,l] * (b[pj, p[l]] - b[pi, p[l]])
    # For l=i, k!=i,j: a[k,i] * (b[p[k], pi] - b[p[k], pj])
    # For l=j, k!=i,j: a[k,j] * (b[p[k], pj] - b[p[k], pi])
    # For (i,j): a[i,j] * (b[pi,pj] - b[pj,pi])
    # For (j,i): a[j,i] * (b[pj,pi] - b[pi,pj])
    # For (i,i): a[i,i] * (b[pi,pi] - b[pj,pj])
    # For (j,j): a[j,j] * (b[pj,pj] - b[pi,pi])
    
    # Simplified:
    # delta = sum_{l != i,j} a[i,l]*(b[pi,p[l]] - b[pj,p[l]]) 
    #       + sum_{l != i,j} a[j,l]*(b[pj,p[l]] - b[pi,p[l]])
    #       + sum_{k != i,j} a[k,i]*(b[p[k],pi] - b[p[k],pj])
    #       + sum_{k != i,j} a[k,j]*(b[p[k],pj] - b[p[k],pi])
    #       + a[i,j]*(b[pi,pj] - b[pj,pi])
    #       + a[j,i]*(b[pj,pi] - b[pi,pj])
    #       + a[i,i]*(b[pi,pi] - b[pj,pj])
    #       + a[j,j]*(b[pj,pj] - b[pi,pi])
    
    # Group terms:
    # = sum_{l != i,j} (a[i,l] - a[j,l]) * (b[pi,p[l]] - b[pj,p[l]])
    #   + sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k],pi] - b[p[k],pj])
    #   + (a[i,j] - a[j,i]) * (b[pi,pj] - b[pj,pi])
    #   + (a[i,i] - a[j,j]) * (b[pi,pi] - b[pj,pj])
    
    # Let's compute this efficiently
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    mask[j] = False
    
    # Row contributions: for each l not in {i,j}
    # sum_l a[i,l]*(b[pi,p[l]] - b[pj,p[l]]) - sum_l a[j,l]*(b[pi,p[l]] - b[pj,p[l]])
    row_diff = a[i, mask] - a[j, mask]  # shape (n-2,)
    b_diff_row = b[pi, p[mask]] - b[pj, p[mask]]  # shape (n-2,)
    row_contrib = np.dot(row_diff, b_diff_row)
    
    # Column contributions: for each k not in {i,j}
    # sum_k a[k,i]*(b[p[k],pi] - b[p[k],pj]) - sum_k a[k,j]*(b[p[k],pi] - b[p[k],pj])
    col_diff = a[mask, i] - a[mask, j]  # shape (n-2,)
    b_diff_col = b[p[mask], pi] - b[p[mask], pj]  # shape (n-2,)
    col_contrib = np.dot(col_diff, b_diff_col)
    
    # Cross terms (i,j) and (j,i)
    cross_contrib = (a[i,j] - a[j,i]) * (b[pi,pj] - b[pj,pi])
    
    # Diagonal terms
    diag_contrib = (a[i,i] - a[j,j]) * (b[pi,pi] - b[pj,pj])
    
    delta = row_contrib + col_contrib + cross_contrib + diag_contrib
    
    return int(delta)


def _local_search_2swap(a, b, p, max_passes=5, time_budget=100):
    """
    Local search using 2-swap moves with efficient delta computation.
    Performs multiple passes until no improvement or time budget exceeded.
    """
    start_time = time.time()
    n = len(p)
    p = p.copy()
    cost = _compute_cost(a, b, p)
    
    for _pass in range(max_passes):
        improved = False
        # Check time
        if time.time() - start_time > time_budget:
            break
        
        # Try all pairs
        for i in range(n):
            for j in range(i + 1, n):
                delta = _compute_delta_swap(a, b, p, i, j)
                if delta < 0:
                    # Accept the swap
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
        
        if not improved:
            break
    
    return p, cost


def _iterated_local_search(a, b, p_init, time_budget=100):
    """
    Iterated Local Search with multiple restarts.
    """
    start_time = time.time()
    n = len(p_init)
    rng = np.random.RandomState(42)
    
    p_best = p_init.copy()
    cost_best = _compute_cost(a, b, p_best)
    
    p_current = p_best.copy()
    
    iteration = 0
    while time.time() - start_time < time_budget:
        # Local search
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time < 5:
            break
            
        p_current, cost_current = _local_search_2swap(
            a, b, p_current, 
            max_passes=3,
            time_budget=remaining_time - 2
        )
        
        # Update best
        if cost_current < cost_best:
            cost_best = cost_current
            p_best = p_current.copy()
        
        # Perturbation
        # Use a random number of swaps based on iteration
        num_swaps = 1 + (iteration % 5)
        for _ in range(num_swaps):
            idx = rng.choice(n, size=2, replace=False)
            p_current[idx[0]], p_current[idx[1]] = p_current[idx[1]], p_current[idx[0]]
        
        iteration += 1
    
    return p_best, cost_best


def solve(a, b):
    n = a.shape[0]
    
    # Try multiple spectral initializations with different seeds
    p_best = None
    cost_best = np.inf
    
    # First, try the main spectral init
    p_spectral = _spectral_init(a, b, seed=0)
    p_best, cost_best = _iterated_local_search(a, b, p_spectral, time_budget=90)
    
    # If we have time left, try another initialization
    if time.time() - 0 < 100:  # This won't work properly, let's use a different approach
        pass
    
    return p_best.tolist()
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I need to fix the time tracking in the solve function. Let me provide a cleaner version:

