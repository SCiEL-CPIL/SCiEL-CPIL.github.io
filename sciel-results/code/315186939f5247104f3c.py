"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    Bp = b[np.ix_(p, p)]
    return int(np.sum(a * Bp))


def _delta_swap(a, b, p, i, j):
    """
    Compute the change in cost if we swap facilities i and j in the assignment p.
    """
    pi = p[i]
    pj = p[j]
    n = len(p)

    mask = np.ones(n, dtype=bool)
    mask[i] = False
    mask[j] = False

    diff1 = b[pj, p[mask]] - b[pi, p[mask]]
    diff2 = b[p[mask], pj] - b[p[mask], pi]

    a_diff_row = a[i, mask] - a[j, mask]
    a_diff_col = a[mask, i] - a[mask, j]

    delta_k = np.sum(a_diff_row * diff1) + np.sum(a_diff_col * diff2)

    delta_self = (a[i, j] * (b[pj, pi] - b[pi, pj])
                  + a[j, i] * (b[pi, pj] - b[pj, pi])
                  + a[i, i] * (b[pj, pj] - b[pi, pi])
                  + a[j, j] * (b[pi, pi] - b[pj, pj]))

    return int(delta_k + delta_self)


def _local_search(a, b, p, deadline, max_passes=50):
    """
    First-improvement local search with 2-opt swaps.
    Returns (p, cost).
    """
    n = len(p)
    cost = _cost(a, b, p)

    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                return p, cost
            for j in range(i + 1, n):
                delta = _delta_swap(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
        if not improved:
            break

    return p, cost


def _spectral_init(a, b, n, rng):
    """
    Spectral initialization: align facility ordering with location ordering
    based on eigenvectors of the flow and distance matrices.
    """
    try:
        evals_a, evecs_a = np.linalg.eigh(a)
        evals_b, evecs_b = np.linalg.eigh(b)
    except np.linalg.LinAlgError:
        return rng.permutation(n).astype(np.int64)

    idx_a = np.argmax(evals_a)
    idx_b = np.argmax(evals_b)

    vec_a = evecs_a[:, idx_a]
    vec_b = evecs_b[:, idx_b]

    facility_order = np.argsort(-vec_a, kind="stable")
    location_order = np.argsort(-vec_b, kind="stable")

    p = np.empty(n, dtype=np.int64)
    p[facility_order] = location_order

    if rng.random() < 0.5:
        location_order = location_order[::-1]
        p = np.empty(n, dtype=np.int64)
        p[facility_order] = location_order

    return p


def _lap_perturbation(a_sym, b, p):
    """
    Generate a new permutation using a linear assignment problem surrogate.
    a_sym is a + a.T.
    """
    n = len(p)
    Bp = b[np.ix_(p, p)]
    L = a_sym @ Bp
    row_ind, col_ind = linear_sum_assignment(L)
    new_p = np.empty(n, dtype=np.int64)
    for i, k in zip(row_ind, col_ind):
        new_p[i] = p[k]
    return new_p


def _hungarian_init(a, b, n, rng):
    """
    Formulate QAP as max-weight perfect matching on bipartite graph.
    Edge weights derived from facility flow centrality * location distance centrality.
    Solve via Hungarian to get a structurally coherent initial assignment.
    """
    # Facility centrality: row sums of flow matrix
    fac_cent = a.sum(axis=1).astype(np.float64)
    # Location centrality: row sums of distance matrix (lower is better, so invert)
    loc_cent = -b.sum(axis=1).astype(np.float64)
    
    # Build weight matrix: W[i, j] = fac_cent[i] * loc_cent[j]
    W = np.outer(fac_cent, loc_cent)
    
    # Solve min-cost assignment (Hungarian minimizes, so negate for max)
    cost_mat = -W
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    
    p = np.empty(n, dtype=np.int64)
    for i, j in zip(row_ind, col_ind):
        p[i] = j
    
    # Add small random perturbation to break ties
    if rng.random() < 0.5:
        k = rng.integers(1, 5)
        for _ in range(k):
            i, j = rng.integers(0, n, size=2)
            if i != j:
                p[i], p[j] = p[j], p[i]
    
    return p


def _simulated_annealing(a, b, p, t_start, deadline, rng, max_iter=None):
    """
    Vectorized Simulated Annealing loop starting from p.
    Uses 2-swap moves with temperature schedule.
    Returns (p, cost).
    """
    n = len(p)
    cost = _cost(a, b, p)
    best_p = p.copy()
    best_cost = cost
    
    # Temperature schedule
    T_start = max(100.0, abs(cost) * 0.01)
    T_end = 1.0
    # Estimate number of iterations we can do
    # Each iteration: one swap evaluation (~0.1ms for n=100)
    # Be conservative: assume ~5000 iterations per second
    time_left = deadline - t_start
    if max_iter is None:
        max_iter = int(time_left * 3000)  # conservative estimate
    
    if max_iter <= 0:
        return p, cost
    
    # Geometric cooling
    if T_start > T_end:
        cooling = (T_end / T_start) ** (1.0 / max(1, max_iter))
    else:
        cooling = 1.0
    
    T = T_start
    current_p = p.copy()
    current_cost = cost
    
    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        # Generate random swap
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        
        delta = _delta_swap(a, b, current_p, i, j)
        
        # Acceptance criterion
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current_p[i], current_p[j] = current_p[j], current_p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = current_p.copy()
        
        T *= cooling
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = t_start + 145.0  # safety margin before 150s kill

    a_sym = a + a.T
    rng = np.random.default_rng(42)

    best_p = None
    best_cost = np.inf

    # --- Phase 1: Multiple diverse initializations ---
    initializations = []
    
    # Init 1: Hungarian (max-weight matching based on centrality products)
    p_hung = _hungarian_init(a, b, n, rng)
    initializations.append(p_hung)
    
    # Init 2: Rank-matching (heavy facilities to central locations)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    initializations.append(p_rank)
    
    # Init 3: Reverse rank-matching
    heavy2 = np.argsort(a_row_sums, kind="stable")
    central2 = np.argsort(-b_row_sums, kind="stable")
    p_rank_rev = np.empty(n, dtype=np.int64)
    p_rank_rev[heavy2] = central2
    initializations.append(p_rank_rev)
    
    # Init 4: Spectral
    p_spec = _spectral_init(a, b, n, rng)
    initializations.append(p_spec)
    
    # Init 5: LAP-based
    p_lap = _lap_perturbation(a_sym, b, p_spec)
    initializations.append(p_lap)
    
    # Init 6: Random
    p_rand = rng.permutation(n).astype(np.int64)
    initializations.append(p_rand)

    # Local search on each initialization
    for p_init in initializations:
        if time.time() > deadline - 60.0:
            break
        p_ls, c_ls = _local_search(a, b, p_init.copy(), deadline - 60.0)
        if c_ls < best_cost:
            best_cost = c_ls
            best_p = p_ls.copy()

    # --- Phase 2: Simulated Annealing from best solution ---
    if time.time() < deadline - 30.0:
        sa_deadline = deadline - 5.0
        p_sa, c_sa = _simulated_annealing(a, b, best_p.copy(), t_start, sa_deadline, rng)
        if c_sa < best_cost:
            best_cost = c_sa
            best_p = p_sa.copy()
        # Local search polish on SA result
        if time.time() < deadline - 5.0:
            p_polish, c_polish = _local_search(a, b, p_sa.copy(), deadline - 3.0)
            if c_polish < best_cost:
                best_cost = c_polish
                best_p = p_polish.copy()

    # --- Phase 3: Iterated Local Search with perturbations ---
    current_p = best_p.copy()
    current_cost = best_cost
    ilse_deadline = deadline - 5.0

    iteration = 0
    while time.time() < ilse_deadline:
        iteration += 1

        if iteration % 7 == 0:
            # Random perturbation
            k = 3 + (iteration % 5)
            perturbed_p = current_p.copy()
            for _ in range(k):
                i, j = rng.integers(0, n, size=2)
                if i != j:
                    perturbed_p[i], perturbed_p[j] = perturbed_p[j], perturbed_p[i]
        else:
            # LAP perturbation
            perturbed_p = _lap_perturbation(a_sym, b, current_p)
            if rng.random() < 0.3:
                i, j = rng.integers(0, n, size=2)
                if i != j:
                    perturbed_p[i], perturbed_p[j] = perturbed_p[j], perturbed_p[i]

        perturbed_p, perturbed_cost = _local_search(a, b, perturbed_p, ilse_deadline)

        if perturbed_cost < best_cost:
            best_cost = perturbed_cost
            best_p = perturbed_p.copy()
            current_p = perturbed_p
            current_cost = perturbed_cost
        elif perturbed_cost == current_cost:
            if rng.random() < 0.1:
                current_p = perturbed_p
                current_cost = perturbed_cost

    # --- Phase 4: Final polish ---
    if time.time() < deadline - 2.0:
        polish_deadline = deadline - 1.0
        polish_p = best_p.copy()
        polish_cost = best_cost
        for _ in range(10):
            if time.time() > polish_deadline:
                break
            improved = False
            for i in range(n):
                if time.time() > polish_deadline:
                    break
                for j in range(i + 1, n):
                    delta = _delta_swap(a, b, polish_p, i, j)
                    if delta < 0:
                        polish_p[i], polish_p[j] = polish_p[j], polish_p[i]
                        polish_cost += delta
                        improved = True
            if not improved:
                break
        if polish_cost < best_cost:
            best_cost = polish_cost
            best_p = polish_p.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
