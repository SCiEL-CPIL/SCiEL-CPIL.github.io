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
    Uses vectorized dot products for speed.
    """
    pi = p[i]
    pj = p[j]

    # Direct pair term
    delta = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])

    # Diagonal terms
    delta += a[i, i] * (b[pj, pj] - b[pi, pi])
    delta += a[j, j] * (b[pi, pi] - b[pj, pj])

    # Cross terms using vectorized dot products
    # Term 1: sum_{k != i,j} (a[i,k] - a[j,k]) * (b[pj, p[k]] - b[pi, p[k]])
    diff_a_row = a[i, :] - a[j, :]
    diff_b_col = b[pj, p] - b[pi, p]
    sum_T1 = np.dot(diff_a_row, diff_b_col)
    # Subtract k=i and k=j contributions
    sum_T1 -= (a[i, i] - a[j, i]) * (b[pj, pi] - b[pi, pi])
    sum_T1 -= (a[i, j] - a[j, j]) * (b[pj, pj] - b[pi, pj])

    # Term 2: sum_{k != i,j} (a[k,i] - a[k,j]) * (b[p[k], pj] - b[p[k], pi])
    diff_a_col = a[:, i] - a[:, j]
    diff_b_row = b[p, pj] - b[p, pi]
    sum_T2 = np.dot(diff_a_col, diff_b_row)
    # Subtract k=i and k=j contributions
    sum_T2 -= (a[i, i] - a[i, j]) * (b[pi, pj] - b[pi, pi])
    sum_T2 -= (a[j, i] - a[j, j]) * (b[pj, pj] - b[pj, pi])

    delta += sum_T1 + sum_T2
    return int(delta)


def _local_search(a, b, p, deadline, max_passes=100):
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
                    break  # First improvement per i, then move to next i
        if not improved:
            break

    return p, cost


def _surrogate_perturb(a, b, p, current_cost, rng, n):
    """
    Surrogate-guided perturbation: identify facilities/locations with high
    contribution to cost and perturb those positions.
    """
    perturbed_p = p.copy()

    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)

    contrib = a_row_sums * b_row_sums[p]

    k = 3 + int(rng.integers(0, 4))

    top_idx = np.argsort(-contrib)[:k]

    for idx in top_idx:
        if rng.random() < 0.5:
            j = rng.integers(0, n)
            if j != idx:
                perturbed_p[idx], perturbed_p[j] = perturbed_p[j], perturbed_p[idx]
        else:
            low_idx = np.argsort(contrib)[:k]
            j = low_idx[rng.integers(0, k)]
            if j != idx:
                perturbed_p[idx], perturbed_p[j] = perturbed_p[j], perturbed_p[idx]

    return perturbed_p


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


def _hungarian_init(a, b, n, rng):
    """
    Hungarian algorithm initialization using a linear surrogate cost matrix.
    Combines row-sum centrality with GNN-inspired embeddings.
    """
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)

    # GNN-inspired embeddings (simple message passing)
    a_norm = a / (np.max(np.abs(a)) + 1e-10)
    b_norm = b / (np.max(np.abs(b)) + 1e-10)

    e_f = a_norm.sum(axis=1)
    e_l = b_norm.sum(axis=1)
    for _ in range(2):
        e_f = a_norm @ e_f
        e_l = b_norm @ e_l

    e_f_norm = np.linalg.norm(e_f)
    e_l_norm = np.linalg.norm(e_l)
    if e_f_norm > 0:
        e_f = e_f / e_f_norm
    if e_l_norm > 0:
        e_l = e_l / e_l_norm

    ars_max = np.max(np.abs(a_row_sums))
    brs_max = np.max(np.abs(b_row_sums))
    if ars_max > 0:
        ars_norm = a_row_sums / ars_max
    else:
        ars_norm = a_row_sums
    if brs_max > 0:
        brs_norm = b_row_sums / brs_max
    else:
        brs_norm = b_row_sums

    w1 = 1.0
    w2 = 0.5

    term1 = w1 * np.outer(ars_norm, brs_norm)
    term2 = w2 * np.outer(e_f, e_l)

    C = -(term1 + term2)

    row_ind, col_ind = linear_sum_assignment(C)

    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind

    return p


def _sa_polish(a, b, p, time_limit, rng=None):
    """
    Simulated annealing polish phase with adaptive restarts.
    Uses vectorized delta computation and batch processing.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(p)
    current_cost = _cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost

    start_time = time.time()

    # Adaptive restart parameters
    t0 = 30000.0
    t_final = 0.05
    no_improve_count = 0
    max_no_improve = 50000
    restart_temp_multiplier = 1.5

    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        frac = min(elapsed / time_limit, 1.0)
        temp = t0 * (t_final / t0) ** frac

        # Check for adaptive restart
        if no_improve_count > max_no_improve:
            # Jump back to best solution and increase temperature
            p = best_p.copy()
            current_cost = best_cost
            t0 *= restart_temp_multiplier
            no_improve_count = 0
            continue

        # Batch of random swaps
        for _ in range(50):
            if time.time() - start_time > time_limit:
                break

            i = rng.integers(n)
            j = rng.integers(n)
            if i == j:
                continue

            delta = _delta_swap(a, b, p, i, j)

            # Acceptance criterion
            if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

    return best_p, best_cost


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin before 150s kill

    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)

    best_p = None
    best_cost = np.inf

    rng = np.random.default_rng(12345)

    # --- Phase 1: Multiple heuristic initializations with local search ---

    # Heuristic 1: Rank-matching (heavy facilities to central locations)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy] = central
    p1, c1 = _local_search(a, b, p1.copy(), deadline)
    if c1 < best_cost:
        best_cost = c1
        best_p = p1.copy()

    # Heuristic 2: Reverse rank-matching
    heavy2 = np.argsort(a_row_sums, kind="stable")
    central2 = np.argsort(-b_row_sums, kind="stable")
    p2 = np.empty(n, dtype=np.int64)
    p2[heavy2] = central2
    p2, c2 = _local_search(a, b, p2.copy(), deadline)
    if c2 < best_cost:
        best_cost = c2
        best_p = p2.copy()

    # Heuristic 3: Spectral initialization
    p3 = _spectral_init(a, b, n, rng)
    p3, c3 = _local_search(a, b, p3, deadline)
    if c3 < best_cost:
        best_cost = c3
        best_p = p3.copy()

    # Heuristic 4: Hungarian algorithm initialization
    p4 = _hungarian_init(a, b, n, rng)
    p4, c4 = _local_search(a, b, p4, deadline)
    if c4 < best_cost:
        best_cost = c4
        best_p = p4.copy()

    # Heuristic 5: Random
    p5 = rng.permutation(n).astype(np.int64)
    p5, c5 = _local_search(a, b, p5, deadline)
    if c5 < best_cost:
        best_cost = c5
        best_p = p5.copy()

    # --- Phase 2: Iterated Local Search (ILS) with surrogate-guided perturbations ---
    current_p = best_p.copy()
    current_cost = best_cost

    ilse_deadline = deadline - 20.0  # reserve time for SA polish

    iteration = 0
    while time.time() < ilse_deadline:
        iteration += 1

        if iteration % 3 == 0:
            perturbed_p = _surrogate_perturb(a, b, current_p, current_cost, rng, n)
        else:
            k = 2 + (iteration % 4)
            perturbed_p = current_p.copy()
            for _ in range(k):
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

    # --- Phase 3: SA Polish with adaptive restarts ---
    if time.time() < deadline - 2.0:
        sa_deadline = deadline - 1.0
        sa_p, sa_cost = _sa_polish(a, b, best_p.copy(), sa_deadline - time.time(), rng)
        if sa_cost < best_cost:
            best_cost = sa_cost
            best_p = sa_p.copy()

    # --- Phase 4: Final high-precision local search pass ---
    if time.time() < deadline - 0.5:
        final_p = best_p.copy()
        final_p, final_cost = _local_search(a, b, final_p, deadline - 0.1, max_passes=50)
        if final_cost < best_cost:
            best_cost = final_cost
            best_p = final_p.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
