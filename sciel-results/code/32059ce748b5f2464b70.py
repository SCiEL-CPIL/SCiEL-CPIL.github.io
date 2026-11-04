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
    Vectorized for speed.
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


def _build_surrogate(a, b, p):
    """
    Build a surrogate cost matrix C[i, l] approximating the marginal cost
    of assigning facility i to location l, holding other assignments fixed.
    C[i, l] = sum_j (a[i, j] + a[j, i]) * b[p[j], l]
    """
    n = len(p)
    a_sym = a + a.T
    # term[i, l] = sum_j a_sym[i, j] * b[p[j], l]
    # b[p, :] gives rows of b indexed by p, so b[p, l] is the column l of b[p, :]
    # We want sum_j a_sym[i, j] * b[p[j], l]
    # This is (a_sym @ b[p, :])[i, l]
    C = a_sym @ b[p, :]
    return C


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
        a_sym = (a + a.T) / 2.0
        b_sym = (b + b.T) / 2.0
        evals_a, evecs_a = np.linalg.eigh(a_sym)
        evals_b, evecs_b = np.linalg.eigh(b_sym)
    except np.linalg.LinAlgError:
        return rng.permutation(n).astype(np.int64)

    k = min(10, n)
    idx_a = np.argsort(-np.abs(evals_a))[:k]
    fac_emb = evecs_a[:, idx_a]

    idx_b = np.argsort(-np.abs(evals_b))[:k]
    loc_emb = evecs_b[:, idx_b]

    fac_norm = np.linalg.norm(fac_emb, axis=1, keepdims=True)
    fac_norm[fac_norm == 0] = 1
    fac_emb = fac_emb / fac_norm

    loc_norm = np.linalg.norm(loc_emb, axis=1, keepdims=True)
    loc_norm[loc_norm == 0] = 1
    loc_emb = loc_emb / loc_norm

    cost_matrix = -fac_emb @ loc_emb.T
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind

    if rng.random() < 0.3:
        subset_size = rng.integers(5, 20)
        subset = rng.choice(n, size=subset_size, replace=False)
        p[subset] = p[subset][::-1]

    return p


def _quantum_perturb(p, rng, n, temperature=0.5):
    """
    Quantum-inspired perturbation: probabilistic multi-swap moves.
    """
    perturbed_p = p.copy()

    base_swaps = 2
    max_swaps = 8
    n_swaps = base_swaps + int(temperature * (max_swaps - base_swaps))
    n_swaps = max(1, min(n_swaps, n // 2))

    for _ in range(n_swaps):
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i != j:
            perturbed_p[i], perturbed_p[j] = perturbed_p[j], perturbed_p[i]

    if rng.random() < 0.1 * temperature:
        block_len = rng.integers(3, min(10, n // 3))
        start = rng.integers(0, n - block_len + 1)
        perturbed_p[start:start + block_len] = perturbed_p[start:start + block_len][::-1]

    return perturbed_p


def _best_improvement_pass(a, b, p, deadline):
    """
    One pass of best-improvement: find the best swap, apply it.
    """
    n = len(p)
    cost = _cost(a, b, p)
    best_delta = 0
    best_i = -1
    best_j = -1

    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            delta = _delta_swap(a, b, p, i, j)
            if delta < best_delta:
                best_delta = delta
                best_i = i
                best_j = j

    if best_i >= 0:
        p[best_i], p[best_j] = p[best_j], p[best_i]
        cost += best_delta

    return p, cost, best_delta


def _lap_perturbation(a_sym, b, p):
    """
    Generate a new permutation using a linear assignment problem surrogate.
    a_sym is a + a.T.
    """
    n = len(p)
    # L[i, k] = sum_j (a[i, j] + a[j, i]) * b[p[j], k]
    L = a_sym @ b[p, :]
    # Solve LAP
    row_ind, col_ind = linear_sum_assignment(L)
    # row_ind is facilities, col_ind is locations (actual location indices)
    new_p = np.empty(n, dtype=np.int64)
    new_p[row_ind] = col_ind
    return new_p


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin before 150s kill

    # Precompute row sums of a for heuristic
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    a_sym = a + a.T

    # --- Phase 1: Multiple heuristic initializations with local search ---
    best_p = None
    best_cost = np.inf

    rng = np.random.default_rng(12345)

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

    # Heuristic 4: LAP-based initialization
    p4 = _spectral_init(a, b, n, rng)
    p4_lap = _lap_perturbation(a_sym, b, p4)
    p4_lap, c4 = _local_search(a, b, p4_lap, deadline)
    if c4 < best_cost:
        best_cost = c4
        best_p = p4_lap.copy()

    # Heuristic 5: Random
    p5 = rng.permutation(n).astype(np.int64)
    p5, c5 = _local_search(a, b, p5, deadline)
    if c5 < best_cost:
        best_cost = c5
        best_p = p5.copy()

    # --- Phase 2: Iterated Local Search (ILS) with mixed perturbations ---
    current_p = best_p.copy()
    current_cost = best_cost

    ilse_deadline = deadline - 5.0

    iteration = 0
    while time.time() < ilse_deadline:
        iteration += 1

        # Choose perturbation strategy
        r = rng.random()
        if r < 0.4:
            # LAP perturbation
            perturbed_p = _lap_perturbation(a_sym, b, current_p)
            # Add a small random jitter to avoid cycles
            if rng.random() < 0.3:
                i, j = rng.integers(0, n, size=2)
                if i != j:
                    perturbed_p[i], perturbed_p[j] = perturbed_p[j], perturbed_p[i]
        elif r < 0.7:
            # Surrogate-guided perturbation
            perturbed_p = _surrogate_perturb(a, b, current_p, current_cost, rng, n)
        else:
            # Quantum-inspired perturbation
            perturbed_p = _quantum_perturb(current_p, rng, n, temperature=0.5)

        # Local search from perturbed solution
        perturbed_p, perturbed_cost = _local_search(a, b, perturbed_p, ilse_deadline)

        # Accept if better
        if perturbed_cost < best_cost:
            best_cost = perturbed_cost
            best_p = perturbed_p.copy()
            current_p = perturbed_p
            current_cost = perturbed_cost
        elif perturbed_cost == current_cost:
            if rng.random() < 0.1:
                current_p = perturbed_p
                current_cost = perturbed_cost

    # --- Phase 3: Final polish with best-improvement on the best solution ---
    if time.time() < deadline - 2.0:
        polish_deadline = deadline - 1.0
        polish_p = best_p.copy()
        polish_cost = best_cost
        for _ in range(20):
            if time.time() > polish_deadline:
                break
            polish_p, polish_cost, bd = _best_improvement_pass(a, b, polish_p, polish_deadline)
            if bd == 0:
                break
        if polish_cost < best_cost:
            best_cost = polish_cost
            best_p = polish_p.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
