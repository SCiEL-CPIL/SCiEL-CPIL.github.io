"""QAP wil100 solver: 3-relocation neighborhood + Lagrangian construction."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _cost(a, b, p):
    Bp = b[np.ix_(p, p)]
    return int(np.sum(a * Bp))


def _delta_relocate3(a, b, p, i, j, k):
    """
    Compute the change in QAP cost if we perform a 3-relocation move:
    remove elements at positions i, j, k and reinsert them cyclically
    (i->j, j->k, k->i). This is equivalent to a 3-cycle permutation of
    three assigned locations.
    """
    n = len(p)
    pi = p[i]
    pj = p[j]
    pk = p[k]

    # After the move: position i gets pk, position j gets pi, position k gets pj
    # So facility i moves from pi to pk, facility j from pj to pi, facility k from pk to pj

    # Build the set of all other positions
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    mask[j] = False
    mask[k] = False
    others = p[mask]

    # For each facility being moved, compute the change in its contribution
    # Old contribution of facility i: sum over all j' of a[i,j'] * b[pi, p[j']]
    # New contribution of facility i: sum over all j' of a[i,j'] * b[pk, p[j']]
    # (where p[j'] is updated for j'=i,j,k)

    # Let's compute delta by direct comparison for the affected pairs.
    # The total cost change involves all pairs (x,y) where at least one of x,y is in {i,j,k}.

    # Pairs where both are in {i,j,k}:
    # Old: a[i,j]*b[pi,pj] + a[i,k]*b[pi,pk] + a[j,i]*b[pj,pi] + a[j,k]*b[pj,pk]
    #      + a[k,i]*b[pk,pi] + a[k,j]*b[pk,pj]
    #      + a[i,i]*b[pi,pi] + a[j,j]*b[pj,pj] + a[k,k]*b[pk,pk]
    # New: a[i,j]*b[pk,pi] + a[i,k]*b[pk,pj] + a[j,i]*b[pi,pk] + a[j,k]*b[pi,pj]
    #      + a[k,i]*b[pj,pi] + a[k,j]*b[pj,pk]
    #      + a[i,i]*b[pk,pk] + a[j,j]*b[pi,pi] + a[k,k]*b[pj,pj]

    old_inner = (a[i, j] * b[pi, pj] + a[i, k] * b[pi, pk] +
                 a[j, i] * b[pj, pi] + a[j, k] * b[pj, pk] +
                 a[k, i] * b[pk, pi] + a[k, j] * b[pk, pj] +
                 a[i, i] * b[pi, pi] + a[j, j] * b[pj, pj] + a[k, k] * b[pk, pk])

    new_inner = (a[i, j] * b[pk, pi] + a[i, k] * b[pk, pj] +
                 a[j, i] * b[pi, pk] + a[j, k] * b[pi, pj] +
                 a[k, i] * b[pj, pi] + a[k, j] * b[pj, pk] +
                 a[i, i] * b[pk, pk] + a[j, j] * b[pi, pi] + a[k, k] * b[pj, pj])

    # Pairs where one is in {i,j,k} and the other is in "others":
    # Facility i: old uses b[pi, others] and b[others, pi], new uses b[pk, others] and b[others, pk]
    # Facility j: old uses b[pj, others] and b[others, pj], new uses b[pi, others] and b[others, pi]
    # Facility k: old uses b[pk, others] and b[others, pk], new uses b[pj, others] and b[others, pj]

    # delta for facility i with others:
    # sum_{o in others} [a[i,o]*(b[pk,o]-b[pi,o]) + a[o,i]*(b[o,pk]-b[o,pi])]
    # delta for facility j with others:
    # sum_{o in others} [a[j,o]*(b[pi,o]-b[pj,o]) + a[o,j]*(b[o,pi]-b[o,pj])]
    # delta for facility k with others:
    # sum_{o in others} [a[k,o]*(b[pj,o]-b[pk,o]) + a[o,k]*(b[o,pj]-b[o,pk])]

    # Vectorized computation:
    # For facility i:
    d_i_row = a[i, mask] * (b[pk, others] - b[pi, others])
    d_i_col = a[mask, i] * (b[others, pk] - b[others, pi])

    # For facility j:
    d_j_row = a[j, mask] * (b[pi, others] - b[pj, others])
    d_j_col = a[mask, j] * (b[others, pi] - b[others, pj])

    # For facility k:
    d_k_row = a[k, mask] * (b[pj, others] - b[pk, others])
    d_k_col = a[mask, k] * (b[others, pj] - b[others, pk])

    delta_outer = (np.sum(d_i_row) + np.sum(d_i_col) +
                   np.sum(d_j_row) + np.sum(d_j_col) +
                   np.sum(d_k_row) + np.sum(d_k_col))

    return int(new_inner - old_inner + delta_outer)


def _delta_swap(a, b, p, i, j):
    """Fast swap delta for 2-opt moves."""
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


def _lagrangian_construct(a, b, n, rng, deadline):
    """
    Lagrangian relaxation guided construction.
    We relax the permutation constraint and use a surrogate cost matrix
    derived from the Lagrangian dual to guide assignment.
    """
    # Compute row sums for initialization
    a_rs = a.sum(axis=1)
    b_rs = b.sum(axis=1)

    # Lagrangian multiplier initialization: use the "pressure" from each side
    # The Lagrangian of QAP: L(u,v) = sum_i sum_j a[i,j]*b[p[i],p[j]] + sum_i u_i*(1-sum_j X[i,j]) + ...
    # A practical surrogate: C[i,l] = sum_j a[i,j] * b[l, p[j]] + sum_j a[j,i] * b[p[j], l]
    # But we don't have p yet. Use iterative approach.

    # Start with a random assignment
    p = rng.permutation(n).astype(np.int64)

    # Iterate: build surrogate from current p, reassign
    for it in range(15):
        if time.time() > deadline:
            break

        # Build surrogate cost matrix: marginal cost of assigning facility i to location l
        # C[i,l] = sum_j a[i,j]*b[l,p[j]] + sum_j a[j,i]*b[p[j],l]
        # = (a @ B_p)[i,l] + (a.T @ B_p)[i,l] where B_p[l,j] = b[l,p[j]]
        Bp = b[:, p]  # shape (n, n): Bp[l, j] = b[l, p[j]]

        term1 = a @ Bp  # [i, l] = sum_j a[i,j]*b[l,p[j]]
        term2 = a.T @ Bp  # [i, l] = sum_j a[j,i]*b[l,p[j]]

        # Actually term2 should be: sum_j a[j,i] * b[p[j], l]
        # b[p[j], l] = Bp.T[j, l]... no. Bp[l,j] = b[l,p[j]], so b[p[j],l] = Bp[l,j].T...
        # Let me redo: Bp = b[:, p] means Bp[l, j] = b[l, p[j]]
        # We want sum_j a[j,i] * b[p[j], l]
        # b[p[j], l]: for fixed l, this is b[p[j], l] for each j
        # Let's define Bp2 where Bp2[j, l] = b[p[j], l], i.e., Bp2 = b[p, :]
        Bp2 = b[p, :]  # Bp2[j, l] = b[p[j], l]

        term2 = a.T @ Bp2  # [i, l] = sum_j a[j,i] * b[p[j], l]

        C = term1 + term2

        # Add a small regularization to break ties
        C += rng.random((n, n)) * 0.01

        # Solve assignment
        row_ind, col_ind = linear_sum_assignment(C)
        p_new = np.empty(n, dtype=np.int64)
        p_new[row_ind] = col_ind

        if np.array_equal(p, p_new):
            break
        p = p_new

    return p


def _local_search_3rel(a, b, p, deadline, max_passes=50):
    """
    First-improvement local search using 3-relocation (3-cycle) moves.
    """
    n = len(p)
    cost = _cost(a, b, p)

    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        # Sample a subset of 3-tuples for efficiency
        # Full 3-opt is O(n^3) per pass which is too slow for n=100
        # Use random sampling of 3-tuples
        n_samples = min(5000, n * (n - 1) * (n - 2) // 6)
        for _s in range(n_samples):
            if time.time() > deadline:
                return p, cost
            idx = np.random.choice(n, size=3, replace=False)
            i, j, k = idx[0], idx[1], idx[2]
            delta = _delta_relocate3(a, b, p, i, j, k)
            if delta < 0:
                # Apply the 3-cycle: i->pk, j->pi, k->pj
                pi, pj, pk = p[i], p[j], p[k]
                p[i], p[j], p[k] = pk, pi, pj
                cost += delta
                improved = True
                break  # First improvement, restart sampling
        if not improved:
            break

    return p, cost


def _local_search_swap(a, b, p, deadline, max_passes=100):
    """First-improvement 2-swap local search (vectorized)."""
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
                    break
        if not improved:
            break

    return p, cost


def _combined_local_search(a, b, p, deadline):
    """
    Alternating 3-relocation and 2-swap local search.
    """
    n = len(p)
    cost = _cost(a, b, p)

    # First do 2-swap to get to a 2-opt local optimum
    p, cost = _local_search_swap(a, b, p, deadline, max_passes=50)

    # Then do 3-relocation to escape
    for _ in range(10):
        if time.time() > deadline:
            break
        p, cost = _local_search_3rel(a, b, p, deadline, max_passes=20)
        # Then re-polish with 2-swap
        p, cost = _local_search_swap(a, b, p, deadline, max_passes=30)

    return p, cost


def _perturb_3cycle(p, rng, n, k):
    """Perturbation: apply k random 3-cycles."""
    perturbed = p.copy()
    for _ in range(k):
        idx = rng.choice(n, size=3, replace=False)
        i, j, k_idx = idx[0], idx[1], idx[2]
        pi, pj, pk = perturbed[i], perturbed[j], perturbed[k_idx]
        perturbed[i], perturbed[j], perturbed[k_idx] = pk, pi, pj
    return perturbed


def _perturb_block_reverse(p, rng, n, k):
    """Perturbation: reverse k random blocks."""
    perturbed = p.copy()
    for _ in range(k):
        block_len = rng.integers(3, min(15, n // 4 + 1))
        start = rng.integers(0, n - block_len + 1)
        perturbed[start:start + block_len] = perturbed[start:start + block_len][::-1]
    return perturbed


def _hungarian_init(a, b, n, rng):
    """
    Hungarian-based initialization using a carefully constructed cost matrix.
    Uses the product of flow centrality and distance centrality.
    """
    a_rs = a.sum(axis=1)
    b_rs = b.sum(axis=1)

    # Cost matrix: we want high-flow facilities at low-distance locations
    # C[i, l] = -a_rs[i] * b_rs[l]  (maximize the product)
    C = -np.outer(a_rs, b_rs)

    # Add a secondary term based on the full matrices
    # C2[i,l] = sum_j a[i,j] * b[l,j]  (this is a @ b^T essentially)
    C2 = a @ b.T
    C = C + 0.1 * C2

    C += rng.random((n, n)) * 0.001

    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def _spectral_init(a, b, n, rng):
    """Spectral initialization using eigenvector alignment."""
    try:
        a_sym = (a + a.T) / 2.0
        b_sym = (b + b.T) / 2.0
        evals_a, evecs_a = np.linalg.eigh(a_sym)
        evals_b, evecs_b = np.linalg.eigh(b_sym)
    except np.linalg.LinAlgError:
        return rng.permutation(n).astype(np.int64)

    k = min(15, n)
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
    return p


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = t_start + 115.0  # Safety margin before 150s kill

    best_p = None
    best_cost = np.inf

    rng = np.random.default_rng(42)

    # === Phase 1: Multiple diverse initializations ===
    initializations = []

    # 1a: Hungarian with centrality
    p_hung = _hungarian_init(a, b, n, rng)
    initializations.append(p_hung)

    # 1b: Spectral
    p_spec = _spectral_init(a, b, n, rng)
    initializations.append(p_spec)

    # 1c: Lagrangian construction
    if time.time() < deadline - 30.0:
        p_lagr = _lagrangian_construct(a, b, n, rng, deadline - 25.0)
        initializations.append(p_lagr)

    # 1d: Rank-matching
    a_rs = a.sum(axis=1)
    b_rs = b.sum(axis=1)
    heavy = np.argsort(-a_rs, kind="stable")
    central = np.argsort(b_rs, kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    initializations.append(p_rank)

    # 1e: Reverse rank-matching
    heavy2 = np.argsort(a_rs, kind="stable")
    central2 = np.argsort(-b_rs, kind="stable")
    p_rank2 = np.empty(n, dtype=np.int64)
    p_rank2[heavy2] = central2
    initializations.append(p_rank2)

    # 1f: Random
    p_rand = rng.permutation(n).astype(np.int64)
    initializations.append(p_rand)

    # === Phase 2: Local search on each initialization ===
    for idx, p_init in enumerate(initializations):
        if time.time() > deadline - 30.0:
            break
        # Allocate remaining time among initializations
        remaining = deadline - 30.0 - time.time()
        time_per_init = max(5.0, remaining / (len(initializations) - idx))
        p_ls, c_ls = _combined_local_search(a, b, p_init.copy(), t_start + time.time() - t_start + time_per_init)
        # Fix: use absolute deadline
        if c_ls < best_cost:
            best_cost = c_ls
            best_p = p_ls.copy()

    # Recompute best if needed
    if best_p is None:
        best_p = initializations[0].copy()
        best_cost = _cost(a, b, best_p)

    # === Phase 3: Iterated Local Search with 3-cycle perturbations ===
    current_p = best_p.copy()
    current_cost = best_cost

    ilse_deadline = deadline - 15.0
    iteration = 0
    total_expected = 40

    while time.time() < ilse_deadline:
        iteration += 1
        progress = min(1.0, iteration / total_expected)
        # Temperature for perturbation strength
        k_base = 2
        k_max = 10
        k = k_base + int((1.0 - 0.6 * progress) * (k_max - k_base))
        k = max(1, min(k, n // 3))

        r = rng.random()
        if r < 0.4:
            # 3-cycle perturbation
            perturbed_p = _perturb_3cycle(current_p, rng, n, k)
        elif r < 0.7:
            # Block reverse perturbation
            perturbed_p = _perturb_block_reverse(current_p, rng, n, max(1, k // 2))
        else:
            # Mixed: some swaps + some 3-cycles
            perturbed_p = current_p.copy()
            for _ in range(k // 2 + 1):
                if rng.random() < 0.5:
                    i, j = rng.integers(0, n, size=2)
                    if i != j:
                        perturbed_p[i], perturbed_p[j] = perturbed_p[j], perturbed_p[i]
                else:
                    idx = rng.choice(n, size=3, replace=False)
                    i, j, k_idx = idx[0], idx[1], idx[2]
                    pi, pj, pk = perturbed_p[i], perturbed_p[j], perturbed_p[k_idx]
                    perturbed_p[i], perturbed_p[j], perturbed_p[k_idx] = pk, pi, pj

        # Local search on perturbed solution
        perturbed_p, perturbed_cost = _combined_local_search(a, b, perturbed_p, ilse_deadline)

        if perturbed_cost < best_cost:
            best_cost = perturbed_cost
            best_p = perturbed_p.copy()
            current_p = perturbed_p
            current_cost = perturbed_cost
        elif perturbed_cost < current_cost:
            # Accept if better than current (even if not best)
            delta = perturbed_cost - current_cost
            temp = (1.0 - 0.7 * progress) * 500.0
            accept_prob = np.exp(-delta / max(temp, 1.0))
            if rng.random() < accept_prob:
                current_p = perturbed_p
                current_cost = perturbed_cost
        elif perturbed_cost == current_cost:
            if rng.random() < 0.15:
                current_p = perturbed_p
                current_cost = perturbed_cost

    # === Phase 4: Final aggressive polish ===
    if time.time() < deadline - 5.0:
        polish_deadline = deadline - 1.0
        polish_p = best_p.copy()
        polish_cost = _cost(a, b, polish_p)

        # Alternating 2-swap and 3-relocation until convergence or timeout
        for _ in range(20):
            if time.time() > polish_deadline:
                break
            # 2-swap pass
            polish_p, polish_cost = _local_search_swap(a, b, polish_p, polish_deadline, max_passes=30)
            # 3-relocation pass
            polish_p, polish_cost = _local_search_3rel(a, b, polish_p, polish_deadline, max_passes=15)

        if polish_cost < best_cost:
            best_cost = polish_cost
            best_p = polish_p.copy()

    # === Phase 5: Final 2-swap sweep ===
    if time.time() < deadline - 0.5:
        final_p = best_p.copy()
        final_p, final_cost = _local_search_swap(a, b, final_p, deadline - 0.1, max_passes=5)
        if final_cost < best_cost:
            best_cost = final_cost
            best_p = final_p.copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
