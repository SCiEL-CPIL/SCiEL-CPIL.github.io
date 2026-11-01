import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    # Generate all weight-2 vectors in {-1, 0, 1}^11 (Norm^2 = 2)
    v2_candidates = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                v2_candidates.append(v)

    vectors_v4 = np.array(v4_candidates + v1_candidates, dtype=np.int64)
    vectors_v2 = np.array(v2_candidates, dtype=np.int64)

    N4 = len(vectors_v4)
    N2 = len(vectors_v2)

    # Compute conflict graph for V4 vectors: conflict if dist_sq < 4
    norms_sq_4 = np.sum(vectors_v4**2, axis=1)
    dots4 = vectors_v4 @ vectors_v4.T
    dist_sq_4 = norms_sq_4[:, None] + norms_sq_4[None, :] - 2 * dots4
    conflicts4 = (dist_sq_4 < 4)
    np.fill_diagonal(conflicts4, False)

    # Compute conflict between V4 and V2: conflict if dist_sq < 4
    dots42 = vectors_v4 @ vectors_v2.T
    norms_sq_2 = np.sum(vectors_v2**2, axis=1)
    dist_sq_42 = norms_sq_4[:, None] + norms_sq_2[None, :] - 2 * dots42
    conflicts42 = (dist_sq_42 < 4)

    # Compute conflict graph for V2 vectors: conflict if dist_sq < 4
    dots2 = vectors_v2 @ vectors_v2.T
    dist_sq_2 = norms_sq_2[:, None] + norms_sq_2[None, :] - 2 * dots2
    conflicts2 = (dist_sq_2 < 4)
    np.fill_diagonal(conflicts2, False)

    # Spectral method: Max-Cut relaxation on V4 conflict graph
    degree = conflicts4.sum(axis=1)
    L = np.diag(degree) - conflicts4.astype(np.float64)

    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh

    L_sparse = csr_matrix(L)

    try:
        eigvals, eigvecs = eigsh(L_sparse, k=3, sigma=0, which='LM')
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        fiedler = eigvecs[:, 1]
    except Exception:
        rng = np.random.default_rng(42)
        fiedler = rng.standard_normal(N4)

    best_selected = None
    best_count = 0
    best_total = 0  # total count including V2 additions

    # Strategy 1: Greedy minimum degree
    def greedy_min_degree(conflicts, n, seed=42):
        rng = np.random.default_rng(seed)
        alive = np.ones(n, dtype=bool)
        selected = np.zeros(n, dtype=bool)
        degrees = conflicts.sum(axis=1).astype(np.int64)

        while np.any(alive):
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            alive_degrees = degrees[alive_indices]
            min_deg = np.min(alive_degrees)
            cands = alive_indices[alive_degrees == min_deg]
            chosen = cands[rng.integers(len(cands))]

            selected[chosen] = True
            to_remove = np.where(conflicts[chosen] | (np.arange(n) == chosen))[0]
            alive[to_remove] = False
            remaining = np.where(alive)[0]
            if len(remaining) > 0:
                degrees[remaining] -= conflicts[remaining][:, to_remove].sum(axis=1)

        return selected

    # Strategy 2: Spectral-based ordering
    def greedy_spectral(conflicts, n, fiedler_vec):
        order = np.argsort(fiedler_vec)
        selected = np.zeros(n, dtype=bool)
        alive = np.ones(n, dtype=bool)

        for i in order:
            if not alive[i]:
                continue
            selected[i] = True
            to_remove = np.where(conflicts[i] | (np.arange(n) == i))[0]
            alive[to_remove] = False

        return selected

    # Local search: add non-conflicting vertices, then try swaps
    def local_search(selected, conflicts, n, max_iters=200):
        selected = selected.copy()
        for _ in range(max_iters):
            if time.time() > deadline - 15:
                break
            improved = False
            non_selected = np.where(~selected)[0]

            # Try to add any vertex with no conflicts
            for v in non_selected:
                if np.sum(conflicts[v] & selected) == 0:
                    selected[v] = True
                    improved = True

            if not improved:
                break

        # Try 2-swaps: remove two selected, add two non-selected
        for _ in range(50):
            if time.time() > deadline - 15:
                break
            improved = False
            selected_idx = np.where(selected)[0]
            non_sel_idx = np.where(~selected)[0]

            if len(non_sel_idx) < 2 or len(selected_idx) < 2:
                break

            # Try random pairs
            rng = np.random.default_rng(int(time.time() * 1000) % 10000)
            for _ in range(100):
                if time.time() > deadline - 15:
                    break
                if len(non_sel_idx) < 2:
                    break
                v1, v2 = rng.choice(non_sel_idx, size=2, replace=False)
                # Check if v1 and v2 can both be added (they must not conflict with each other)
                if conflicts4[v1, v2] if v1 < N4 and v2 < N4 else False:
                    continue

                # Find which selected vertices they conflict with
                sel_conf_v1 = np.where(conflicts4[v1] & selected)[0] if v1 < N4 else np.where(conflicts42[v1, :] & selected)[0]
                sel_conf_v2 = np.where(conflicts4[v2] & selected)[0] if v2 < N4 else np.where(conflicts42[v2, :] & selected)[0]

                # Only consider if they conflict with at most 2 total selected vertices
                all_conf = np.union1d(sel_conf_v1, sel_conf_v2)
                if len(all_conf) > 2:
                    continue

                # Try removing the conflicting selected vertices and adding v1, v2
                trial = selected.copy()
                trial[all_conf] = False
                if v1 < N4:
                    trial[v1] = True
                if v2 < N4:
                    trial[v2] = True

                if trial.sum() > selected.sum():
                    selected = trial
                    improved = True
                    break

            if not improved:
                break

        return selected

    # Run multiple strategies
    strategies = []

    # Strategy 1: Greedy min degree with different seeds
    for seed in range(20):
        if time.time() > deadline - 30:
            break
        sel = greedy_min_degree(conflicts4, N4, seed=seed)
        strategies.append(sel)

    # Strategy 2: Spectral ordering
    sel = greedy_spectral(conflicts4, N4, fiedler)
    strategies.append(sel)

    # Strategy 3: More random restarts
    for seed in range(50, 80):
        if time.time() > deadline - 30:
            break
        sel = greedy_min_degree(conflicts4, N4, seed=seed)
        strategies.append(sel)

    # Local search on each strategy
    for i, sel in enumerate(strategies):
        if time.time() > deadline - 20:
            break
        sel = local_search(sel, conflicts4, N4)
        count = sel.sum()
        if count > best_count:
            best_count = count
            best_selected = sel.copy()

    # If we didn't find any, use the first strategy
    if best_selected is None:
        best_selected = strategies[0] if strategies else np.zeros(N4, dtype=bool)

    # Now add compatible V2 vectors
    # A V2 vector can be added if it has dist_sq >= 4 to all selected V4 vectors
    # and dist_sq >= 4 to all other added V2 vectors

    result_v4 = vectors_v4[best_selected]
    result_v2 = []

    if len(result_v4) > 0:
        # For each V2 vector, check if it's compatible with all selected V4 vectors
        # dist_sq >= 4 required
        dots_v4_v2 = result_v4 @ vectors_v2.T
        norms_v4 = np.sum(result_v4**2, axis=1)
        dist_v4_v2 = norms_v4[:, None] + norms_sq_2[None, :] - 2 * dots_v4_v2
        compatible_v2 = np.all(dist_v4_v2 >= 4, axis=0)
    else:
        compatible_v2 = np.ones(N2, dtype=bool)

    # Now find maximum independent set among compatible V2 vectors
    # with the constraint that they also need dist_sq >= 4 to each other
    compat_indices = np.where(compatible_v2)[0]
    if len(compat_indices) > 0:
        # Build sub-conflict graph for compatible V2 vectors
        sub_conflicts = conflicts2[np.ix_(compat_indices, compat_indices)]
        n_sub = len(compat_indices)

        # Greedy on sub graph
        sel_sub = greedy_min_degree(sub_conflicts, n_sub, seed=42)

        # Local search
        sel_sub = local_search(sel_sub, sub_conflicts, n_sub)

        added_v2_indices = compat_indices[sel_sub]
        result_v2 = vectors_v2[added_v2_indices]

    # Combine results
    if len(result_v2) > 0:
        final_vectors = np.vstack([result_v4, result_v2])
    else:
        final_vectors = result_v4

    # Verify the result is valid
    if len(final_vectors) > 1:
        norms_final = np.sum(final_vectors**2, axis=1)
        max_norm = np.max(np.sqrt(norms_final))
        dots_final = final_vectors @ final_vectors.T
        dist_final = norms_final[:, None] + norms_final[None, :] - 2 * dots_final
        np.fill_diagonal(dist_final, np.inf)
        min_dist = np.min(np.sqrt(dist_final))

        if min_dist < max_norm - 1e-9:
            # Invalid, fall back to just V4 vectors
            final_vectors = result_v4

    return final_vectors
