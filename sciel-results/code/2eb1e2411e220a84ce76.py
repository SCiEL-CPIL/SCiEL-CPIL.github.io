import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Generate all t-subsets and map them to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}

    # 2. Generate all k-subsets (candidate blocks)
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # 3. Build coverage matrix: n_ksubs x n_tsubs (uint8), where M[j, i] = 1 if k-sub j covers t-sub i
    #    This is memory-heavy but enables vectorized operations
    #    For v=19, k=7, t=4: n_ksubs = C(19,7) = 50388, n_tsubs = C(19,4) = 3876
    #    Matrix: 50388 * 3876 bytes = ~195 MB as uint8 - manageable
    cov_matrix = np.zeros((n_ksubs, n_tsubs), dtype=np.uint8)
    for j, ks in enumerate(all_ksubs):
        for tsub in itertools.combinations(ks, t):
            idx = tsub_to_idx[tsub]
            cov_matrix[j, idx] = 1

    # Precompute row sums (total coverage of each block)
    total_cov = cov_matrix.sum(axis=1)

    def greedy_cover(seed, max_time):
        """Run greedy set cover with randomization for tie-breaking."""
        rng = np.random.RandomState(seed)
        uncovered = np.ones(n_tsubs, dtype=np.uint8)
        remaining = n_tsubs
        used = np.zeros(n_ksubs, dtype=bool)
        selected = []
        t_start = time.time()

        while remaining > 0 and time.time() < max_time:
            # Vectorized gain computation: for each unused block, count how many uncovered t-subs it covers
            gains = (cov_matrix & uncovered[None, :]).sum(axis=1)
            gains[used] = 0

            # Find max gain
            max_gain = gains.max()
            if max_gain == 0:
                break

            # Get all blocks with max gain (or top-k for randomization)
            top_k = min(20, n_ksubs)
            # Use random tie-breaking: add small random noise to top candidates
            candidates = np.where(gains >= max_gain)[0]
            if len(candidates) > top_k:
                # Randomly pick from top candidates
                chosen_idx = candidates[rng.randint(len(candidates))]
            else:
                chosen_idx = candidates[rng.randint(len(candidates))]

            selected.append(chosen_idx)
            used[chosen_idx] = True
            uncovered = uncovered & (1 - cov_matrix[chosen_idx])
            remaining = uncovered.sum()

        return selected

    def remove_redundant(selected, max_time):
        """Remove redundant blocks from the selected set."""
        if not selected:
            return selected

        active = list(selected)
        t_start = time.time()
        changed = True

        while changed and time.time() < max_time:
            changed = False
            new_active = []
            for i, bi in enumerate(active):
                # Check if bi is redundant: all its covered t-subs are covered by other active blocks
                if len(active) <= 1:
                    new_active.append(bi)
                    continue

                # Build union mask of all other blocks
                other_mask = np.zeros(n_tsubs, dtype=np.uint8)
                for j, bj in enumerate(active):
                    if j != i:
                        other_mask |= cov_matrix[bj]

                # Check if bi's coverage is subset of other_mask
                if (cov_matrix[bi] & ~other_mask).sum() == 0:
                    changed = True
                else:
                    new_active.append(bi)

            active = new_active

        return active

    # Main search: multiple restarts with different random seeds
    best_blocks = None
    best_count = float('inf')

    # Estimate time per restart
    # First, do one quick greedy to estimate
    t0 = time.time()
    quick_selected = greedy_cover(0, deadline - 10.0)
    t1 = time.time()
    quick_time = t1 - t0

    if quick_selected:
        quick_active = remove_redundant(quick_selected, deadline - 5.0)
        if quick_active:
            best_blocks = [list(all_ksubs[idx]) for idx in quick_active]
            best_count = len(quick_active)

    # Now do additional restarts if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 15.0:
        n_restarts = min(50, int(remaining_time / max(quick_time, 1.0)))
        for r in range(n_restarts):
            if time.time() > deadline - 5.0:
                break
            seed = r + 1
            time_budget = min(deadline - time.time() - 2.0, quick_time * 1.2)
            if time_budget < 2.0:
                break

            selected = greedy_cover(seed, time_budget)
            if not selected:
                continue

            active = remove_redundant(selected, deadline - 2.0)
            if active and len(active) < best_count:
                best_count = len(active)
                best_blocks = [list(all_ksubs[idx]) for idx in active]

    if best_blocks is None:
        # Fallback: return a valid but potentially large covering
        best_blocks = [list(all_ksubs[idx]) for idx in quick_selected]

    return best_blocks
