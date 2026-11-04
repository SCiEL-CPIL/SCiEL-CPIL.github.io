import itertools
import time
import sys
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using vectorized greedy with
    multiple randomized restarts, local-search polish, and an optional ILP
    refinement. Returns a list of blocks (each a list of k distinct ints).
    """
    deadline = time.monotonic() + 145.0  # safety margin before 150s kill
    points = list(range(v))

    # Precomputation: all t-subsets and all k-blocks
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)

    all_kblocks = list(itertools.combinations(points, k))
    num_kblocks = len(all_kblocks)

    # Build sparse incidence matrix A (num_tsubs x num_kblocks)
    row_indices = []
    col_indices = []
    data = []
    for j, b in enumerate(all_kblocks):
        for ts in itertools.combinations(b, t):
            i = tsub_to_idx[ts]
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)

    A = csr_matrix((data, (row_indices, col_indices)), shape=(num_tsubs, num_kblocks))
    A_dense = A.toarray()  # for fast vectorized ops; num_tsubs=3876, num_kblocks=50388 -> ~155 MB as int8? use uint8

    # Precompute per-block t-subset index lists for pruning/local search
    block_tsub_indices = []
    for i, b in enumerate(all_kblocks):
        idxs = [tsub_to_idx[ts] for ts in itertools.combinations(b, t)]
        block_tsub_indices.append(idxs)

    full_mask = (1 << num_tsubs) - 1

    # Helper: compute coverage mask for a list of block indices
    def coverage_mask(block_indices):
        m = 0
        for bi in block_indices:
            for ti in block_tsub_indices[bi]:
                m |= (1 << ti)
        return m

    # Helper: check if a set of blocks covers everything
    def is_complete(block_indices):
        return coverage_mask(block_indices) == full_mask

    # Helper: prune redundant blocks from a list
    def prune(block_indices):
        if not block_indices:
            return block_indices
        # Count coverage per t-subset
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for bi in block_indices:
            for ti in block_tsub_indices[bi]:
                counts[ti] += 1
        changed = True
        while changed:
            if time.monotonic() > deadline - 10:
                break
            changed = False
            new_list = []
            for bi in block_indices:
                redundant = True
                for ti in block_tsub_indices[bi]:
                    if counts[ti] == 1:
                        redundant = False
                        break
                if redundant:
                    for ti in block_tsub_indices[bi]:
                        counts[ti] -= 1
                    changed = True
                else:
                    new_list.append(bi)
            block_indices = new_list
        return block_indices

    # Vectorized greedy with optional random tie-breaking
    def greedy_random(noise_scale=0.0, seed=None):
        """
        Vectorized greedy. Returns list of block indices.
        If noise_scale > 0, adds random perturbation to gains for diversity.
        """
        rng = np.random.default_rng(seed)
        uncovered = np.ones(num_tsubs, dtype=np.uint8)
        selected = []
        remaining = np.ones(num_kblocks, dtype=bool)

        while uncovered.any() and time.monotonic() < deadline - 30:
            # Compute gain for each remaining block: number of newly covered t-subs
            # gain_j = sum over i where uncovered[i]==1 of A[i,j]
            gains = (A_dense[:, remaining].T @ uncovered).astype(np.float64)
            if noise_scale > 0:
                gains += rng.uniform(0, noise_scale, size=gains.shape)
            best_local = int(np.argmax(gains))
            best_gain = gains[best_local]
            if best_gain <= 0:
                break
            best_global = np.where(remaining)[0][best_local]
            selected.append(best_global)
            # Update uncovered
            uncovered[A_dense[:, best_global] > 0] = 0
            remaining[best_global] = False

        return selected

    # Local search: try to swap a block for another that covers same or better
    def local_search(block_indices, max_iters=200):
        """
        Try to reduce block count via swaps and drops.
        """
        block_indices = list(block_indices)
        n = len(block_indices)
        if n == 0:
            return block_indices

        # Precompute coverage counts
        counts = np.zeros(num_tsubs, dtype=np.int32)
        for bi in block_indices:
            for ti in block_tsub_indices[bi]:
                counts[ti] += 1

        improved = True
        iters = 0
        while improved and iters < max_iters and time.monotonic() < deadline - 25:
            improved = False
            iters += 1

            # Try dropping each block
            for idx in range(len(block_indices)):
                bi = block_indices[idx]
                # Check if any t-sub is only covered by this block
                can_drop = True
                for ti in block_tsub_indices[bi]:
                    if counts[ti] == 1:
                        can_drop = False
                        break
                if can_drop:
                    # Drop it
                    for ti in block_tsub_indices[bi]:
                        counts[ti] -= 1
                    block_indices.pop(idx)
                    improved = True
                    break

            if improved:
                continue

            # Try swapping: remove block i, add block j that covers the now-uncovered t-subs
            for i in range(len(block_indices)):
                bi = block_indices[i]
                # t-subs that would become uncovered if we drop bi
                critical = [ti for ti in block_tsub_indices[bi] if counts[ti] == 1]
                if not critical:
                    continue
                # Find a block that covers all critical t-subs and isn't already selected
                critical_set = set(critical)
                for j in range(num_kblocks):
                    if j in block_indices:
                        continue
                    # Check if block j covers all critical
                    covers = True
                    for ti in critical_set:
                        if A_dense[ti, j] == 0:
                            covers = False
                            break
                    if covers:
                        # Perform swap
                        for ti in block_tsub_indices[bi]:
                            counts[ti] -= 1
                        for ti in block_tsub_indices[j]:
                            counts[ti] += 1
                        block_indices[i] = j
                        improved = True
                        break
                if improved:
                    break

        return block_indices

    # Run multiple greedy restarts
    best_blocks = None
    best_count = float('inf')
    num_restarts = 5
    restart_time_budget = 10.0  # seconds per restart for greedy

    for r in range(num_restarts):
        if time.monotonic() > deadline - 40:
            break
        seed = 42 + r * 7
        noise = 0.0 if r == 0 else 0.5  # first is deterministic, rest randomized
        t0 = time.monotonic()
        sel = greedy_random(noise_scale=noise, seed=seed)
        if not sel:
            continue
        sel = prune(sel)
        if is_complete(sel):
            # Local search
            sel_ls = local_search(sel, max_iters=100)
            if is_complete(sel_ls) and len(sel_ls) < len(sel):
                sel = sel_ls
            if len(sel) < best_count:
                best_count = len(sel)
                best_blocks = sel
                sys.stderr.write(f"Restart {r}: {best_count} blocks\n")

    if best_blocks is None:
        # Fallback: pure greedy
        sel = greedy_random(noise_scale=0.0, seed=0)
        sel = prune(sel)
        best_blocks = sel
        best_count = len(sel)

    # Optional ILP polish if time permits
    remaining_time = deadline - time.monotonic() - 5
    if remaining_time > 15 and best_count > 130:
        try:
            # Only include blocks that are "useful" (cover at least one t-sub not fully covered by others)
            # For simplicity, use all blocks but limit time
            c = np.ones(num_kblocks)
            constraint = LinearConstraint(A, lb=1, ub=np.inf)
            bounds = Bounds(0, 1)
            integrality = np.ones(num_kblocks)
            options = {'time_limit': int(min(remaining_time, 30)), 'presolve': True}
            result = milp(c, constraints=[constraint], integrality=integrality,
                         bounds=bounds, options=options)
            if result.x is not None:
                x = result.x
                selected_indices = [i for i in range(num_kblocks) if x[i] > 0.5]
                if selected_indices and is_complete(selected_indices):
                    if len(selected_indices) < best_count:
                        best_blocks = selected_indices
                        best_count = len(selected_indices)
                        sys.stderr.write(f"ILP: {best_count} blocks\n")
        except Exception:
            pass

    if not best_blocks:
        return []

    return [list(all_kblocks[i]) for i in best_blocks]
