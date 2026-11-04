import itertools
import random
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin

    # Precompute all triples and their bitmasks
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_idx = {tri: i for i, tri in enumerate(all_triples)}

    # Precompute for each k-subset its triple bitmask (as a numpy boolean array)
    # We'll use this lazily - only for blocks we actually consider
    def block_triple_mask(block):
        """Return a numpy bool array of length num_triples, True where covered."""
        mask = np.zeros(num_triples, dtype=bool)
        for tri in itertools.combinations(block, t):
            mask[triple_idx[tri]] = True
        return mask

    # Build an array: for each triple, its bitmask of which points it contains
    triple_points = np.zeros((num_triples, v), dtype=np.int8)
    for i, tri in enumerate(all_triples):
        for p in tri:
            triple_points[i, p] = 1

    # For gain computation: given a partial block (set of points), and a candidate point x,
    # gain = number of uncovered triples that would be covered by adding x
    # A triple is covered by adding x if: triple contains x AND all other elements of triple are in block
    # We precompute for each point x: which triples contain x
    triples_containing = [[] for _ in range(v)]
    for i, tri in enumerate(all_triples):
        for p in tri:
            triples_containing[p].append(i)

    def fast_greedy(seed=None):
        """Deterministic-ish greedy with optional randomization for tie-breaking."""
        rng = random.Random(seed)
        uncovered = np.ones(num_triples, dtype=bool)
        blocks = []

        while uncovered.any():
            # Pick an uncovered triple to start
            unc_idx = np.where(uncovered)[0]
            if len(unc_idx) == 0:
                break
            start_i = unc_idx[rng.randint(0, len(unc_idx) - 1)]
            start_tri = all_triples[start_i]
            block = list(start_tri)
            block_set = set(block)

            while len(block) < k:
                # Compute gains for each candidate point
                best_gain = 0
                best_candidates = []
                for x in range(v):
                    if x in block_set:
                        continue
                    # Count uncovered triples that contain x and whose other t-1 points are in block
                    gain = 0
                    for ti in triples_containing[x]:
                        if not uncovered[ti]:
                            continue
                        tri = all_triples[ti]
                        # Check all other points are in block
                        ok = True
                        for p in tri:
                            if p != x and p not in block_set:
                                ok = False
                                break
                        if ok:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain and gain > 0:
                        best_candidates.append(x)

                if best_candidates:
                    x = best_candidates[rng.randint(0, len(best_candidates) - 1)]
                else:
                    available = [x for x in range(v) if x not in block_set]
                    x = available[rng.randint(0, len(available) - 1)]
                block.append(x)
                block_set.add(x)

            block.sort()
            blocks.append(block)

            # Update uncovered
            for tri in itertools.combinations(block, t):
                uncovered[triple_idx[tri]] = False

        return blocks

    def build_counts(blocks):
        counts = np.zeros(num_triples, dtype=np.int32)
        for b in blocks:
            for tri in itertools.combinations(b, t):
                counts[triple_idx[tri]] += 1
        return counts

    def remove_redundant(blocks, counts):
        """Remove blocks whose all triples are covered by >= 2 blocks."""
        changed = True
        while changed:
            changed = False
            n = len(blocks)
            # Shuffle order
            order = list(range(n))
            random.shuffle(order)
            for i in order:
                if i >= len(blocks):
                    continue
                b = blocks[i]
                # Check if all triples of b have count > 1
                redundant = True
                for tri in itertools.combinations(b, t):
                    if counts[triple_idx[tri]] <= 1:
                        redundant = False
                        break
                if redundant:
                    for tri in itertools.combinations(b, t):
                        counts[triple_idx[tri]] -= 1
                    blocks.pop(i)
                    changed = True
        return blocks, counts

    def local_optimize(blocks, max_time=8.0):
        """Iterative redundancy removal + block replacement."""
        counts = build_counts(blocks)
        blocks, counts = remove_redundant(blocks, counts)

        # Try to replace blocks
        while time.time() < start_time + max_time + (start_time - start_time) + 8.0:
            if time.time() > deadline - 2:
                break
            counts = build_counts(blocks)
            # Find blocks with no critical triples
            improvable = False
            for i, b in enumerate(blocks):
                has_critical = False
                for tri in itertools.combinations(b, t):
                    if counts[triple_idx[tri]] == 1:
                        has_critical = True
                        break
                if not has_critical:
                    # Try to find a better replacement
                    old_triple_set = set(itertools.combinations(b, t))
                    best_gain = 0
                    best_block = None
                    for _ in range(30):
                        new_block = list(b)
                        # Replace 1-2 random elements
                        n_swap = random.randint(1, 2)
                        for _s in range(n_swap):
                            pos = random.randint(0, k - 1)
                            old_val = new_block[pos]
                            new_val = random.randint(0, v - 1)
                            if new_val not in new_block:
                                new_block[pos] = new_val
                        new_block.sort()
                        if set(new_block) == set(b):
                            continue
                        new_triple_set = set(itertools.combinations(new_block, t))
                        # Must cover all critical triples of old block
                        critical = set(tri for tri in old_triple_set if counts[triple_idx[tri]] == 1)
                        if not critical.issubset(new_triple_set):
                            continue
                        new_only = new_triple_set - old_triple_set
                        old_only = old_triple_set - new_triple_set
                        gain = len(new_only) - len(old_only)
                        if gain > best_gain:
                            best_gain = gain
                            best_block = new_block
                    if best_block is not None and best_gain > 0:
                        blocks[i] = best_block
                        improvable = True
            if not improvable:
                break
            blocks, counts = remove_redundant(blocks, build_counts(blocks))

        return blocks

    def ilp_refine(blocks, time_limit=10.0):
        """Use column generation + MILP to refine the covering."""
        if time.time() > deadline - 12:
            return blocks

        # Build initial column pool from current blocks + some random columns
        counts = build_counts(blocks)
        # Only keep columns that cover at least one currently under-covered or sparsely covered triple
        pool = []
        seen = set()
        for b in blocks:
            key = tuple(b)
            if key not in seen:
                seen.add(key)
                pool.append(b)

        # Add random columns to expand the pool
        for _ in range(200):
            if time.time() > deadline - 12:
                break
            b = sorted(random.sample(range(v), k))
            key = tuple(b)
            if key not in seen:
                seen.add(key)
                pool.append(b)

        # Build the ILP: minimize sum x_j subject to A @ x >= 1
        # A[i, j] = 1 if column j covers triple i
        # To keep size manageable, only include triples that are not yet over-covered
        # Actually, include all triples but this may be large.
        # num_triples = 2024 for C(24,3), pool size ~ 200-276
        # A is 2024 x ~276 - manageable

        n_cols = len(pool)
        if n_cols < len(blocks) + 1:
            return blocks

        # Build A matrix
        A = np.zeros((num_triples, n_cols), dtype=np.int8)
        for j, b in enumerate(pool):
            for tri in itertools.combinations(b, t):
                A[triple_idx[tri], j] = 1

        # Objective: minimize sum x_j
        c = np.ones(n_cols)

        # Constraints: A @ x >= 1  =>  -A @ x <= -1
        cons = LinearConstraint(-A, lb=-np.ones(num_triples), ub=0)
        bounds = Bounds(0, 1)

        try:
            res = milp(c=c, constraints=cons, integrality=np.ones(n_cols), bounds=bounds)
            if res.status == 0 and res.x is not None:
                selected = [pool[j] for j in range(n_cols) if res.x[j] > 0.5]
                if selected and len(selected) < len(blocks):
                    # Verify validity
                    sel_counts = build_counts(selected)
                    if (sel_counts >= 1).all():
                        return selected
        except Exception:
            pass

        return blocks

    # Main search loop
    best_blocks = None

    # First: a deterministic greedy (seed=0)
    blocks = fast_greedy(seed=0)
    blocks = local_optimize(blocks, max_time=6.0)
    best_blocks = blocks

    if time.time() > deadline - 20:
        return best_blocks

    # ILP refinement on the best
    refined = ilp_refine(best_blocks, time_limit=12.0)
    if refined is not None and len(refined) < len(best_blocks):
        best_blocks = refined

    if time.time() > deadline - 5:
        return best_blocks

    # Multiple randomized restarts
    for seed in range(1, 50):
        if time.time() > deadline - 5:
            break
        blocks = fast_greedy(seed=seed)
        blocks = local_optimize(blocks, max_time=4.0)
        if best_blocks is None or len(blocks) < len(best_blocks):
            best_blocks = blocks

        if len(best_blocks) <= 69:
            break

        # Periodic ILP refinement
        if seed % 5 == 0 and time.time() < deadline - 12:
            refined = ilp_refine(best_blocks, time_limit=10.0)
            if refined is not None and len(refined) < len(best_blocks):
                best_blocks = refined

    # Final ILP pass if time permits
    if time.time() < deadline - 12:
        refined = ilp_refine(best_blocks, time_limit=10.0)
        if refined is not None and len(refined) < len(best_blocks):
            best_blocks = refined

    return best_blocks
