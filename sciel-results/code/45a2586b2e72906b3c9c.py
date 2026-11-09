import itertools
import random
import time
import math
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using a probabilistic random
    hypergraph model to jump near optimal size, followed by targeted
    greedy repair and local search refinement.
    """
    deadline = time.time() + 115.0  # Safety margin from 120s limit

    # Precompute t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute for each t-subset, the list of k-subsets that contain it
    tsub_to_ksubs = {}
    for ts in all_tsubs:
        ts_set = set(ts)
        remaining = [p for p in range(v) if p not in ts_set]
        ksubs = []
        for extra in itertools.combinations(remaining, k - t):
            block = tuple(sorted(ts + tuple(extra)))
            ksubs.append(block)
        tsub_to_ksubs[ts] = ksubs

    # Precompute for each k-subset, the list of t-subset indices it covers
    ksub_to_tsub_indices = {}

    def get_ksub_indices(block):
        block = tuple(block)
        if block not in ksub_to_tsub_indices:
            idxs = [tsub_to_idx[comb] for comb in itertools.combinations(block, t)]
            ksub_to_tsub_indices[block] = idxs
        return ksub_to_tsub_indices[block]

    def validate_covering(blocks):
        """Check if blocks form a valid covering."""
        if not blocks:
            return n_tsubs == 0
        covered = np.zeros(n_tsubs, dtype=bool)
        for b in blocks:
            idxs = get_ksub_indices(tuple(b))
            covered[idxs] = True
        return np.all(covered)

    def remove_redundant(blocks, max_iter=10):
        """Remove redundant blocks iteratively."""
        if not blocks:
            return blocks

        for _ in range(max_iter):
            if time.time() > deadline - 2.0:
                break

            # Count coverage of each t-subset
            cover_count = np.zeros(n_tsubs, dtype=np.int32)
            block_indices = []
            for b in blocks:
                idxs = get_ksub_indices(tuple(b))
                block_indices.append(idxs)
                cover_count[idxs] += 1

            valid_blocks = []
            removed_any = False

            for i, b in enumerate(blocks):
                if time.time() > deadline - 2.0:
                    return blocks

                idxs = block_indices[i]
                if np.all(cover_count[idxs] > 1):
                    # Redundant: remove it
                    cover_count[idxs] -= 1
                    removed_any = True
                else:
                    valid_blocks.append(b)

            if not removed_any:
                break
            blocks = valid_blocks

        return blocks

    def greedy_repair(blocks, uncovered_mask, rng, max_blocks=None):
        """
        Greedily add blocks to cover remaining uncovered t-subsets.
        """
        if max_blocks is None:
            max_blocks = 10000

        current_blocks = list(blocks)
        uncovered = uncovered_mask.copy()

        added = 0
        while np.any(uncovered) and added < max_blocks:
            if time.time() > deadline - 5.0:
                break

            # Pick a random uncovered t-subset
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            start_idx = rng.choice(uncovered_indices)
            S = all_tsubs[start_idx]

            # Get precomputed k-subsets containing S
            candidates = list(tsub_to_ksubs[S])
            rng.shuffle(candidates)

            # Evaluate gain for each candidate
            best_block = None
            best_gain = -1

            for B in candidates:
                idxs = get_ksub_indices(B)
                gain = np.sum(uncovered[idxs])

                if gain > best_gain:
                    best_gain = gain
                    best_block = B

            if best_block is None:
                break

            current_blocks.append(best_block)
            added += 1

            # Update uncovered mask
            idxs = get_ksub_indices(best_block)
            uncovered[idxs] = False

        return current_blocks

    def local_search(blocks, rng, time_limit=10.0):
        """
        Local search: try to remove blocks and replace with better ones.
        """
        if not blocks:
            return blocks

        best = blocks[:]
        best_len = len(best)
        start_time = time.time()

        for _ in range(200):
            if time.time() > time_limit:
                break

            # Compute coverage
            cover_count = np.zeros(n_tsubs, dtype=np.int32)
            block_idx_map = []
            for b in best:
                idxs = get_ksub_indices(tuple(b))
                block_idx_map.append(idxs)
                cover_count[idxs] += 1

            # Find blocks that can be removed (all their t-subsets are covered by others)
            removable = []
            for i, idxs in enumerate(block_idx_map):
                if np.all(cover_count[idxs] > 1):
                    removable.append(i)

            if not removable:
                break

            # Try removing a random removable block
            remove_idx = rng.choice(removable)
            new_blocks = best[:remove_idx] + best[remove_idx+1:]

            # Check if still valid
            if len(new_blocks) < best_len:
                # Verify validity
                covered = np.zeros(n_tsubs, dtype=bool)
                for b in new_blocks:
                    idxs = get_ksub_indices(tuple(b))
                    covered[idxs] = True
                if np.all(covered):
                    best = new_blocks
                    best_len = len(best)
                    continue

            # If not valid or same length, try a swap
            # Find a block to replace
            replace_candidates = [i for i in range(len(best)) if i != remove_idx]
            if replace_candidates:
                replace_idx = rng.choice(replace_candidates)
                old_block = best[replace_idx]

                # Find a better replacement
                # Look at t-subsets covered by old_block that are under-covered
                old_idxs = block_idx_map[replace_idx]
                under_covered = [idx for idx in old_idxs if cover_count[idx] == 1]

                if under_covered:
                    target_idx = rng.choice(under_covered)
                    target_tsub = all_tsubs[target_idx]

                    candidates = list(tsub_to_ksubs[target_tsub])
                    rng.shuffle(candidates)

                    best_new = None
                    best_gain = -1
                    for B in candidates:
                        if B == tuple(old_block):
                            continue
                        idxs = get_ksub_indices(B)
                        gain = np.sum(cover_count[idxs] == 1)
                        if gain > best_gain:
                            best_gain = gain
                            best_new = B

                    if best_new is not None and best_gain > 0:
                        trial = best[:]
                        trial[replace_idx] = best_new
                        # Validate
                        covered = np.zeros(n_tsubs, dtype=bool)
                        for b in trial:
                            idxs = get_ksub_indices(tuple(b))
                            covered[idxs] = True
                        if np.all(covered) and len(trial) < best_len:
                            best = trial
                            best_len = len(best)

        return best

    # ====================================================================
    # PHASE 1: Probabilistic random hypergraph construction
    # ====================================================================
    # Use Poisson approximation to estimate optimal number of blocks
    # Expected uncovered t-subsets after m random blocks:
    # E[uncovered] = C(v,t) * (1 - C(k,t)/C(v,t))^m
    # Solve for m such that E[uncovered] < 1:
    # m > log(1/C(v,t)) / log(1 - C(k,t)/C(v,t))

    # Calculate theoretical lower bound on m
    prob_cover_single = math.comb(k, t) / math.comb(v, t)
    if prob_cover_single > 0:
        m_estimate = math.ceil(math.log(1.0 / n_tsubs) / math.log(1.0 - prob_cover_single))
    else:
        m_estimate = 300

    # Add some margin to ensure good coverage
    m_candidates = [
        max(264, m_estimate),           # Near theoretical optimum
        max(270, m_estimate + 10),      # Slightly above
        max(280, m_estimate + 20),      # More margin
        max(290, m_estimate + 30),      # Even more margin
        max(300, m_estimate + 40),      # Conservative
    ]

    best_blocks = None
    best_len = float('inf')

    # Try multiple random constructions with different m values
    for m_target in m_candidates:
        if time.time() > deadline - 30.0:
            break

        rng = random.Random(m_target * 1000 + 42)

        # Sample m_target random blocks
        random_blocks = []
        seen = set()
        attempts = 0
        max_attempts = m_target * 10

        while len(random_blocks) < m_target and attempts < max_attempts:
            # Sample a random k-subset
            block = tuple(sorted(rng.sample(range(v), k)))
            if block not in seen:
                seen.add(block)
                random_blocks.append(block)
            attempts += 1

        if not random_blocks:
            continue

        # Check coverage
        covered = np.zeros(n_tsubs, dtype=bool)
        for b in random_blocks:
            idxs = get_ksub_indices(tuple(b))
            covered[idxs] = True

        uncovered_count = np.sum(~covered)

        # If fully covered, we have a valid covering
        if uncovered_count == 0:
            # Try to reduce
            reduced = remove_redundant(random_blocks)
            if len(reduced) < best_len:
                best_len = len(reduced)
                best_blocks = reduced[:]
        else:
            # Repair: greedily add blocks to cover remaining uncovered t-subsets
            uncovered_mask = ~covered
            repaired = greedy_repair(random_blocks, uncovered_mask, rng, max_blocks=50)

            if validate_covering(repaired):
                reduced = remove_redundant(repaired)
                if len(reduced) < best_len:
                    best_len = len(reduced)
                    best_blocks = reduced[:]

    # ====================================================================
    # PHASE 2: Multi-start greedy (fallback and refinement)
    # ====================================================================
    if best_blocks is None or len(best_blocks) > 350:
        # Use greedy if random approach didn't work well
        for seed in range(10):
            if time.time() > deadline - 20.0:
                break

            rng = random.Random(seed)
            uncovered = np.ones(n_tsubs, dtype=bool)
            blocks = []

            while np.any(uncovered):
                if time.time() > deadline - 10.0:
                    break

                # Pick a random uncovered t-subset
                uncovered_indices = np.where(uncovered)[0]
                if len(uncovered_indices) == 0:
                    break
                start_idx = rng.choice(uncovered_indices)
                S = all_tsubs[start_idx]

                # Get precomputed k-subsets containing S
                candidates = list(tsub_to_ksubs[S])
                rng.shuffle(candidates)

                # Evaluate gain for each candidate
                best_block = None
                best_gain = -1

                for B in candidates:
                    idxs = get_ksub_indices(B)
                    gain = np.sum(uncovered[idxs])

                    if gain > best_gain:
                        best_gain = gain
                        best_block = B

                if best_block is None:
                    break

                blocks.append(best_block)

                # Update uncovered mask
                idxs = get_ksub_indices(best_block)
                uncovered[idxs] = False

            if blocks:
                reduced = remove_redundant(blocks)
                if len(reduced) < best_len:
                    best_len = len(reduced)
                    best_blocks = reduced[:]

    # ====================================================================
    # PHASE 3: Local search refinement
    # ====================================================================
    if best_blocks is not None and time.time() < deadline - 5.0:
        rng = random.Random(12345)
        refined = local_search(best_blocks, rng, time_limit=min(15.0, deadline - time.time() - 3.0))
        if len(refined) < len(best_blocks):
            best_blocks = refined

    # Final validation and return
    if best_blocks is None:
        # Emergency fallback: pure greedy
        rng = random.Random(0)
        uncovered = np.ones(n_tsubs, dtype=bool)
        blocks = []
        while np.any(uncovered):
            if time.time() > deadline - 1.0:
                break
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            start_idx = rng.choice(uncovered_indices)
            S = all_tsubs[start_idx]
            candidates = list(tsub_to_ksubs[S])
            rng.shuffle(candidates)
            best_block = None
            best_gain = -1
            for B in candidates:
                idxs = get_ksub_indices(B)
                gain = np.sum(uncovered[idxs])
                if gain > best_gain:
                    best_gain = gain
                    best_block = B
            if best_block is None:
                break
            blocks.append(best_block)
            idxs = get_ksub_indices(best_block)
            uncovered[idxs] = False
        best_blocks = blocks

    # Ensure validity
    if not validate_covering(best_blocks):
        # This shouldn't happen, but just in case
        rng = random.Random(999)
        uncovered = np.zeros(n_tsubs, dtype=bool)
        for b in best_blocks:
            idxs = get_ksub_indices(tuple(b))
            uncovered[idxs] = True
        uncovered = ~uncovered
        repaired = greedy_repair(best_blocks, uncovered, rng, max_blocks=100)
        best_blocks = repaired

    return best_blocks
