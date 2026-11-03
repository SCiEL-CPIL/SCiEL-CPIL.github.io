# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct(v, k, t):
    """
    Covering design C(v,k,t): find a small set of k-subsets (blocks) of {0,...,v-1}
    such that every t-subset is contained in at least one block.
    Uses a greedy construction followed by local search (removal + block swaps).
    """
    deadline = time.time() + 115  # safety margin below the 120s wall-clock limit
    rng = random.Random(42)

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # Helper: coverage count for a set of blocks (as frozensets of block indices)
    def build_coverage(block_list):
        """Return a list of coverage counts for each t-subset, and the set of blocks used."""
        cov = [0] * n_tsubs
        for b in block_list:
            b_set = frozenset(b)
            for ts in itertools.combinations(b, t):
                idx = tsub_index[ts]
                cov[idx] += 1
        return cov

    # Greedy construction: always add the block covering the most uncovered t-subsets
    def greedy_construct():
        cov = [0] * n_tsubs
        blocks = []
        # Precompute for each k-subset the list of t-subset indices it covers
        # This is expensive for large v, so we compute on the fly but cache per block
        # For v=19, k=8: C(19,8)=75582 blocks, C(8,5)=56 t-subsets each -> manageable
        # But we only need to consider blocks that improve coverage, so use greedy
        # Start with uncovered = all t-subsets
        uncovered_count = n_tsubs
        # We'll track which t-subsets are uncovered
        # Use a list of 0/1 for uncovered
        uncov = [1] * n_tsubs

        while uncovered_count > 0:
            best_block = None
            best_gain = 0
            # Greedy: pick block that covers most uncovered t-subsets
            # To speed up, we can sample or use a smarter heuristic
            # For now, iterate over candidate blocks
            # Optimization: only consider blocks that include at least one point from an uncovered t-subset
            # Simpler: just iterate all possible k-subsets (75582 for v=19,k=8)
            for b in itertools.combinations(range(v), k):
                gain = 0
                for ts in itertools.combinations(b, t):
                    idx = tsub_index[ts]
                    if uncov[idx] == 1:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
                # Early stop if we found a block covering all remaining
                if best_gain >= uncovered_count:
                    break
            if best_gain == 0:
                break
            blocks.append(list(best_block))
            # Update coverage
            for ts in itertools.combinations(best_block, t):
                idx = tsub_index[ts]
                if uncov[idx] == 1:
                    uncov[idx] = 0
                    uncovered_count -= 1
            if time.time() > deadline - 30:
                break
        return blocks

    # Local search: try removing blocks and swapping blocks to reduce count
    def local_search(blocks):
        # Convert to set of frozensets for fast lookup
        block_set = set(frozenset(b) for b in blocks)
        # Build coverage
        cov = [0] * n_tsubs
        for b in block_set:
            for ts in itertools.combinations(b, t):
                idx = tsub_index[ts]
                cov[idx] += 1

        best_size = len(block_set)
        best_blocks = set(block_set)
        improved = True
        iters = 0
        max_iters = 5000

        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try removing each block
            for b in list(block_set):
                # Check if removing b leaves all t-subsets covered
                can_remove = True
                for ts in itertools.combinations(b, t):
                    idx = tsub_index[ts]
                    if cov[idx] == 1:
                        can_remove = False
                        break
                if can_remove:
                    block_set.remove(b)
                    for ts in itertools.combinations(b, t):
                        idx = tsub_index[ts]
                        cov[idx] -= 1
                    best_size = len(block_set)
                    best_blocks = set(block_set)
                    improved = True
                    if time.time() > deadline - 5:
                        return [list(b) for b in best_blocks]

            # Try swapping: remove one block, add another
            if not improved:
                # Try block swaps
                # For each block in the set, try replacing it with a new block
                # that covers at least the t-subsets not covered by other blocks
                # This is expensive, so do it carefully
                current_blocks = list(block_set)
                for i, old_b in enumerate(current_blocks):
                    # Find t-subsets only covered by old_b
                    only_ts = []
                    for ts in itertools.combinations(old_b, t):
                        idx = tsub_index[ts]
                        if cov[idx] == 1:
                            only_ts.append(idx)
                    if not only_ts:
                        # Can remove old_b
                        block_set.remove(old_b)
                        for ts in itertools.combinations(old_b, t):
                            idx = tsub_index[ts]
                            cov[idx] -= 1
                        best_size = len(block_set)
                        best_blocks = set(block_set)
                        improved = True
                        break
                    else:
                        # Need to find a new block that covers all only_ts
                        # And doesn't add too much redundancy
                        # Try some candidate blocks
                        # Candidate: blocks that contain the intersection of points
                        # This is complex; skip for now to keep time
                        pass
                if time.time() > deadline - 5:
                    break

        return [list(b) for b in best_blocks]

    # Phase 1: Greedy construction
    blocks = greedy_construct()

    # Phase 2: Local search
    blocks = local_search(blocks)

    # Check if we have time for a second round
    if time.time() < deadline - 30:
        # Try a different greedy order: start from random points
        rng.shuffle(all_tsubs)
        # Re-greedy with different tie-breaking
        uncov = [1] * n_tsubs
        uncovered_count = n_tsubs
        blocks2 = []
        while uncovered_count > 0:
            best_block = None
            best_gain = 0
            for b in itertools.combinations(range(v), k):
                gain = 0
                for ts in itertools.combinations(b, t):
                    idx = tsub_index[ts]
                    if uncov[idx] == 1:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
                if best_gain >= uncovered_count:
                    break
            if best_gain == 0:
                break
            blocks2.append(list(best_block))
            for ts in itertools.combinations(best_block, t):
                idx = tsub_index[ts]
                if uncov[idx] == 1:
                    uncov[idx] = 0
                    uncovered_count -= 1
            if time.time() > deadline - 15:
                break
        blocks2 = local_search(blocks2)
        if len(blocks2) < len(blocks):
            blocks = blocks2

    return blocks
# EVOLVE-BLOCK-END
