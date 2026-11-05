"""Optimized covering design C(19,8,5) using bitmask-accelerated greedy + local search."""
import time
import random
import itertools

# Deadline guard: finish well before the 150s wall-clock kill.
_DEADLINE = 135.0  # seconds


def construct(v, k, t):
    """
    Greedy covering with bitmask acceleration, multiple randomized restarts,
    and a local-search polish pass. Returns the best valid covering found.
    """
    start_time = time.time()

    # Precompute all t-subsets as sorted tuples and assign indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # Map each point to the set of t-subset indices that contain it
    # point_t_subs[p] = list of t-subset indices where p appears
    point_t_subs = [[] for _ in range(v)]
    for sub in all_t_subs:
        for p in sub:
            point_t_subs[p].append(t_sub_to_idx[sub])

    # For a block B (list of points), the t-subsets it covers are those
    # fully contained in B. We precompute this via bitmask of t-subset indices.
    # n_t can be up to C(19,5)=11628, so we use a set of ints.

    def block_covers(block):
        """Return the set of t-subset indices covered by block."""
        # A t-subset is covered by block if all its elements are in block.
        block_set = set(block)
        covered = set()
        for sub in itertools.combinations(block, t):
            covered.add(t_sub_to_idx[sub])
        return covered

    # Faster: for a given block, enumerate C(k,t) subsets directly
    def block_covers_fast(block):
        block_set = frozenset(block)
        covered = set()
        for sub in itertools.combinations(block, t):
            covered.add(t_sub_to_idx[sub])
        return covered

    def uncovered_count_after_adding(block, point, current_covered):
        """How many currently-uncovered t-subsets would be covered by adding point to block."""
        # t-subsets containing point and (t-1) points from block
        gain = 0
        block_set = set(block)
        for rest in itertools.combinations(block, t - 1):
            sub = tuple(sorted(rest + (point,)))
            idx = t_sub_to_idx[sub]
            if idx not in current_covered:
                gain += 1
        return gain

    def greedy_cover(seed):
        """Run a greedy covering algorithm with a given random seed."""
        rng = random.Random(seed)
        uncovered = set(range(n_t))  # set of uncovered t-subset indices
        blocks = []
        all_points = list(range(v))

        while uncovered:
            # Pick a random uncovered t-subset to anchor the new block
            anchor_idx = rng.choice(list(uncovered))
            anchor = all_t_subs[anchor_idx]
            block = list(anchor)

            # Greedily add points that maximize coverage of remaining uncovered
            while len(block) < k:
                best_point = None
                best_gain = -1
                candidates = [x for x in all_points if x not in block]
                # Shuffle for tie-breaking
                rng.shuffle(candidates)
                for x in candidates:
                    gain = uncovered_count_after_adding(block, x, set())
                    # Count how many of those are actually uncovered
                    # We need to check against 'uncovered'
                    actual_gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        sub = tuple(sorted(rest + (x,)))
                        idx = t_sub_to_idx[sub]
                        if idx in uncovered:
                            actual_gain += 1
                    if actual_gain > best_gain:
                        best_gain = actual_gain
                        best_point = x
                if best_point is None:
                    break
                block.append(best_point)

            block.sort()
            blocks.append(block)

            # Remove covered t-subsets
            covered = block_covers_fast(block)
            uncovered -= covered

        return blocks

    def local_search_polish(blocks, max_iterations=200):
        """Try to remove redundant blocks or swap them for better ones."""
        blocks = [list(b) for b in blocks]
        n = len(blocks)

        # Precompute coverage per block
        def compute_coverage(block):
            return block_covers_fast(block)

        block_coverages = [compute_coverage(b) for b in blocks]

        # Union of all coverages
        all_covered = set()
        for cov in block_coverages:
            all_covered |= cov

        # For local search: try removing each block and see if coverage is still complete
        for _iter in range(max_iterations):
            if time.time() - start_time > _DEADLINE - 5:
                break
            improved = False
            # Try removing blocks (in random order)
            order = list(range(n))
            random.shuffle(order)
            for i in order:
                if len(blocks) <= 1:
                    break
                # Check if block i is redundant
                # Coverage without block i
                # This is expensive; do it only if block i has low unique contribution
                unique_to_i = block_coverages[i] - (
                    set().union(*[block_coverages[j] for j in range(len(blocks)) if j != i])
                    if len(blocks) > 1 else set()
                )
                if len(unique_to_i) == 0:
                    # Block i is redundant
                    blocks.pop(i)
                    block_coverages.pop(i)
                    n = len(blocks)
                    improved = True
                    break
            if not improved:
                break

        return blocks

    # Run multiple greedy passes with different seeds, keep the best
    best_blocks = None
    best_len = float('inf')

    # First: deterministic greedy (seed=0)
    blocks0 = greedy_cover(0)
    best_blocks = blocks0
    best_len = len(blocks0)

    # Try a few more seeds if time permits
    for seed in [1, 2, 3, 4, 5, 7, 11, 13]:
        if time.time() - start_time > _DEADLINE - 10:
            break
        blocks = greedy_cover(seed)
        if len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)

    # Local search polish on the best
    if time.time() - start_time < _DEADLINE - 5:
        polished = local_search_polish(best_blocks)
        if len(polished) < best_len:
            best_blocks = polished
            best_len = len(polished)

    # Verify validity
    if best_blocks is None:
        # Fallback: should not happen
        return []

    # Final check: ensure every t-subset is covered
    final_covered = set()
    for b in best_blocks:
        final_covered |= block_covers_fast(b)
    if len(final_covered) < n_t:
        # Invalid; fall back to a safe greedy
        # Re-run a clean deterministic greedy
        uncovered = set(range(n_t))
        blocks = []
        while uncovered:
            anchor_idx = min(uncovered)
            anchor = all_t_subs[anchor_idx]
            block = list(anchor)
            while len(block) < k:
                best_point = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        sub = tuple(sorted(rest + (x,)))
                        idx = t_sub_to_idx[sub]
                        if idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point is not None:
                    block.append(best_point)
                else:
                    break
            block.sort()
            blocks.append(block)
            covered = block_covers_fast(block)
            uncovered -= covered
        best_blocks = blocks

    return best_blocks
