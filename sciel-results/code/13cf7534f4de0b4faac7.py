"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time


def construct(v, k, t):
    # Deadline: evaluator kills after 150s; keep safety margin.
    deadline = time.time() + 125.0

    # Precompute all t-subsets and map to indices.
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_to_idx = {comb: i for i, comb in enumerate(all_t_subsets)}

    # uncovered: set of indices of uncovered t-subsets
    uncovered = set(range(n_t))
    blocks = []

    def block_t_indices(block_tuple):
        """Return set of t-subset indices covered by block_tuple."""
        return {t_to_idx[c] for c in itertools.combinations(block_tuple, t)}

    def greedy_extend(block, uncov):
        """Extend block to size k, minimizing uncovered t-subsets."""
        block = list(block)
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # t-subsets of block+[x] that include x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    idx = t_to_idx[tuple(sorted(rest + (x,)))]
                    if idx in uncov:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
        return tuple(block)

    # Phase 1: Greedy set cover
    while uncovered:
        if time.time() > deadline - 20:
            break
        # Pick the uncovered t-subset with highest "difficulty" (fewest candidate blocks)
        # For simplicity, pick the first uncovered one
        first_idx = min(uncovered)
        initial = all_t_subsets[first_idx]
        block = greedy_extend(initial, uncovered)
        block = tuple(sorted(block))
        blocks.append(block)
        uncovered -= block_t_indices(block)

    # Phase 2: Finish if time allows
    if uncovered and time.time() < deadline - 5:
        while uncovered:
            if time.time() > deadline - 5:
                break
            first_idx = min(uncovered)
            initial = all_t_subsets[first_idx]
            block = greedy_extend(initial, uncovered)
            block = tuple(sorted(block))
            blocks.append(block)
            uncovered -= block_t_indices(block)

    # Phase 3: Local search - remove redundant blocks
    if time.time() < deadline - 5:
        # Build coverage count for each t-subset
        cov_count = [0] * n_t
        for blk in blocks:
            for idx in block_t_indices(blk):
                cov_count[idx] += 1

        # Try to remove blocks that are redundant
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                # Check if removing block i leaves all its t-subsets covered
                redundant = True
                for idx in block_t_indices(blocks[i]):
                    if cov_count[idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for idx in block_t_indices(blocks[i]):
                        cov_count[idx] -= 1
                    blocks.pop(i)
                    improved = True

    return [list(b) for b in blocks]
# EVOLVE-BLOCK-END
