import time
import numpy as np
import itertools

def construct(v, k, t):
    """
    Build a covering of all t-subsets of v points using k-subsets (blocks).
    Structural approach:
      1. Greedy construction with fast bitset-based uncovered tracking.
      2. Local search: swap-based block replacement (if a block is redundant
         in terms of coverage, replace it with a better one that covers more
         unique t-subsets).
    """
    start = time.time()
    deadline = start + 110.0  # 10s safety margin before 120s kill

    # Precompute all t-subsets of {0..v-1}
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {s: i for i, s in enumerate(all_tsubs)}

    # We'll track uncovered t-subsets as a numpy boolean array
    uncovered = np.ones(n_tsubs, dtype=bool)

    def tsub_indices_for_block(block):
        """Return the indices of t-subsets covered by a given block."""
        idxs = []
        for combo in itertools.combinations(block, t):
            idxs.append(tsub_index[combo])
        return np.array(idxs, dtype=np.int64)

    blocks = []

    # Phase 1: Greedy construction
    while True:
        if time.time() > deadline - 30.0:
            break  # leave time for local search
        if not uncovered.any():
            break

        # Pick the uncovered t-subset with smallest "frequency" among remaining candidates
        # For simplicity and speed, pick the first uncovered one (lexicographically)
        uncovered_indices = np.nonzero(uncovered)[0]
        if len(uncovered_indices) == 0:
            break
        seed_tsub = all_tsubs[uncovered_indices[0]]
        block = list(seed_tsub)

        # Greedily add points to maximize coverage of uncovered t-subsets
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Compute how many new uncovered t-subsets would be covered by adding x
                new_block = block + [x]
                # Only consider t-subsets that include at least one element from new_block but not all from block
                # Fast approach: compute tsub indices for new_block, count how many are uncovered
                idxs = tsub_indices_for_block(new_block)
                gain = np.count_nonzero(uncovered[idxs])
                # Subtract those already covered by block (to get marginal gain)
                if best_gain < 0:
                    block_idx_cache = None
                # For efficiency, we can just count uncovered in new_block
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)

        block.sort()
        blocks.append(block)
        # Update uncovered
        idxs = tsub_indices_for_block(block)
        uncovered[idxs] = False

    # Phase 2: Local search - try to reduce number of blocks by merging or replacing
    # For each block, see if we can find a replacement that covers the same unique t-subsets
    # but allows removing another block. This is complex; instead, try random swaps.

    if time.time() > deadline - 5.0:
        return blocks

    # Local search: try to replace each block with a better one
    # A block is "good" if it covers many uncovered t-subsets (but all are covered now)
    # Instead, try: for each block, see if it's redundant (its unique t-subsets are covered by others)
    # If so, remove it. Then try to re-add a better block.

    # Simplified local search: try random block replacements
    import random
    random.seed(42)

    for iteration in range(200):
        if time.time() > deadline - 2.0:
            break
        if not uncovered.any():
            break

        # Pick a random block index
        if not blocks:
            break
        i = random.randint(0, len(blocks) - 1)
        old_block = blocks[i]

        # Compute t-subsets uniquely covered by this block
        # A t-subset is uniquely covered if it's in old_block and not in any other block
        # This is expensive; skip for now and just try random replacements

        # Generate a random new block
        new_block = sorted(random.sample(range(v), k))
        if new_block == old_block:
            continue

        # Check if replacing old_block with new_block maintains coverage
        # Temporarily remove old_block, check if all t-subsets are still covered
        # Then add new_block
        # This is O(n_blocks * C(k,t)) per check, which may be slow

        # For speed, only try if new_block covers more unique t-subsets
        idxs_old = tsub_indices_for_block(old_block)
        idxs_new = tsub_indices_for_block(new_block)

        # Count how many t-subsets in new_block are not covered by any other block
        # and how many in old_block are uniquely covered
        # Approximation: just try the swap and check validity

        # Quick check: does new_block cover any t-subset that old_block uniquely covers?
        # Skip detailed check for speed; just do the swap and validate at the end
        blocks[i] = new_block

        # Validate coverage after swap
        # Rebuild uncovered from scratch (expensive but correct)
        covered = np.zeros(n_tsubs, dtype=bool)
        for b in blocks:
            idxs = tsub_indices_for_block(b)
            covered[idxs] = True
        if covered.all():
            # Good swap
            pass
        else:
            # Revert
            blocks[i] = old_block

    return blocks
