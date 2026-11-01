"""Structural change: global bitset greedy with precomputed masks + local improvement."""
# EVOLVE-BLOCK-START
import itertools
import time


def construct(v, k, t):
    deadline = time.time() + 140  # safety margin before 150s kill

    # Precompute all k-subsets and their t-subset bitmasks
    # Represent each t-subset as an integer index via combinatorial number system
    # Map t-combination to index
    t_combos = list(itertools.combinations(range(v), t))
    t_combo_to_idx = {c: i for i, c in enumerate(t_combos)}
    n_t = len(t_combos)

    # Build bitmask for each k-block: which t-subsets it covers
    # Use Python integers as bitsets (n_t can be up to C(19,4)=3876, fits in int)
    k_combos = list(itertools.combinations(range(v), k))
    n_k = len(k_combos)

    # Precompute mask for each k-block
    block_masks = []
    for blk in k_combos:
        mask = 0
        for ts in itertools.combinations(blk, t):
            mask |= (1 << t_combo_to_idx[ts])
        block_masks.append(mask)

    # Also precompute: for each t-subset, which k-blocks contain it (for fast gain calc)
    # t_to_blocks[i] = list of block indices that cover t-subset i
    t_to_blocks = [[] for _ in range(n_t)]
    for bi, blk in enumerate(k_combos):
        for ts in itertools.combinations(blk, t):
            t_to_blocks[t_combo_to_idx[ts]].append(bi)

    # Greedy: repeatedly pick the block with maximum uncovered t-subsets
    uncovered = (1 << n_t) - 1  # all t-subsets initially uncovered
    selected = []
    selected_set = set()

    while uncovered and time.time() < deadline:
        best_block = -1
        best_gain = 0
        for bi, mask in enumerate(block_masks):
            if bi in selected_set:
                continue
            gain = (uncovered & mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block = bi
            elif gain == best_gain and gain > 0:
                # tie-break: prefer blocks with fewer total t-subsets (more focused)
                pass
        if best_block == -1:
            break
        selected.append(best_block)
        selected_set.add(best_block)
        uncovered &= ~block_masks[best_block]

    # Convert selected indices back to blocks
    blocks = [list(k_combos[i]) for i in selected]

    # Local improvement: try to swap out a block for a different one that reduces count
    # This is a simple 1-swap heuristic: for each selected block, see if removing it
    # and adding a different block still covers everything.
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for si in range(len(selected)):
                # Try removing selected[si] and finding a replacement
                trial_uncovered = 0
                for sj in range(len(selected)):
                    if sj != si:
                        trial_uncovered &= ~block_masks[selected[sj]]
                # trial_uncovered now has bits set for t-subsets NOT covered by others
                # We need a block that covers all of trial_uncovered
                # Actually: uncovered_by_others = bits set in trial_uncovered
                # We need a block whose mask covers all those bits
                need = trial_uncovered
                if need == 0:
                    # Removing this block doesn't uncover anything -> remove it
                    selected.pop(si)
                    improved = True
                    break
                # Find a single block that covers all of 'need'
                found = False
                for bi, mask in enumerate(block_masks):
                    if bi in selected_set and bi != selected[si]:
                        continue
                    if (need & ~mask) == 0:  # mask covers all of need
                        # Replace selected[si] with bi
                        selected_set.discard(selected[si])
                        selected[si] = bi
                        selected_set.add(bi)
                        improved = True
                        found = True
                        break
                if found:
                    break

    blocks = [list(k_combos[i]) for i in selected]
    return blocks
# EVOLVE-BLOCK-END
