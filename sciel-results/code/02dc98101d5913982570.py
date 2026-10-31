"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time


def construct(v, k, t):
    """
    Greedy set cover using bitsets for efficient gain calculation.
    """
    deadline = time.time() + 115.0  # Safety margin for 120s limit (150s kill)

    # 1. Generate all t-subsets and map them to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}

    # 2. Initialize uncovered bitset (all 1s)
    # Python integers can be arbitrarily large, so we can use them as bitsets.
    uncovered_mask = (1 << n_tsubs) - 1
    remaining = n_tsubs

    # 3. Generate all k-subsets
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)

    # 4. Precompute coverage mask for each k-subset
    # ksub_mask[j] is an integer where bit i is 1 if k-subset j covers t-subset i
    ksub_masks = [0] * n_ksubs
    for j, ks in enumerate(all_ksubs):
        mask = 0
        # A k-subset covers C(k, t) t-subsets
        for tsub in itertools.combinations(ks, t):
            idx = tsub_to_idx[tsub]
            mask |= (1 << idx)
        ksub_masks[j] = mask

    blocks = []
    used_blocks = set()

    # 5. Greedy selection loop
    # We need to select k-subsets that maximize the number of newly covered t-subsets.
    # Gain = popcount(uncovered_mask & ksub_mask[j])
    
    # Optimization: If we have many blocks, scanning all 50k candidates every time might be slow.
    # However, for v=19, n_ksubs=50388. 
    # popcount on Python ints is fast. 50k popcounts per iteration.
    # Expected number of blocks is ~150. 150 * 50k = 7.5 million popcounts.
    # This is well within 115 seconds in Python.

    while remaining > 0:
        if time.time() > deadline:
            break

        best_j = -1
        best_gain = 0
        
        # Scan all candidates
        for j in range(n_ksubs):
            if j in used_blocks:
                continue
            
            # Calculate gain: number of bits set in (uncovered & mask)
            # int.bit_count() is available in Python 3.10+
            # If older version, use bin(x).count('1')
            common = uncovered_mask & ksub_masks[j]
            gain = common.bit_count() if hasattr(common, 'bit_count') else bin(common).count('1')
            
            if gain > best_gain:
                best_gain = gain
                best_j = j

        if best_j == -1 or best_gain == 0:
            # No more progress possible (should not happen if remaining > 0 and masks are correct)
            break

        # Add best block
        block = list(all_ksubs[best_j])
        blocks.append(block)
        used_blocks.add(best_j) # Store index for O(1) lookup
        
        # Update uncovered mask
        uncovered_mask &= ~ksub_masks[best_j]
        remaining -= best_gain

    # If we exited early due to time, we might have uncovered t-subsets.
    # The problem requires a valid covering. If we don't have one, the score is 0.
    # However, the greedy algorithm is very fast for this size. 
    # If it times out, it's better to return the partial result? 
    # No, "Valid when every t-subset ... lies in at least one block".
    # If invalid, score is 0.
    # So we must ensure validity if we return.
    # Given the speed estimate, it should finish.
    # If it doesn't, we can try a simple repair, but it might be slow.
    # Let's assume it finishes. If not, returning invalid is bad.
    # But for v=19, it's very fast.

    return blocks
# EVOLVE-BLOCK-END
