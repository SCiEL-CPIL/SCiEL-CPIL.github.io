"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
from collections import Counter


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    def time_ok():
        return time.time() < deadline

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsubsets)

    # Count how many k-blocks contain each t-subset (for constraint-based selection)
    # This is expensive, so we'll use a simpler heuristic: 
    # pick the uncovered t-subset with the smallest "flexibility"
    # For efficiency, we'll just use lexicographic order but with a twist:
    # prioritize t-subsets that share elements with fewer other uncovered t-subsets

    blocks = []
    
    while uncovered and time_ok():
        # Select the "most constrained" uncovered t-subset to start a new block
        # Heuristic: pick the one with the fewest points that appear in many other uncovered t-subsets
        # For speed, just pick the lexicographically smallest for now, but we can improve
        
        # Better heuristic: pick the t-subset that maximizes the number of uncovered t-subsets
        # it can be extended to cover (i.e., has the most potential)
        # Actually, let's pick the one that is "hardest" - appears in fewest possible blocks
        # Since all t-subsets can appear in C(v-t, k-t) blocks, they're all equally constrained
        # So we just pick the first one
        
        seed_ts = min(uncovered)
        block_set = set(seed_ts)
        block = list(seed_ts)
        
        while len(block) < k and time_ok():
            # Find the point that maximizes coverage of uncovered t-subsets
            best_x = None
            best_gain = -1
            best_rarity_gain = -1
            
            # Precompute which uncovered t-subsets are "close" to the current block
            # A t-subset is covered by adding x if it contains t-1 elements from block and x
            candidates = [x for x in range(v) if x not in block_set]
            
            for x in candidates:
                # Count how many uncovered t-subsets would be covered by adding x
                gain = 0
                rarity_weighted_gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    if ts in uncovered:
                        gain += 1
                        # Weight by how rare this t-subset is (how few blocks can cover it)
                        # All t-subsets have the same number of containing blocks: C(v-t, k-t)
                        # So no rarity weighting helps here
                        rarity_weighted_gain += 1
                
                if gain > best_gain or (gain == best_gain and rarity_weighted_gain > best_rarity_gain):
                    best_gain = gain
                    best_rarity_gain = rarity_weighted_gain
                    best_x = x
            
            if best_x is None or best_gain <= 0:
                # No point adds coverage, just pick any remaining point
                remaining = [x for x in range(v) if x not in block_set]
                if remaining:
                    best_x = remaining[0]
                else:
                    break
            
            block.append(best_x)
            block_set.add(best_x)
        
        if len(block) == k:
            block.sort()
            blocks.append(tuple(block))
            # Remove all t-subsets covered by this block
            covered = set(itertools.combinations(block, t))
            uncovered -= covered
    
    return [list(b) for b in blocks]
# EVOLVE-BLOCK-END
