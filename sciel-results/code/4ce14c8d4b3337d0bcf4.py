import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before 120s

    # Precomputation
    # 1. Enumerate all t-subsets and assign indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)

    # 2. Enumerate all k-subsets and compute their coverage bitmask
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)

    # masks[i] is an integer bitmask where bit j is 1 if k_subset i covers t_subset j
    masks = [0] * num_k_subsets
    # For each t-subset, store the list of k-subsets that cover it
    t_to_k = defaultdict(list)
    for i, ks in enumerate(k_subsets):
        mask = 0
        for ts in itertools.combinations(ks, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
            t_to_k[idx].append(i)
        masks[i] = mask

    # Helper to count bits
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    best_blocks = None
    best_count = float('inf')

    # Run multiple random restarts with different strategies
    num_restarts = 50000  # We will limit by time

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        uncovered = (1 << num_t_subsets) - 1
        current_blocks = []
        used_blocks = set()
        # Count how many times each k-subset is a candidate for each uncovered t-subset
        # We'll use a different approach: maintain a gain array for each k-subset
        # gain[i] = number of currently uncovered t-subsets covered by k-subset i
        gain = [0] * num_k_subsets
        for i in range(num_k_subsets):
            gain[i] = popcount(masks[i] & uncovered)

        while uncovered:
            if time.time() > deadline:
                break

            # Find max gain among unused blocks
            max_gain = 0
            candidates = []
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                if gain[i] > max_gain:
                    max_gain = gain[i]
                    candidates = [i]
                elif gain[i] == max_gain and gain[i] > 0:
                    candidates.append(i)

            if max_gain == 0:
                break

            # Pick a random candidate to add diversity
            chosen_idx = random.choice(candidates)
            current_blocks.append(k_subsets[chosen_idx])
            used_blocks.add(chosen_idx)
            
            # Update gains for all k-subsets
            new_uncovered = uncovered & ~masks[chosen_idx]
            # The newly covered t-subsets are: uncovered & masks[chosen_idx]
            newly_covered = uncovered & masks[chosen_idx]
            uncovered = new_uncovered
            
            # For each k-subset, subtract the number of newly covered t-subsets it also covers
            # This is expensive, so we do it lazily or use a different structure
            # Actually, let's just recompute gains periodically or use a smarter update
            # For now, recompute gains for all unused blocks
            if len(used_blocks) % 10 == 0:  # Recompute every 10 steps
                for i in range(num_k_subsets):
                    if i not in used_blocks:
                        gain[i] = popcount(masks[i] & uncovered)
            else:
                # Quick update: for each newly covered t-subset, decrement gain of k-subsets that cover it
                # But this is also expensive. Let's just recompute when needed.
                # Actually, the current approach of scanning all k_subsets each step is O(num_k_subsets * num_t_subsets)
                # which is too slow. Let's use a different strategy.
                
                # Alternative: maintain a count of uncovered t-subsets per k-subset
                # and update it when we add a block.
                # When we add block b, for each k-subset i, gain[i] -= |masks[i] & masks[b] & uncovered_old|
                # This is still expensive.
                
                # Let's use the t_to_k structure: for each newly covered t-subset, 
                # decrement the gain of all k-subsets that cover it.
                # But we need to know which k-subsets cover each newly covered t-subset.
                # We have t_to_k for that.
                
                # However, we need to be careful: we should only decrement if the k-subset
                # was not already used.
                
                # Let's iterate over the newly covered t-subsets and update gains
                # This might be faster if the number of newly covered t-subsets is small.
                # But in the beginning, it could be large.
                
                # For now, let's just recompute gains from scratch each step.
                # This is O(num_k_subsets * num_t_subsets) which is too slow for large problems.
                # But for v=19, k=8, t=5, num_t_subsets = C(19,5) = 11628, num_k_subsets = C(19,8) = 75582
                # So this is about 75582 * 11628 / 64 ≈ 13.6 million bit operations per step, which might be okay in Python?
                # Actually, popcount is fast, but the loop is in Python.
                
                # Let's try a hybrid: recompute gains every few steps, and for the intermediate steps,
                # use a lazy update.
                
                # For now, let's just recompute gains each step and see if it's fast enough.
                for i in range(num_k_subsets):
                    if i not in used_blocks:
                        gain[i] = popcount(masks[i] & uncovered)

        if uncovered == 0 and len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = [list(b) for b in current_blocks]

    if best_blocks is None:
        # Fallback to simple greedy if randomization failed (unlikely)
        uncovered = (1 << num_t_subsets) - 1
        current_blocks = []
        used_blocks = set()
        while uncovered:
            max_gain = 0
            best_block_idx = -1
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                gain = popcount(masks[i] & uncovered)
                if gain > max_gain:
                    max_gain = gain
                    best_block_idx = i
            if max_gain == 0:
                break
            current_blocks.append(k_subsets[best_block_idx])
            used_blocks.add(best_block_idx)
            uncovered &= ~masks[best_block_idx]
        best_blocks = [list(b) for b in current_blocks]

    return best_blocks
