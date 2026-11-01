import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer
    
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
    for i, ks in enumerate(k_subsets):
        mask = 0
        # For each k-subset, find all t-subsets contained in it
        # Since k=8, t=5, we can iterate over t-combinations of the k-subset
        for ts in itertools.combinations(ks, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
        masks[i] = mask
        
    # Helper to count bits
    # Python 3.10+ has int.bit_count(), but let's be safe or use bin(x).count('1')
    # Given the size (11628 bits), bin().count() is fast enough, but bit_count is better.
    # We assume modern Python.
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    best_blocks = None
    best_count = float('inf')
    
    # Run multiple random restarts
    # Each restart: Greedy selection with random tie-breaking or randomization of weights
    num_restarts = 10000 # We will limit by time
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        uncovered = (1 << num_t_subsets) - 1
        current_blocks = []
        used_blocks = set()
        
        while uncovered:
            if time.time() > deadline:
                break
                
            best_gain = -1
            best_block_idx = -1
            
            # To add randomness, we can shuffle the indices or pick a random subset of candidates
            # But iterating all 75k blocks is fast.
            # To break ties randomly, we can collect all max-gain blocks and pick one randomly.
            # Or, simply add a small random noise to the gain? No, that changes the objective.
            # Better: If multiple blocks have the same max gain, pick one randomly.
            
            max_gain = 0
            candidates = []
            
            # Optimization: If we have many restarts, iterating all blocks might be slow.
            # 75582 blocks * 1000 restarts = 75M operations. In Python, this might be too slow.
            # Let's check speed. 75M simple integer ops in Python ~ 5-10s.
            # So 1000 restarts might take 5-10s just for the inner loop.
            # We need to be careful.
            
            # Let's limit the number of blocks we consider or optimize the loop.
            # Actually, we can just run fewer restarts if it's slow.
            
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                gain = popcount(masks[i] & uncovered)
                if gain > max_gain:
                    max_gain = gain
                    candidates = [i]
                elif gain == max_gain and gain > 0:
                    candidates.append(i)
            
            if max_gain == 0:
                # Should not happen if the problem is solvable and we haven't used all blocks
                # But if we did, break
                break
                
            # Pick a random candidate to add diversity
            chosen_idx = random.choice(candidates)
            current_blocks.append(k_subsets[chosen_idx])
            used_blocks.add(chosen_idx)
            uncovered &= ~masks[chosen_idx]
            
        if uncovered == 0 and len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = [list(b) for b in current_blocks]
            
        # If we found a good solution, we might want to stop early?
        # The known best is 335. If we get close, we can increase effort.
        # For now, just keep running until deadline.

    if best_blocks is None:
        # Fallback to simple greedy if randomization failed (unlikely)
        # Or return an empty list? No, must be valid.
        # The greedy will always find a solution if one exists.
        # If we timed out in the middle of a restart, we might have no complete solution.
        # We need to ensure we return a valid solution.
        # Let's run one deterministic greedy without random tie-breaking to ensure validity.
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
