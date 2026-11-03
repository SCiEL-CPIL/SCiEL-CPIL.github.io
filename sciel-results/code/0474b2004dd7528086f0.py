import itertools
import time
import random
import sys

# Precomputation for specific v, k, t is not possible since construct(v,k,t) is the interface.
# However, the task says "the evaluator passes this task's parameters v, k and t".
# We assume v=19, k=8, t=5 based on the prompt, but the code must be generic or handle the specific case efficiently.
# Given the 120s limit, we can check if v, k, t match the known hard case and use optimized logic, 
# or just run the general bitset greedy which is fast enough for v=19.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Total number of t-subsets
    total_t_subsets = 0
    for i in range(v - t + 1):
        # Calculate C(v, t) iteratively or use math.comb
        pass
    
    # Use math.comb for clarity and speed
    try:
        import math
        num_t_sets = math.comb(v, t)
        num_k_blocks = math.comb(v, k)
    except ImportError:
        # Fallback for older Python
        def nCr(n, r):
            if r < 0 or r > n:
                return 0
            if r == 0:
                return 1
            r = min(r, n - r)
            num = 1
            den = 1
            for i in range(r):
                num *= n - i
                den *= i + 1
            return num // den
        num_t_sets = nCr(v, t)
        num_k_blocks = nCr(v, k)

    # Map each t-subset to an index 0..num_t_sets-1
    t_subset_to_idx = {}
    idx_to_t_subset = []
    idx = 0
    for comb in itertools.combinations(range(v), t):
        t_subset_to_idx[comb] = idx
        idx_to_t_subset.append(comb)
        idx += 1
        
    # Map each k-block to its index and precompute the bitmask of t-subsets it covers
    # Bitmask is an integer where bit i is 1 if t-subset i is covered by this k-block
    k_blocks = list(itertools.combinations(range(v), k))
    block_masks = [0] * num_k_blocks
    block_indices = {}
    
    for i, block in enumerate(k_blocks):
        block_set = set(block)
        mask = 0
        # Iterate through all t-subsets of this block
        for t_comb in itertools.combinations(block, t):
            t_idx = t_subset_to_idx[t_comb]
            mask |= (1 << t_idx)
        block_masks[i] = mask
        block_indices[block] = i
        
    # Initial full mask of all t-subsets
    full_mask = (1 << num_t_sets) - 1
    
    best_blocks = None
    best_count = float('inf')
    
    # Number of restarts depends on time available
    # Estimate time per restart:
    # Steps ~ num_t_sets / avg_gain. Avg gain is roughly (k/t)^t * C(v-t, k-t)? No, simpler.
    # Let's just run as many restarts as we can before deadline.
    
    restarts = 0
    while time.time() < deadline:
        uncovered_mask = full_mask
        current_blocks = []
        
        # To add randomness, we can shuffle the blocks or add noise to the gain.
        # Shuffling blocks changes the order of ties.
        # For better diversity, we can randomly select from top 5 candidates.
        
        while uncovered_mask != 0:
            max_gain = 0
            best_block_idx = -1
            # We can optimize this by only checking blocks that have potential to cover new subsets
            # But checking all is O(num_k_blocks) per step.
            # num_k_blocks ~ 75k. Steps ~ 300. 22M ops. Fast.
            
            # Randomized selection: 
            # Find the max gain. If there are multiple with max gain, pick one randomly.
            # Or pick from top K.
            
            # For speed, we find the single max. To add randomness, we can perturb the gain slightly?
            # Better: Shuffle the list of block indices once per restart? No, that changes the search space order.
            # Let's just pick the best. The randomness comes from... wait, greedy is deterministic.
            # To get different results, we need to change the input or the tie-breaking.
            # If we break ties randomly, we get different paths.
            
            candidates = []
            for b_idx in range(num_k_blocks):
                gain = bin(block_masks[b_idx] & uncovered_mask).count('1')
                if gain > max_gain:
                    max_gain = gain
                    candidates = [b_idx]
                elif gain == max_gain and gain > 0:
                    candidates.append(b_idx)
                    
            if not candidates:
                # Should not happen if covering is possible, but break to avoid infinite loop
                break
                
            # Randomly select from candidates with max gain
            selected_idx = random.choice(candidates)
            current_blocks.append(k_blocks[selected_idx])
            uncovered_mask &= ~block_masks[selected_idx]
            
        # End while uncovered
        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks
            # Optional: stop early if we hit the theoretical lower bound?
            # Lower bound for C(19,8,5) is 264.
            # If best_count == 264, we can't do better.
            if best_count <= 264:
                break
                
        restarts += 1
        
    if best_blocks is None:
        # Fallback to simple greedy if something went wrong
        return []
        
    return best_blocks
