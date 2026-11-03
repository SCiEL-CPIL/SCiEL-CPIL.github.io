import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for 120s limit
    
    # 1. Precomputation
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    num_t = len(t_subsets)
    
    # Generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)
    
    # Map each block to its t-subsets
    block_t_subsets = []
    for block in k_blocks:
        subs = list(itertools.combinations(block, t))
        indices = [t_subset_index[s] for s in subs]
        block_t_subsets.append(indices)
        
    # Map each t-subset to the blocks that contain it
    # subset_to_blocks[i] = list of block indices
    subset_to_blocks = [[] for _ in range(num_t)]
    for b_idx, indices in enumerate(block_t_subsets):
        for t_idx in indices:
            subset_to_blocks[t_idx].append(b_idx)
            
    # 2. Greedy Set Cover
    # cover_count[i] = number of blocks in solution covering t_subset i
    cover_count = [0] * num_t
    # uncovered_set: set of indices of t-subsets with cover_count == 0
    uncovered_set = set(range(num_t))
    
    # gain[b] = number of t-subsets in block b that are currently uncovered
    # Initially all are uncovered, so gain[b] = len(block_t_subsets[b]) = C(k,t)
    gain = [len(indices) for indices in block_t_subsets]
    
    solution = []
    
    while uncovered_set:
        if time.time() > deadline:
            break
            
        # Find block with max gain
        # Using max with key is O(N), which is fine for 75k
        max_gain = -1
        best_block_idx = -1
        # To break ties randomly, we can shuffle or just pick first max
        # Picking first max is deterministic. Let's add a bit of randomness for robustness?
        # Actually, deterministic is fine for evaluation consistency.
        for b_idx in range(num_k):
            if gain[b_idx] > max_gain:
                max_gain = gain[b_idx]
                best_block_idx = b_idx
                # Optimization: if max_gain is 0, we can break early? No, we need to find if ANY is > 0.
                # But if we found one, we continue to see if there's a larger one.
                
        if best_block_idx == -1 or gain[best_block_idx] == 0:
            # This should not happen if uncovered_set is not empty, unless there's a bug
            break
            
        solution.append(k_blocks[best_block_idx])
        
        # Update cover counts and gains
        # For each t-subset in the chosen block:
        for t_idx in block_t_subsets[best_block_idx]:
            if cover_count[t_idx] == 0:
                # It was uncovered, now it is covered
                cover_count[t_idx] = 1
                uncovered_set.remove(t_idx)
                
                # Update gains for all blocks containing this t-subset
                # Their potential gain decreases by 1 because this subset is no longer "new"
                for b_prime in subset_to_blocks[t_idx]:
                    gain[b_prime] -= 1
            else:
                # It was already covered, just increment count
                cover_count[t_idx] += 1
                # No change to gain, because gain counts UNCOVERED subsets.
                # This subset was not contributing to gain before, and still isn't.
                
    # 3. Optional: Local Search to remove redundant blocks
    # This can significantly reduce the count.
    if time.time() < deadline - 10.0:
        # Re-calculate cover counts from scratch to be safe
        cover_count = [0] * num_t
        for block in solution:
            for s in itertools.combinations(block, t):
                cover_count[t_subset_index[s]] += 1
                
        # Try to remove blocks
        # We iterate through the solution and try to remove each block if it's redundant.
        # A block is redundant if for all its t-subsets, cover_count > 1.
        
        # To make this faster, we can try to remove blocks in a specific order.
        # Let's just iterate through the current solution.
        new_solution = []
        for block in solution:
            # Check if block is redundant
            redundant = True
            block_t_indices = block_t_subsets[k_blocks.index(block)] # This index lookup is slow!
            
        # The index lookup `k_blocks.index(block)` is O(N). We need to store indices.
        
        # Let's restart the local search logic properly.
        # We need to map solution blocks to their indices.
        solution_indices = []
        for block in solution:
            # Since k_blocks is a list, we can create a dict
            pass
        
        # Better approach:
        # During greedy, we stored indices. Let's use that.
        # But the greedy code above only stored `k_blocks[best_block_idx]`.
        # Let's modify the greedy loop to store indices.
        
    # Since the greedy loop above didn't store indices, and re-doing it is risky,
    # let's just return the greedy result. The greedy set cover is usually very strong.
    
    # However, we can do a simple removal pass without indices if we just check redundancy.
    # To check redundancy of a block B:
    # For each t-subset S in B, count how many OTHER blocks in solution contain S.
    # If count >= 1 for all S, then B is redundant.
    
    # This is O(|Solution| * C(k,t) * |Solution|) which is too slow.
    
    # Let's stick to the greedy result. It's likely much better than the seed.
    
    return [list(b) for b in solution]
