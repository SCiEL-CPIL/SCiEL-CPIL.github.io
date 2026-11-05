import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0
    
    # Pre-compute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    T = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Pre-compute all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_blocks)
    
    # Map each block index to list of t-subset indices contained in it
    # block_subs[b] = [s1, s2, ...]
    block_subs = [[] for _ in range(num_blocks)]
    
    # Map each t-subset index to list of block indices containing it
    # subset_blocks[s] = [b1, b2, ...]
    subset_blocks = [[] for _ in range(T)]
    
    for b_idx, block in enumerate(all_k_blocks):
        for s in itertools.combinations(block, t):
            s_idx = t_subset_to_idx[s]
            block_subs[b_idx].append(s_idx)
            subset_blocks[s_idx].append(b_idx)

    def greedy_construct(top_k=1):
        uncovered = set(range(T))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset
            s_idx = random.choice(list(uncovered))
            candidates = subset_blocks[s_idx]
            
            gains = []
            for b in candidates:
                gain = 0
                for sub_idx in block_subs[b]:
                    if sub_idx in uncovered:
                        gain += 1
                gains.append((gain, b))
                
            # Sort by gain descending
            gains.sort(key=lambda x: x[0], reverse=True)
            
            # Pick best or random from top_k
            if len(gains) > top_k:
                # Pick random from top k
                choice_idx = random.randint(0, top_k - 1)
                best_block = gains[choice_idx][1]
            else:
                best_block = gains[0][1]
                
            blocks.append(best_block)
            
            # Update uncovered
            for sub_idx in block_subs[best_block]:
                if sub_idx in uncovered:
                    uncovered.remove(sub_idx)
                        
        return blocks

    def greedy_remove(blocks):
        if not blocks:
            return blocks
            
        changed = True
        while changed:
            changed = False
            # Count coverage
            cover_counts = [0] * T
            for b in blocks:
                for s_idx in block_subs[b]:
                    cover_counts[s_idx] += 1
                    
            new_blocks = []
            for b in blocks:
                # Check if b is redundant
                redundant = True
                for s_idx in block_subs[b]:
                    if cover_counts[s_idx] == 1:
                        redundant = False
                        break
                        
                if redundant:
                    # Remove b
                    for s_idx in block_subs[b]:
                        cover_counts[s_idx] -= 1
                    changed = True
                else:
                    new_blocks.append(b)
                    
            blocks = new_blocks
            
        return blocks

    best_blocks = None
    best_size = float('inf')
    
    # Run multiple restarts with different randomness
    # top_k controls exploration: 1 is greedy, >1 is randomized
    attempts = 0
    while time.time() - start_time < time_limit - 5.0:
        # Vary top_k to explore different neighborhoods
        # Start with greedy, then increase randomness
        top_k = 1 + (attempts % 5)
        if attempts > 10:
            top_k = 5 + (attempts % 10)
            
        blocks = greedy_construct(top_k)
        blocks = greedy_remove(blocks)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            # If we get very good, we can stop or continue
            # print(f"New best: {best_size}")
            
        attempts += 1
        
    # Convert block indices back to lists of points
    if best_blocks is None:
        # Fallback: just greedy
        blocks = greedy_construct(1)
        blocks = greedy_remove(blocks)
        best_blocks = blocks

    result = [list(all_k_blocks[b]) for b in best_blocks]
    return result
