import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Precomputation
    # Map t-subsets to indices
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_index = {sub: i for i, sub in enumerate(t_subs)}
    num_t_subs = len(t_subs)
    
    # Map k-subsets to indices and their t-subsets
    k_subs = list(itertools.combinations(range(v), k))
    block_t_subs = []
    # For each k-subset, find which t-subsets it contains
    # To speed up, we can iterate k_subs and check t_subs? No, too slow.
    # Better: For each k-subset, generate its t-subsets and map to index.
    for k_sub in k_subs:
        t_list = []
        for t_sub in itertools.combinations(k_sub, t):
            t_list.append(t_sub_index[t_sub])
        block_t_subs.append(t_list)
        
    num_blocks = len(k_subs)
    
    # Map t-subset to blocks containing it
    blocks_in_t_sub = [[] for _ in range(num_t_subs)]
    for i, t_list in enumerate(block_t_subs):
        for t_idx in t_list:
            blocks_in_t_sub[t_idx].append(i)
            
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple greedy attempts
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        uncovered = set(range(num_t_subs))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset
            t_idx = random.choice(list(uncovered))
            candidates = blocks_in_t_sub[t_idx]
            
            best_gain = -1
            best_block = -1
            
            for b_idx in candidates:
                gain = 0
                for t_i in block_t_subs[b_idx]:
                    if t_i in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    
            if best_block == -1:
                break # Should not happen if uncovered is not empty and mappings are correct
                
            blocks.append(best_block)
            # Update uncovered
            for t_i in block_t_subs[best_block]:
                if t_i in uncovered:
                    uncovered.remove(t_i)
                    
        # Local Search: Remove redundant blocks
        # Maintain coverage counts
        coverage = [0] * num_t_subs
        for b_idx in blocks:
            for t_i in block_t_subs[b_idx]:
                coverage[t_i] += 1
                
        changed = True
        while changed:
            changed = False
            # Shuffle to avoid order bias
            random.shuffle(blocks)
            for i in range(len(blocks) - 1, -1, -1):
                b_idx = blocks[i]
                # Check if any t-subset in this block has coverage 1
                is_redundant = True
                for t_i in block_t_subs[b_idx]:
                    if coverage[t_i] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove block
                    for t_i in block_t_subs[b_idx]:
                        coverage[t_i] -= 1
                    blocks.pop(i)
                    changed = True
                    if time.time() > deadline:
                        break
                        
        if time.time() > deadline:
            break
            
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
            
    # Convert block indices back to lists of points
    if best_blocks is None:
        # Fallback to simple greedy if something went wrong
        return []
        
    result = [list(k_subs[b_idx]) for b_idx in best_blocks]
    return result
