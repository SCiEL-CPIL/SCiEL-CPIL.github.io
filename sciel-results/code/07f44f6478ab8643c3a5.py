import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 145  # Leave 5s margin
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute for each point, the indices of t-subsets containing it
    point_to_subsets = [[] for _ in range(v)]
    for idx, subset in enumerate(all_t_subsets):
        for p in subset:
            point_to_subsets[p].append(idx)
            
    def greedy_construct(seed):
        random.seed(seed)
        counts = [0] * n_t_subsets
        uncovered = set(range(n_t_subsets))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset
            target_idx = random.choice(list(uncovered))
            target_subset = all_t_subsets[target_idx]
            
            block = list(target_subset)
            
            while len(block) < k:
                best_gain = -1
                best_point = None
                block_set = set(block)
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    gain = 0
                    # Check subsets containing x
                    for s_idx in point_to_subsets[x]:
                        if s_idx in uncovered:
                            # Check if other t-1 points are in block
                            s = all_t_subsets[s_idx]
                            # Since x is in s, check if s \ {x} is subset of block
                            # s is a tuple of length t.
                            # We can check: all(p in block_set for p in s if p != x)
                            if all(p in block_set for p in s if p != x):
                                gain += 1
                                
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                    elif gain == best_gain and gain > 0:
                        # Tie break randomly
                        if random.random() < 0.5:
                            best_point = x
                            
                if best_point is None:
                    # Should not happen if uncovered is not empty and k < v
                    break
                block.append(best_point)
                
            block.sort()
            blocks.append(block)
            
            # Update counts
            for s in itertools.combinations(block, t):
                idx = subset_to_idx[s]
                counts[idx] += 1
                if counts[idx] == 1:
                    uncovered.discard(idx)
                    
        return blocks, counts

    def local_search(blocks, counts, seed):
        random.seed(seed)
        best_blocks = blocks
        best_counts = counts
        best_len = len(blocks)
        
        # Copy state
        cur_blocks = list(blocks)
        cur_counts = list(counts)
        
        while time.time() - start_time < time_limit:
            improved = False
            
            # 1. Reduction Phase
            while True:
                any_removed = False
                for i in range(len(cur_blocks)):
                    b = cur_blocks[i]
                    # Check if all t-subsets in b have count > 1
                    redundant = True
                    for s in itertools.combinations(b, t):
                        if cur_counts[subset_to_idx[s]] == 1:
                            redundant = False
                            break
                    if redundant:
                        # Remove block i
                        for s in itertools.combinations(b, t):
                            cur_counts[subset_to_idx[s]] -= 1
                        cur_blocks.pop(i)
                        any_removed = True
                        improved = True
                        break # Restart loop as indices changed
                if not any_removed:
                    break
            
            if len(cur_blocks) < best_len:
                best_len = len(cur_blocks)
                best_blocks = list(cur_blocks)
                best_counts = list(cur_counts)
                
            # 2. Swap Phase
            # Try to replace a block to enable more reductions
            # We iterate through blocks and try to find a better replacement
            for i in range(len(cur_blocks)):
                if time.time() - start_time > time_limit:
                    break
                b_i = cur_blocks[i]
                # Find critical subsets in b_i
                criticals = []
                for s in itertools.combinations(b_i, t):
                    if cur_counts[subset_to_idx[s]] == 1:
                        criticals.append(s)
                
                if not criticals:
                    continue # Should have been removed in reduction
                
                # Construct a candidate replacement block that covers criticals
                # and tries to cover other critical subsets
                new_block = []
                must_cover = set(criticals)
                # Also consider other critical subsets from other blocks to maximize potential reduction
                # But primarily we MUST cover 'must_cover'
                
                # To speed up, we can just try to cover 'must_cover' first
                # Greedy selection of points
                while must_cover and len(new_block) < k:
                    best_gain = -1
                    best_point = None
                    new_block_set = set(new_block)
                    
                    for x in range(v):
                        if x in new_block_set:
                            continue
                        gain = 0
                        for s_idx in point_to_subsets[x]:
                            if all_t_subsets[s_idx] in must_cover:
                                s = all_t_subsets[s_idx]
                                if all(p in new_block_set for p in s if p != x):
                                    gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_point = x
                        elif gain == best_gain and gain > 0 and random.random() < 0.5:
                            best_point = x
                            
                    if best_point is None:
                        break
                    new_block.append(best_point)
                    # Update must_cover
                    new_block_set.add(best_point)
                    must_cover = {s for s in must_cover if not (best_point in s and all(p in new_block_set for p in s if p != best_point))}
                    
                if len(new_block) < k or must_cover:
                    continue # Couldn't cover criticals with k points? Or failed.
                
                new_block.sort()
                
                # If new_block is same as b_i, skip
                if new_block == b_i:
                    continue
                    
                # Check if replacement leads to reduction
                # 1. Remove b_i from counts
                for s in itertools.combinations(b_i, t):
                    cur_counts[subset_to_idx[s]] -= 1
                # 2. Add new_block to counts
                for s in itertools.combinations(new_block, t):
                    cur_counts[subset_to_idx[s]] += 1
                # 3. Replace in list
                cur_blocks[i] = new_block
                
                # 4. Run reduction
                reduction_improved = False
                while True:
                    any_removed = False
                    for j in range(len(cur_blocks)):
                        b = cur_blocks[j]
                        redundant = True
                        for s in itertools.combinations(b, t):
                            if cur_counts[subset_to_idx[s]] == 1:
                                redundant = False
                                break
                        if redundant:
                            for s in itertools.combinations(b, t):
                                cur_counts[subset_to_idx[s]] -= 1
                            cur_blocks.pop(j)
                            any_removed = True
                            reduction_improved = True
                            break
                    if not any_removed:
                        break
                    
                if reduction_improved:
                    improved = True
                    if len(cur_blocks) < best_len:
                        best_len = len(cur_blocks)
                        best_blocks = list(cur_blocks)
                        best_counts = list(cur_counts)
                else:
                    # Revert
                    for s in itertools.combinations(new_block, t):
                        cur_counts[subset_to_idx[s]] -= 1
                    for s in itertools.combinations(b_i, t):
                        cur_counts[subset_to_idx[s]] += 1
                    cur_blocks[i] = b_i
                    
            if not improved:
                # Stuck, maybe randomize?
                # For now, break or try different seed
                # Let's just break to save time, greedy + one pass of LS is often enough for small v
                break
                
        return best_blocks, best_counts

    # Run multiple iterations with different seeds to escape local optima
    best_blocks = None
    best_counts = None
    best_len = float('inf')
    
    num_iterations = 10
    for i in range(num_iterations):
        if time.time() - start_time > time_limit:
            break
        g_blocks, g_counts = greedy_construct(i)
        l_blocks, l_counts = local_search(g_blocks, g_counts, i)
        if len(l_blocks) < best_len:
            best_len = len(l_blocks)
            best_blocks = l_blocks
            best_counts = l_counts
            
    if best_blocks is None:
        # Fallback to simple greedy
        g_blocks, g_counts = greedy_construct(0)
        best_blocks = g_blocks
        
    return best_blocks
