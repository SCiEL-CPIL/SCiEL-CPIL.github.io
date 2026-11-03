import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute for each point, the list of subset indices containing it
    # This is not strictly necessary for the greedy gain if we use pair_to_subsets for t=3
    # but good for general t
    point_to_subsets = [[] for _ in range(v)]
    for i, s in enumerate(all_t_subsets):
        for p in s:
            point_to_subsets[p].append(i)
            
    # Precompute for each pair of points (if t=3), the list of subset indices containing both
    pair_to_subsets = {}
    if t == 3:
        for i, s in enumerate(all_t_subsets):
            # s is (a, b, c)
            pairs = [(s[0], s[1]), (s[0], s[2]), (s[1], s[2])]
            for p in pairs:
                if p[0] > p[1]:
                    p = (p[1], p[0])
                if p not in pair_to_subsets:
                    pair_to_subsets[p] = []
                pair_to_subsets[p].append(i)

    def greedy_construct(seed):
        random.seed(seed)
        uncovered = set(range(num_t_subsets))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset to start the block
            # Using list() is O(N), but N=2024 is small
            start_subset_idx = random.choice(list(uncovered))
            start_subset = all_t_subsets[start_subset_idx]
            
            block = list(start_subset)
            
            # Build the block
            while len(block) < k:
                best_gain = -1
                best_point = -1
                candidates = [p for p in range(v) if p not in block]
                
                for p in candidates:
                    gain = 0
                    if t == 3:
                        for i in range(len(block)):
                            for j in range(i+1, len(block)):
                                pair = (block[i], block[j])
                                if pair[0] > pair[1]:
                                    pair = (pair[1], pair[0])
                                if pair in pair_to_subsets:
                                    for sub_idx in pair_to_subsets[pair]:
                                        if sub_idx in uncovered:
                                            gain += 1
                    else:
                        # General case: iterate over all t-1 subsets of block
                        for rest in itertools.combinations(block, t - 1):
                            candidate_subset = tuple(sorted(rest + (p,)))
                            if candidate_subset in subset_to_idx:
                                sub_idx = subset_to_idx[candidate_subset]
                                if sub_idx in uncovered:
                                    gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                    elif gain == best_gain and best_point != -1:
                        # Random tie break
                        if random.random() < 0.5:
                            best_point = p
                            
                if best_point == -1:
                    break
                block.append(best_point)
                
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            if t == 3:
                for i in range(len(block)):
                    for j in range(i+1, len(block)):
                        pair = (block[i], block[j])
                        if pair[0] > pair[1]:
                            pair = (pair[1], pair[0])
                        if pair in pair_to_subsets:
                            for sub_idx in pair_to_subsets[pair]:
                                uncovered.discard(sub_idx)
            else:
                for sub in itertools.combinations(block, t):
                    idx = subset_to_idx[sub]
                    uncovered.discard(idx)
            
        return blocks

    def reduce_covering(blocks):
        # Compute coverage counts
        coverage = [0] * num_t_subsets
        for block in blocks:
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                coverage[idx] += 1
                
        # Iteratively remove redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                block = blocks[i]
                is_redundant = True
                for sub in itertools.combinations(block, t):
                    idx = subset_to_idx[sub]
                    if coverage[idx] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove block i
                    for sub in itertools.combinations(block, t):
                        idx = subset_to_idx[sub]
                        coverage[idx] -= 1
                    blocks.pop(i)
                    changed = True
                    break # Restart loop as indices changed
                    
        return blocks

    best_blocks = None
    best_size = float('inf')
    seed = 0
    
    while time.time() < deadline:
        blocks = greedy_construct(seed)
        blocks = reduce_covering(blocks)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            
        seed += 1

    if best_blocks is None:
        # Fallback
        return [list(range(k))]
        
    return best_blocks
