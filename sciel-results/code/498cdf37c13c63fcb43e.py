"""Portfolio search: generate varied candidate coverings and keep the best within a strict time budget."""
import time
import random
import itertools

def construct(v, k, t):
    deadline = time.time() + 130.0  # safety margin below 150s kill
    start_time = time.time()
    best_blocks = None
    best_count = float('inf')
    
    # Precompute all t-subsets for fast checking
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered and time.time() < deadline:
            if not uncovered:
                break
            # Choose a random uncovered subset to seed the block
            start_subset = rng.choice(list(uncovered))
            block = list(start_subset)
            block_set = set(block)
            
            # Greedily add points to maximize coverage of uncovered t-subsets
            while len(block) < k:
                best_gain = -1
                best_point = -1
                candidates = [x for x in range(v) if x not in block_set]
                
                if len(block) >= t - 1:
                    block_subsets = set(itertools.combinations(block, t - 1))
                else:
                    block_subsets = set()
                
                for x in candidates:
                    count = 0
                    for sub in block_subsets:
                        new_t_sub = tuple(sorted(sub + (x,)))
                        if new_t_sub in uncovered:
                            count += 1
                    if count > best_gain:
                        best_gain = count
                        best_point = x
                
                if best_point == -1 or best_gain <= 0:
                    best_point = rng.choice(candidates) if candidates else -1
                    if best_point == -1:
                        break
                
                block.append(best_point)
                block_set.add(best_point)
            
            if len(block) == k:
                block.sort()
                blocks.append(block)
                for sub in itertools.combinations(block, t):
                    uncovered.discard(sub)
        
        return blocks

    def local_search_remove(blocks):
        """
        Attempt to remove redundant blocks from a valid covering.
        """
        if not blocks:
            return blocks
        
        cover_count = {}
        for block in blocks:
            for sub in itertools.combinations(block, t):
                cover_count[sub] = cover_count.get(sub, 0) + 1
        
        changed = True
        current_blocks = list(blocks)
        
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(current_blocks)):
                if time.time() >= deadline:
                    break
                block = current_blocks[i]
                is_redundant = True
                for sub in itertools.combinations(block, t):
                    if cover_count.get(sub, 0) <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for sub in itertools.combinations(block, t):
                        cover_count[sub] -= 1
                    current_blocks.pop(i)
                    changed = True
                    break
        
        return current_blocks

    for seed in range(1500):
        if time.time() >= deadline:
            break
        
        blocks = greedy_cover(seed)
        
        if len(blocks) < best_count:
            covered = set()
            for block in blocks:
                for sub in itertools.combinations(block, t):
                    covered.add(sub)
            
            if len(covered) == n_t_subsets:
                optimized_blocks = local_search_remove(blocks)
                
                if len(optimized_blocks) < best_count:
                    best_blocks = optimized_blocks
                    best_count = len(optimized_blocks)
    
    if best_blocks is None:
        uncovered = set(all_t_subsets)
        best_blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                best_gain = -1
                best_point = -1
                for x in range(v):
                    if x in block:
                        continue
                    count = 0
                    for sub in itertools.combinations(block, t - 1):
                        if tuple(sorted(sub + (x,))) in uncovered:
                            count += 1
                    if count > best_gain:
                        best_gain = count
                        best_point = x
                block.append(best_point)
            block.sort()
            best_blocks.append(block)
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
        best_blocks = local_search_remove(best_blocks)
    
    return best_blocks
