import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    all_points = list(range(v))
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    
    def greedy_once(seed_val):
        random.seed(seed_val)
        uncovered = set(all_tsubsets)
        blocks = []
        
        while uncovered:
            if time.time() > deadline - 3:
                return None
            
            target = random.choice(list(uncovered))
            block = list(target)
            block_set = set(block)
            
            while len(block) < k:
                best_gain = -1
                best_points = []
                
                for x in all_points:
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        new_t = tuple(sorted(rest + (x,)))
                        if new_t in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain:
                        best_points.append(x)
                
                if best_gain <= 0:
                    break
                
                chosen = random.choice(best_points)
                block.append(chosen)
                block_set.add(chosen)
            
            if len(block) == k:
                block.sort()
                blocks.append(block)
                for ts in itertools.combinations(block, t):
                    uncovered.discard(ts)
        
        return blocks
    
    best_blocks = None
    best_count = float('inf')
    
    # Try multiple seeds
    for seed in range(500):
        if time.time() > deadline - 5:
            break
        
        blocks = greedy_once(seed)
        if blocks is not None and len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
        
        if best_count <= 76:
            break
    
    # Fallback: deterministic greedy
    if best_blocks is None:
        uncovered = set(all_tsubsets)
        best_blocks = []
        while uncovered:
            target = min(uncovered)
            block = list(target)
            block_set = set(block)
            while len(block) < k:
                best_gain = -1
                best_x = None
                for x in all_points:
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        new_t = tuple(sorted(rest + (x,)))
                        if new_t in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is not None and best_gain > 0:
                    block.append(best_x)
                    block_set.add(best_x)
                else:
                    for x in all_points:
                        if x not in block_set:
                            block.append(x)
                            block_set.add(x)
                            break
            block.sort()
            best_blocks.append(block)
            for ts in itertools.combinations(block, t):
                uncovered.discard(ts)
    
    # Remove redundant blocks
    if best_blocks is not None:
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for i in range(len(best_blocks)):
                if time.time() > deadline - 2:
                    break
                remaining = best_blocks[:i] + best_blocks[i+1:]
                covered = set()
                for b in remaining:
                    for ts in itertools.combinations(b, t):
                        covered.add(ts)
                if len(covered) == num_tsubsets:
                    best_blocks = remaining
                    changed = True
                    break
    
    return best_blocks
