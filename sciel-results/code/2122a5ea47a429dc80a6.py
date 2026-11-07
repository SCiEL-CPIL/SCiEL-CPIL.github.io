import time
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 78  # 78s safety margin
    
    def try_greedy(order, max_size=60):
        """Greedily build set by trying candidates in given order."""
        # sums[k] is a Python int bitset: bit s set means some subset of size k has sum s
        sums = [0] * (max_size + 1)
        sums[0] = 1  # empty subset: sum 0
        
        selected = []
        used = [False] * 257
        
        for candidate in order:
            if time.time() > deadline:
                break
            if candidate < 1 or candidate > 256:
                continue
            if used[candidate]:
                continue
            
            # Check collision: for each k >= 1, new sums at cardinality k are (sums[k-1] << candidate)
            # Collision if (sums[k-1] << candidate) & sums[k] != 0
            conflict = False
            for k in range(len(selected) + 1, 0, -1):
                new_sums = (sums[k-1] << candidate)
                if new_sums & sums[k]:
                    conflict = True
                    break
            
            if not conflict:
                used[candidate] = True
                selected.append(candidate)
                for k in range(len(selected), 0, -1):
                    sums[k] |= (sums[k-1] << candidate)
                
                if len(selected) >= max_size:
                    break
        
        return selected
    
    # Strategy 1: Descending order
    result1 = try_greedy(list(range(256, 0, -1)))
    if time.time() > deadline:
        return result1
    
    # Strategy 2: Ascending order  
    result2 = try_greedy(list(range(1, 257)))
    if len(result2) > len(result1):
        result1 = result2
    if time.time() > deadline:
        return result1
    
    # Strategy 3: Powers of 2 first, then rest descending
    powers = [2**i for i in range(8) if 2**i <= 256]
    rest = [x for x in range(256, 0, -1) if x not in set(powers)]
    result3 = try_greedy(powers + rest)
    if len(result3) > len(result1):
        result1 = result3
    if time.time() > deadline:
        return result1
    
    # Strategy 4: Try to extend current best
    best = result1
    used_set = set(best)
    remaining = [x for x in range(256, 0, -1) if x not in used_set]
    
    max_k = len(best) + 1
    sums = [0] * (max_k + 1)
    sums[0] = 1
    for v in best:
        for k in range(len(best), 0, -1):
            sums[k] |= (sums[k-1] << v)
    
    extended = list(best)
    for candidate in remaining:
        if time.time() > deadline:
            break
        conflict = False
        for k in range(len(extended) + 1, 0, -1):
            new_sums = (sums[k-1] << candidate)
            if new_sums & sums[k]:
                conflict = True
                break
        if not conflict:
            extended.append(candidate)
            for k in range(len(extended), 0, -1):
                sums[k] |= (sums[k-1] << candidate)
    
    if len(extended) > len(best):
        best = extended
    
    # Strategy 5: Randomized restarts
    for attempt in range(50):
        if time.time() > deadline:
            break
        order = list(range(256, 0, -1))
        random.shuffle(order)
        result = try_greedy(order)
        if len(result) > len(best):
            best = result
    
    return best

# EVOLVE-BLOCK-END
