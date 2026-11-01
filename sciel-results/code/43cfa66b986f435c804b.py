"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for 120s limit
    
    # Encode t-subsets as integers using combinatorial number system
    # Or simpler: use a dictionary mapping frozenset to id
    # For speed, use integer encoding via bitmasks
    # v=19 fits in 19 bits, so we can use integers directly
    
    # Generate all t-subsets and assign IDs
    all_t_subs = list(itertools.combinations(range(v), t))
    sub_to_id = {s: i for i, s in enumerate(all_t_subs)}
    n_subs = len(all_t_subs)
    
    # uncovered: boolean array, True if uncovered
    uncovered = bytearray([1]) * n_subs
    uncovered_count = n_subs
    
    blocks = []
    
    # Precompute for each point x, which t-subsets contain x
    # This helps with gain calculation
    # For a point x not in block, gain(x) = number of (t-1)-subsets P of block
    # such that P ∪ {x} is uncovered
    # = number of (t-1)-subsets P of block where sub_to_id has P∪{x} and uncovered[id]
    
    while uncovered_count > 0:
        if time.time() > deadline:
            break
        
        # Pick an uncovered subset - use the first one for simplicity
        # Could use a heuristic here
        target_id = -1
        for i in range(n_subs):
            if uncovered[i]:
                target_id = i
                break
        if target_id == -1:
            break
        
        target = all_t_subs[target_id]
        block = list(target)
        block_set = set(block)
        
        # Grow block to size k
        while len(block) < k:
            if time.time() > deadline:
                break
            
            best_point = -1
            best_gain = -1
            
            # Get all (t-1)-subsets of current block
            t_minus_1_subs = list(itertools.combinations(block, t - 1))
            
            for x in range(v):
                if x in block_set:
                    continue
                
                gain = 0
                for p in t_minus_1_subs:
                    # Create p ∪ {x} as a sorted tuple
                    # p is sorted, insert x
                    new_sub = list(p)
                    # Insert x in sorted position
                    inserted = False
                    for i in range(len(new_sub)):
                        if x < new_sub[i]:
                            new_sub.insert(i, x)
                            inserted = True
                            break
                    if not inserted:
                        new_sub.append(x)
                    
                    new_tuple = tuple(new_sub)
                    sid = sub_to_id.get(new_tuple)
                    if sid is not None and uncovered[sid]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1 or best_gain <= 0:
                # Pick any point not in block that appears in some uncovered subset
                found = False
                for i in range(n_subs):
                    if uncovered[i]:
                        s = all_t_subs[i]
                        for x in s:
                            if x not in block_set:
                                best_point = x
                                found = True
                                break
                        if found:
                            break
                if best_point == -1:
                    # Fallback
                    for x in range(v):
                        if x not in block_set:
                            best_point = x
                            break
            
            block.append(best_point)
            block_set.add(best_point)
        
        block.sort()
        blocks.append(tuple(block))
        
        # Update uncovered: remove all t-subsets of this block
        for s in itertools.combinations(block, t):
            sid = sub_to_id.get(tuple(s))
            if sid is not None and uncovered[sid]:
                uncovered[sid] = 0
                uncovered_count -= 1
    
    # Convert back to lists
    return [list(b) for b in blocks]
# EVOLVE-BLOCK-END
