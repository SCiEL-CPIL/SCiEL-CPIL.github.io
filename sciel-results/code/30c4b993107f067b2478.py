import itertools
import random
import math
import time

def construct(v, k, t):
    # 1. Precompute
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    def get_t_indices(block):
        indices = []
        for t_sub in itertools.combinations(block, t):
            indices.append(sub_to_idx[t_sub])
        return indices

    # 2. Greedy Initialization
    counts = [0] * n_t_subs
    deficit = n_t_subs
    blocks = []
    
    # To speed up finding uncovered subsets, we can just scan.
    # C(19,5) is small.
    
    while deficit > 0:
        # Find first uncovered
        idx = -1
        for i in range(n_t_subs):
            if counts[i] == 0:
                idx = i
                break
        
        if idx == -1:
            break # Should not happen
            
        t_sub = all_t_subs[idx]
        block = list(t_sub)
        
        # Greedy fill
        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                # Check t-subsets formed by x and t-1 from block
                for rest in itertools.combinations(block, t-1):
                    candidate_t_sub = tuple(sorted(rest + (x,)))
                    cand_idx = sub_to_idx[candidate_t_sub]
                    if counts[cand_idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)
            
        block.sort()
        blocks.append(block)
        
        # Update counts
        for t_idx in get_t_indices(block):
            if counts[t_idx] == 0:
                deficit -= 1
            counts[t_idx] += 1

    # 3. Simulated Annealing
    current_blocks = blocks
    block_indices = [get_t_indices(b) for b in current_blocks]
    current_score = len(current_blocks)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    start_time = time.time()
    time_limit = 110
    
    # Penalty for invalidity
    PENALTY = 1000
    
    while T > T_min and time.time() - start_time < time_limit:
        move_type = random.random()
        
        if move_type < 0.6:
            # Remove Block
            if len(current_blocks) <= 1:
                T *= cooling
                continue
            i = random.randint(0, len(current_blocks) - 1)
            
            # Check if removal causes deficit
            new_deficit = 0
            for t_idx in block_indices[i]:
                if counts[t_idx] == 1:
                    new_deficit += 1
            
            # Old score: len(current_blocks) + PENALTY * 0 (assuming valid)
            # New score: (len-1) + PENALTY * new_deficit
            # Delta: -1 + PENALTY * new_deficit
            
            delta_score = -1 + PENALTY * new_deficit
            
            if delta_score < 0 or (T > 0 and random.random() < math.exp(-delta_score / T)):
                # Accept
                for t_idx in block_indices[i]:
                    counts[t_idx] -= 1
                current_blocks.pop(i)
                block_indices.pop(i)
                current_score += delta_score
            else:
                # Reject
                pass

        else:
            # Add Block
            # If valid, add random block? 
            # If invalid, add block covering uncovered subset.
            
            # Check if currently valid
            # We don't track deficit explicitly in SA loop, 
            # but we can infer from current_score? 
            # No, current_score includes penalty.
            # Let's just check if there are any counts==0.
            # This is O(N).
            
            # Optimization: Keep track of deficit?
            # Let's add a variable `current_deficit`
            # But we didn't initialize it in SA section.
            # Let's calculate it once.
            
            # Actually, simpler: 
            # If we assume we start valid, and only accept moves that keep it valid 
            # or pay penalty, we can just try to add a block that helps.
            
            # Pick a target t-subset.
            # If we are invalid, pick an uncovered one.
            # If valid, pick a random one (to explore).
            
            # To avoid O(N) scan every time, let's just pick random index.
            # If it's uncovered, great. If covered, we add a redundant block (bad, but SA might remove it later).
            
            idx = random.randint(0, n_t_subs - 1)
            t_sub = all_t_subs[idx]
            
            # Construct block
            block = list(t_sub)
            candidates = [x for x in range(v) if x not in block]
            random.shuffle(candidates)
            block.extend(candidates[:k-t])
            block.sort()
            
            new_t_indices = get_t_indices(block)
            
            # Calculate delta
            # Adding block: len + 1
            # Coverage change:
            covered_uncovered = 0
            for t_idx in new_t_indices:
                if counts[t_idx] == 0:
                    covered_uncovered += 1
            
            delta_score = 1 - PENALTY * covered_uncovered
            
            if delta_score < 0 or (T > 0 and random.random() < math.exp(-delta_score / T)):
                # Accept
                for t_idx in new_t_indices:
                    counts[t_idx] += 1
                current_blocks.append(block)
                block_indices.append(new_t_indices)
                current_score += delta_score

        T *= cooling

    # Final Repair
    # Check if valid
    is_valid = True
    for c in counts:
        if c == 0:
            is_valid = False
            break
            
    if not is_valid:
        # Greedy repair
        while True:
            # Find uncovered
            idx = -1
            for i in range(n_t_subs):
                if counts[i] == 0:
                    idx = i
                    break
            if idx == -1:
                break
                
            t_sub = all_t_subs[idx]
            block = list(t_sub)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t-1):
                        candidate_t_sub = tuple(sorted(rest + (x,)))
                        cand_idx = sub_to_idx[candidate_t_sub]
                        if counts[cand_idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                block.append(best_x)
            block.sort()
            current_blocks.append(block)
            for t_idx in get_t_indices(block):
                counts[t_idx] += 1

    return current_blocks
