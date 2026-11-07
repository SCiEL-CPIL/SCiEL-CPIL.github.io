import random
import time
import math

def construct():
    n = 12
    num_sets = 40
    max_mask = 1 << n
    
    # Precompute list of all masks
    all_masks = list(range(max_mask))
    
    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b

    def count_comparable_pairs(chosen):
        count = 0
        for i in range(len(chosen)):
            for j in range(i+1, len(chosen)):
                if is_comparable(chosen[i], chosen[j]):
                    count += 1
        return count

    def get_gain(mask, chosen):
        gain = 0
        for c in chosen:
            if is_comparable(mask, c):
                gain += 1
        return gain

    # 1. Initial Solution: Greedy Chain Expansion
    # This mimics finding an Independent Set in the Conflict Graph (which is a Chain)
    order = list(range(n))
    random.shuffle(order)
    current = 0
    chain = [0]
    for i in order:
        current |= (1 << i)
        chain.append(current)
    
    chosen = set(chain)
    
    # Greedily add remaining sets
    while len(chosen) < num_sets:
        best_mask = -1
        best_gain = -1
        candidates = []
        max_g = -1
        for m in all_masks:
            if m in chosen:
                continue
            gain = get_gain(m, list(chosen))
            if gain > max_g:
                max_g = gain
                candidates = [m]
            elif gain == max_g:
                candidates.append(m)
        
        if candidates:
            best_mask = random.choice(candidates)
            chosen.add(best_mask)
        else:
            break
        
    best_chosen = list(chosen)
    best_score = count_comparable_pairs(best_chosen)
    
    # 2. Local Search (Simulated Annealing)
    start_time = time.time()
    limit = 50.0
    
    # Initial temperature
    T = 5.0
    cooling = 0.9995
    
    while time.time() - start_time < limit and T > 0.05:
        # Pick a random index to remove
        idx = random.randint(0, len(best_chosen) - 1)
        removed_mask = best_chosen[idx]
        
        # Remove it
        temp_chosen = [best_chosen[i] for i in range(len(best_chosen)) if i != idx]
        temp_set = set(temp_chosen)
        
        # Find best replacement
        best_new_mask = -1
        best_new_gain = -1
        
        for m in all_masks:
            if m in temp_set:
                continue
            gain = get_gain(m, temp_chosen)
            if gain > best_new_gain:
                best_new_gain = gain
                best_new_mask = m
            elif gain == best_new_gain and random.random() < 0.5:
                best_new_mask = m

        removed_gain = get_gain(removed_mask, temp_chosen)
        delta = best_new_gain - removed_gain
        
        # Accept if improvement, or with probability exp(delta/T) if worse
        if delta > 0 or random.random() < math.exp(delta / T):
            best_chosen[idx] = best_new_mask
            best_score += delta
            
        T *= cooling

    # Convert to list of lists
    result = []
    for mask in best_chosen:
        subset = [i for i in range(n) if (mask >> i) & 1]
        result.append(subset)
        
    return result
