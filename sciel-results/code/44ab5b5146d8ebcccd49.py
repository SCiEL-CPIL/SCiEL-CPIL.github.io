import random
import time

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
    # Start with a random chain
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
        # Optimization: We can limit search to masks that are "close" to chosen?
        # For now, scan all.
        for m in all_masks:
            if m in chosen:
                continue
            gain = get_gain(m, list(chosen))
            if gain > best_gain:
                best_gain = gain
                best_mask = m
        chosen.add(best_mask)
        
    best_chosen = list(chosen)
    best_score = count_comparable_pairs(best_chosen)
    
    # 2. Local Search (Hill Climbing)
    # Time limit: 50 seconds
    start_time = time.time()
    limit = 50.0
    
    while time.time() - start_time < limit:
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
            # Tie-breaker: prefer masks with smaller size? Or random?
            # Let's just take the first max or random among max.
            if gain > best_new_gain:
                best_new_gain = gain
                best_new_mask = m
            elif gain == best_new_gain and random.random() < 0.5:
                best_new_mask = m

        # Calculate new score
        # We know the score change:
        # Old score had pairs involving removed_mask.
        # New score has pairs involving best_new_mask.
        # It's faster to recalculate or use delta.
        # Delta = gain(best_new) - gain(removed) + (comparability between new and others is accounted in gain)
        # Wait, gain(m, chosen) is exactly the number of new pairs added.
        # So new_score = old_score - gain(removed, chosen_without_removed) + gain(new, chosen_without_removed)
        # But gain(removed, chosen_without_removed) is the number of comparable pairs removed_mask had with the rest.
        
        removed_gain = get_gain(removed_mask, temp_chosen)
        
        if best_new_gain > removed_gain:
            # Improvement
            best_chosen[idx] = best_new_mask
            best_score += (best_new_gain - removed_gain)
        elif best_new_gain == removed_gain:
            # Neutral, might accept to escape plateaus?
            if random.random() < 0.1:
                best_chosen[idx] = best_new_mask
        else:
            # Worse, reject (or anneal, but let's keep it simple first)
            pass

    # Convert to list of lists
    result = []
    for mask in best_chosen:
        subset = [i for i in range(n) if (mask >> i) & 1]
        result.append(subset)
        
    return result
