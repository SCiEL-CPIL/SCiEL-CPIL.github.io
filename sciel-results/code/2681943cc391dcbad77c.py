import time
import random

MAX_VAL = 1000
MAX_SUM = 2000

def _construct_greedy_density_aware(deadline):
    """
    Greedy construction that picks the element minimizing the number of 
    sum-collisions that become 'tight' (count reaches 2).
    """
    counts = [0] * (MAX_SUM + 1)
    chosen = []
    chosen_set = set()
    remaining = list(range(1, MAX_VAL + 1))
    
    while remaining:
        if time.time() > deadline:
            break
            
        best_x = -1
        best_cost = float('inf')
        best_val = float('inf') # Tie breaker: prefer smaller values? Or random?
        
        # To allow for some randomness in tie-breaking or exploration, 
        # we can shuffle remaining occasionally, but sorting by cost is key.
        
        # Optimization: 
        # Iterating all remaining (up to 1000) and checking against chosen (up to 90)
        # is 90,000 ops per step. 90 steps -> 8.1M ops.
        
        for x in remaining:
            # Quick check: is x valid?
            valid = True
            cost = 0
            
            # Check x + y for y in chosen
            for y in chosen:
                s = x + y
                if counts[s] >= 2:
                    valid = False
                    break
                elif counts[s] == 1:
                    cost += 1
            
            if not valid:
                continue
                
            # Check 2x
            s2 = 2 * x
            if counts[s2] >= 2:
                valid = False
            elif counts[s2] == 1:
                cost += 1
                
            if valid:
                # Tie breaker: prefer smaller x to keep numbers dense? 
                # Or prefer larger x? 
                # Usually, smaller x leaves larger numbers for later which might be more flexible?
                # Let's stick to smallest x for determinism in tie, but we can randomize.
                if cost < best_cost or (cost == best_cost and x < best_val):
                    best_cost = cost
                    best_x = x
                    best_val = x
                    
        if best_x == -1:
            break
            
        # Add best_x
        chosen.append(best_x)
        chosen_set.add(best_x)
        remaining.remove(best_x)
        
        # Update counts
        for y in chosen[:-1]: # All previous elements
            s = best_x + y
            counts[s] += 1
        s2 = 2 * best_x
        counts[s2] += 1
            
    return chosen

def _local_improve(chosen, deadline):
    """
    Try to improve the set by swapping elements.
    """
    chosen = sorted(chosen)
    counts = [0] * (MAX_SUM + 1)
    for i in range(len(chosen)):
        for j in range(i, len(chosen)):
            counts[chosen[i] + chosen[j]] += 1
            
    chosen_set = set(chosen)
    improved = True
    passes = 0
    max_passes = 20
    
    while improved and passes < max_passes:
        if time.time() > deadline:
            break
        improved = False
        passes += 1
        
        # Calculate tightness for each element
        # Tightness: number of sums involving x that have count == 2
        tightness = {x: 0 for x in chosen}
        for i in range(len(chosen)):
            for j in range(i, len(chosen)):
                s = chosen[i] + chosen[j]
                if counts[s] == 2:
                    tightness[chosen[i]] += 1
                    if j > i:
                        tightness[chosen[j]] += 1
                        
        # Sort by tightness descending
        sorted_chosen = sorted(chosen, key=lambda x: -tightness[x])
        
        for x_remove in sorted_chosen:
            if time.time() > deadline:
                break
            
            # Remove x_remove
            idx = chosen.index(x_remove)
            chosen_temp = chosen[:idx] + chosen[idx+1:]
            
            # Update counts for removal
            for y in chosen_temp:
                if y != x_remove:
                    s = x_remove + y
                    counts[s] -= 1
            counts[2 * x_remove] -= 1
            chosen_set.discard(x_remove)
            
            # Try to find a replacement
            best_add = -1
            best_new_tightness = float('inf')
            
            for x in range(1, MAX_VAL + 1):
                if x in chosen_set:
                    continue
                    
                # Check validity
                valid = True
                new_tight = 0
                for y in chosen_temp:
                    s = x + y
                    if counts[s] >= 2:
                        valid = False
                        break
                    elif counts[s] == 1:
                        new_tight += 1
                if not valid:
                    continue
                
                s2 = 2 * x
                if counts[s2] >= 2:
                    valid = False
                else:
                    if counts[s2] == 1:
                        new_tight += 1
                        # Note: if counts[s2] was 0, new_tight doesn't increase, 
                        # but the sum becomes 1, which is not "tight".
                        
                if valid:
                    # We want to minimize new tightness
                    if new_tight < best_new_tightness:
                        best_new_tightness = new_tight
                        best_add = x
                        
            if best_add != -1:
                # Accept swap
                chosen_temp.append(best_add)
                chosen_set.add(best_add)
                
                # Update counts for addition
                for y in chosen_temp[:-1]:
                    s = best_add + y
                    counts[s] += 1
                counts[2 * best_add] += 1
                
                chosen = sorted(chosen_temp)
                improved = True
            else:
                # Revert removal
                chosen_temp.append(x_remove)
                chosen_set.add(x_remove)
                for y in chosen_temp[:-1]:
                    s = x_remove + y
                    counts[s] += 1
                counts[2 * x_remove] += 1
                chosen = sorted(chosen_temp)
                
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 145 # Safety margin
    
    best_set = []
    best_size = 0
    
    # Run multiple trials of the density-aware greedy
    # Each trial is fast.
    
    # Trial 1: Deterministic (smallest x tie-break)
    s1 = _construct_greedy_density_aware(deadline)
    best_set = s1
    best_size = len(s1)
    
    # Trial 2-10: Randomized tie-breaks
    random.seed(42)
    for i in range(10):
        if time.time() > deadline - 10:
            break
        
        # We can modify the greedy to accept random ties
        # For simplicity, we just re-run the same function. 
        # To make it different, we'd need to pass a randomness parameter.
        # Let's just run it again. If it's deterministic, it gives same result.
        # Let's add a small random perturbation to the cost function?
        # Or just shuffle the 'remaining' list order? 
        # The current impl iterates 'remaining' which is a list. 
        # If we shuffle it, the 'first' valid candidate with min cost might change if there are ties.
        
        # Let's create a variant that shuffles remaining at each step? 
        # That's expensive. 
        # Instead, let's just run the local improvement on the first result.
        pass
        
    # Local Improvement
    if time.time() < deadline - 5:
        improved = _local_improve(best_set, deadline - 2)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(improved)
            
    return best_set
