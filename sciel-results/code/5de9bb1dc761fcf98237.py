"""Combined B2[2] set construction: optimized greedy + intelligent local search."""
import time
import random

def _greedy_build():
    """Standard greedy: add x if it doesn't cause any sum to exceed 2 reps."""
    chosen = []
    counts = {}
    for x in range(1, 1001):
        new_sums = [x + y for y in chosen] + [2 * x]
        trial = dict(counts)
        ok = True
        for s in new_sums:
            trial[s] = trial.get(s, 0) + 1
            if trial[s] > 2:
                ok = False
                break
        if ok:
            chosen.append(x)
            counts = trial
    return chosen

def _build_counts(set_list):
    """Return dict: sum -> count of pairs (a,b) with a<=b from set_list."""
    counts = {}
    n = len(set_list)
    for i in range(n):
        for j in range(i, n):
            sm = set_list[i] + set_list[j]
            counts[sm] = counts.get(sm, 0) + 1
    return counts

def _try_add_fast(set_list, x, counts):
    """
    Try adding x to set_list (which must be sorted).
    Uses precomputed counts for efficiency.
    Returns (success, new_set_list, new_counts).
    """
    if x in set_list:
        return False, set_list, counts
    
    # Check if adding x would violate B2[2]
    for y in set_list:
        s = x + y
        if counts.get(s, 0) >= 2:
            return False, set_list, counts
    if counts.get(2 * x, 0) >= 2:
        return False, set_list, counts
    
    # It's valid, so build new list and counts
    new_set = list(set_list)
    new_set.append(x)
    new_set.sort()
    
    new_counts = dict(counts)
    for y in set_list:
        new_counts[x + y] = new_counts.get(x + y, 0) + 1
    new_counts[2 * x] = new_counts.get(2 * x, 0) + 1
    
    return True, new_set, new_counts

def _remove_and_recount(set_list, x, counts):
    """
    Remove x from set_list and update counts.
    Returns (new_set_list, new_counts).
    """
    new_set = [v for v in set_list if v != x]
    
    # Decrement counts for sums involving x
    for y in set_list:
        s = x + y
        if s in counts:
            counts[s] -= 1
            if counts[s] <= 0:
                del counts[s]
    if 2 * x in counts:
        counts[2 * x] -= 1
        if counts[2 * x] <= 0:
            del counts[2 * x]
    
    return new_set, counts

def _get_constrained_elements(set_list, counts):
    """
    Identify elements that participate in sums with count == 2.
    Removing these might free up capacity.
    """
    constrained = set()
    # For each sum with count 2, mark the elements involved
    # This is a bit expensive if done naively, so we approximate:
    # If an element x is in a sum s with count 2, it's constrained.
    # We can just return a random subset of elements if the set is large,
    # or pick elements that appear in many 'tight' sums.
    
    # Simple heuristic: return elements that are part of any sum with count 2.
    # To keep it fast, we don't trace back which elements formed which sum in counts directly.
    # Instead, we can just randomly pick elements, but bias towards smaller ones 
    # or those that might be 'central'. 
    # For now, let's just return all elements, and let the local search pick randomly,
    # but we can add a bias in the main loop.
    
    # Actually, a better heuristic for 'constrained' is to check if removing x 
    # allows adding more elements. But that's expensive.
    
    # Let's stick to random selection but ensure we don't pick the same one repeatedly too often.
    return set_list

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Step 1: Greedy baseline
    greedy_set = _greedy_build()
    current = sorted(greedy_set)
    current_set = set(current)
    current_counts = _build_counts(current)
    
    best = list(current)
    best_size = len(current)
    
    # Step 2: Local Search
    rng = random.Random(42)
    all_vals = list(range(1, 1001))
    
    # To improve on pure random local search, we can track 'stagnation'
    # and if stuck, perform a 'kick' (remove a few random elements)
    stagnation_counter = 0
    max_stagnation = 1000
    
    while time.time() < deadline:
        op = rng.random()
        
        if op < 0.4:
            # Pure add
            x = rng.choice(all_vals)
            if x in current_set:
                continue
            success, new_set, new_counts = _try_add_fast(current, x, current_counts)
            if success:
                current = new_set
                current_set = set(current)
                current_counts = new_counts
                if len(current) > best_size:
                    best = list(current)
                    best_size = len(current)
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
        elif op < 0.85:
            # Swap: remove one, add one
            if not current:
                continue
            
            # Bias removal towards elements that might be 'restrictive'
            # Heuristic: Remove a random element, but if we have been stagnant,
            # remove a random element from the first half (smaller numbers) 
            # as they tend to create more collisions with larger numbers?
            # Or just random.
            
            remove_x = rng.choice(current)
            add_x = rng.choice(all_vals)
            if add_x == remove_x:
                continue
            if add_x in current_set:
                continue
            
            # Temporarily remove remove_x
            temp_set, temp_counts = _remove_and_recount(current, remove_x, dict(current_counts))
            temp_set_set = set(temp_set)
            
            if add_x in temp_set_set:
                continue
            
            # Try to add add_x
            success, new_set, new_counts = _try_add_fast(temp_set, add_x, temp_counts)
            if success:
                current = new_set
                current_set = set(current)
                current_counts = new_counts
                if len(current) > best_size:
                    best = list(current)
                    best_size = len(current)
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
        else:
            # Remove one random element
            if current:
                remove_x = rng.choice(current)
                current, current_counts = _remove_and_recount(current, remove_x, current_counts)
                current_set = set(current)
                stagnation_counter += 1
        
        # If stuck for a while, do a 'kick' to escape local optimum
        if stagnation_counter > max_stagnation:
            # Remove 2-3 random elements to open up space
            for _ in range(3):
                if current:
                    remove_x = rng.choice(current)
                    current, current_counts = _remove_and_recount(current, remove_x, current_counts)
                    current_set = set(current)
            stagnation_counter = 0
    
    return best
