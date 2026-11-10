import time
import random
import numpy as np

def _try_greedy_fast(order, deadline):
    """Fast greedy using numpy array for sum counts."""
    chosen = []
    # counts[s] = number of representations of s, s ranges from 2 to 2000
    counts = np.zeros(2001, dtype=np.int32)
    
    for x in order:
        if time.time() > deadline:
            break
        
        if chosen:
            # Check if any new sum would exceed 2
            # New sums: x + y for y in chosen, and 2*x
            # Vectorized check
            sums_with_chosen = x + np.array(chosen)
            # Check 2*x
            two_x = 2 * x
            # Check all new sums
            if np.any(counts[sums_with_chosen] >= 2) or counts[two_x] >= 2:
                continue
        
        # Valid: add x
        chosen.append(x)
        if chosen:
            counts[x + np.array(chosen[:-1])] += 1
        counts[2 * x] += 1
    
    return chosen

def _try_greedy_weighted(order, weights, deadline):
    """Greedy that prioritizes elements with higher 'desirability' scores."""
    # Sort by weight descending, then process
    indexed = list(enumerate(order))
    indexed.sort(key=lambda i: -weights[i[0]])
    
    chosen = []
    counts = np.zeros(2001, dtype=np.int32)
    
    for _, x in indexed:
        if time.time() > deadline:
            break
        
        if chosen:
            sums_with_chosen = x + np.array(chosen)
            two_x = 2 * x
            if np.any(counts[sums_with_chosen] >= 2) or counts[two_x] >= 2:
                continue
        
        chosen.append(x)
        if chosen:
            counts[x + np.array(chosen[:-1])] += 1
        counts[2 * x] += 1
    
    return chosen

def _local_search(initial_set, deadline, max_iter=5000):
    """Try to improve a set by swapping elements or adding/removing."""
    chosen = sorted(initial_set)
    chosen_set = set(chosen)
    
    # Build counts using numpy array
    counts = np.zeros(2001, dtype=np.int32)
    for i, a in enumerate(chosen):
        for b in chosen[i:]:
            s = a + b
            counts[s] += 1
    
    # Verify validity
    if np.any(counts > 2):
        return list(initial_set)  # Invalid, return as-is
    
    best_size = len(chosen)
    best_set = list(chosen)
    
    all_elements = set(range(1, 1001))
    candidates_to_add = sorted(all_elements - chosen_set)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try adding a candidate
        for x in candidates_to_add:
            if time.time() > deadline:
                break
            
            # Check if adding x is valid
            if chosen:
                new_sums = x + np.array(chosen)
                if np.any(counts[new_sums] >= 2) or counts[2 * x] >= 2:
                    continue
            
            # Add x
            chosen.append(x)
            chosen_set.add(x)
            if chosen:
                counts[x + np.array(chosen[:-1])] += 1
            counts[2 * x] += 1
            best_size = len(chosen)
            best_set = list(chosen)
            improved = True
            break
        
        if improved:
            candidates_to_add = sorted(all_elements - chosen_set)
            continue
        
        # Try swapping: remove one, add another
        for i, x in enumerate(chosen):
            if time.time() > deadline:
                break
            
            # Temporarily remove x
            if chosen:
                other_chosen = [c for j, c in enumerate(chosen) if j != i]
                removed_sums = [x + y for y in other_chosen] + [2 * x]
            
            # Actually remove
            if chosen:
                other_chosen = [c for j, c in enumerate(chosen) if j != i]
                for y in other_chosen:
                    counts[x + y] -= 1
                counts[2 * x] -= 1
            chosen.pop(i)
            chosen_set.discard(x)
            
            # Try adding each candidate
            for y in candidates_to_add:
                if time.time() > deadline:
                    break
                
                # Check if adding y is valid
                if chosen:
                    new_sums = y + np.array(chosen)
                    if np.any(counts[new_sums] >= 2) or counts[2 * y] >= 2:
                        continue
                
                # Perform swap
                chosen.append(y)
                chosen_set.add(y)
                if chosen:
                    counts[y + np.array(chosen[:-1])] += 1
                counts[2 * y] += 1
                best_size = len(chosen)
                best_set = list(chosen)
                improved = True
                break
            
            if not improved:
                # Restore x
                chosen.insert(i, x)
                chosen_set.add(x)
                if chosen:
                    other_chosen = [c for j, c in enumerate(chosen) if j != i]
                    for y in other_chosen:
                        counts[x + y] += 1
                    counts[2 * x] += 1
            
            if improved:
                candidates_to_add = sorted(all_elements - chosen_set)
                break
    
    return best_set

def _polish_additions(current_set, deadline, max_rounds=50):
    """Try to add more elements by checking all candidates multiple times."""
    chosen = sorted(current_set)
    chosen_set = set(chosen)
    counts = np.zeros(2001, dtype=np.int32)
    for i, a in enumerate(chosen):
        for b in chosen[i:]:
            s = a + b
            counts[s] += 1
    
    if np.any(counts > 2):
        return list(current_set)
    
    all_elements = set(range(1, 1001))
    candidates = sorted(all_elements - chosen_set)
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        
        added_any = False
        new_candidates = []
        for x in candidates:
            if time.time() > deadline:
                break
            
            if chosen:
                new_sums = x + np.array(chosen)
                if np.any(counts[new_sums] >= 2) or counts[2 * x] >= 2:
                    new_candidates.append(x)
                    continue
            
            chosen.append(x)
            chosen_set.add(x)
            if chosen:
                counts[x + np.array(chosen[:-1])] += 1
            counts[2 * x] += 1
            added_any = True
        
        if not added_any:
            break
        candidates = new_candidates
    
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_set = []
    best_size = 0
    
    # 1. Deterministic Greedy (Ascending)
    s1 = _try_greedy_fast(range(1, 1001), deadline)
    if len(s1) > best_size:
        best_set = s1
        best_size = len(s1)
    
    # 2. Deterministic Greedy (Descending)
    s2 = _try_greedy_fast(range(1000, 0, -1), deadline)
    if len(s2) > best_size:
        best_set = s2
        best_size = len(s2)
    
    # 3. Randomized Greedy with multiple seeds
    rng = random.Random(42)
    
    # Run multiple randomized restarts
    restart_count = 0
    while time.time() < deadline - 30.0:  # Leave time for local search
        order = list(range(1, 1001))
        rng.shuffle(order)
        s = _try_greedy_fast(order, deadline - 30.0)
        if len(s) > best_size:
            best_set = s
            best_size = len(s)
        
        restart_count += 1
        # Do local search every few restarts
        if restart_count % 3 == 0 and time.time() < deadline - 10.0:
            improved = _local_search(best_set, deadline - 5.0, max_iter=2000)
            if len(improved) > best_size:
                best_set = improved
                best_size = len(improved)
    
    # Final polish pass
    if time.time() < deadline - 2.0:
        polished = _polish_additions(best_set, deadline - 1.0, max_rounds=100)
        if len(polished) > best_size:
            best_set = polished
            best_size = len(polished)
    
    # One more local search on the best result
    if time.time() < deadline - 1.0:
        improved = _local_search(best_set, deadline - 0.5, max_iter=1000)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(improved)
    
    return best_set
