import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 150s wall-clock, 5s safety margin
    
    N = 1000
    best_set = []
    best_size = 0
    
    def run_greedy(seed):
        """
        Greedy construction with optimized cost computation.
        Uses numpy for vectorized cost calculation where possible.
        """
        random.seed(seed)
        chosen = []
        chosen_set = set()
        counts = np.zeros(2 * N + 1, dtype=np.int32)
        
        # Precompute candidate list
        candidates = list(range(1, N + 1))
        cand_set = set(candidates)
        
        while candidates:
            # Check time
            if time.time() > deadline - 10:
                break
            
            best_x = None
            min_cost = float('inf')
            best_candidates = []
            
            # For each candidate, compute the cost (number of sums that would go from 1->2)
            # and validity (no sum goes from 2->3)
            # cost = number of y in chosen where counts[x+y] == 1
            # valid = no y in chosen where counts[x+y] >= 2, and counts[2x] < 2
            
            if not chosen:
                # First element: always valid, cost=0
                best_x = random.choice(candidates)
            else:
                chosen_arr = np.array(chosen, dtype=np.int32)
                
                for x in candidates:
                    if counts[2 * x] >= 2:
                        continue
                    
                    # Check validity and compute cost
                    # sums = x + chosen
                    sums = x + chosen_arr
                    sum_counts = counts[sums]
                    
                    # Valid if no sum has count >= 2
                    if np.any(sum_counts >= 2):
                        continue
                    
                    # Cost: number of sums with count == 1
                    cost = np.sum(sum_counts == 1)
                    if counts[2 * x] == 1:
                        cost += 1
                    
                    if cost < min_cost:
                        min_cost = cost
                        best_candidates = [x]
                    elif cost == min_cost:
                        best_candidates.append(x)
            
            if not best_candidates:
                break
            
            best_x = random.choice(best_candidates)
            chosen.append(best_x)
            chosen_set.add(best_x)
            counts[2 * best_x] += 1
            for y in chosen[:-1]:
                counts[best_x + y] += 1
            candidates.remove(best_x)
            cand_set.discard(best_x)
        
        return chosen
    
    def improve_set(chosen, time_limit):
        """
        Local search: Try swaps and additions to grow the set.
        """
        chosen = list(chosen)
        chosen_set = set(chosen)
        counts = np.zeros(2 * N + 1, dtype=np.int32)
        
        for x in chosen:
            counts[2 * x] += 1
            for y in chosen:
                if y > x:
                    counts[x + y] += 1
        
        end_time = time.time() + time_limit
        steps = 0
        
        while time.time() < end_time:
            steps += 1
            if steps % 100 == 0 and time.time() > deadline - 5:
                break
            
            # Strategy: Try to add a new element directly
            # If that fails, try a swap to enable addition
            
            # First, try direct addition
            added = False
            # Sample some candidates to speed up
            if len(chosen) < 50:
                candidates_to_try = list(range(1, N + 1))
            else:
                # Random sample for speed
                candidates_to_try = random.sample(range(1, N + 1), min(200, N))
            
            for c in candidates_to_try:
                if c in chosen_set:
                    continue
                if counts[2 * c] >= 2:
                    continue
                
                valid = True
                for y in chosen:
                    s = c + y
                    if counts[s] >= 2:
                        valid = False
                        break
                
                if valid:
                    # Add c
                    chosen.append(c)
                    chosen_set.add(c)
                    counts[2 * c] += 1
                    for y in chosen[:-1]:
                        counts[c + y] += 1
                    added = True
                    break
            
            if added:
                continue
            
            # Try a swap: remove one element, add another
            if len(chosen) == 0:
                break
            
            i = random.randint(0, len(chosen) - 1)
            a = chosen[i]
            
            # Remove a
            for y in chosen:
                if y == a:
                    counts[2 * a] -= 1
                else:
                    counts[a + y] -= 1
            chosen.pop(i)
            chosen_set.discard(a)
            
            # Try to find a valid replacement b
            valid_replacements = []
            candidates_to_try = random.sample(range(1, N + 1), min(300, N))
            
            for b in candidates_to_try:
                if b in chosen_set:
                    continue
                if counts[2 * b] >= 2:
                    continue
                
                valid = True
                for y in chosen:
                    s = b + y
                    if counts[s] >= 2:
                        valid = False
                        break
                
                if valid:
                    valid_replacements.append(b)
            
            if not valid_replacements:
                # Put a back
                chosen.insert(i, a)
                chosen_set.add(a)
                for y in chosen:
                    if y == a:
                        counts[2 * a] += 1
                    else:
                        counts[a + y] += 1
                continue
            
            b = random.choice(valid_replacements)
            
            # Add b
            chosen.append(b)
            chosen_set.add(b)
            counts[2 * b] += 1
            for y in chosen[:-1]:
                counts[b + y] += 1
            
            # Now try to add a new element c
            candidates_to_try = random.sample(range(1, N + 1), min(200, N))
            for c in candidates_to_try:
                if c in chosen_set:
                    continue
                if counts[2 * c] >= 2:
                    continue
                
                valid = True
                for y in chosen:
                    s = c + y
                    if counts[s] >= 2:
                        valid = False
                        break
                
                if valid:
                    chosen.append(c)
                    chosen_set.add(c)
                    counts[2 * c] += 1
                    for y in chosen[:-1]:
                        counts[c + y] += 1
                    break
        
        return chosen
    
    # Phase 1: Greedy Restarts
    seed = 0
    while time.time() < deadline - 30:
        current_set = run_greedy(seed)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
        seed += 1
    
    # Phase 2: Local Search on Best Set
    if best_set and time.time() < deadline - 5:
        remaining_time = min(25, deadline - 5 - time.time())
        improved_set = improve_set(best_set, remaining_time)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set
    
    return best_set
