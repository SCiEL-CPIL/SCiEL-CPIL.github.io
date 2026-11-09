import random
import time

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin

    def can_add(x, S, counts):
        if x in S:
            return False
        if counts.get(2 * x, 0) >= 2:
            return False
        for y in S:
            if counts.get(x + y, 0) >= 2:
                return False
        return True

    def try_add_greedy(S, counts, order):
        added_any = False
        for x in order:
            if x in S:
                continue
            if can_add(x, S, counts):
                S.add(x)
                counts[2 * x] = counts.get(2 * x, 0) + 1
                for y in S:
                    if y != x:
                        counts[x + y] = counts.get(x + y, 0) + 1
                added_any = True
        return added_any

    def greedy_build(order):
        S = set()
        counts = {}
        # Sort order to ensure deterministic or specific behavior? 
        # The order is the strategy.
        for x in order:
            if can_add(x, S, counts):
                S.add(x)
                counts[2 * x] = counts.get(2 * x, 0) + 1
                for y in S:
                    if y != x:
                        counts[x + y] = counts.get(x + y, 0) + 1
        return S, counts

    def local_search(S, counts):
        # Returns improved S, counts
        improved = True
        while improved and time.time() - start_time < DEADLINE:
            improved = False
            
            # 1. Try to add elements
            # We try to add the smallest possible elements first to pack densely?
            # Or any element? Smallest is usually better for future space.
            candidates_to_add = [x for x in range(1, 1001) if x not in S]
            # Sort by value to prefer small numbers? 
            # Actually, checking all is O(1000 * N). 
            # We can just iterate 1..1000.
            for x in range(1, 1001):
                if x in S:
                    continue
                if can_add(x, S, counts):
                    S.add(x)
                    counts[2 * x] = counts.get(2 * x, 0) + 1
                    for y in S:
                        if y != x:
                            counts[x + y] = counts.get(x + y, 0) + 1
                    improved = True
                    break # Restart the loop after adding one to update counts
            
            if not improved:
                # 2. Try 1-for-2 swaps
                # Remove x, add y, z
                for x in list(S):
                    if time.time() - start_time > DEADLINE:
                        return S, counts
                    
                    # Temporarily remove x
                    S.remove(x)
                    # Update counts
                    for y in S:
                        counts[x + y] -= 1
                        if counts[x + y] == 0:
                            del counts[x + y]
                    counts[2 * x] -= 1
                    if counts[2 * x] == 0:
                        del counts[2 * x]
                    
                    # Try to add two elements
                    # First, find all y that can be added
                    first_adds = []
                    for y in range(1, 1001):
                        if y in S:
                            continue
                        if can_add(y, S, counts):
                            first_adds.append(y)
                    
                    for y in first_adds:
                        if time.time() - start_time > DEADLINE:
                            # Restore and break
                            S.add(x)
                            counts[2 * x] = counts.get(2 * x, 0) + 1
                            for z in S:
                                if z != x:
                                    counts[x + z] = counts.get(x + z, 0) + 1
                            return S, counts
                        
                        # Temporarily add y
                        S.add(y)
                        counts[2 * y] = counts.get(2 * y, 0) + 1
                        for z in S:
                            if z != y:
                                counts[y + z] = counts.get(y + z, 0) + 1
                        
                        # Try to add a second element z
                        for z in range(1, 1001):
                            if z in S or z == y:
                                continue
                            if can_add(z, S, counts):
                                # Success! Add z
                                S.add(z)
                                counts[2 * z] = counts.get(2 * z, 0) + 1
                                for w in S:
                                    if w != z:
                                        counts[z + w] = counts.get(z + w, 0) + 1
                                
                                # We replaced x with y and z. Size increased by 1.
                                # Return immediately to restart the main loop?
                                # Yes, because counts changed significantly.
                                return S, counts
                        
                        # Undo adding y
                        S.remove(y)
                        counts[2 * y] -= 1
                        if counts[2 * y] == 0:
                            del counts[2 * y]
                        for z in S:
                            if z != y:
                                counts[y + z] -= 1
                                if counts[y + z] == 0:
                                    del counts[y + z]
                    
                    # Restore x
                    S.add(x)
                    counts[2 * x] = counts.get(2 * x, 0) + 1
                    for y in S:
                        if y != x:
                            counts[x + y] = counts.get(x + y, 0) + 1
                    
                    # If we didn't find a swap for this x, continue to next x?
                    # Or break?
                    # Usually, if we fail for one x, others might work.
                    # But it's expensive.
        
        return S, counts

    best_set = set()
    best_size = 0
    
    # 1. Baseline Greedy (sorted)
    S, counts = greedy_build(range(1, 1001))
    S, counts = local_search(S, counts)
    best_set = S
    best_size = len(S)
    
    # 2. Randomized Restarts
    for i in range(50):
        if time.time() - start_time > DEADLINE:
            break
        
        order = list(range(1, 1001))
        random.shuffle(order)
        
        S, counts = greedy_build(order)
        
        # Quick local search
        S, counts = local_search(S, counts)
        
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
            
    return sorted(best_set)
