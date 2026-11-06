import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    N = 1000
    best_set = []
    best_size = 0
    
    def run_greedy(seed):
        random.seed(seed)
        chosen = []
        counts = [0] * (2 * N + 1)
        candidates = list(range(1, N + 1))
        
        while candidates:
            best_x = None
            min_cost_found = float('inf')
            best_candidates = []
            
            for x in candidates:
                if counts[2 * x] >= 2:
                    continue
                valid = True
                cost = 0
                if counts[2 * x] == 1:
                    cost += 1
                for y in chosen:
                    s = x + y
                    if counts[s] >= 2:
                        valid = False
                        break
                    if counts[s] == 1:
                        cost += 1
                if not valid:
                    continue
                
                if cost < min_cost_found:
                    min_cost_found = cost
                    best_candidates = [x]
                elif cost == min_cost_found:
                    best_candidates.append(x)
            
            if not best_candidates:
                break
            
            best_x = random.choice(best_candidates)
            chosen.append(best_x)
            counts[2 * best_x] += 1
            for y in chosen[:-1]:
                s = best_x + y
                counts[s] += 1
            candidates.remove(best_x)
            
        return chosen

    def improve_set(chosen):
        """
        Local search: Try to swap elements to allow adding new ones.
        Returns a set potentially larger than the input.
        """
        chosen = list(chosen)
        counts = [0] * (2 * N + 1)
        for x in chosen:
            counts[2 * x] += 1
            for y in chosen:
                if y > x:
                    counts[x + y] += 1
        
        # We will try to perform swaps that allow addition
        # We iterate for a fixed number of steps or time
        steps = 0
        while time.time() < deadline - 5: # Leave 5s for return
            if steps > 10000: break
            steps += 1
            
            # Pick a random element to remove
            i = random.randint(0, len(chosen) - 1)
            a = chosen[i]
            
            # Remove a
            for y in chosen:
                if y == a:
                    counts[2 * a] -= 1
                else:
                    s = a + y
                    counts[s] -= 1
            
            chosen.pop(i)
            
            # Try to find a replacement b that is valid
            # And ideally, allows adding a new element c
            # For simplicity, just find a valid b first
            valid_replacements = []
            # Sample a subset of candidates to speed up?
            # Or iterate all?
            # Iterating all N=1000 is fast.
            for b in range(1, N + 1):
                if b in chosen:
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
                for y in chosen:
                    if y == a:
                        counts[2 * a] += 1
                    else:
                        s = a + y
                        counts[s] += 1
                continue
            
            b = random.choice(valid_replacements)
            
            # Add b
            chosen.append(b)
            counts[2 * b] += 1
            for y in chosen[:-1]:
                s = b + y
                counts[s] += 1
            
            # Now, try to add a new element c
            # Check if any element not in chosen is valid
            added_new = False
            # To speed up, we can just pick a random few?
            # Or check all?
            # Checking all is O(N*K) = 1000*50 = 50k. Fast.
            for c in range(1, N + 1):
                if c in chosen:
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
                    counts[2 * c] += 1
                    for y in chosen[:-1]:
                        s = c + y
                        counts[s] += 1
                    added_new = True
                    break # Only add one at a time to keep control
            
            # If we didn't add a new element, we have the same size.
            # We can keep this swap or revert?
            # Keeping it might lead to better future swaps.
            # So we keep it.
            
        return chosen

    # Phase 1: Greedy Restarts
    seed = 0
    while time.time() < deadline - 20: # Reserve 20s for local search
        current_set = run_greedy(seed)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
        seed += 1
        
    # Phase 2: Local Search on Best Set
    if best_set and time.time() < deadline - 5:
        improved_set = improve_set(best_set)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set
            
    return best_set
