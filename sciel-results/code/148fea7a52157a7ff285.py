"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    N = 1000
    
    def is_valid_add(chosen_set, counts, x):
        """Check if x can be added to chosen."""
        for y in chosen_set:
            s = x + y
            if counts.get(s, 0) >= 2:
                return False
        if counts.get(2 * x, 0) >= 2:
            return False
        return True
    
    def greedy(order, chosen_set, counts):
        """Greedy construction given an order."""
        for x in order:
            if x not in chosen_set and is_valid_add(chosen_set, counts, x):
                for y in chosen_set:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                counts[2 * x] = counts.get(2 * x, 0) + 1
                chosen_set.add(x)
        return chosen_set, counts
    
    def build_counts(chosen_list):
        """Build sum counts for a given sorted list."""
        counts = {}
        for i in range(len(chosen_list)):
            a = chosen_list[i]
            for j in range(i, len(chosen_list)):
                s = a + chosen_list[j]
                counts[s] = counts.get(s, 0) + 1
        return counts
    
    # Phase 1: Multiple greedy passes with different orderings
    best_chosen = set()
    best_size = 0
    
    # Forward greedy
    chosen_set = set()
    counts = {}
    for x in range(1, N + 1):
        if is_valid_add(chosen_set, counts, x):
            for y in chosen_set:
                s = x + y
                counts[s] = counts.get(s, 0) + 1
            counts[2 * x] = counts.get(2 * x, 0) + 1
            chosen_set.add(x)
    best_chosen = chosen_set.copy()
    best_size = len(chosen_set)
    
    # Backward greedy
    if time.time() < deadline - 15:
        chosen_set = set()
        counts = {}
        for x in range(N, 0, -1):
            if is_valid_add(chosen_set, counts, x):
                for y in chosen_set:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                counts[2 * x] = counts.get(2 * x, 0) + 1
                chosen_set.add(x)
        if len(chosen_set) > best_size:
            best_chosen = chosen_set.copy()
            best_size = len(chosen_set)
    
    # Random greedy passes
    random.seed(42)
    num_random_passes = 50
    for trial in range(num_random_passes):
        if time.time() > deadline - 15:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        chosen_set = set()
        counts = {}
        for x in order:
            if is_valid_add(chosen_set, counts, x):
                for y in chosen_set:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                counts[2 * x] = counts.get(2 * x, 0) + 1
                chosen_set.add(x)
        if len(chosen_set) > best_size:
            best_chosen = chosen_set.copy()
            best_size = len(chosen_set)
    
    # Phase 2: Try to add any remaining elements to the best solution
    chosen_set = best_chosen.copy()
    chosen_list = sorted(chosen_set)
    counts = build_counts(chosen_list)
    
    added = True
    while added and time.time() < deadline - 10:
        added = False
        for x in range(1, N + 1):
            if x not in chosen_set and is_valid_add(chosen_set, counts, x):
                for y in chosen_set:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                counts[2 * x] = counts.get(2 * x, 0) + 1
                chosen_set.add(x)
                added = True
                break
    
    best_size = len(chosen_set)
    best_chosen = chosen_set.copy()
    
    # Phase 3: Local search - try to improve by swapping elements
    if time.time() < deadline - 3:
        chosen_list = sorted(best_chosen)
        counts = build_counts(chosen_list)
        
        for iteration in range(300):
            if time.time() > deadline - 2:
                break
            
            if len(best_chosen) == 0:
                break
            
            # Pick a random element to remove
            old_x = random.choice(list(best_chosen))
            
            # Temporarily remove it
            for y in best_chosen:
                if y == old_x:
                    continue
                s = old_x + y
                counts[s] -= 1
                if counts[s] == 0:
                    del counts[s]
            counts[2 * old_x] -= 1
            if counts[2 * old_x] == 0:
                del counts[2 * old_x]
            best_chosen.discard(old_x)
            
            # Try to add a different element
            found = False
            candidates = [x for x in range(1, N + 1) if x != old_x and x not in best_chosen]
            random.shuffle(candidates)
            
            for x in candidates[:150]:
                if is_valid_add(best_chosen, counts, x):
                    for y in best_chosen:
                        s = x + y
                        counts[s] = counts.get(s, 0) + 1
                    counts[2 * x] = counts.get(2 * x, 0) + 1
                    best_chosen.add(x)
                    found = True
                    break
            
            if not found:
                # Put old_x back
                for y in best_chosen:
                    s = old_x + y
                    counts[s] = counts.get(s, 0) + 1
                counts[2 * old_x] = counts.get(2 * old_x, 0) + 1
                best_chosen.add(old_x)
    
    return sorted(best_chosen)
# EVOLVE-BLOCK-END
