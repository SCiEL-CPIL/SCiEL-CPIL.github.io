import random
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit, 10s safety margin
    
    # Base set: all odd numbers
    base_set = list(range(1, 257, 2))
    evens = list(range(2, 257, 2))
    
    best_set = base_set[:]
    
    # Greedy function: adds evens to base_set if no conflict
    def greedy_add(evens_order, current_set):
        s = current_set[:]
        # Convert to set for O(1) lookup? No, we need to check conflicts with all existing.
        # Actually, we just check conflicts with existing members.
        for a in evens_order:
            conflict = False
            for b in s:
                if a != b and (a * b) % (a + b) == 0:
                    conflict = True
                    break
            if not conflict:
                s.append(a)
        return s
    
    # Try ascending order
    candidate = greedy_add(evens, base_set)
    if len(candidate) > len(best_set):
        best_set = candidate
    
    # Try descending order
    candidate = greedy_add(evens[::-1], base_set)
    if len(candidate) > len(best_set):
        best_set = candidate
        
    # Try random orders
    while time.time() < deadline:
        random_evens = evens[:]
        random.shuffle(random_evens)
        candidate = greedy_add(random_evens, base_set)
        if len(candidate) > len(best_set):
            best_set = candidate
            # Optional: local search on best_set? 
            # For now, just update best.

    return best_set
