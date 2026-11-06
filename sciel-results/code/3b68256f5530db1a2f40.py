import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit to be safe within 60/90s
    
    N = 256
    # Initial set: all odd numbers
    S = set(range(1, N + 1, 2))
    S_list = sorted(S)
    
    # Precompute conflict sets for each number
    # conflict[x] = set of y in S such that (x*y) % (x+y) == 0
    # We'll maintain this dynamically
    
    def can_add(x, s_set):
        for y in s_set:
            if (x * y) % (x + y) == 0:
                return False
        return True
    
    # 1. Greedy Add Evens
    evens = list(range(2, N + 1, 2))
    random.shuffle(evens)
    
    for e in evens:
        if time.time() > deadline:
            break
        if can_add(e, S):
            S.add(e)
    
    best_size = len(S)
    best_S = set(S)
    
    # 2. Simulated Annealing / Local Search
    # We try to swap elements or add/remove to improve size
    # Since we want to maximize cardinality, we focus on adding elements
    # If we can't add any, we try to swap: remove one element, add two others
    
    all_numbers = list(range(1, N + 1))
    random.shuffle(all_numbers)
    
    T = 1.0
    cooling = 0.9995
    
    # Current state
    current_S = set(best_S)
    current_size = len(current_S)
    
    while time.time() < deadline:
        # Try random additions
        added = False
        for x in all_numbers:
            if time.time() > deadline:
                break
            if x not in current_S and can_add(x, current_S):
                current_S.add(x)
                current_size += 1
                added = True
                if current_size > best_size:
                    best_size = current_size
                    best_S = set(current_S)
        
        if not added:
            # Try swaps: remove one, add two
            # Pick a random element to remove
            if current_size == 0:
                break
                
            remove_candidates = list(current_S)
            random.shuffle(remove_candidates)
            
            for rem in remove_candidates[:5]:  # Try a few removals
                if time.time() > deadline:
                    break
                    
                trial_S = set(current_S)
                trial_S.discard(rem)
                
                # Try to add two elements
                additions = []
                for x in all_numbers:
                    if x not in trial_S and can_add(x, trial_S):
                        additions.append(x)
                        if len(additions) >= 2:
                            break
                
                if len(additions) >= 2:
                    # Accept swap
                    current_S = trial_S
                    for x in additions[:2]:
                        current_S.add(x)
                    current_size = len(current_S)
                    if current_size > best_size:
                        best_size = current_size
                        best_S = set(current_S)
                else:
                    # Maybe accept with probability based on energy (size change)
                    # But since we want max size, only accept if size doesn't decrease
                    # Or use simulated annealing to escape local optima
                    pass
        
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    # Final greedy pass
    for x in all_numbers:
        if time.time() > deadline:
            break
        if x not in best_S and can_add(x, best_S):
            best_S.add(x)
    
    return list(best_S)
