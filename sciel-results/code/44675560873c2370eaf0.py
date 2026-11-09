import time
import random
import numpy as np
from itertools import combinations

N = 512
TIME_LIMIT = 75  # seconds, safety margin before 90s kill

def construct():
    start_time = time.time()
    
    # Precompute all 4-term arithmetic progressions in [1, 512]
    # Store for each element x, the list of other 3 elements in each 4-AP containing x
    aps_containing = {i: [] for i in range(1, N + 1)}
    
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # The AP is (a, b, c, e)
            # For each element, record the other three
            aps_containing[a].append((b, c, e))
            aps_containing[b].append((a, c, e))
            aps_containing[c].append((a, b, e))
            aps_containing[e].append((a, b, c))

    def can_add(x, current_set):
        """Check if adding x to current_set creates a 4-term AP"""
        for others in aps_containing[x]:
            if others[0] in current_set and others[1] in current_set and others[2] in current_set:
                return False
        return True

    def greedy_randomized():
        """Greedy with random ordering"""
        elements = list(range(1, N + 1))
        random.shuffle(elements)
        current_set = set()
        for x in elements:
            if can_add(x, current_set):
                current_set.add(x)
        return current_set

    def local_search(current_set):
        """
        Improved local search:
        1. Try to add any valid element (greedy fill).
        2. If stuck, try swapping: remove a random element, try to add multiple new ones.
        """
        current_set = set(current_set)
        improved = True
        while improved and time.time() - start_time < TIME_LIMIT:
            improved = False
            
            # Phase 1: Greedy fill
            added_any = True
            while added_any and time.time() - start_time < TIME_LIMIT:
                added_any = False
                # Shuffle to avoid bias
                candidates = [x for x in range(1, N + 1) if x not in current_set]
                random.shuffle(candidates)
                for x in candidates:
                    if can_add(x, current_set):
                        current_set.add(x)
                        added_any = True
                        break

            # Phase 2: Swap
            if not added_any:
                # Pick a random element to remove
                if current_set:
                    remove_x = random.choice(list(current_set))
                    trial_set = current_set - {remove_x}
                    # Try to fill with new elements
                    added_any = False
                    candidates = [x for x in range(1, N + 1) if x not in trial_set]
                    random.shuffle(candidates)
                    for x in candidates:
                        if can_add(x, trial_set):
                            trial_set.add(x)
                            added_any = True
                    if len(trial_set) > len(current_set):
                        current_set = trial_set
                        improved = True
                        break

        return current_set

    best_set = set()
    best_size = 0
    
    # Run multiple restarts
    num_restarts = 100 # Can afford more due to speedup
    for i in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        current_set = greedy_randomized()
        current_set = local_search(current_set)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
    
    # If we have time, try a few more targeted improvements
    if time.time() - start_time < TIME_LIMIT - 2:
        # Base-3 starting point
        base3_set = set()
        for i in range(1, N + 1):
            n = i
            valid = True
            while n > 0:
                if n % 3 == 2:
                    valid = False
                    break
                n //= 3
            if valid:
                base3_set.add(i)
        
        if len(base3_set) > best_size:
            best_set = base3_set
            best_size = len(base3_set)
        
        improved_set = local_search(best_set)
        if len(improved_set) > best_size:
            best_set = improved_set
            best_size = len(improved_set)
    
    return sorted(best_set)
