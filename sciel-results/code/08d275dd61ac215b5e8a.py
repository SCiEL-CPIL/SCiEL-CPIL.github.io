N = 256

import random
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 25.0  # Leave 5s buffer
    
    def greedy_randomized():
        # Shuffle candidates
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        selected = []
        sums = set()
        
        for x in candidates:
            # Check if x can be added
            added_sums = [x + a for a in selected] + [2 * x]
            # Optimization: check intersection first
            if sums.intersection(added_sums):
                continue
            if len(set(added_sums)) != len(added_sums):
                continue
            
            selected.append(x)
            sums.update(added_sums)
            
        return selected

    best_set = None
    best_size = float('inf')
    
    # Run several restarts
    while time.time() < deadline:
        current_set = greedy_randomized()
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = current_set
            
    # Fallback to deterministic if time ran out or no result (unlikely)
    if best_set is None:
        # Deterministic ascending greedy
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        best_set = selected

    return best_set
# EVOLVE-BLOCK-END
