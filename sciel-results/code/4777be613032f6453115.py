import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer for safety
    
    # Initial prescribed elements
    initial = [1, 2, 5, 11]
    used_sums = set()
    for i in range(len(initial)):
        for j in range(i, len(initial)):
            used_sums.add(initial[i] + initial[j])
    
    best_set = initial[:]
    best_size = len(initial)
    
    def can_add(x, S, sums):
        # Check if 2*x conflicts
        if 2 * x in sums:
            return False
        # Check if x + y conflicts for any y in S
        for y in S:
            if x + y in sums:
                return False
        return True

    def backtrack(S, sums, start_idx, candidates_order):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return
        
        if len(S) > best_size:
            best_set = S[:]
            best_size = len(S)
            
        # Filter valid candidates from the provided order that are >= start_idx
        # We iterate through the order and pick valid ones.
        # To avoid re-computing can_add for invalid ones repeatedly in deep recursion,
        # we just check on the fly.
        
        for x in candidates_order:
            if x < start_idx:
                continue
                
            if time.time() > deadline:
                return
            
            if can_add(x, S, sums):
                new_S = S + [x]
                new_sums = sums | set()
                for y in S:
                    new_sums.add(x + y)
                new_sums.add(2 * x)
                
                backtrack(new_S, new_sums, x + 1, candidates_order)
                
                if time.time() > deadline:
                    return

    # First pass: Deterministic increasing order
    candidates = list(range(12, 285))
    backtrack(initial, used_sums, 12, candidates)
    
    # If we have time, try a few random restarts to escape local optima
    # The increasing order is a strong heuristic, but randomization can find better packings
    if time.time() < deadline - 5:
        for _ in range(10):
            if time.time() > deadline:
                break
            random.shuffle(candidates)
            backtrack(initial, used_sums, 12, candidates)
            
    return best_set
