import time
import random
import numpy as np

def construct():
    n = 512
    start_time = time.time()
    time_limit = 57.0  # Leave 3s buffer before 60s kill
    
    # Precompute all 4-APs in [1, n]
    # For each x, store the list of (y1, y2, y3) such that {x, y1, y2, y3} is a 4-AP
    conflict_triples = [None] * (n + 1)
    for x in range(1, n + 1):
        triples = []
        # Case 1: x is first term: x, x+d, x+2d, x+3d
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            triples.append((x + d, x + 2 * d, x + 3 * d))
        
        # Case 2: x is second term: x-d, x, x+d, x+2d
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            triples.append((x - d, x + d, x + 2 * d))
        
        # Case 3: x is third term: x-2d, x-d, x, x+d
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            triples.append((x - 2 * d, x - d, x + d))
        
        # Case 4: x is fourth term: x-3d, x-2d, x-d, x
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            triples.append((x - 3 * d, x - 2 * d, x - d))
        
        conflict_triples[x] = triples
    
    # Helper: Check if adding x to set S creates a 4-AP using precomputed triples
    def creates_4ap(S, x):
        for (a, b, c) in conflict_triples[x]:
            if a in S and b in S and c in S:
                return True
        return False
    
    # Helper: Greedily fill the set given a starting set
    def greedy_fill(S, order=None):
        if order is None:
            order = list(range(1, n + 1))
        S = set(S)
        for x in order:
            if x not in S:
                if not creates_4ap(S, x):
                    S.add(x)
        return S

    # Initial Greedy Construction
    best_set = set()
    best_size = 0
    
    random.seed(42)
    
    # Standard order
    current = greedy_fill(set(), list(range(1, n + 1)))
    if len(current) > best_size:
        best_set = set(current)
        best_size = len(current)
        
    # Reverse order
    current = greedy_fill(set(), list(range(n, 0, -1)))
    if len(current) > best_size:
        best_set = set(current)
        best_size = len(current)
        
    # Random order - generate many diverse seeds
    num_random_tries = 100
    for _ in range(num_random_tries):
        if time.time() - start_time > time_limit * 0.3:  # 30% time for init
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        current = greedy_fill(set(), order)
        if len(current) > best_size:
            best_set = set(current)
            best_size = len(current)

    # Local Search
    # Randomized Hill Climbing with simulated annealing-like kicks
    current_set = set(best_set)
    
    while time.time() - start_time < time_limit:
        # Try to add
        added = False
        candidates = [i for i in range(1, n + 1) if i not in current_set]
        
        if not candidates:
            break
        
        # Try a few random candidates
        random.shuffle(candidates)
        for x in candidates[:100]:  # Check up to 100 random candidates
            if not creates_4ap(current_set, x):
                current_set.add(x)
                added = True
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(current_set)
                break
        
        if not added:
            # Stuck. Perform a kick.
            # Remove 1-3 random elements
            k = random.randint(1, 3)
            if len(current_set) > k:
                to_remove = random.sample(list(current_set), k)
                for x in to_remove:
                    current_set.remove(x)
                
                # Now try to add as many as possible
                # Greedy fill from current state
                candidates = [i for i in range(1, n + 1) if i not in current_set]
                random.shuffle(candidates)
                for x in candidates:
                    if not creates_4ap(current_set, x):
                        current_set.add(x)
                
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(current_set)

    return sorted(best_set)
