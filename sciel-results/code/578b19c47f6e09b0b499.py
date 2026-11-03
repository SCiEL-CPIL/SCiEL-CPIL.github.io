import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    best_set = [0, 1, 3]
    best_score = 1.079 # Approx seed score
    
    def score_set(U):
        if not U or 0 not in U:
            return 0
        U = sorted(set(U))
        if U[0] != 0:
            return 0
        max_u = U[-1]
        if max_u == 0:
            return 1.0
        sums = set()
        diffs = set()
        for u in U:
            for v in U:
                sums.add(u + v)
                diffs.add(u - v)
        ratio = len(diffs) / len(sums)
        if ratio <= 1:
            return 1.0
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1.0
        return 1.0 + math.log(ratio) / denom

    current_score = score_set(best_set)
    
    # Try some known good structures
    candidates = [
        [0, 1, 3],
        [0, 1, 3, 9],
        [0, 1, 3, 9, 27],
        [0, 1, 2, 4],
        [0, 1, 2, 4, 8],
        [0, 1, 4, 10], # Random
        [0, 1, 3, 7],
        [0, 1, 4, 9],
        [0, 1, 2, 5, 11],
    ]
    
    for c in candidates:
        if time.time() > deadline:
            break
        s = score_set(c)
        if s > current_score:
            current_score = s
            best_set = c
    
    # Local search
    # Randomly add or remove elements
    for _ in range(500):
        if time.time() > deadline:
            break
        U = best_set[:]
        if random.random() < 0.5 and len(U) < 8:
            # Add element
            max_u = U[-1]
            new_u = random.randint(1, max_u * 2 + 10)
            if new_u not in U:
                U.append(new_u)
        elif len(U) > 2:
            # Remove element (not 0)
            idx = random.randint(1, len(U)-1)
            U.pop(idx)
        
        U = sorted(set(U))
        s = score_set(U)
        if s > current_score:
            current_score = s
            best_set = U
            
    return best_set
