import time
import random
from math import log
import itertools

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
        
    sums = set()
    diffs = set()
    
    # Optimization: use local variables for speed
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def construct():
    deadline = time.time() + 110  # Safety margin
    best_score = -1
    best_set = [0, 1, 4, 9, 11]
    
    # Update best if initial is good
    s = score_set(best_set)
    if s > best_score:
        best_score = s
        best_set = list(best_set)
        
    # Try some structured sets
    structured_candidates = [
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81], # Squares
        [0, 1, 2, 4, 8, 16, 32, 64], # Powers of 2
        [0, 1, 3, 7, 15, 31, 63], # Mersenne-like
        [0, 1, 4, 10, 20, 35, 56], # Triangular
    ]
    for cand in structured_candidates:
        s = score_set(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Exhaustive search for small sets
    max_val_exhaustive = 50
    for n in range(3, 6): # Sizes 3 to 5
        if time.time() > deadline:
            break
        for combo in itertools.combinations(range(1, max_val_exhaustive + 1), n - 1):
            if time.time() > deadline:
                break
            U = (0,) + combo
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
                
    # Local search for larger sets or wider range
    # We will perform random restarts with local search
    max_n_local = 12
    max_val_local = 200
    iterations_per_restart = 100
    
    # Seed random for reproducibility? No, let it vary or fix it.
    # random.seed(42) 
    
    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1
        # Choose a size
        n = random.randint(4, max_n_local)
        # Generate a random set
        # Ensure 0 is included
        # Generate n-1 random distinct integers from 1..max_val_local
        if n - 1 > max_val_local:
            continue
        elements = random.sample(range(1, max_val_local + 1), n - 1)
        U = [0] + elements
        U.sort()
        
        current_score = score_set(U)
        
        # Local search
        for _ in range(iterations_per_restart):
            if time.time() > deadline - 5:
                break
            
            # Pick a random element to perturb (not 0)
            idx = random.randint(1, n - 1)
            old_val = U[idx]
            
            # Perturbation: change to a nearby value or random value
            if random.random() < 0.5:
                # Small step
                step = random.choice([-2, -1, 1, 2, 5, 10])
                new_val = old_val + step
            else:
                # Large jump
                new_val = random.randint(1, max_val_local)
                
            if new_val < 1 or new_val > max_val_local:
                continue
            if new_val in U:
                continue
                
            U[idx] = new_val
            U.sort()
            
            new_score = score_set(U)
            
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(U)
            else:
                # Revert
                U[idx] = old_val
                U.sort()
                
    return best_set
