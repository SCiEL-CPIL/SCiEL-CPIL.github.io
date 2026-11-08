import time
import math
import random

def compute_score(U):
    """
    Computes the lower bound exponent: 1 + log(|U-U|/|U+U|)/log(2 max U + 1)
    """
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    
    # For small to medium U, set operations are fine.
    sum_set = set()
    diff_set = set()
    
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
            
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0
    return 1 + log_ratio / log_denom

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s limit
    
    best_score = -1
    best_U = [0]
    
    def try_update(U_candidate):
        nonlocal best_score, best_U
        if not U_candidate:
            return
        if 0 not in U_candidate:
            U_candidate = list(set(U_candidate) | {0})
        U_candidate = sorted(set(U_candidate))
        if len(U_candidate) < 2:
            return
        s = compute_score(U_candidate)
        if s > best_score:
            best_score = s
            best_U = U_candidate[:]
            
    # 1. Try specific known good patterns and small structured sets
    
    # Powers of 2
    for k in range(2, 30):
        if time.time() > deadline: break
        U = [2**i for i in range(k)]
        if 0 not in U: U.insert(0, 0)
        try_update(U)
        
    # Powers of 3
    for k in range(2, 20):
        if time.time() > deadline: break
        U = [3**i for i in range(k)]
        if 0 not in U: U.insert(0, 0)
        try_update(U)
        
    # Mersenne-like: 2^k - 1
    for k in range(2, 20):
        if time.time() > deadline: break
        U = [2**i - 1 for i in range(k)]
        U = sorted(set(U) | {0})
        try_update(U)
        
    # Small exhaustive search for M <= 15 (subsets of {0..M} containing 0)
    # This might be too slow if we do 2^15, but let's try up to M=12 (2^12=4096 subsets)
    for M in range(2, 13):
        if time.time() > deadline: break
        elements = list(range(1, M+1))
        n = len(elements)
        for mask in range(1 << n):
            if time.time() > deadline: break
            U = [0]
            for i in range(n):
                if mask & (1 << i):
                    U.append(elements[i])
            try_update(U)
            
    # 2. Local Search (Hill Climbing)
    # Start from the best found so far
    current_U = best_U[:] if best_U else [0, 1]
    if 0 not in current_U: current_U.append(0)
    current_score = compute_score(current_U)
    
    random.seed(42)
    max_val_bound = 1000 # Search space for new elements
    
    # To allow exploring larger M if it helps, but usually small M is better.
    # We'll bias towards small M initially, then expand.
    
    iterations = 0
    while time.time() < deadline - 5:
        iterations += 1
        if iterations % 1000 == 0 and time.time() - start_time > 120:
            break
            
        # Generate a neighbor
        # Moves:
        # 1. Add a random element x in [1, max_val_bound]
        # 2. Remove a random non-zero element
        # 3. Replace a random non-zero element with another
        
        move_type = random.random()
        
        if move_type < 0.4 and len(current_U) < 50:
            # Add
            # Bias towards smaller numbers? Or uniform?
            # Let's try uniform in [1, max_val_bound] but also some local moves
            if random.random() < 0.5:
                new_val = random.randint(1, max_val_bound)
            else:
                # Local move: near existing elements
                if len(current_U) > 1:
                    base = random.choice(current_U)
                    offset = random.randint(-10, 10)
                    new_val = base + offset
                    if new_val < 0: new_val = 0
            if new_val not in current_U:
                candidate = current_U + [new_val]
                s = compute_score(candidate)
                if s > current_score:
                    current_U = candidate
                    current_score = s
                    if current_score > best_score:
                        best_score = current_score
                        best_U = current_U[:]
                        
        elif move_type < 0.7 and len(current_U) > 1:
            # Remove
            idx = random.randint(1, len(current_U)-1) # Don't remove 0
            if current_U[idx] != 0:
                candidate = current_U[:]
                candidate.pop(idx)
                s = compute_score(candidate)
                if s > current_score:
                    current_U = candidate
                    current_score = s
                    if current_score > best_score:
                        best_score = current_score
                        best_U = current_U[:]
                        
        else:
            # Replace
            if len(current_U) > 1:
                idx = random.randint(1, len(current_U)-1)
                if current_U[idx] != 0:
                    if random.random() < 0.5:
                        new_val = random.randint(1, max_val_bound)
                    else:
                        base = random.choice(current_U)
                        offset = random.randint(-20, 20)
                        new_val = base + offset
                        if new_val < 0: new_val = 0
                    if new_val not in current_U:
                        candidate = current_U[:]
                        candidate[idx] = new_val
                        s = compute_score(candidate)
                        if s > current_score:
                            current_U = candidate
                            current_score = s
                            if current_score > best_score:
                                best_score = current_score
                                best_U = current_U[:]
        
        # Occasionally expand search space
        if iterations % 100 == 0:
             max_val_bound = min(max_val_bound * 2, 10000)
             # Also try shrinking if we've been stuck?
             if random.random() < 0.1:
                 max_val_bound = max(100, max_val_bound // 2)
                 
    return best_U

if __name__ == "__main__":
    print(construct())
