import math
import time

def calc_score(U):
    if not U:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0.0
    # log base 2
    num = math.log(ratio, 2)
    den = math.log(2 * max_u + 1, 2)
    return 1.0 + num / den

def construct():
    start_time = time.time()
    deadline = 140.0
    
    best_U = [0, 1, 3, 9]
    best_score = calc_score(best_U)
    
    # Phase 1: Brute force small M
    # Try M up to 12
    for M in range(1, 13):
        if time.time() > deadline - 10:
            break
        # Generate all subsets of {0, ..., M} containing 0
        # Represent subset as bitmask of 1..M
        for mask in range(1 << M):
            if time.time() > deadline - 10:
                break
            U = [0]
            for i in range(M):
                if (mask >> i) & 1:
                    U.append(i + 1)
            score = calc_score(U)
            if score > best_score:
                best_score = score
                best_U = U[:]
                
    # Phase 2: Local Search on larger M
    # Use best_U as starting point, allow elements up to 100
    current_U = best_U[:]
    current_score = best_score
    max_val = 100
    
    # Simple hill climbing
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try adding
        for x in range(1, max_val + 1):
            if x in current_U:
                continue
            new_U = current_U + [x]
            new_score = calc_score(new_U)
            if new_score > current_score:
                current_U = new_U
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_U = current_U[:]
        
        # Try removing
        for x in current_U:
            if x == 0:
                continue
            new_U = [y for y in current_U if y != x]
            new_score = calc_score(new_U)
            if new_score > current_score:
                current_U = new_U
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_U = current_U[:]
                    
        # Try swapping
        for i in range(len(current_U)):
            if current_U[i] == 0:
                continue
            for x in range(1, max_val + 1):
                if x in current_U:
                    continue
                new_U = current_U[:]
                new_U[i] = x
                new_score = calc_score(new_U)
                if new_score > current_score:
                    current_U = new_U
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_U = current_U[:]
                        
    return best_U
