import time
import random
import math
from itertools import combinations

def get_score(U):
    if not U:
        return 0.0
    U = sorted(set(U))
    if 0 not in U:
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
    
    len_sum = len(sum_set)
    len_diff = len(diff_set)
    
    if len_sum == 0:
        return 0.0
        
    ratio = len_diff / len_sum
    if ratio <= 1:
        # Score will be <= 1, but we want to maximize. 
        # If ratio < 1, log is negative.
        pass
        
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
        
    score = 1 + math.log(ratio) / denom
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Initial candidates
    candidates = [
        [0, 1, 3],
        [0, 1, 3, 7],
        [0, 1, 3, 7, 15],
        [0, 1, 3, 7, 15, 31],
        [0, 1, 3, 9],
        [0, 1, 3, 9, 27]
    ]
    
    for c in candidates:
        s = get_score(c)
        if s > best_score:
            best_score = s
            best_set = c[:]

    # Local search / Hill Climbing
    # We will try to improve best_set
    
    current_set = best_set[:]
    current_score = best_score
    
    # Define neighbors
    def get_neighbors(U, max_val_limit=100):
        neighbors = []
        U = sorted(U)
        # Add element
        for i in range(1, max_val_limit + 1):
            if i not in U:
                new_U = U + [i]
                new_U.sort()
                neighbors.append(new_U)
        # Remove element (keep 0)
        for i in range(1, len(U)):
            new_U = U[:i] + U[i+1:]
            neighbors.append(new_U)
        # Change element
        for i in range(1, len(U)):
            old_val = U[i]
            for new_val in range(1, max_val_limit + 1):
                if new_val != old_val and new_val not in U:
                    new_U = U[:i] + [new_val] + U[i+1:]
                    new_U.sort()
                    neighbors.append(new_U)
        return neighbors

    # Simulated Annealing or Hill Climbing
    # Given the small search space for small n, Hill Climbing with random restarts is good.
    
    def local_search(start_set, max_iter=5000):
        current = start_set[:]
        current_s = get_score(current)
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            neighbors = get_neighbors(current, max_val_limit=50)
            # Shuffle to avoid bias
            random.shuffle(neighbors)
            best_neighbor = None
            best_neighbor_score = current_s
            for n in neighbors:
                ns = get_score(n)
                if ns > best_neighbor_score:
                    best_neighbor_score = ns
                    best_neighbor = n
                if time.time() > deadline:
                    break
            if best_neighbor:
                current = best_neighbor
                current_s = best_neighbor_score
            else:
                break
        return current, current_s

    # Run local search from best initial candidate
    if time.time() < deadline:
        res_set, res_score = local_search(best_set)
        if res_score > best_score:
            best_score = res_score
            best_set = res_set

    # Try other starting points if time permits
    other_starts = [
        [0, 1, 3, 7, 15, 31, 63],
        [0, 1, 3, 7, 15, 31, 63, 127],
        [0, 1, 4, 10, 22], # Random-ish
        [0, 1, 3, 6, 12]
    ]
    
    for start in other_starts:
        if time.time() > deadline - 10:
            break
        res_set, res_score = local_search(start)
        if res_score > best_score:
            best_score = res_score
            best_set = res_set

    return best_set
