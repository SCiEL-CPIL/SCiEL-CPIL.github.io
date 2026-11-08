import time
from math import log
import random

def score_set(U):
    """
    Computes the score for a set U.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U)+1)
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    # Optimize loops
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def local_search(U, deadline, max_val=200, max_n=30):
    """
    Local search to improve the set U.
    """
    U = sorted(set(U))
    current_score = score_set(U)
    best_score = current_score
    best_set = list(U)
    
    no_improvement = 0
    max_no_improvement = 50
    
    while time.time() < deadline - 1:
        if no_improvement > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        if move_type < 0.4:
            # Add element
            if len(U) < max_n:
                # Choose a candidate
                # Try values near existing elements or random
                if random.random() < 0.5:
                    base = random.choice(U)
                    candidate = base + random.randint(1, 10)
                else:
                    candidate = random.randint(1, max_val)
                
                if candidate not in U and candidate > 0:
                    new_U = U + [candidate]
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
        elif move_type < 0.7:
            # Remove element
            if len(U) > 2:
                removable = [x for x in U if x != 0]
                if removable:
                    candidate = random.choice(removable)
                    new_U = [x for x in U if x != candidate]
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
        else:
            # Replace element
            if len(U) > 1:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                # Try nearby values
                for delta in range(-5, 6):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val <= 0 or new_val in U:
                        continue
                    new_U = U.copy()
                    new_U[idx] = new_val
                    new_U.sort()
                    new_score = score_set(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        break
        
        if improved:
            no_improvement = 0
            if current_score > best_score:
                best_score = current_score
                best_set = list(U)
        else:
            no_improvement += 1
            
    return best_score, best_set

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Initial seeds
    seeds = [
        [0, 1, 4, 9, 16, 25, 36],
        [0, 1, 2, 4, 8, 16, 32],
        [0, 1, 3, 7, 15, 31, 63],
        [0, 1, 4, 10, 20, 35, 56],
        [0, 1, 3, 9, 27, 81],
    ]
    
    # Add some random seeds
    for _ in range(5):
        n = random.randint(5, 15)
        U = {0}
        current_max = 0
        for _ in range(n - 1):
            # Geometric growth
            current_max = current_max * random.randint(2, 3) + random.randint(1, 3)
            U.add(current_max)
        seeds.append(sorted(U))
        
    best_score = -1
    best_set = [0, 1]
    
    # Evaluate seeds
    for seed in seeds:
        s = score_set(seed)
        if s > best_score:
            best_score = s
            best_set = seed
            
    # Local search loop
    # Use the best found so far as primary seed
    seeds = [best_set] + seeds[:3]
    
    while time.time() < deadline - 5:
        # Pick a seed
        seed = random.choice(seeds)
        s, new_set = local_search(seed, deadline)
        if s > best_score:
            best_score = s
            best_set = new_set
            # Update seeds list
            if len(seeds) < 5:
                seeds.append(new_set)
            else:
                # Replace worst seed? Or just keep best
                seeds[-1] = new_set
                
    # Final check
    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
