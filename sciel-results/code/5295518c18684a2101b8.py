import time
import random
from math import log

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    sums = set()
    diffs = set()
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
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def local_search(start_set, deadline, max_iter=5000):
    current = sorted(set(start_set))
    if 0 not in current:
        current = sorted(set(current + [0]))
    current_score = score_set(current)
    if current_score < 0:
        return current, -1
    
    best_set = current[:]
    best_score = current_score
    
    while time.time() < deadline - 2 and max_iter > 0:
        max_iter -= 1
        # Pick a random element to change (except 0)
        if len(current) <= 1:
            break
            
        i = random.randint(1, len(current) - 1)
        old_val = current[i]
        
        # Generate a new value
        # Strategy: small perturbation or jump
        if random.random() < 0.5:
            # Small perturbation
            delta = random.randint(-5, 5)
            new_val = old_val + delta
        else:
            # Jump
            new_val = random.randint(1, current[-1] + 10)
            
        if new_val <= 0:
            continue
            
        new_set = current[:]
        new_set[i] = new_val
        new_set = sorted(set(new_set))
        
        if len(new_set) != len(current):
            # If size changed, we might want to keep size fixed or allow it?
            # The problem asks for a set U. Size is not fixed.
            # However, changing size changes the ratio significantly.
            # Let's allow size change but be careful.
            pass
            
        new_score = score_set(new_set)
        
        if new_score > best_score:
            best_score = new_score
            best_set = new_set[:]
            
        # Accept if better or with some probability (Simulated Annealing)
        if new_score > current_score:
            current = new_set
            current_score = new_score
        else:
            # Accept with probability exp((new_score - current_score) / T)
            # T should decrease
            # Simple: accept if close
            if new_score > current_score - 0.01:
                if random.random() < 0.2:
                    current = new_set
                    current_score = new_score
                    
    return best_set, best_score

def construct():
    deadline = time.time() + 110
    
    best_set = [0, 1, 4]
    best_score = score_set(best_set)
    
    # Seeds
    seeds = [
        [0, 1, 4],
        [0, 1, 4, 16],
        [0, 1, 3, 9, 27],
        [0, 1, 2, 4, 8, 16], # Not Sidon, but try
        [0, 1, 3, 7, 15],
        [0, 1, 4, 9, 16], # Squares
        [0, 1, 5, 11, 25],
    ]
    
    # Add random seeds
    random.seed(42)
    for _ in range(10):
        n = random.randint(3, 6)
        U = [0]
        max_val = random.randint(10, 50)
        for _ in range(n-1):
            U.append(random.randint(1, max_val))
        seeds.append(sorted(set(U)))
        
    for seed in seeds:
        if time.time() > deadline - 10:
            break
        local_best, local_score = local_search(seed, deadline, max_iter=2000)
        if local_score > best_score:
            best_score = local_score
            best_set = local_best[:]
            
    return best_set
