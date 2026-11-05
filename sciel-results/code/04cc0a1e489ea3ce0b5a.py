import math
import time

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    # Precompute weights
    weights = [1.0 / (512 - a) for a in range(512)]
    
    # Greedy initialization
    chosen = []
    for a in range(511, 0, -1):
        if all(math.gcd(a, b) == 1 for b in chosen):
            chosen.append(a)
            
    best_set = chosen[:]
    best_score = sum(weights[a] for a in chosen)
    
    # Local search: try removing one element and re-greedy-filling
    improved = True
    while improved:
        improved = False
        for i, a in enumerate(best_set):
            # Try removing a
            remaining = [x for j, x in enumerate(best_set) if j != i]
            # Re-greedy fill from 511 down to 1, skipping those in remaining
            new_set = remaining[:]
            for b in range(511, 0, -1):
                if b in new_set:
                    continue
                if all(math.gcd(b, c) == 1 for c in new_set):
                    new_set.append(b)
            
            new_score = sum(weights[b] for b in new_set)
            if new_score > best_score:
                best_score = new_score
                best_set = new_set
                improved = True
                # Restart loop to check for further improvements
                break
        
        if time.time() > deadline:
            break
            
    return best_set
