import time
from math import log
import random
import numpy as np

def score_set(U):
    """Compute the score for a set U."""
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    # Use sets for sums and diffs
    # Optimization: if n is large, this is O(n^2). 
    # For n ~ 50, 2500 ops is fine.
    
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

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_set = [0, 1]
    best_score = score_set(best_set)
    
    # Phase 1: Greedy Construction
    # Start with 0, then add elements one by one to maximize score
    U = [0]
    current_score = score_set(U)
    
    # Candidate pool for new elements
    # We want elements that create sum collisions but not difference collisions.
    # Heuristic: try small integers first, then expand.
    
    max_val = 200
    candidates = list(range(1, max_val + 1))
    random.shuffle(candidates)
    
    # Try to build a set of size ~20-50
    target_size = 30
    
    while len(U) < target_size and time.time() < deadline - 60:
        best_candidate = None
        best_candidate_score = current_score
        
        # Sample a subset of candidates to speed up
        sample_size = min(50, len(candidates))
        sampled = random.sample(candidates, sample_size)
        
        for c in sampled:
            if c in U:
                continue
            new_U = U + [c]
            new_U = sorted(set(new_U))
            s = score_set(new_U)
            if s > best_candidate_score:
                best_candidate_score = s
                best_candidate = c
        
        if best_candidate is not None:
            U.append(best_candidate)
            U.sort()
            # Remove chosen candidate from pool
            candidates.remove(best_candidate)
            current_score = best_candidate_score
            if current_score > best_score:
                best_score = current_score
                best_set = U[:]
        else:
            # If no improvement found in sample, break or try different range
            # Expand range
            new_range = list(range(max_val + 1, max_val * 2 + 1))
            candidates.extend(new_range)
            random.shuffle(candidates)
            if max_val > 1000:
                break

    # Phase 2: Local Search Refinement
    # We will perform local search on the best set found so far.
    # We'll use a simulated annealing approach with fast scoring.
    
    current_U = best_set[:]
    current_score = best_score
    
    T = 1.0
    cooling = 0.999
    
    # Pre-compute some stats if needed? No, just re-evaluate.
    
    while time.time() < deadline - 5:
        if T < 0.001:
            break
            
        # Generate a move
        move_type = random.random()
        new_U = current_U[:]
        
        if move_type < 0.6:
            # Replace an element
            if len(new_U) > 1:
                idx = random.randint(1, len(new_U) - 1)
                old_val = new_U[idx]
                # Try nearby values
                delta = random.randint(-10, 10)
                new_val = old_val + delta
                if new_val > 0 and new_val not in new_U:
                    new_U[idx] = new_val
                    new_U = sorted(set(new_U))
                else:
                    # Try random value in range
                    new_val = random.randint(1, max(new_U) + 20)
                    if new_val not in new_U:
                        new_U[idx] = new_val
                        new_U = sorted(set(new_U))
        
        elif move_type < 0.8:
            # Add an element
            if len(new_U) < 80:
                new_val = random.randint(1, max(new_U) + 50)
                if new_val not in new_U:
                    new_U.append(new_val)
                    new_U = sorted(set(new_U))
        
        else:
            # Remove an element
            if len(new_U) > 2:
                idx = random.randint(1, len(new_U) - 1)
                new_U.pop(idx)
                new_U = sorted(set(new_U))
        
        if len(new_U) < 2 or 0 not in new_U:
            continue
            
        new_score = score_set(new_U)
        
        # Acceptance
        if new_score >= current_score:
            current_U = new_U
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_set = current_U[:]
        else:
            delta = new_score - current_score
            if random.random() < pow(2.718281828, delta / T):
                current_U = new_U
                current_score = new_score
        
        T *= cooling

    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
