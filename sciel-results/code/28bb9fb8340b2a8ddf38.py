"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import log
import sys

_start_time = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _compute_score(U):
    """Compute the score for a given set U (must contain 0, nonnegative ints)."""
    U = sorted(set(U))
    if 0 not in U:
        U = sorted([0] + U)
    n = len(U)
    if n < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Compute sumset U+U
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    
    # Compute difference set U-U
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0.0
    
    denom = log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    
    return 1.0 + log(ratio) / denom

def construct():
    global _start_time
    best_score = 0.0
    best_set = [0, 1]
    
    # Helper to update best
    def update_best(U, score):
        nonlocal best_score, best_set
        if score > best_score:
            best_score = score
            best_set = list(U)

    # 1. Exhaustive/Structured search for small n
    # Try specific structured sets first
    structured_sets = []
    
    # Squares
    for n in range(3, 15):
        structured_sets.append([i*i for i in range(n)])
        
    # Cubes
    for n in range(3, 10):
        structured_sets.append([i**3 for i in range(n)])
        
    # Powers of 2
    for n in range(3, 15):
        structured_sets.append([1 << i for i in range(n)])
        
    # Powers of 3
    for n in range(3, 10):
        structured_sets.append([3**i for i in range(n)])
        
    # Fibonacci
    fibs = [0, 1]
    for i in range(20):
        fibs.append(fibs[-1] + fibs[-2])
    for n in range(3, 12):
        structured_sets.append(fibs[:n])
        
    # Triangular
    for n in range(3, 15):
        structured_sets.append([i*(i+1)//2 for i in range(n)])
        
    # Random-ish structured
    random.seed(42)
    for _ in range(50):
        n = random.randint(3, 12)
        U = [0]
        current = 1
        for i in range(1, n):
            current += random.randint(1, 5)
            U.append(current)
        structured_sets.append(U)

    for U in structured_sets:
        s = _compute_score(U)
        update_best(U, s)

    # 2. Local Search / Hill Climbing
    # We will run local search from multiple starting points.
    # Mutation strategies:
    # - Add a new random element
    # - Remove a random non-zero element
    # - Replace a random non-zero element with a new one
    
    def local_search(start_U, max_iters=5000):
        current_U = list(start_U)
        current_score = _compute_score(current_U)
        update_best(current_U, current_score)
        
        no_improve_count = 0
        for i in range(max_iters):
            if time.time() - _start_time > TIME_LIMIT:
                return
            
            # Generate a neighbor
            neighbor = list(current_U)
            max_val = max(neighbor) if neighbor else 0
            
            # Choose mutation type
            mut_type = random.randint(0, 2)
            
            if mut_type == 0 and len(neighbor) < 15:
                # Add element
                # Try to add an element that might increase diffset/sumset ratio
                # Usually adding a large gap helps difference set uniqueness
                # But increases max_u
                # Try small additions first, then larger
                added = False
                for _ in range(10):
                    if random.random() < 0.5:
                        new_val = random.randint(1, max_val + 5)
                    else:
                        new_val = random.randint(max_val + 1, max_val + 20)
                    
                    if new_val not in neighbor:
                        neighbor.append(new_val)
                        neighbor.sort()
                        added = True
                        break
                if not added:
                    continue
                    
            elif mut_type == 1 and len(neighbor) > 2:
                # Remove element
                idx = random.randint(1, len(neighbor) - 1) # Keep 0
                neighbor.pop(idx)
                
            else:
                # Replace element
                if len(neighbor) < 2:
                    continue
                idx = random.randint(1, len(neighbor) - 1)
                old_val = neighbor[idx]
                # New value
                if random.random() < 0.5:
                    new_val = random.randint(1, max_val + 5)
                else:
                    new_val = random.randint(1, max_val * 2)
                    
                if new_val == old_val:
                    continue
                if new_val in neighbor:
                    continue
                    
                neighbor[idx] = new_val
                neighbor.sort()

            # Evaluate
            new_score = _compute_score(neighbor)
            
            if new_score > current_score:
                current_U = neighbor
                current_score = new_score
                no_improve_count = 0
                update_best(current_U, current_score)
            else:
                no_improve_count += 1
                # Simple simulated annealing step? Or just accept if close?
                # For now, pure hill climbing with random restarts is safer.
                # But to escape local minima, we can occasionally accept worse moves.
                if no_improve_count > 100 and random.random() < 0.1:
                    current_U = neighbor
                    current_score = new_score
                    no_improve_count = 0

    # Run local search from best found so far
    local_search(best_set)
    
    # Run local search from random starts
    random.seed(123)
    starts = 20
    for i in range(starts):
        if time.time() - _start_time > TIME_LIMIT:
            break
            
        n = random.randint(3, 12)
        U = [0]
        current = 0
        for j in range(1, n):
            # Generate sparse set
            step = random.randint(1, 5)
            current += step
            U.append(current)
            
        local_search(U, max_iters=2000)

    # Final polish: Try to optimize the best set found
    # Try specific tweaks on best_set
    current_best = list(best_set)
    improved = True
    while improved and time.time() - _start_time < TIME_LIMIT:
        improved = False
        max_val = max(current_best)
        
        # Try adding small values
        for v in range(1, max_val + 10):
            if v not in current_best:
                cand = sorted(current_best + [v])
                s = _compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = cand
                    current_best = cand
                    improved = True
                    
        # Try removing
        for i in range(1, len(current_best)):
            cand = current_best[:i] + current_best[i+1:]
            if len(cand) >= 2:
                s = _compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_set = cand
                    current_best = cand
                    improved = True

    return sorted(set(best_set))

# EVOLVE-BLOCK-END
