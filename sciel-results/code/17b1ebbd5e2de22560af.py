"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import log
import sys

_start_time = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s kill

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
    
    # Compute sumset U+U using sorted list for efficiency
    sumset = set()
    for i, a in enumerate(U):
        for j in range(i, n):
            s = a + U[j]
            sumset.add(s)
            if i != j:
                sumset.add(U[j] + a)
    
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
    for n in range(3, 20):
        structured_sets.append([i*i for i in range(n)])
        
    # Cubes
    for n in range(3, 12):
        structured_sets.append([i**3 for i in range(n)])
        
    # Powers of 2
    for n in range(3, 18):
        structured_sets.append([1 << i for i in range(n)])
        
    # Powers of 3
    for n in range(3, 12):
        structured_sets.append([3**i for i in range(n)])
        
    # Powers of 5
    for n in range(3, 10):
        structured_sets.append([5**i for i in range(n)])
        
    # Fibonacci
    fibs = [0, 1]
    for i in range(25):
        fibs.append(fibs[-1] + fibs[-2])
    for n in range(3, 15):
        structured_sets.append(fibs[:n])
        
    # Triangular
    for n in range(3, 18):
        structured_sets.append([i*(i+1)//2 for i in range(n)])
        
    # Primes
    def is_prime(x):
        if x < 2: return False
        if x < 4: return True
        if x % 2 == 0 or x % 3 == 0: return False
        i = 5
        while i * i <= x:
            if x % i == 0 or x % (i+2) == 0: return False
            i += 6
        return True
    primes = [0] + [p for p in range(2, 200) if is_prime(p)]
    for n in range(3, 15):
        structured_sets.append(primes[:n])
        
    # Gaps: 0, 1, 1+g, 1+g+2g, ...
    for g in range(2, 15):
        for n in range(3, 12):
            structured_sets.append([0] + [1 + i*g for i in range(n-1)])
    
    # Random-ish structured
    random.seed(42)
    for _ in range(100):
        n = random.randint(3, 15)
        U = [0]
        current = 1
        for i in range(1, n):
            current += random.randint(1, 8)
            U.append(current)
        structured_sets.append(U)

    for U in structured_sets:
        s = _compute_score(U)
        update_best(U, s)

    # 2. Local Search / Hill Climbing with simulated annealing
    def local_search(start_U, max_iters=8000):
        current_U = list(start_U)
        current_score = _compute_score(current_U)
        update_best(current_U, current_score)
        
        no_improve_count = 0
        temperature = 0.01
        for i in range(max_iters):
            if time.time() - _start_time > TIME_LIMIT:
                return
            
            # Generate a neighbor
            neighbor = list(current_U)
            max_val = max(neighbor) if neighbor else 0
            n = len(neighbor)
            
            # Choose mutation type with weighted probabilities
            r = random.random()
            if r < 0.35:
                # Replace element - most likely to improve
                if n < 2:
                    continue
                idx = random.randint(1, n - 1)
                old_val = neighbor[idx]
                # Try values near existing elements or at strategic positions
                if random.random() < 0.4:
                    # Near an existing element
                    ref = random.choice(neighbor[1:])
                    new_val = ref + random.randint(-5, 5)
                elif random.random() < 0.5:
                    # Small value
                    new_val = random.randint(1, max(1, max_val // 3))
                else:
                    # Larger value to expand range
                    new_val = random.randint(max_val, max_val + max(5, max_val // 2))
                
                new_val = max(1, new_val)
                if new_val == old_val or new_val in neighbor:
                    # Try a few more times
                    for _ in range(3):
                        new_val = old_val + random.randint(-10, 10)
                        new_val = max(1, new_val)
                        if new_val not in neighbor:
                            break
                    else:
                        continue
                neighbor[idx] = new_val
                neighbor.sort()
                
            elif r < 0.60:
                # Add element
                if n >= 18:
                    continue
                # Add at strategic position
                if random.random() < 0.3:
                    new_val = max_val + random.randint(1, 10)
                elif random.random() < 0.5:
                    # Fill a gap
                    gaps = []
                    for j in range(1, n-1):
                        if neighbor[j+1] - neighbor[j] > 2:
                            gaps.append(random.randint(neighbor[j]+1, neighbor[j+1]-1))
                    if gaps:
                        new_val = random.choice(gaps)
                    else:
                        new_val = random.randint(1, max_val + 5)
                else:
                    new_val = random.randint(1, max_val + 5)
                
                new_val = max(1, new_val)
                if new_val in neighbor:
                    continue
                neighbor.append(new_val)
                neighbor.sort()
                
            elif r < 0.80:
                # Remove element
                if n <= 3:
                    continue
                idx = random.randint(1, n - 1)
                neighbor.pop(idx)
                
            else:
                # Swap two elements
                if n < 3:
                    continue
                idx1 = random.randint(1, n - 1)
                idx2 = random.randint(1, n - 1)
                if idx1 == idx2:
                    continue
                neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
                neighbor.sort()

            # Evaluate
            new_score = _compute_score(neighbor)
            
            if new_score > current_score:
                current_U = neighbor
                current_score = new_score
                no_improve_count = 0
                temperature = min(0.01, temperature * 1.1)
                update_best(current_U, current_score)
            else:
                no_improve_count += 1
                # Simulated annealing: accept worse moves with decreasing probability
                delta = current_score - new_score
                if delta < temperature and random.random() < 0.3:
                    current_U = neighbor
                    current_score = new_score
                temperature = max(0.001, temperature * 0.999)

    # Run local search from best found so far with more iterations
    local_search(best_set, max_iters=10000)
    
    # Run local search from multiple diverse starts
    random.seed(123)
    starts = 30
    for i in range(starts):
        if time.time() - _start_time > TIME_LIMIT:
            break
            
        # Diverse starting points
        strategy = i % 5
        if strategy == 0:
            # Sparse arithmetic progression
            n = random.randint(4, 14)
            step = random.randint(2, 10)
            U = [i * step for i in range(n)]
        elif strategy == 1:
            # Geometric-like
            n = random.randint(4, 12)
            U = [0]
            current = 1
            for j in range(1, n):
                current *= random.randint(2, 4)
                U.append(current)
        elif strategy == 2:
            # Mixed: some small, some large
            n = random.randint(5, 15)
            U = [0]
            small = random.randint(1, 5)
            U.append(small)
            current = small
            for j in range(2, n):
                current += random.randint(3, 15)
                U.append(current)
        elif strategy == 3:
            # Gaps with occasional small jumps
            n = random.randint(5, 15)
            U = [0, 1]
            current = 1
            for j in range(2, n):
                if random.random() < 0.2:
                    current += random.randint(1, 3)
                else:
                    current += random.randint(5, 20)
                U.append(current)
        else:
            # Random walk
            n = random.randint(4, 14)
            U = [0]
            current = random.randint(1, 3)
            U.append(current)
            for j in range(2, n):
                current += random.randint(1, 10)
                U.append(current)
        
        U = sorted(set(U))
        local_search(U, max_iters=5000)

    # Final high-precision polish: exhaustive single-element tweaks
    current_best = list(best_set)
    improved = True
    polish_rounds = 0
    while improved and time.time() - _start_time < TIME_LIMIT and polish_rounds < 5:
        improved = False
        polish_rounds += 1
        max_val = max(current_best)
        n = len(current_best)
        
        # Try replacing each element with nearby values
        for i in range(1, n):
            if time.time() - _start_time > TIME_LIMIT:
                break
            old_val = current_best[i]
            for delta in range(-15, 16):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= 0 or new_val in current_best:
                    continue
                cand = current_best[:]
                cand[i] = new_val
                cand.sort()
                s = _compute_score(cand)
                if s > best_score + 1e-12:
                    best_score = s
                    best_set = cand
                    current_best = cand
                    improved = True
                    break
        
        # Try adding values
        if time.time() - _start_time < TIME_LIMIT:
            for v in range(1, max_val + 20):
                if v in current_best:
                    continue
                cand = sorted(current_best + [v])
                s = _compute_score(cand)
                if s > best_score + 1e-12:
                    best_score = s
                    best_set = cand
                    current_best = cand
                    improved = True
        
        # Try removing
        if time.time() - _start_time < TIME_LIMIT:
            for i in range(1, len(current_best)):
                cand = current_best[:i] + current_best[i+1:]
                if len(cand) >= 2:
                    s = _compute_score(cand)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_set = cand
                        current_best = cand
                        improved = True

    return sorted(set(best_set))

# EVOLVE-BLOCK-END
