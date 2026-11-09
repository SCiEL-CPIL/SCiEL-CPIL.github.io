import random
import time
from math import log, log2
from itertools import combinations

def compute_score(U):
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    n = len(U)
    if n == 0:
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
            
    max_u = max(U)
    if max_u == 0:
        return 0.0
        
    ratio = len(diffset) / len(sumset)
    if ratio <= 1:
        return 1.0
        
    denominator = log(2 * max_u + 1)
    if denominator <= 0:
        return 0.0
        
    return 1.0 + log(ratio) / denominator

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    best_set = [0, 1, 3]
    best_score = compute_score(best_set)
    
    # Predefined candidates from known constructions
    candidates = [
        [0, 1, 3],
        [0, 1, 4, 9],
        [0, 1, 3, 7],
        [0, 1, 2, 4, 8],
        [0, 1, 3, 6, 12],
        [0, 1, 3, 7, 15],
        [0, 1, 4, 10],
        [0, 1, 5, 11],
        [0, 1, 3, 7, 15, 31],
        [0, 1, 3, 7, 15, 31, 63],
        [0, 1, 4, 16, 64],
        [0, 1, 4, 16],
        [0, 1, 3, 9, 27],
        [0, 1, 3, 9],
        [0, 2, 6, 18],
        [0, 2, 6],
        [0, 1, 2, 5, 10],
        [0, 1, 3, 6, 10],
    ]
    
    for c in candidates:
        s = compute_score(c)
        if s > best_score:
            best_score = s
            best_set = list(c)
    
    # Random search with local improvement
    # Key insight: lacunary sequences (powers of small integers) tend to have
    # large difference sets relative to sum sets
    
    # Try lacunary sequences: {0, a, a^2, a^3, ...}
    for base in range(2, 15):
        for length in range(2, 8):
            if base ** (length - 1) > 1000:
                break
            U = [base ** i for i in range(length)]
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
    
    # Random search over small sets
    random.seed(42)
    
    # Phase 1: Random subsets of small ranges
    for n in range(2, 8):
        max_range = min(50, 2 * n * n)
        for _ in range(2000):
            if time.time() > deadline:
                return best_set
            subset = random.sample(range(1, max_range + 1), n - 1)
            U = [0] + sorted(subset)
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
    
    # Phase 2: Local search - hill climbing
    # Try to improve by adding/removing/swapping elements
    for iteration in range(500):
        if time.time() > deadline:
            return best_set
        
        current = list(best_set)
        improved = False
        
        # Try adding a random element
        if len(current) < 10:
            max_val = max(current) if current else 1
            for _ in range(5):
                new_elem = random.randint(1, max_val * 2 + 10)
                if new_elem not in current:
                    candidate = sorted(current + [new_elem])
                    s = compute_score(candidate)
                    if s > best_score:
                        best_score = s
                        best_set = candidate
                        current = candidate
                        improved = True
                        break
        
        # Try removing an element (if size > 2)
        if len(current) > 2:
            for i in range(len(current)):
                if current[i] == 0:
                    continue  # Keep 0
                candidate = [x for j, x in enumerate(current) if j != i]
                s = compute_score(candidate)
                if s > best_score:
                    best_score = s
                    best_set = candidate
                    current = candidate
                    improved = True
                    break
        
        # Try swapping an element
        if len(current) > 1:
            for i in range(len(current)):
                if current[i] == 0:
                    continue
                new_val = random.randint(1, max(current) * 2 + 5)
                if new_val not in current:
                    candidate = list(current)
                    candidate[i] = new_val
                    candidate = sorted(candidate)
                    s = compute_score(candidate)
                    if s > best_score:
                        best_score = s
                        best_set = candidate
                        current = candidate
                        improved = True
                        break
        
        if not improved:
            # Reset to a random perturbation of best
            current = list(best_set)
            if len(current) > 1:
                idx = random.randint(1, len(current) - 1)
                current[idx] = random.randint(1, max(current) * 2 + 10)
                current = sorted(set(current))
                if 0 not in current:
                    current = [0] + [x for x in current if x != 0]
    
    # Phase 3: Try structured sequences with random parameters
    for _ in range(500):
        if time.time() > deadline:
            return best_set
        
        # Geometric-like sequences
        a = random.randint(2, 10)
        r = random.randint(1, 5)
        length = random.randint(3, 7)
        U = [0]
        val = a
        for i in range(length - 1):
            U.append(val)
            val *= (a + r)
            if val > 5000:
                break
        if len(U) >= 2:
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_set = list(U)
    
    # Phase 4: Final polish - try small perturbations of best set
    for _ in range(1000):
        if time.time() > deadline:
            return best_set
        
        current = list(best_set)
        if len(current) < 2:
            break
        
        # Random small perturbation
        idx = random.randint(1, len(current) - 1)
        delta = random.randint(-3, 3)
        new_val = current[idx] + delta
        if new_val > 0 and new_val not in current:
            candidate = list(current)
            candidate[idx] = new_val
            candidate = sorted(candidate)
            s = compute_score(candidate)
            if s > best_score:
                best_score = s
                best_set = candidate
    
    return best_set
