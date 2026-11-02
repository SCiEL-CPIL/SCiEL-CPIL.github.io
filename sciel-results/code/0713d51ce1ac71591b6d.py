import itertools
import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_set = [0, 1, 4, 9]
    best_score = 1.2
    
    def calc_score(U):
        if not U or 0 not in U:
            return 0.0
        U = sorted(set(U))
        max_u = max(U)
        if max_u == 0:
            return 1.0
        
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
        
        size_s = len(sums)
        size_d = len(diffs)
        
        if size_s == 0:
            return 0.0
            
        ratio = size_d / size_s
        return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

    def local_search(U, steps=20):
        U = list(U)
        current_score = calc_score(U)
        for _ in range(steps):
            if time.time() > deadline:
                return U, current_score
            improved = False
            for i in range(len(U)):
                if U[i] == 0:
                    continue
                # Try small perturbations
                for delta in range(-5, 6):
                    if delta == 0:
                        continue
                    new_val = U[i] + delta
                    if new_val < 0:
                        continue
                    new_U = list(U)
                    new_U[i] = new_val
                    new_U = sorted(set(new_U))
                    if 0 not in new_U or len(new_U) != len(U):
                        continue
                    new_score = calc_score(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return U, current_score

    # 1. Test specific sparse structures
    candidates = []
    
    # Squares
    for n in range(3, 8):
        candidates.append([i**2 for i in range(n)])
        
    # Primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for n in range(3, 8):
        candidates.append(primes[:n])
        
    # Geometric
    for r in range(2, 6):
        for n in range(3, 8):
            candidates.append([r**i for i in range(n)])
            
    # Random sparse
    random.seed(42)
    for _ in range(500):
        n = random.randint(3, 7)
        max_val = random.randint(n * 2, 50)
        sample = random.sample(range(1, max_val + 1), n - 1)
        candidates.append(sorted([0] + sample))
        
    # Evaluate candidates
    for U in candidates:
        if time.time() > deadline:
            break
        s = calc_score(U)
        if s > best_score:
            best_score = s
            best_set = U
            
    # 2. Local search on the best candidate found so far
    best_set, best_score = local_search(best_set, steps=100)
    
    # 3. Local search on top few other candidates
    scored_candidates = []
    for U in candidates:
        if time.time() > deadline:
            break
        s = calc_score(U)
        scored_candidates.append((s, U))
        
    scored_candidates.sort(key=lambda x: -x[0])
    
    for s, U in scored_candidates[:5]:
        if time.time() > deadline:
            break
        new_U, new_s = local_search(U, steps=50)
        if new_s > best_score:
            best_score = new_s
            best_set = new_U

    return best_set
