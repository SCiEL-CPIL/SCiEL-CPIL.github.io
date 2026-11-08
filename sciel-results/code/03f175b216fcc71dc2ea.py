import time
import math
import random
from itertools import combinations

def compute_score(U):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)"""
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    maxU = U[-1]
    if maxU == 0:
        return 0.0
    
    # Compute U+U and U-U
    sum_set = set()
    diff_set = set()
    for i in range(n):
        for j in range(n):
            sum_set.add(U[i] + U[j])
            diff_set.add(U[i] - U[j])
    
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0.0
    score = 1.0 + math.log(ratio) / math.log(2 * maxU + 1)
    return score

def exhaustive_search_small(n, max_val, deadline):
    """Exhaustive search for small sets of size n"""
    best_score = 0.0
    best_set = None
    # Try all subsets of {0, 1, ..., max_val} of size n that contain 0
    candidates = list(range(1, max_val + 1))
    for combo in combinations(candidates, n - 1):
        if time.time() > deadline:
            break
        U = sorted([0] + list(combo))
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    return best_score, best_set

def parametric_families(deadline):
    """Try various parametric families"""
    best_score = 0.0
    best_set = None
    
    # Family 1: {0, 1, 2, ..., n-1} - arithmetic progression
    for n in range(3, 50):
        if time.time() > deadline:
            break
        U = list(range(n))
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # Family 2: {0, 1, 3, 7, 15, ...} - Mersenne-like (2^k - 1)
    for n in range(3, 20):
        if time.time() > deadline:
            break
        U = [0] + [2**k - 1 for k in range(1, n)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # Family 3: {0, 1, 2, 4, 8, ...} - powers of 2
    for n in range(3, 20):
        if time.time() > deadline:
            break
        U = [0] + [2**k for k in range(1, n)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # Family 4: {0, 1, 3, 6, 10, ...} - triangular numbers
    for n in range(3, 30):
        if time.time() > deadline:
            break
        U = [0] + [k*(k+1)//2 for k in range(1, n)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # Family 5: {0, 1, 2, 4, 7, 11, ...} - quadratic
    for n in range(3, 30):
        if time.time() > deadline:
            break
        U = [0] + [k*k//2 for k in range(1, n)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # Family 6: Mixed - {0} + small dense set + sparse large elements
    for base_n in range(3, 10):
        if time.time() > deadline:
            break
        for gap in range(2, 20):
            if time.time() > deadline:
                break
            for extra in range(0, 5):
                if time.time() > deadline:
                    break
                U = list(range(base_n))
                for i in range(extra):
                    U.append(base_n * gap * (i + 1))
                U = sorted(set(U))
                s = compute_score(U)
                if s > best_score:
                    best_score = s
                    best_set = U
    
    return best_score, best_set

def hill_climb(initial_set, deadline, max_iter=5000):
    """Local search via element replacement"""
    U = sorted(set(initial_set))
    if len(U) < 2:
        return 0.0, U
    best_score = compute_score(U)
    best_set = U[:]
    
    max_val = max(U)
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        n = len(U)
        
        # Try replacing each element with a random value
        for i in range(n):
            if time.time() > deadline:
                break
            # Try several random replacements
            for _ in range(10):
                if time.time() > deadline:
                    break
                new_val = random.randint(0, max_val * 2)
                if new_val in U and new_val != U[i]:
                    continue
                new_U = U[:i] + [new_val] + U[i+1:]
                new_U = sorted(set(new_U))
                if 0 not in new_U:
                    new_U = sorted([0] + [x for x in new_U if x > 0])
                if len(new_U) < 2:
                    continue
                s = compute_score(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_set = new_U
                    U = new_U
                    max_val = max(U)
                    improved = True
                    break
        
        if not improved:
            # Try adding a new element
            if len(U) < 50:
                for _ in range(5):
                    if time.time() > deadline:
                        break
                    new_val = random.randint(1, max_val * 3)
                    new_U = sorted(U + [new_val])
                    s = compute_score(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_set = new_U
                        U = new_U
                        max_val = max(U)
                        improved = True
                        break
            
            if not improved:
                # Try removing an element
                if len(U) > 3:
                    for i in range(len(U)):
                        if time.time() > deadline:
                            break
                        if U[i] == 0:
                            continue
                        new_U = U[:i] + U[i+1:]
                        s = compute_score(new_U)
                        if s > best_score + 1e-12:
                            best_score = s
                            best_set = new_U
                            U = new_U
                            max_val = max(U)
                            improved = True
                            break
            
            if not improved:
                break
    
    return best_score, best_set

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # 1. Try small exhaustive search
    for n in range(3, 7):
        if time.time() > deadline:
            break
        max_val = 30 if n <= 5 else 20
        s, U = exhaustive_search_small(n, max_val, deadline)
        if s > best_score:
            best_score = s
            best_set = U
    
    # 2. Try parametric families
    s, U = parametric_families(deadline)
    if s > best_score:
        best_score = s
        best_set = U
    
    # 3. Hill climb from best found so far
    if time.time() < deadline - 10:
        s, U = hill_climb(best_set, deadline, max_iter=3000)
        if s > best_score:
            best_score = s
            best_set = U
    
    # 4. Try more hill climbing from different starting points
    if time.time() < deadline - 5:
        # Start from a few different random sets
        random.seed(42)
        for seed in range(5):
            if time.time() > deadline:
                break
            random.seed(seed)
            n = random.randint(4, 12)
            U = sorted(random.sample(range(1, 100), n - 1) + [0])
            s, U2 = hill_climb(U, deadline, max_iter=1000)
            if s > best_score:
                best_score = s
                best_set = U2
    
    return best_set

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
