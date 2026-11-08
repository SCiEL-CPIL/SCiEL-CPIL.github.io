import time
import math
import numpy as np
import random
from itertools import combinations

def compute_score(U):
    """
    Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
    Uses NumPy for efficient set difference and sum calculations.
    """
    if len(U) < 2:
        return 0.0
    
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0.0
    
    n = len(U)
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute all pairwise sums and differences
    # sums[i,j] = U[i] + U[j]
    # diffs[i,j] = U[i] - U[j]
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    
    sum_set_size = len(np.unique(sums))
    diff_set_size = len(np.unique(diffs))
    
    ratio = diff_set_size / sum_set_size
    if ratio <= 0:
        return 0.0
        
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    
    if log_denom <= 0:
        return 0.0
        
    return 1 + log_ratio / log_denom

def exhaustive_search_small(n, max_val, deadline):
    """
    Exhaustive search for sets of size n containing 0, with other elements < max_val.
    Returns (best_score, best_set).
    """
    best_score = 0.0
    best_set = None
    
    # Candidates for the non-zero elements
    candidates = list(range(1, max_val + 1))
    
    # We need to choose n-1 elements from candidates
    for combo in combinations(candidates, n - 1):
        if time.time() > deadline:
            break
        
        U = sorted([0] + list(combo))
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
            
    return best_score, best_set

def parametric_families(deadline):
    """
    Try various parametric families of sets.
    Returns (best_score, best_set).
    """
    best_score = 0.0
    best_set = None
    
    def update(U):
        nonlocal best_score, best_set
        if time.time() > deadline:
            return
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]

    # Family 1: {0, 1, 2, ..., n-1} - Arithmetic Progression
    for n in range(3, 50):
        if time.time() > deadline:
            break
        U = list(range(n))
        update(U)
        
    # Family 2: {0, 1, 3, 7, 15, ...} - Mersenne-like (2^k - 1)
    for n in range(3, 20):
        if time.time() > deadline:
            break
        U = [0] + [2**k - 1 for k in range(1, n)]
        update(U)
        
    # Family 3: {0, 1, 2, 4, 8, ...} - Powers of 2
    for n in range(3, 20):
        if time.time() > deadline:
            break
        U = [0] + [2**k for k in range(1, n)]
        update(U)
        
    # Family 4: {0, 1, 3, 6, 10, ...} - Triangular Numbers
    for n in range(3, 30):
        if time.time() > deadline:
            break
        U = [0] + [k*(k+1)//2 for k in range(1, n)]
        update(U)
        
    # Family 5: {0, 1, 2, 4, 7, 11, ...} - Quadratic-like (k^2 // 2)
    for n in range(3, 30):
        if time.time() > deadline:
            break
        U = [0] + [k*k//2 for k in range(1, n)]
        update(U)
        
    # Family 6: {0, 1, 3, 9, 27, ...} - Powers of 3
    for n in range(3, 20):
        if time.time() > deadline:
            break
        U = [0] + [3**k for k in range(1, n)]
        update(U)
        
    # Family 7: Mixed - Dense small part + Sparse large part
    # {0, ..., k-1} U {m, 2m, ...}
    for k in range(3, 15):
        if time.time() > deadline:
            break
        for gap in range(2, 20):
            if time.time() > deadline:
                break
            base_max = k * gap
            for extra_count in range(1, 5):
                if time.time() > deadline:
                    break
                U = list(range(k))
                for i in range(extra_count):
                    U.append(base_max * (i + 1))
                U = sorted(set(U))
                update(U)
                
    # Family 8: Geometric with offset
    for base in [2, 3, 4, 5]:
        for offset in range(0, base):
            if time.time() > deadline:
                break
            for n in range(3, 10):
                if time.time() > deadline:
                    break
                U = [0]
                x = 1
                for i in range(n):
                    U.append(x + offset)
                    x *= base
                U = sorted(set(U))
                if len(U) == len(set(U)):
                    update(U)

    return best_score, best_set

def hill_climb(initial_set, deadline, max_iter=5000):
    """
    Local search via element replacement, addition, and removal.
    """
    U = sorted(set(initial_set))
    if len(U) < 2:
        return 0.0, U
        
    best_score = compute_score(U)
    best_set = U[:]
    
    max_val = max(U)
    no_improve_count = 0
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
            
        improved = False
        n = len(U)
        
        # 1. Try replacing each element with a random value
        # We prioritize replacing non-zero elements first, but 0 must stay 0? 
        # Actually, the problem says "containing 0". It doesn't say ONLY 0 is fixed, 
        # but usually 0 is part of the structure. Let's allow moving 0? 
        # No, "containing 0" implies 0 is in the set. If we replace 0, we might lose it.
        # Let's keep 0 fixed to be safe, or re-add it if lost.
        
        for i in range(n):
            if time.time() > deadline:
                break
            
            # Skip 0? Or allow moving it?
            # If we move 0, we must ensure 0 is still in the set.
            # Let's just try replacing non-zero elements first.
            if U[i] == 0:
                continue
                
            # Try several random replacements
            for _ in range(10):
                if time.time() > deadline:
                    break
                
                # Sample from a range that might be useful
                # Often optimal sets have elements spread out.
                new_val = random.randint(1, max_val * 2)
                
                if new_val in U:
                    continue
                    
                new_U = U[:i] + [new_val] + U[i+1:]
                new_U = sorted(set(new_U))
                
                # Ensure 0 is in the set
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
                    no_improve_count = 0
                    break
        
        # 2. Try adding a new element
        if not improved and len(U) < 50:
            for _ in range(5):
                if time.time() > deadline:
                    break
                new_val = random.randint(1, max_val * 3)
                if new_val in U:
                    continue
                new_U = sorted(U + [new_val])
                s = compute_score(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_set = new_U
                    U = new_U
                    max_val = max(U)
                    improved = True
                    no_improve_count = 0
                    break
        
        # 3. Try removing an element (except 0)
        if not improved and len(U) > 3:
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
                    no_improve_count = 0
                    break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 20:
                break
                
    return best_score, best_set

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # 1. Exhaustive search for small sets
    # This is crucial as the global optimum is often a small set
    for n in range(3, 7):
        if time.time() > deadline:
            break
        # Limit search space based on n
        # For n=3, max_val can be larger
        # For n=6, max_val must be smaller to fit in time
        max_val = 30 if n <= 5 else 20
        s, U = exhaustive_search_small(n, max_val, deadline)
        if s > best_score:
            best_score = s
            best_set = U if U else best_set
            
    # 2. Try parametric families
    if time.time() < deadline - 5:
        s, U = parametric_families(deadline)
        if U and s > best_score:
            best_score = s
            best_set = U
            
    # 3. Hill climb from the best found so far
    if time.time() < deadline - 10:
        s, U = hill_climb(best_set, deadline, max_iter=3000)
        if s > best_score:
            best_score = s
            best_set = U
            
    # 4. Hill climb from a few other promising starting points
    # Start from known good structures like {0, 1, 3, 9} or {0, 1, 2, 4}
    starting_points = [
        [0, 1, 3, 9],
        [0, 1, 2, 4, 8],
        [0, 1, 3, 6, 10],
        [0, 1, 2, 5, 10]
    ]
    
    for start in starting_points:
        if time.time() > deadline - 5:
            break
        s, U = hill_climb(start, deadline, max_iter=1000)
        if s > best_score:
            best_score = s
            best_set = U
            
    # 5. Final random restart hill climbs
    random.seed(42)
    for _ in range(5):
        if time.time() > deadline - 2:
            break
        n = random.randint(4, 10)
        # Generate a random set with 0
        candidates = random.sample(range(1, 100), n - 1)
        U = sorted([0] + candidates)
        s, U2 = hill_climb(U, deadline, max_iter=500)
        if s > best_score:
            best_score = s
            best_set = U2
            
    return best_set

if __name__ == "__main__":
    # Quick test
    U = construct()
    print(f"Best Set: {U}")
    print(f"Score: {compute_score(U)}")
