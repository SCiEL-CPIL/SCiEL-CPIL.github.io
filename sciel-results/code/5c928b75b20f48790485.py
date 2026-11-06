import time
import random
import math

def _compute_score(A):
    n = len(A)
    sums = set()
    prods = set()
    a = A
    for i in range(n):
        ai = a[i]
        for j in range(i, n):
            aj = a[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return max(len(sums), len(prods))

def construct():
    start_time = time.time()
    # Use 55s deadline to be safe against 60s limit
    deadline = start_time + 55
    
    # Good initial set: mix of powers of 2 and 3*2^i
    # 2^0..2^13 (14 elements) + 3*2^0..3*2^5 (6 elements) = 20
    A = [2**i for i in range(14)] + [3 * 2**i for i in range(6)]
    A = sorted(set(A))
    if len(A) != 20:
        # Fallback if collision (unlikely with these specific ranges)
        A = sorted([2**i for i in range(20)])
        
    best_A = A[:]
    best_score = _compute_score(best_A)
    
    # Precompute a pool of "structural" candidates
    # Powers of 2, 3, 5, 7 and small multiples
    pool = set()
    for p in [2, 3, 5, 7]:
        v = 1
        while v <= 10000:
            pool.add(v)
            v *= p
    # Add small multiples to allow mixing
    for base in [2, 3, 5]:
        for mult in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            v = base * mult
            if v <= 10000:
                pool.add(v)
    
    # Filter pool to valid integers
    pool_list = [x for x in pool if 1 <= x <= 10000]
    random.seed(123)
    
    # Local search
    while time.time() < deadline:
        idx = random.randint(0, 19)
        old_val = best_A[idx]
        
        # Generate candidate
        # 50% chance from pool, 50% chance local neighbor
        if random.random() < 0.5:
            new_val = random.choice(pool_list)
        else:
            # Local neighbor
            delta = random.randint(-10, 10)
            new_val = old_val + delta
            if new_val < 1: new_val = 1
            if new_val > 10000: new_val = 10000
        
        if new_val == old_val:
            continue
            
        # Check if new_val is already in best_A
        # If it is, and it's not at idx, we have a collision.
        # Reject collisions to maintain distinctness simply and fast.
        found = False
        for v in best_A:
            if v == new_val:
                found = True
                break
        if found:
            continue
            
        # Construct new A
        new_A = best_A[:]
        new_A[idx] = new_val
        new_A = sorted(new_A)
        
        # It is distinct by construction and size 20
        new_score = _compute_score(new_A)
        
        if new_score < best_score:
            best_A = new_A
            best_score = new_score
        elif new_score == best_score:
            # Accept equal to explore? Maybe with low prob
            if random.random() < 0.1:
                best_A = new_A
                # best_score stays same
        
    return best_A
