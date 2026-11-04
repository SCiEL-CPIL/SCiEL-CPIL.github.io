"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    def compute_score(U):
        U = sorted(set(U))
        if 0 not in U:
            return 0
        n = len(U)
        if n < 2:
            return 0
        max_u = max(U)
        # Compute U+U
        sum_set = set()
        for a in U:
            for b in U:
                sum_set.add(a + b)
        # Compute U-U
        diff_set = set()
        for a in U:
            for b in U:
                diff_set.add(a - b)
        
        sum_size = len(sum_set)
        diff_size = len(diff_set)
        
        if sum_size == 0 or max_u < 0:
            return 0
        
        ratio = diff_size / sum_size
        if ratio <= 1:
            return 1.0
        
        denom = math.log(2 * max_u + 1)
        if denom <= 0:
            return 1.0
        
        return 1.0 + math.log(ratio) / denom
    
    best_set = [0, 1, 3]
    best_score = compute_score(best_set)
    
    # Try various structured families
    # Family 1: {0, 1, 3, 7, ...} (Mersenne-like, 2^k - 1)
    for k in range(2, 12):
        if time.time() > deadline:
            break
        U = [0] + [2**i - 1 for i in range(1, k+1)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    # Family 2: {0, 1, 3, 9, 27, ...} (geometric, ratio 3)
    for k in range(2, 10):
        if time.time() > deadline:
            break
        U = [0]
        val = 1
        for i in range(k):
            U.append(val)
            val *= 3
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    # Family 3: {0, 1, 3, 6, 10, ...} (triangular numbers)
    for k in range(2, 15):
        if time.time() > deadline:
            break
        U = [i*(i+1)//2 for i in range(k+1)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    # Family 4: {0, 1, 2, 4, 8, ...} (powers of 2)
    for k in range(2, 12):
        if time.time() > deadline:
            break
        U = [2**i for i in range(k+1)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    # Family 5: Random search in small ranges
    random.seed(42)
    for trial in range(2000):
        if time.time() > deadline:
            break
        # Try sets of various sizes in [0, M]
        n = random.randint(3, 12)
        M = random.randint(n, 50)
        U = [0] + random.sample(range(1, M+1), n-1)
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    # Family 6: Structured sets with gaps
    # Try {0, 1, 3, 6, 11, 18, ...} (quadratic gaps)
    for k in range(2, 12):
        if time.time() > deadline:
            break
        U = [0]
        val = 0
        for i in range(1, k+1):
            val += i
            U.append(val)
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    # Family 7: {0, 1, 3, 7, 15, ...} (2^k - 1) already tried, try {0, 2, 4, 8, ...}
    for k in range(2, 10):
        if time.time() > deadline:
            break
        U = [0] + [2**(i+1) for i in range(k)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    # Family 8: Hybrid - small AP plus a large jump
    for k in range(3, 10):
        if time.time() > deadline:
            break
        for jump in range(1, 20):
            U = list(range(k)) + [k + jump]
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    # Family 9: {0, 1, 3, 6, 12, 24, ...} (mixed growth)
    for k in range(3, 10):
        if time.time() > deadline:
            break
            U = [0, 1, 3]
        val = 6
        for i in range(k - 3):
            U.append(val)
            val *= 2
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    # Final polish: local search around best_set
    if time.time() < deadline - 10:
        U = best_set[:]
        for _ in range(500):
            if time.time() > deadline:
                break
            # Try adding a random element
            if random.random() < 0.5:
                max_u = max(U) if U else 0
                new_elem = random.randint(1, max_u * 2 + 5)
                U_new = U + [new_elem]
            else:
                # Try removing a random non-zero element
                if len(U) > 2:
                    idx = random.randint(1, len(U) - 1)
                    U_new = U[:idx] + U[idx+1:]
                else:
                    continue
            
            s = compute_score(U_new)
            if s > best_score:
                best_score = s
                best_set = U_new[:]
    
    return best_set
# EVOLVE-BLOCK-END
