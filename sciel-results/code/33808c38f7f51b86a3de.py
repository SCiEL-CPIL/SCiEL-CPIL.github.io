import time
import math
import numpy as np
from itertools import combinations

def compute_score(U):
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    if len(U) > 30:
        # Use numpy for larger sets
        U_arr = np.array(U)
        sums = U_arr[:, None] + U_arr[None, :]
        diffs = U_arr[:, None] - U_arr[None, :]
        sum_set_size = len(np.unique(sums))
        diff_set_size = len(np.unique(diffs))
    else:
        sum_set = set()
        diff_set = set()
        for a in U:
            for b in U:
                sum_set.add(a + b)
                diff_set.add(a - b)
        sum_set_size = len(sum_set)
        diff_set_size = len(diff_set)
    ratio = diff_set_size / sum_set_size
    if ratio <= 0:
        return 0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0
    return 1 + log_ratio / log_denom

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Theoretical best found: {0, 1, 3, 9}
    best_U = [0, 1, 3, 9]
    best_score = compute_score(best_U)
    
    def try_set(U):
        nonlocal best_U, best_score
        if time.time() > deadline:
            return False
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
        return time.time() <= deadline
    
    # Try structured candidates with geometric growth
    for base in [2, 3, 4, 5, 7, 8, 10, 12, 15, 20]:
        for n in range(3, 10):
            if time.time() > deadline:
                break
            U = [0]
            x = 1
            for _ in range(n):
                U.append(x)
                x *= base
            if U[-1] > 5000:
                break
            try_set(U)
    
    # Try {0, 1, a, a*b} type sets with wider range
    for a in range(2, 60):
        if time.time() > deadline:
            break
        for b in range(2, 30):
            if time.time() > deadline:
                break
            c = a * b
            if c > 2000:
                break
            try_set([0, 1, a, c])
    
    # Try {0, 1, a, b, c} with geometric-like spacing
    for a in range(2, 25):
        if time.time() > deadline:
            break
        for b in range(a + 1, 50):
            if time.time() > deadline:
                break
            for c in range(b + 1, 150):
                if time.time() > deadline:
                    break
                try_set([0, 1, a, b, c])
    
    # Try {0, 1, a, b, c, d} with geometric-like spacing
    for a in range(2, 15):
        if time.time() > deadline:
            break
        for b in range(a + 1, 30):
            if time.time() > deadline:
                break
            for c in range(b + 1, 60):
                if time.time() > deadline:
                    break
                for d in range(c + 1, 100):
                    if time.time() > deadline:
                        break
                    try_set([0, 1, a, b, c, d])
    
    # Exhaustive search over 4-element sets {0, a, b, c} with 0 < a < b < c <= 100
    for a in range(1, 35):
        if time.time() > deadline:
            break
        for b in range(a + 1, 70):
            if time.time() > deadline:
                break
            for c in range(b + 1, 101):
                if not try_set([0, a, b, c]):
                    break
    
    # Exhaustive search over 5-element sets {0, a, b, c, d} with 0 < a < b < c < d <= 50
    for a in range(1, 12):
        if time.time() > deadline:
            break
        for b in range(a + 1, 25):
            if time.time() > deadline:
                break
            for c in range(b + 1, 40):
                if time.time() > deadline:
                    break
                for d in range(c + 1, 51):
                    if not try_set([0, a, b, c, d]):
                        break
    
    # Try 6-element sets with geometric growth
    for a in range(1, 8):
        if time.time() > deadline:
            break
        for b in range(a + 1, 15):
            if time.time() > deadline:
                break
            for c in range(b + 1, 25):
                if time.time() > deadline:
                    break
                for d in range(c + 1, 40):
                    if time.time() > deadline:
                        break
                    for e in range(d + 1, 60):
                        if not try_set([0, a, b, c, d, e]):
                            break
    
    # Try sets with one large element and small dense part
    for small_max in range(5, 30):
        if time.time() > deadline:
            break
        for large in range(small_max + 5, min(small_max * 8 + 1, 500)):
            if time.time() > deadline:
                break
            U = list(range(small_max + 1)) + [large]
            try_set(U)
    
    # Try {0, 1, 2, ..., k} + {m} for various k, m
    for k in range(3, 20):
        if time.time() > deadline:
            break
        for m in range(k + 2, min(k * 8 + 1, 300)):
            if time.time() > deadline:
                break
            U = list(range(k + 1)) + [m]
            try_set(U)
    
    # Try arithmetic progressions
    for a in range(1, 15):
        if time.time() > deadline:
            break
        for d in range(1, 25):
            if time.time() > deadline:
                break
            for n in range(4, 10):
                U = [a * i * d for i in range(n)]
                U[0] = 0
                if U[-1] > 1000:
                    break
                try_set(U)
    
    # Try quadratic sequences {0, 1, 4, 9, 16, ...}
    for n in range(3, 10):
        if time.time() > deadline:
            break
        U = [i * i for i in range(n)]
        try_set(U)
    
    # Try factorial-like sequences
    for n in range(3, 7):
        if time.time() > deadline:
            break
        U = [1]
        x = 1
        for i in range(2, n + 1):
            x *= i
            U.append(x)
        try_set(U)
    
    # Try prime-based sequences
    def first_primes(n):
        primes = []
        x = 2
        while len(primes) < n:
            if all(x % p != 0 for p in primes if p * p <= x):
                primes.append(x)
            x += 1
        return primes
    
    for n in range(3, 8):
        if time.time() > deadline:
            break
        U = [0] + first_primes(n)
        try_set(U)
    
    # Local improvement: hill-climb on the best set found
    for iteration in range(100):
        if time.time() > deadline:
            break
        improved = False
        U_sorted = sorted(set(best_U))
        n = len(U_sorted)
        
        # Try replacing each element (except 0) with nearby values
        for i in range(1, n):
            if time.time() > deadline:
                break
            old_val = U_sorted[i]
            for delta in range(-25, 26):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                # Check if new_val would cause a duplicate
                if new_val in U_sorted:
                    continue
                new_U = U_sorted[:i] + [new_val] + U_sorted[i+1:]
                if len(set(new_U)) != len(new_U):
                    continue
                s = compute_score(new_U)
                if s > best_score:
                    best_score = s
                    best_U = new_U[:]
                    improved = True
                    break
        
        # Try adding a new element
        if time.time() <= deadline:
            max_val = U_sorted[-1]
            for new_val in range(1, min(max_val + 50, 500)):
                if time.time() > deadline:
                    break
                if new_val in U_sorted:
                    continue
                new_U = sorted(U_sorted + [new_val])
                s = compute_score(new_U)
                if s > best_score:
                    best_score = s
                    best_U = new_U[:]
                    improved = True
                    break
        
        # Try removing a non-zero element
        if time.time() <= deadline and n > 2:
            for i in range(1, n):
                if time.time() > deadline:
                    break
                new_U = U_sorted[:i] + U_sorted[i+1:]
                if len(new_U) < 2:
                    continue
                s = compute_score(new_U)
                if s > best_score:
                    best_score = s
                    best_U = new_U[:]
                    improved = True
                    break
        
        if not improved:
            break
    
    # Random restarts to escape local optima
    if time.time() < deadline - 10:
        import random
        random.seed(42)
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            
            # Generate a random set
            size = random.randint(3, 7)
            max_val = random.randint(20, 200)
            U = [0]
            remaining = max_val
            for i in range(size - 1):
                if remaining <= 0:
                    break
                max_pick = min(remaining, max_val)
                val = random.randint(1, max_pick)
                if val not in U:
                    U.append(val)
                    remaining -= val
            U.sort()
            
            if len(U) < 2:
                continue
            
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
            
            # Local improvement from this random start
            for iteration in range(20):
                if time.time() > deadline - 5:
                    break
                U_sorted = sorted(set(best_U))
                n = len(U_sorted)
                
                improved = False
                for i in range(1, n):
                    if time.time() > deadline - 5:
                        break
                    old_val = U_sorted[i]
                    for delta in range(-10, 11):
                        if delta == 0:
                            continue
                        new_val = old_val + delta
                        if new_val <= 0:
                            continue
                        if new_val in U_sorted:
                            continue
                        new_U = U_sorted[:i] + [new_val] + U_sorted[i+1:]
                        if len(set(new_U)) != len(new_U):
                            continue
                        s = compute_score(new_U)
                        if s > best_score:
                            best_score = s
                            best_U = new_U[:]
                            improved = True
                            break
                    if improved:
                        break
                
                if not improved:
                    break
    
    return best_U
