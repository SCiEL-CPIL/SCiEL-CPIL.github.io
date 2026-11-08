import time
import math
import random
import numpy as np

def compute_score(U):
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0
    return 1 + log_ratio / log_denom

def construct():
    start = time.time()
    deadline = start + 140
    
    best_score = 0
    best_U = [0, 1, 3]
    
    # Exhaustive search over small sets to find optimal pattern
    # Try all subsets of {0, 1, 2, ..., M} with 0 included, size 3-6
    # For small M, enumerate all combinations to find optimal small sets
    
    from itertools import combinations
    
    # Exhaustive search for 3-element sets {0, a, b} with 0 < a < b <= 40
    for a in range(1, 41):
        if time.time() > deadline - 30:
            break
        for b in range(a + 1, 41):
            U = [0, a, b]
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
    
    # Exhaustive search for 4-element sets {0, a, b, c} with 0 < a < b < c <= 30
    for a in range(1, 25):
        if time.time() > deadline - 30:
            break
        for b in range(a + 1, 31):
            for c in range(b + 1, 31):
                U = [0, a, b, c]
                s = compute_score(U)
                if s > best_score:
                    best_score = s
                    best_U = U[:]
    
    # Exhaustive search for 5-element sets {0, a, b, c, d} with 0 < a < b < c < d <= 20
    for a in range(1, 15):
        if time.time() > deadline - 30:
            break
        for b in range(a + 1, 21):
            for c in range(b + 1, 21):
                for d in range(c + 1, 21):
                    U = [0, a, b, c, d]
                    s = compute_score(U)
                    if s > best_score:
                        best_score = s
                        best_U = U[:]
    
    # Try various structured families
    families = []
    
    # Consecutive integers
    for n in range(2, 50):
        families.append(list(range(n)))
    
    # Powers of 2
    for n in range(2, 30):
        families.append([0] + [2**i for i in range(1, n)])
    
    # Mersenne: 2^k - 1
    for n in range(2, 30):
        families.append([0] + [2**i - 1 for i in range(1, n)])
    
    # Arithmetic progressions
    for n in range(2, 50):
        for step in range(1, 10):
            families.append([i * step for i in range(n)])
    
    # Mixed: {0, 1, a, 2a, 3a, ...}
    for a in range(2, 50):
        for n in range(3, 15):
            families.append([0, 1] + [i * a for i in range(1, n)])
    
    # {0, 1, a, a^2, a^3, ...}
    for a in range(2, 20):
        for n in range(3, 10):
            U = [0, 1]
            val = a
            for i in range(n - 2):
                U.append(val)
                val *= a
            families.append(U)
    
    # Geometric-like with small bases
    for base in range(2, 12):
        for n in range(3, 12):
            U = [0]
            val = 1
            for i in range(n - 1):
                U.append(val)
                val *= base
            families.append(U)
    
    # Sparse sets with gaps
    for gap in range(2, 15):
        for n in range(3, 20):
            families.append([i * gap for i in range(n)])
    
    # Evaluate all
    for U in families:
        if time.time() > deadline - 15:
            break
        U = sorted(set(U))
        if len(U) < 2:
            continue
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U
    
    # Now do local search from the best
    U = list(best_U)
    for _ in range(200000):
        if time.time() > deadline - 5:
            break
        U_new = U[:]
        op = random.choice(['add', 'remove', 'modify'])
        max_val = max(U_new) if U_new else 1
        
        if op == 'add':
            if len(U_new) < 40:
                U_new.append(random.randint(1, 3 * max_val + 20))
                U_new = sorted(set(U_new))
        elif op == 'remove':
            if len(U_new) > 2:
                idx = random.randint(1, len(U_new) - 1)
                U_new.pop(idx)
        else:
            idx = random.randint(1, len(U_new) - 1)
            U_new[idx] = random.randint(1, 3 * max_val + 20)
            U_new = sorted(set(U_new))
        
        if len(U_new) >= 2:
            s = compute_score(U_new)
            if s > best_score:
                best_score = s
                best_U = U_new
                U = U_new[:]
    
    return best_U
