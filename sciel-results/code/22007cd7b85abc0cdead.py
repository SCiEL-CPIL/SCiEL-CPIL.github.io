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
    
    best_score = compute_score([0, 1, 3])
    best_U = [0, 1, 3]
    
    # Try structured sets: powers, Mersenne, etc.
    candidates = []
    
    # Powers of 2
    for n in range(2, 20):
        U = [0] + [2**i for i in range(1, n)]
        candidates.append(U)
    
    # Mersenne numbers
    for n in range(2, 20):
        U = [0] + [2**i - 1 for i in range(1, n)]
        candidates.append(U)
    
    # Random sets
    random.seed(42)
    for _ in range(5000):
        if time.time() > deadline - 5:
            break
        n = random.randint(3, 15)
        max_val = random.randint(10, 500)
        U = set([0])
        for _ in range(n - 1):
            U.add(random.randint(1, max_val))
        candidates.append(sorted(U))
    
    # Evaluate all candidates
    for U in candidates:
        if time.time() > deadline - 5:
            break
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U
    
    # Local search from best
    if time.time() < deadline - 10:
        U = list(best_U)
        for _ in range(100000):
            if time.time() > deadline - 5:
                break
            # Try adding, removing, or modifying an element
            op = random.choice(['add', 'remove', 'modify'])
            U_new = U[:]
            if op == 'add':
                if len(U_new) < 30:
                    U_new.append(random.randint(1, 2 * max(U_new) + 10))
                    U_new = sorted(set(U_new))
            elif op == 'remove':
                if len(U_new) > 2:
                    idx = random.randint(1, len(U_new) - 1)  # don't remove 0
                    U_new.pop(idx)
            else:
                idx = random.randint(1, len(U_new) - 1)
                U_new[idx] = random.randint(1, 2 * max(U_new) + 10)
                U_new = sorted(set(U_new))
            
            if len(U_new) >= 2:
                s = compute_score(U_new)
                if s > best_score:
                    best_score = s
                    best_U = U_new
                    U = U_new[:]
    
    return best_U
