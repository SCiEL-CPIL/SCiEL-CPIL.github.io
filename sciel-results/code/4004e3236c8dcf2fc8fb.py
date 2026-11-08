import time
import math
import numpy as np

def compute_score(U):
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    arr = np.array(U, dtype=np.int64)
    sums = arr[:, None] + arr[None, :]
    diffs = arr[:, None] - arr[None, :]
    sum_count = len(np.unique(sums))
    diff_count = len(np.unique(diffs))
    ratio = diff_count / sum_count
    if ratio <= 0:
        return 0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0
    return 1 + log_ratio / log_denom

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
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
    
    def local_search(U, max_iters=200):
        """Hill-climb by adding/removing/adjusting elements."""
        nonlocal best_U, best_score
        current_U = sorted(set(U))
        current_score = compute_score(current_U)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding a new element
            max_val = max(current_U) if current_U else 0
            for new_val in range(1, max_val * 3 + 10):
                if new_val in current_U:
                    continue
                candidate = sorted(current_U + [new_val])
                s = compute_score(candidate)
                if s > current_score:
                    current_U = candidate
                    current_score = s
                    improved = True
                    break
            
            # Try removing an element (except 0)
            if len(current_U) > 2:
                for i, elem in enumerate(current_U):
                    if elem == 0:
                        continue
                    candidate = [x for x in current_U if x != elem]
                    s = compute_score(candidate)
                    if s > current_score:
                        current_U = candidate
                        current_score = s
                        improved = True
                        break
            
            # Try adjusting an element
            if len(current_U) > 2:
                for i, elem in enumerate(current_U):
                    if elem == 0:
                        continue
                    for delta in [-3, -2, -1, 1, 2, 3, 5, 10]:
                        new_val = elem + delta
                        if new_val < 0 or new_val in current_U:
                            continue
                        candidate = sorted([new_val if x == elem else x for x in current_U])
                        s = compute_score(candidate)
                        if s > current_score:
                            current_U = candidate
                            current_score = s
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        if current_score > best_score:
            best_score = current_score
            best_U = current_U[:]
        return current_U
    
    # Phase 1: Structured candidates with geometric growth
    for base in range(2, 15):
        for n in range(3, 12):
            if time.time() > deadline:
                break
            U = [0]
            x = 1
            for _ in range(n):
                U.append(x)
                x *= base
            try_set(U)
    
    # Phase 2: {0, 1, a, a*b} type sets with wider ranges
    for a in range(2, 80):
        if time.time() > deadline:
            break
        for b in range(2, 40):
            if time.time() > deadline:
                break
            c = a * b
            if c > 2000:
                break
            try_set([0, 1, a, c])
    
    # Phase 3: {0, 1, a, b, c} with wider geometric-like spacing
    for a in range(2, 30):
        if time.time() > deadline:
            break
        for b in range(a + 1, 80):
            if time.time() > deadline:
                break
            for c in range(b + 1, 200):
                if time.time() > deadline:
                    break
                try_set([0, 1, a, b, c])
    
    # Phase 4: Exhaustive search over 4-element sets {0, a, b, c} with 0 < a < b < c <= 150
    for a in range(1, 40):
        if time.time() > deadline:
            break
        for b in range(a + 1, 80):
            if time.time() > deadline:
                break
            for c in range(b + 1, 151):
                if not try_set([0, a, b, c]):
                    break
    
    # Phase 5: Exhaustive search over 5-element sets {0, a, b, c, d} with 0 < a < b < c < d <= 60
    for a in range(1, 12):
        if time.time() > deadline:
            break
        for b in range(a + 1, 25):
            if time.time() > deadline:
                break
            for c in range(b + 1, 45):
                if time.time() > deadline:
                    break
                for d in range(c + 1, 61):
                    if not try_set([0, a, b, c, d]):
                        break
    
    # Phase 6: 6-element sets with geometric growth
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
    
    # Phase 7: Sets with one large element and small dense part
    for small_max in range(3, 30):
        if time.time() > deadline:
            break
        for large in range(small_max + 3, small_max * 10 + 1):
            if time.time() > deadline:
                break
            U = list(range(small_max + 1)) + [large]
            try_set(U)
    
    # Phase 8: {0, 1, 2, ..., k} + {m} for various k, m
    for k in range(2, 25):
        if time.time() > deadline:
            break
        for m in range(k + 2, k * 10 + 1):
            if time.time() > deadline:
                break
            U = list(range(k + 1)) + [m]
            try_set(U)
    
    # Phase 9: Arithmetic progressions
    for a in range(1, 20):
        if time.time() > deadline:
            break
        for d in range(1, 30):
            if time.time() > deadline:
                break
            for n in range(4, 12):
                U = [a * i * d for i in range(n)]
                U[0] = 0
                try_set(U)
    
    # Phase 10: Sets of the form {0, 1, p, p^2} for prime p
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in primes:
        if time.time() > deadline:
            break
        try_set([0, 1, p, p * p])
        if p * p * p < 10000:
            try_set([0, 1, p, p * p, p * p * p])
    
    # Phase 11: Fibonacci-like growth
    fib_sets = []
    a, b = 1, 1
    for _ in range(12):
        fib_sets.append(a)
        a, b = b, a + b
    for n in range(3, 10):
        if time.time() > deadline:
            break
        U = [0] + fib_sets[:n]
        try_set(U)
    
    # Phase 12: Lucas-like growth
    lucas = [2, 1]
    for _ in range(12):
        lucas.append(lucas[-1] + lucas[-2])
    for n in range(3, 10):
        if time.time() > deadline:
            break
        U = [0] + lucas[:n]
        try_set(U)
    
    # Phase 13: Powers of 2
    for n in range(3, 15):
        if time.time() > deadline:
            break
        U = [2**i for i in range(n)]
        try_set(U)
    
    # Phase 14: Powers of 3
    for n in range(3, 10):
        if time.time() > deadline:
            break
        U = [3**i for i in range(n)]
        try_set(U)
    
    # Phase 15: Powers of 4, 5, 6
    for base in [4, 5, 6]:
        for n in range(3, 8):
            if time.time() > deadline:
                break
            U = [base**i for i in range(n)]
            try_set(U)
    
    # Phase 16: Mixed growth patterns
    for r1 in [2, 3, 4]:
        for r2 in [2, 3, 4, 5]:
            for n in range(4, 10):
                if time.time() > deadline:
                    break
                U = [0, 1]
                x = 1
                for i in range(n - 2):
                    x *= r1 if i % 2 == 0 else r2
                    U.append(x)
                try_set(U)
    
    # Phase 17: Sets with one element much larger than the rest
    for base_size in range(3, 10):
        if time.time() > deadline:
            break
        for max_val in range(50, 300):
            if time.time() > deadline:
                break
            U = list(range(base_size)) + [max_val]
            try_set(U)
            U = [0, 1]
            x = 3
            for _ in range(base_size - 2):
                U.append(x)
                x *= 3
            U.append(max_val)
            try_set(U)
    
    # Phase 18: Local search from best solution found
    if time.time() < deadline:
        local_search(best_U, max_iters=500)
    
    # Phase 19: Local search from other promising starting points
    if time.time() < deadline - 30:
        for start in [[0, 1, 3, 9], [0, 1, 2, 6], [0, 1, 4, 16], [0, 1, 2, 8], [0, 1, 3, 12]]:
            if time.time() > deadline:
                break
            local_search(start, max_iters=200)
    
    return best_U
