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
    n = len(U)
    U_arr = np.array(U, dtype=np.int64)
    # Compute all sums and differences efficiently
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    sum_set_size = len(np.unique(sums))
    diff_set_size = len(np.unique(diffs))
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
    deadline = start_time + 145
    
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
    for base in [2, 3, 4, 5, 6, 7, 8]:
        for n in range(3, 9):
            if time.time() > deadline:
                break
            U = [0]
            x = 1
            for _ in range(n):
                U.append(x)
                x *= base
            try_set(U)
    
    # Try {0, 1, a, a*b} type sets
    for a in range(2, 40):
        if time.time() > deadline:
            break
        for b in range(2, 25):
            if time.time() > deadline:
                break
            c = a * b
            if c > 800:
                break
            try_set([0, 1, a, c])
    
    # Try {0, 1, a, b, c} with geometric-like spacing
    for a in range(2, 20):
        if time.time() > deadline:
            break
        for b in range(a + 1, 50):
            if time.time() > deadline:
                break
            for c in range(b + 1, 150):
                if time.time() > deadline:
                    break
                try_set([0, 1, a, b, c])
    
    # Exhaustive search over 4-element sets {0, a, b, c} with 0 < a < b < c <= 80
    for a in range(1, 30):
        if time.time() > deadline:
            break
        for b in range(a + 1, 60):
            if time.time() > deadline:
                break
            for c in range(b + 1, 81):
                if not try_set([0, a, b, c]):
                    break
    
    # Exhaustive search over 5-element sets {0, a, b, c, d} with 0 < a < b < c < d <= 50
    for a in range(1, 10):
        if time.time() > deadline:
            break
        for b in range(a + 1, 20):
            if time.time() > deadline:
                break
            for c in range(b + 1, 35):
                if time.time() > deadline:
                    break
                for d in range(c + 1, 51):
                    if not try_set([0, a, b, c, d]):
                        break
    
    # Try 6-element sets with geometric growth
    for a in range(1, 6):
        if time.time() > deadline:
            break
        for b in range(a + 1, 12):
            if time.time() > deadline:
                break
            for c in range(b + 1, 20):
                if time.time() > deadline:
                    break
                for d in range(c + 1, 35):
                    if time.time() > deadline:
                        break
                    for e in range(d + 1, 60):
                        if not try_set([0, a, b, c, d, e]):
                            break
    
    # Try sets with one large element and small dense part
    for small_max in range(5, 25):
        if time.time() > deadline:
            break
        for large in range(small_max + 5, small_max * 6 + 1):
            if time.time() > deadline:
                break
            U = list(range(small_max + 1)) + [large]
            try_set(U)
    
    # Try {0, 1, 2, ..., k} + {m} for various k, m
    for k in range(3, 20):
        if time.time() > deadline:
            break
        for m in range(k + 2, k * 6 + 1):
            if time.time() > deadline:
                break
            U = list(range(k + 1)) + [m]
            try_set(U)
    
    # Try arithmetic progressions
    for a in range(1, 10):
        if time.time() > deadline:
            break
        for d in range(1, 20):
            if time.time() > deadline:
                break
            for n in range(4, 10):
                U = [a * i * d for i in range(n)]
                U[0] = 0
                try_set(U)
    
    # Try Fibonacci-like sequences
    fib_candidates = []
    a, b = 1, 2
    while a <= 500 and len(fib_candidates) < 10:
        fib_candidates.append(a)
        a, b = b, a + b
    for n in range(3, len(fib_candidates) + 1):
        if time.time() > deadline:
            break
        U = [0] + fib_candidates[:n]
        try_set(U)
    
    # Try powers with offsets
    for base in [2, 3, 4, 5]:
        for offset in range(0, base):
            if time.time() > deadline:
                break
            for n in range(3, 8):
                U = [0]
                x = 1
                for i in range(n):
                    U.append(x + offset)
                    x *= base
                if len(set(U)) == len(U):
                    try_set(U)
    
    # Try {0, 1, 3, 9, x} for various x
    for x in range(10, 300):
        if time.time() > deadline:
            break
        try_set([0, 1, 3, 9, x])
    
    # Try {0, 1, 3, 9, 27, x} for various x
    for x in range(28, 500):
        if time.time() > deadline:
            break
        try_set([0, 1, 3, 9, 27, x])
    
    # Try {0, 1, 3, 9, 27, 81, x} for various x
    for x in range(82, 1000):
        if time.time() > deadline:
            break
        try_set([0, 1, 3, 9, 27, 81, x])
    
    # Try {0, 1, 3, 9, 27, 81, 243, x} for various x
    for x in range(244, 2000):
        if time.time() > deadline:
            break
        try_set([0, 1, 3, 9, 27, 81, 243, x])
    
    # Try sets with two large elements: {0, 1, ..., k, m, n}
    for k in range(3, 15):
        if time.time() > deadline:
            break
        for m in range(k + 2, k * 4 + 1):
            if time.time() > deadline:
                break
            for n in range(m + 2, m * 3 + 1):
                if time.time() > deadline:
                    break
                U = list(range(k + 1)) + [m, n]
                if len(set(U)) == len(U):
                    try_set(U)
    
    # Try {0, 1, a, b, c, d} with geometric progression
    for a in range(2, 15):
        if time.time() > deadline:
            break
        for b in range(a + 1, a * 3 + 1):
            if time.time() > deadline:
                break
            for c in range(b + 1, b * 3 + 1):
                if time.time() > deadline:
                    break
                for d in range(c + 1, min(c * 3 + 1, 500)):
                    if time.time() > deadline:
                        break
                    try_set([0, 1, a, b, c, d])
    
    # Try {0, 1, 2, 4, 8, 16, x} - powers of 2 with extra element
    for x in range(17, 500):
        if time.time() > deadline:
            break
        try_set([0, 1, 2, 4, 8, 16, x])
    
    # Try {0, 1, 2, 4, 8, 16, 32, x}
    for x in range(33, 1000):
        if time.time() > deadline:
            break
        try_set([0, 1, 2, 4, 8, 16, 32, x])
    
    # Try {0, 1, 2, 4, 8, 16, 32, 64, x}
    for x in range(65, 2000):
        if time.time() > deadline:
            break
        try_set([0, 1, 2, 4, 8, 16, 32, 64, x])
    
    # Try mixed base: {0, 1, 3, 9, 27, 81, 243, x} with x in specific ranges
    for x in list(range(244, 400)) + list(range(400, 800, 10)):
        if time.time() > deadline:
            break
        try_set([0, 1, 3, 9, 27, 81, 243, x])
    
    # Try {0, 1, 3, 9, 27, 81, 243, 729, x}
    for x in list(range(730, 1000)) + list(range(1000, 2000, 20)):
        if time.time() > deadline:
            break
        try_set([0, 1, 3, 9, 27, 81, 243, 729, x])
    
    # Local improvement: try adding/removing elements from best_U
    for _ in range(50):
        if time.time() > deadline:
            break
        # Try adding a random element
        import random
        max_val = max(best_U) * 2 + 100
        for _ in range(20):
            if time.time() > deadline:
                break
            new_elem = random.randint(1, max_val)
            candidate = sorted(set(best_U + [new_elem]))
            try_set(candidate)
        
        # Try removing an element (except 0)
        if len(best_U) > 2:
            for i in range(1, len(best_U)):
                if time.time() > deadline:
                    break
                candidate = best_U[:i] + best_U[i+1:]
                try_set(candidate)
    
    return best_U
