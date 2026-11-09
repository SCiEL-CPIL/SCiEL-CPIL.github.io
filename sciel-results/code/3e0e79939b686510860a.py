import numpy as np
from itertools import product

def construct():
    # We will search for a better set than the first 20 3-smooth numbers.
    # The current score is 1.3451.
    # We will try:
    # 1. Pure Arithmetic Progressions
    # 2. Pure Geometric Progressions
    # 3. Hybrid AP + GP
    # 4. Subsets of 3-smooth numbers (maybe not the smallest 20?)
    
    best_set = None
    best_score = float('inf')
    
    # Helper to score a set
    def get_score(A):
        if len(A) != 20 or len(set(A)) != 20 or max(A) > 10000 or min(A) < 1:
            return float('inf')
        sums = set()
        prods = set()
        for a in A:
            for b in A:
                sums.add(a + b)
                prods.add(a * b)
        return max(len(sums), len(prods))
    
    # Candidate 1: Current 3-smooth
    candidates = set()
    for i in range(15):
        for j in range(10):
            val = (2 ** i) * (3 ** j)
            if val <= 10000:
                candidates.add(val)
    A_smooth = sorted(list(candidates))[:20]
    s = get_score(A_smooth)
    if s < best_score:
        best_score = s
        best_set = A_smooth

    # Candidate 2: Pure APs
    # A = {a + i*d for i in 0..19}
    # a >= 1, d >= 1
    # Max element a + 19d <= 10000
    # We want small max element to reduce product range?
    # Try small a, d.
    for a in range(1, 100):
        for d in range(1, 100):
            A = [a + i * d for i in range(20)]
            if A[-1] > 10000:
                continue
            s = get_score(A)
            if s < best_score:
                best_score = s
                best_set = A

    # Candidate 3: Pure GPs
    # A = {a * r^i for i in 0..19}
    # r >= 2 (integer)
    for r in range(2, 10):
        for a in range(1, 100):
            A = [a * (r ** i) for i in range(20)]
            if A[-1] > 10000:
                continue
            s = get_score(A)
            if s < best_score:
                best_score = s
                best_set = A

    # Candidate 4: Hybrid AP + GP
    # A = AP_part union GP_part
    # Sizes k and m, k+m=20
    # To keep search space small, try k=10, m=10; k=5, m=15; etc.
    # AP: a + i*d, i in 0..k-1
    # GP: b * r^j, j in 0..m-1
    # Ensure disjointness.
    
    for k in [5, 10, 15]:
        m = 20 - k
        # Search small parameters
        for a in range(1, 20):
            for d in range(1, 5):
                AP_part = [a + i * d for i in range(k)]
                if AP_part[-1] > 10000:
                    continue
                for r in range(2, 5):
                    for b in range(1, 20):
                        GP_part = [b * (r ** j) for j in range(m)]
                        if GP_part[-1] > 10000:
                            continue
                        A = list(set(AP_part + GP_part))
                        if len(A) != 20:
                            continue
                        s = get_score(A)
                        if s < best_score:
                            best_score = s
                            best_set = A

    # Candidate 5: Random search around 3-smooth?
    # Maybe perturb the 3-smooth set?
    # Given time constraints, stick to structured search.
    
    return best_set

def score(A):
    sums = set()
    prods = set()
    for a in A:
        for b in A:
            sums.add(a + b)
            prods.add(a * b)
    return max(len(sums), len(prods))
