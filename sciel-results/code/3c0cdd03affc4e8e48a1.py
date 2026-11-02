import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def score(U):
        if not U or 0 not in U:
            return 0.0
        U = sorted(set(U))
        n = len(U)
        if n < 2:
            return 0.0
        max_u = U[-1]
        if max_u == 0:
            return 0.0
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 0.0
        
        sums = set()
        diffs = set()
        for i in range(n):
            for j in range(n):
                sums.add(U[i] + U[j])
                diffs.add(U[i] - U[j])
        
        ratio = len(diffs) / len(sums)
        if ratio <= 1:
            return 1.0 # Or 0? Task says maximize. If ratio < 1, score < 1.
        return 1.0 + math.log(ratio) / denom

    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Evaluate seed
    best_score = score(best_set)
    
    # Try powers of 2 minus 1
    for n in range(2, 12):
        if time.time() > deadline:
            break
        U = [2**i - 1 for i in range(n)]
        s = score(U)
        if s > best_score:
            best_score = s
            best_set = U
        
    # Try small variations around the best n (likely n=5)
    # Base: 0, 1, 3, 7, 15
    base = [0, 1, 3, 7, 15]
    
    # Try shifting or small modifications
    # Since the search space is small for n=5, we can do a local search
    # But 120s limit is tight for large searches.
    # The powers of 2 are very strong.
    
    # Check n=4,5,6 specifically with exact calculation
    candidates = []
    for n in [3, 4, 5, 6, 7]:
        U = [2**i - 1 for i in range(n)]
        candidates.append(U)
        
    # Add a few other known good structures?
    # E.g. Mian-Chowla sequence start: 0, 1, 3, 7, 15, 31... (same as 2^i-1)
    # 0, 1, 3, 7, 15 is optimal for this family.
    
    # Let's verify n=5 is best
    for U in candidates:
        s = score(U)
        if s > best_score:
            best_score = s
            best_set = U

    return best_set
