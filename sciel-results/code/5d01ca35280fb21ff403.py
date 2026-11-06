"""
Program for integer sum-product extremal sets.
Approach: Deterministic grid search over truncated geometric progressions with
rational ratio p/q (small integers), combined with structured candidates
(1..N, powers of 2, etc.) and a fast local search to refine the best candidate.
"""
import math
import random
import time

def _score_set(A):
    """Compute max(|A+A|, |A*A|) for a set/list of distinct positive integers."""
    if len(set(A)) != len(A):
        return float('inf')
    if any(x < 1 for x in A):
        return float('inf')
    # Sumset
    sset = set()
    for i in range(len(A)):
        ai = A[i]
        for j in range(i, len(A)):
            sset.add(ai + A[j])
    # Productset
    pset = set()
    for i in range(len(A)):
        ai = A[i]
        for j in range(i, len(A)):
            pset.add(ai * A[j])
    return max(len(sset), len(pset))

def construct():
    n = 20
    max_val = 10000
    best_score = float('inf')
    best_set = None
    
    # 1. Structured candidates
    candidates = []
    
    # 1..20
    candidates.append(list(range(1, n + 1)))
    
    # Powers of 2 (as many as fit, pad with small ints)
    p2 = [2**i for i in range(n)]
    if p2[-1] <= max_val:
        candidates.append(p2)
    else:
        # Take as many powers of 2 as fit, then add smallest missing
        p2_fit = [2**i for i in range(n) if 2**i <= max_val]
        remaining = sorted(set(range(1, max_val + 1)) - set(p2_fit))
        p2_fit += remaining[:n - len(p2_fit)]
        candidates.append(p2_fit[:n])
    
    # Powers of 3
    p3 = [3**i for i in range(n)]
    if p3[-1] <= max_val:
        candidates.append(p3)
    else:
        p3_fit = [3**i for i in range(n) if 3**i <= max_val]
        remaining = sorted(set(range(1, max_val + 1)) - set(p3_fit))
        p3_fit += remaining[:n - len(p3_fit)]
        candidates.append(p3_fit[:n])
    
    # 2..21
    candidates.append(list(range(2, n + 2)))
    
    # 3..22
    candidates.append(list(range(3, n + 3)))
    
    # 1, 2, 4, 8, ... + small numbers
    for start in [1, 2, 3, 4, 5]:
        for ratio in [1.5, 1.6, 1.7, 1.8, 1.9]:
            A = []
            for i in range(n):
                term = int(round(start * (ratio ** i)))
                if term > max_val:
                    break
                if term not in A:
                    A.append(term)
            if len(A) < n:
                # Pad with smallest missing
                remaining = sorted(set(range(1, max_val + 1)) - set(A))
                A += remaining[:n - len(A)]
            if len(A) == n and all(1 <= x <= max_val for x in A) and len(set(A)) == n:
                candidates.append(A)
    
    # Evaluate all structured candidates
    for cand in candidates:
        if len(cand) != n:
            continue
        if len(set(cand)) != n:
            continue
        if any(x < 1 or x > max_val for x in cand):
            continue
        s = _score_set(cand)
        if s < best_score:
            best_score = s
            best_set = cand[:]
    
    # 2. Grid search over rational GPs p/q
    # For a GP with ratio p/q, terms are a * (p/q)^i
    # To keep integers, a must be divisible by q^(n-1)
    # Term i = a * p^i / q^i
    # We need all terms <= max_val and distinct
    
    for q in range(1, 8):
        for p in range(q + 1, q + 15):
            if math.gcd(p, q) != 1:
                continue
            q_pow = q ** (n - 1)
            p_pow = p ** (n - 1)
            if p_pow > max_val:
                continue
            max_k = max_val // p_pow
            if max_k < 1:
                continue
            # Try a few values of k
            for k in range(1, min(max_k + 1, 20)):
                a = k * q_pow
                A = []
                valid = True
                for i in range(n):
                    num = a * (p ** i)
                    den = q ** i
                    if num % den != 0:
                        valid = False
                        break
                    term = num // den
                    if term > max_val:
                        valid = False
                        break
                    A.append(term)
                if not valid:
                    continue
                if len(set(A)) != n:
                    continue
                s = _score_set(A)
                if s < best_score:
                    best_score = s
                    best_set = A[:]
    
    # 3. Local search to improve best_set
    if best_set is None:
        best_set = list(range(1, n + 1))
    
    # Ensure best_set is sorted
    best_set.sort()
    current_score = best_score
    
    # Time-limited local search
    start_time = time.time()
    time_limit = 50  # seconds, leaving buffer for evaluation
    
    random.seed(42)
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Try swapping each element with a random other value
        for i in range(n):
            if (time.time() - start_time) > time_limit:
                break
            old_val = best_set[i]
            # Try replacing with a random value
            for _ in range(50):
                new_val = random.randint(1, max_val)
                if new_val == old_val or new_val in best_set:
                    continue
                best_set[i] = new_val
                if len(set(best_set)) != n:
                    best_set[i] = old_val
                    continue
                s = _score_set(best_set)
                if s < current_score:
                    current_score = s
                    improved = True
                else:
                    best_set[i] = old_val
    
    return best_set
