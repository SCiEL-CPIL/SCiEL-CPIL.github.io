"""Direct construction: find A, B with 9 distinct pairwise-coprime sums (all primes)."""
import math
import time
import itertools

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _pairwise_coprime(vals):
    """Check all pairs are coprime. Returns True if valid."""
    n = len(vals)
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(vals[i], vals[j]) > 1:
                return False
    return True

def construct():
    """Return two lists A,B, each containing exactly 3 distinct integers in [1,128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximize the cardinality of A+B."""
    t_start = time.time()
    DEADLINE = 85  # safety margin before 90s kill

    # Precompute primes up to 256
    MAX_SUM = 256
    is_prime = [False] * (MAX_SUM + 1)
    is_prime[2] = True
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, MAX_SUM + 1, i):
                is_prime[j] = False
    
    # Strategy: Fix B to small prime sets, search for A such that all 9 sums are distinct primes.
    # This reduces the search space dramatically.
    
    # Candidate B sets (small, with distinct differences to ensure distinct sums)
    # B differences must be such that a+b values don't collide.
    # If B = {b1, b2, b3} with differences d1=b2-b1, d2=b3-b2, d1+d2=b3-b1
    # Then for A = {a1, a2, a3} with differences e1=a2-a1, e2=a3-a2, e1+e2=a3-a1
    # Sums collide iff some difference in A equals some difference in B (with sign).
    # So we need A's difference set and B's difference set to be disjoint.
    
    # Try several small B sets
    candidate_Bs = [
        [2, 3, 5],      # diffs: 1, 2, 3
        [2, 3, 7],      # diffs: 1, 4, 5
        [2, 5, 7],      # diffs: 3, 2, 5
        [3, 5, 7],      # diffs: 2, 2, 4 -> not distinct, skip
        [2, 3, 11],     # diffs: 1, 8, 9
        [2, 5, 11],     # diffs: 3, 6, 9
        [3, 5, 11],     # diffs: 2, 6, 8
        [2, 7, 11],     # diffs: 5, 4, 9
        [3, 7, 11],     # diffs: 4, 4, 8 -> not distinct
        [5, 7, 11],     # diffs: 2, 4, 6
        [2, 3, 13],     # diffs: 1, 10, 11
        [2, 5, 13],     # diffs: 3, 8, 11
        [3, 5, 13],     # diffs: 2, 8, 10
        [2, 7, 13],     # diffs: 5, 6, 11
        [5, 7, 13],     # diffs: 2, 6, 8
        [7, 11, 13],    # diffs: 4, 2, 6
    ]
    
    # Filter B sets with distinct differences (so 3 distinct elements give 3 distinct pairwise diffs)
    valid_Bs = []
    for B in candidate_Bs:
        B = sorted(B)
        if len(set(B)) != 3:
            continue
        d1 = B[1] - B[0]
        d2 = B[2] - B[1]
        d3 = B[2] - B[0]
        if len({d1, d2, d3}) == 3:  # all three pairwise differences distinct
            valid_Bs.append(B)
    
    if not valid_Bs:
        valid_Bs = [[2, 3, 5]]
    
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = 0
    
    def score_fn(A, B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        distinct = list(sums)
        if not _pairwise_coprime(distinct):
            return 0
        return len(distinct)
    
    best_score = score_fn(best_A, best_B)
    
    # For each valid B, search for A such that all 9 sums are prime
    found = False
    for B in valid_Bs:
        if time.time() - t_start > DEADLINE:
            break
        
        b_diffs = set()
        for i in range(3):
            for j in range(i+1, 3):
                b_diffs.add(B[j] - B[i])
        
        # A must have differences disjoint from b_diffs
        # A = {a, a+d1, a+d1+d2} where {d1, d2, d1+d2} disjoint from b_diffs
        # Try small d1, d2
        max_d = 128 - max(B)  # to keep a+d1+d2+max(B) <= 128+256
        
        for d1 in range(1, 40):
            if time.time() - t_start > DEADLINE:
                break
            for d2 in range(1, 40):
                if d2 <= d1:
                    continue
                if time.time() - t_start > DEADLINE:
                    break
                a_diffs = {d1, d2, d1 + d2}
                if a_diffs & b_diffs:
                    continue
                
                # Try small base a values
                min_a = 1
                max_a = 128 - (d1 + d2)
                # Also need a + min(B) >= 2 (always true for a>=1, B>=2)
                # and a + d1 + d2 + max(B) <= 256 (for prime check)
                
                for a in range(min_a, min(max_a + 1, 129)):
                    if time.time() - t_start > DEADLINE:
                        break
                    A = [a, a + d1, a + d1 + d2]
                    if A[0] < 1 or A[2] > 128:
                        continue
                    
                    # Check all 9 sums are prime
                    all_prime = True
                    for x in A:
                        for y in B:
                            s = x + y
                            if s > MAX_SUM or not is_prime[s]:
                                all_prime = False
                                break
                        if not all_prime:
                            break
                    
                    if all_prime:
                        best_A = A
                        best_B = B
                        best_score = 9
                        found = True
                        break
                
                if found:
                    break
            
            if found:
                break
        
        if found:
            break
    
    if found:
        return [best_A, best_B]
    
    # Fallback: broader search with pairwise coprime (not necessarily all prime)
    odds = list(range(1, 51, 2))
    evens = list(range(2, 51, 2))
    
    for A in itertools.combinations(odds, 3):
        if time.time() - t_start > DEADLINE:
            break
        A = list(A)
        a_diffs = set()
        for i in range(3):
            for j in range(i+1, 3):
                a_diffs.add(A[j] - A[i])
        
        for B in itertools.combinations(evens, 3):
            if time.time() - t_start > DEADLINE:
                break
            B = list(B)
            b_diffs = set()
            for i in range(3):
                for j in range(i+1, 3):
                    b_diffs.add(B[j] - B[i])
            
            if a_diffs & b_diffs:
                continue
            
            sums = set()
            for a in A:
                for b in B:
                    sums.add(a + b)
            
            if len(sums) != 9:
                continue
            
            if _pairwise_coprime(list(sums)):
                best_A = A
                best_B = B
                best_score = 9
                found = True
                break
        
        if found:
            break
    
    if found:
        return [best_A, best_B]
    
    return [best_A, best_B]
