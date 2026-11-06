import math
import time

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
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            if math.gcd(vals[i], vals[j]) > 1:
                return False
    return True

def _validate(A, B):
    if len(A) != 3 or len(B) != 3:
        return 0
    if len(set(A)) != 3 or len(set(B)) != 3:
        return 0
    for x in A:
        if x < 1 or x > 128:
            return 0
    for x in B:
        if x < 1 or x > 128:
            return 0
    sums = [a + b for a in A for b in B]
    distinct = list(set(sums))
    if not _pairwise_coprime(distinct):
        return 0
    return len(distinct)

def construct():
    """Return two lists A,B, each containing exactly 3 distinct integers in [1,128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximize the cardinality of A+B."""
    
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    # Precomputed optimal solution:
    # A = [4, 16, 40], B = [1, 3, 7]
    # Sums:
    # 4+1=5, 4+3=7, 4+7=11
    # 16+1=17, 16+3=19, 16+7=23
    # 40+1=41, 40+3=43, 40+7=47
    # All 9 sums are distinct primes, hence pairwise coprime.
    # Cardinality is 9 (maximum possible for 3x3 sets with distinct sums).
    
    A = [4, 16, 40]
    B = [1, 3, 7]
    
    # Verification
    sums = [a + b for a in A for b in B]
    distinct_sums = list(set(sums))
    
    valid = True
    if len(distinct_sums) != 9:
        valid = False
    else:
        for i in range(9):
            for j in range(i + 1, 9):
                if math.gcd(distinct_sums[i], distinct_sums[j]) > 1:
                    valid = False
                    break
            if not valid:
                break
                
    if valid:
        return [A, B]
        
    # Fallback search if hardcoded solution fails (shouldn't happen)
    best_A, best_B = None, None
    best_score = 0
    
    # Try known valid fallback
    fa, fb = [1, 7, 13], [4, 10, 16]
    s = _validate(fa, fb)
    if s > best_score:
        best_score = s
        best_A, best_B = fa, fb
    
    if best_A is not None and time.time() < deadline:
        return [best_A, best_B]
    
    # Last resort: return something valid even if suboptimal
    return [[1, 7, 13], [4, 10, 16]]
