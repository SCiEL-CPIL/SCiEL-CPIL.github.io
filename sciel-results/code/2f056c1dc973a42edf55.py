import math
from itertools import combinations

def construct():
    best_score = 0.0
    best_set = [0, 1, 3, 7]  # fallback
    
    # Strategy: search for small sets A containing 0 that maximize
    # score = 1 + log(|A-A| / |A+A|) / log(2*max(A)+1)
    #
    # Key insight: Sidon sets (B_2 sequences) have |A+A| ~ n^2/2 and |A-A| ~ n^2,
    # giving ratio ~ 2. The log(2) in numerator is constant.
    # The denominator log(2*max(A)+1) grows with max(A).
    # So we want sets with high ratio AND small max(A).
    #
    # For small n, we can brute-force search subsets of {0, ..., L} of size n.
    
    def calc_score(A):
        if not A or 0 not in A:
            return -1
        n = len(A)
        if n < 2:
            return -1
        max_a = max(A)
        if max_a == 0:
            return -1
        # Compute sumset and diffset
        sum_set = set()
        diff_set = set()
        for a in A:
            for b in A:
                sum_set.add(a + b)
                diff_set.add(a - b)
        P = len(sum_set)
        D = len(diff_set)
        if P == 0:
            return -1
        ratio = D / P
        if ratio <= 0:
            return -1
        denom = math.log(2 * max_a + 1)
        if denom <= 0:
            return -1
        return 1.0 + math.log(ratio) / denom
    
    # Brute force search for small sets
    # For each max range L and size n, enumerate all subsets of {0,...,L} of size n containing 0
    
    # We'll search L from 1 to 60, n from 2 to 10
    # But C(L, n-1) can be large. Let's be selective.
    
    # First, try known good constructions: Sidon sequences
    # Minimal Sidon sequences (starting with 0):
    sidon_sequences = [
        [0, 1, 3],
        [0, 1, 3, 7],
        [0, 1, 3, 7, 12],
        [0, 1, 3, 7, 12, 18],
        [0, 1, 3, 7, 12, 18, 25],
        [0, 1, 3, 7, 12, 18, 25, 33],
        [0, 1, 3, 7, 12, 18, 25, 33, 42],
        [0, 1, 3, 7, 12, 18, 25, 33, 42, 52],
    ]
    
    for A in sidon_sequences:
        s = calc_score(A)
        if s > best_score:
            best_score = s
            best_set = A[:]
    
    # Now brute force search for potentially better sets
    # For small L and n, enumerate all subsets
    
    # L up to 30, n up to 8: C(30, 7) ~ 2M which is too slow in pure Python
    # Let's limit to L up to 20, n up to 6: C(20, 5) = 15504, fast
    # L up to 25, n up to 7: C(25, 6) = 177100, manageable
    
    for L in range(2, 26):
        for n in range(2, min(8, L + 1)):
            # Enumerate all subsets of {1, ..., L} of size n-1, prepend 0
            # This gives all sets of size n with max <= L and containing 0
            count = 0
            for combo in combinations(range(1, L + 1), n - 1):
                A = [0] + list(combo)
                s = calc_score(A)
                count += 1
                if s > best_score:
                    best_score = s
                    best_set = A[:]
    
    # Also try some structured families
    # Geometric-like: {0, 1, 3, 9, 27, ...}
    for base in [2, 3, 4, 5]:
        A = [0]
        val = 1
        while val < 200:
            A.append(val)
            val *= base
        s = calc_score(A)
        if s > best_score:
            best_score = s
            best_set = A[:]
    
    # Try {0, 1, k, k^2, ...} style
    for k in range(2, 10):
        A = [0, 1]
        val = k
        while val < 200:
            A.append(val)
            val *= k
        s = calc_score(A)
        if s > best_score:
            best_score = s
            best_set = A[:]
    
    # Try sets with a large gap: {0, 1, 2, ..., m, M, M+1, ..., M+k}
    # This can sometimes help
    for m in range(1, 10):
        for k in range(1, 10):
            for M in range(m + 2, m + k + 20):
                A = list(range(m + 1)) + list(range(M, M + k + 1))
                if 0 in A:
                    s = calc_score(A)
                    if s > best_score:
                        best_score = s
                        best_set = A[:]
    
    return best_set
