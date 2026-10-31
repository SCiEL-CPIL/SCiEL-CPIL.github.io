"""Construct a finite set U of nonnegative integers containing 0 to maximize
the sumset/difference-set exponent lower bound.

Strategy: U = A ∪ B where A and B are scaled arithmetic progressions with
multiplicatively independent scaling factors. This minimizes sumset collisions
while maximizing difference-set spread.
"""
import math
import time

def _score_set(U):
    """Compute the score for a set U (must contain 0)."""
    if 0 not in U:
        return 0.0
    U = list(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    maxU = max(U)
    if maxU < 1:
        return 0.0
    
    # Compute |U+U| and |U-U|
    U_set = set(U)
    sum_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
    
    diff_set = set()
    for a in U:
        for b in U:
            diff_set.add(a - b)
    
    sum_size = len(sum_set)
    diff_size = len(diff_set)
    
    if sum_size == 0:
        return 0.0
    
    log2_max = math.log2(2 * maxU + 1)
    if log2_max == 0:
        return 0.0
    
    ratio = diff_size / sum_size
    if ratio <= 0:
        return 0.0
    
    score = 1.0 + math.log(ratio) / log2_max
    return score


def _construct_two_block(a, b, m, n):
    """
    Construct U = {i*a : 0 <= i < m} ∪ {b + j*a*b : 0 <= j < n}
    where a, b are scaling factors (multiplicatively independent ideally).
    """
    A = [i * a for i in range(m)]
    B = [b + j * a * b for j in range(n)]
    U = list(set(A + B))
    U.sort()
    if 0 not in U:
        U.append(0)
        U.sort()
    return U


def _construct_two_block_v2(a, b, m, n, offset):
    """
    Variant: U = {i*a : 0 <= i < m} ∪ {offset + j*b : 0 <= j < n}
    More general two-block construction.
    """
    A = [i * a for i in range(m)]
    B = [offset + j * b for j in range(n)]
    U = list(set(A + B))
    U.sort()
    if 0 not in U:
        U.append(0)
        U.sort()
    return U


def construct():
    """
    Construct a finite set U of nonnegative integers containing 0
    to maximize the score: 1 + log(|U-U|/|U+U|) / log(2*max(U)+1).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    # Strategy: Search over two-block constructions
    # U = {i*a : 0 <= i < m} ∪ {b + j*(a*b) : 0 <= j < n}
    # The key insight: if a and b are multiplicatively independent (e.g., powers of different primes),
    # then U+U has size ~m^2 + n^2 (few collisions) while U-U has size ~m^2 + n^2 + 2*m*n (more spread)
    
    # First, try the paper's known constructions as reference
    # Size 2003 construction (from literature): use powers of 2 and 3
    # A = {0, 1, 2, ..., m-1} * a, B = {b, b + a*b, ...}
    
    # Let's systematically search parameter space
    # For two-block: |U+U| ≈ m^2 + n^2 + 2*m*n (if no overlaps in sums)
    # |U-U| ≈ m^2 + n^2 + 2*m*n (similar but with more spread due to negative values)
    # Actually for differences, we get more distinct values because -a*i + b + j*a*b 
    # can take many more values
    
    # Key: max(U) = max((m-1)*a, b + (n-1)*a*b)
    # Score = 1 + log(|U-U|/|U+U|) / log(2*max(U)+1)
    
    # To maximize: want |U-U|/|U+U| large, and max(U) small
    
    # Let's try: a = p (small prime), b = q^k for some prime q
    # m and n balanced
    
    # Parameter search
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # Try various combinations
    candidates = []
    
    # Type 1: A = {0, a, 2a, ..., (m-1)a}, B = {b, b+ab, b+2ab, ..., b+(n-1)ab}
    for a in primes[:15]:
        for b in primes[:15]:
            if a == b:
                continue
            for m in range(2, 50):
                for n in range(2, 50):
                    if time.time() > deadline:
                        break
                    U = _construct_two_block(a, b, m, n)
                    s = _score_set(U)
                    if s > best_score:
                        best_score = s
                        best_set = U
                    candidates.append((s, U))
                if time.time() > deadline:
                    break
        if time.time() > deadline:
            break
    
    if time.time() > deadline:
        return best_set
    
    # Type 2: More general offset
    for a in primes[:10]:
        for b in primes[:10]:
            if a == b:
                continue
            for offset in [b, b*2, b*3, a*b, a*b + b]:
                for m in range(3, 40):
                    for n in range(3, 40):
                        if time.time() > deadline:
                            break
                        U = _construct_two_block_v2(a, b, m, n, offset)
                        s = _score_set(U)
                        if s > best_score:
                            best_score = s
                            best_set = U
                    if time.time() > deadline:
                        break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if time.time() > deadline:
        return best_set
    
    # Type 3: Three-block or other structures
    # U = {i*a} ∪ {b + j*c} ∪ {d + k*e}
    # But let's focus on two-block with better parameters
    
    # Try larger ranges with coarser grid
    for a in [2, 3, 5, 7, 11, 13]:
        for b in [3, 5, 7, 11, 13, 17, 19, 23]:
            if a == b:
                continue
            for m in range(5, 100, 3):
                for n in range(5, 100, 3):
                    if time.time() > deadline:
                        break
                    U = _construct_two_block(a, b, m, n)
                    s = _score_set(U)
                    if s > best_score:
                        best_score = s
                        best_set = U
                if time.time() > deadline:
                    break
        if time.time() > deadline:
            break
    
    if time.time() > deadline:
        return best_set
    
    # Type 4: Use geometric-like progressions within blocks
    # A = {a^i : i in range(m)} type structures mixed with arithmetic
    # Actually, let's try: U = {i * a^i : 0 <= i < m} ∪ {b * j * b^j : 0 <= j < n}
    # This gives very sparse sets with large max but potentially good ratio
    
    for a in [2, 3]:
        for b in [2, 3, 5]:
            for m in range(3, 20):
                for n in range(3, 20):
                    if time.time() > deadline:
                        break
                    A = [i * (a ** i) for i in range(m)]
                    B = [b + j * (b ** j) for j in range(n)]
                    U = list(set(A + B))
                    U.sort()
                    if 0 not in U:
                        U.append(0)
                        U.sort()
                    s = _score_set(U)
                    if s > best_score:
                        best_score = s
                        best_set = U
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Type 5: Mixed base representation
    # U = {i * p + j * q : 0 <= i < m, 0 <= j < n} where p, q are coprime
    # This is a 2D grid, |U+U| ~ 2m * 2n = 4mn, |U-U| ~ (2m-1)(2n-1) ≈ 4mn
    # Ratio ≈ 1, not great. But with non-coprime or structured offsets...
    
    for p in [2, 3, 5, 7]:
        for q in [3, 5, 7, 11, 13]:
            if p == q:
                continue
            for m in range(2, 30):
                for n in range(2, 30):
                    if time.time() > deadline:
                        break
                    U = set()
                    for i in range(m):
                        for j in range(n):
                            U.add(i * p + j * q)
                    U = list(U)
                    U.sort()
                    if 0 not in U:
                        U.append(0)
                        U.sort()
                    s = _score_set(U)
                    if s > best_score:
                        best_score = s
                        best_set = U
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return best_set


# Keep the original function name as required
# The construct() function above is the entry point

if __name__ == "__main__":
    U = construct()
    print(f"Set size: {len(U)}")
    print(f"Max element: {max(U)}")
    print(f"Score: {_score_set(U)}")
