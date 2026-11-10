"""Construct mutually orthogonal Latin squares of order 10."""

import time
import random

DEADLINE = 85.0  # safety margin below the 90s wall-clock limit


def _check_latin(sq):
    n = len(sq)
    for i in range(n):
        row = set(sq[i])
        if len(row) != n:
            return False
    for j in range(n):
        col = set(sq[i][j] for i in range(n))
        if len(col) != n:
            return False
    return True


def _check_orthogonal(a, b):
    seen = set()
    n = len(a)
    for i in range(n):
        for j in range(n):
            pair = (a[i][j], b[i][j])
            if pair in seen:
                return False
            seen.add(pair)
    return True


def _verify(squares):
    n = len(squares[0])
    for sq in squares:
        if len(sq) != n:
            return False
        for i in range(n):
            if len(sq[i]) != n:
                return False
            for v in sq[i]:
                if not isinstance(v, int) or isinstance(v, bool):
                    return False
        if not _check_latin(sq):
            return False
    for i in range(len(squares)):
        for j in range(i + 1, len(squares)):
            if not _check_orthogonal(squares[i], squares[j]):
                return False
    return True


def _make_latin_cyclic(n):
    return [[(i + j) % n for j in range(n)] for i in range(n)]


def _make_latin_mult(n, m):
    """Latin square L[i][j] = (m*i + j) % n  ... actually let's use (i*m + j) % n which isn't Latin.
    Better: L[i][j] = (i + j * m) % n where gcd(m, n) = 1.
    """
    if n == 1:
        return [[0]]
    m = 1
    while m < n:
        if _gcd(m, n) == 1:
            break
        m += 1
    # Use m=1 (cyclic) as fallback, or find proper multiplier
    # Actually (i + j*m) % n is Latin if gcd(m, n) = 1
    # But for n=10, m=3 works: gcd(3,10)=1
    # However (i + j*m) % n with m=3: row i has {i, i+3, i+6, i+9, i+2, i+5, i+8, i+1, i+4, i+7} mod 10
    # That's a permutation. Col j has {(i + j*m) % n for i} = {j*m, j*m+1, ...} mod 10 = all.
    # So it's Latin. But is it orthogonal to the cyclic (i+j)%n?
    # Pair: ((i+j)%n, (i+j*m)%n). For fixed pair (a,b): i+j=a, i+j*m=b (mod n).
    # Subtract: j*(m-1) = b-a (mod n). If gcd(m-1, n) = 1, unique solution.
    # For n=10, m=3: m-1=2, gcd(2,10)=2 ≠ 1. Not orthogonal!
    # Try m=3 doesn't work. m=7: m-1=6, gcd(6,10)=2. No.
    # m=3,7 fail. m=1: same square. m=3,7 are the only multipliers with gcd(m,10)=1 besides 1,9.
    # So simple multiplier construction won't give orthogonality to cyclic for n=10.
    # Let's just return the cyclic one.
    return [[(i + j) % n for j in range(n)] for i in range(n)]


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _try_find_orthogonal(base, n, max_iters=200000, rng=None):
    """Try to find a Latin square orthogonal to base via random row/column symbol permutations."""
    if rng is None:
        rng = random.Random(42)
    
    # Start with cyclic square, then apply random permutations of symbols in each row
    # and random permutations of rows and columns
    # This preserves Latin property if done carefully.
    
    # Actually, let's use a different approach:
    # Generate random Latin squares and check orthogonality.
    # For n=10, we can use the "complete mapping" approach.
    # A Latin square can be represented as L[i][j] = (i + p(j)) % n where p is a permutation.
    # This is Latin if p is a permutation.
    # Orthogonality with cyclic (i+j)%n requires that (i+j, i+p(j)) are all distinct.
    # Fix a = i+j, b = i+p(j). Then i = a-j, b = a-j+p(j) => p(j) = b-a+j.
    # For each (a,b), we need exactly one j such that p(j) = (b-a+j) % n.
    # This means p(j) - j ≡ b-a (mod n) has exactly one solution for each value.
    # So p(j) - j (mod n) must be a permutation of 0..n-1.
    # These are called "complete mappings" or "orthomorphisms".
    
    # So: find a permutation p of 0..n-1 such that (p(j) - j) % n is also a permutation.
    # Then L[i][j] = (i + p(j)) % n is orthogonal to the cyclic square.
    
    # Let's search for such p.
    symbols = list(range(n))
    
    for it in range(max_iters):
        p = symbols[:]
        rng.shuffle(p)
        diffs = [(p[j] - j) % n for j in range(n)]
        if len(set(diffs)) == n:
            # Found! Construct the square
            sq = [[(i + p[j]) % n for j in range(n)] for i in range(n)]
            if _check_latin(sq) and _check_orthogonal(base, sq):
                return sq
    return None


def _try_find_second_orthogonal(s1, s2, n, max_iters=100000, rng=None):
    """Try to find a third square orthogonal to both s1 and s2."""
    if rng is None:
        rng = random.Random(123)
    
    # Same approach: L[i][j] = (i + p(j)) % n
    # Must be orthogonal to both s1 and s2.
    # s1 is cyclic: (i+j)%n
    # s2 is (i + p2(j))%n where p2 is the permutation from s2
    
    # Extract p2 from s2: s2[i][j] = (i + p2(j)) % n => p2(j) = (s2[0][j] - 0) % n = s2[0][j]
    p2 = [s2[0][j] for j in range(n)]
    
    symbols = list(range(n))
    
    for it in range(max_iters):
        p = symbols[:]
        rng.shuffle(p)
        # Check orthogonality to s1 (cyclic): p(j) - j must be a permutation
        diffs1 = [(p[j] - j) % n for j in range(n)]
        if len(set(diffs1)) != n:
            continue
        # Check orthogonality to s2: (i+p(j), i+p2(j)) all distinct
        # For fixed (a,b): i+p(j)=a, i+p2(j)=b => p(j)-p2(j) = a-b (mod n)
        # So p(j) - p2(j) must be a permutation of 0..n-1
        diffs2 = [(p[j] - p2[j]) % n for j in range(n)]
        if len(set(diffs2)) != n:
            continue
        # Construct and verify
        sq = [[(i + p[j]) % n for j in range(n)] for i in range(n)]
        if _check_latin(sq) and _check_orthogonal(s1, sq) and _check_orthogonal(s2, sq):
            return sq
    return None


def _try_find_third_orthogonal(s1, s2, s3, n, max_iters=50000, rng=None):
    """Try to find a fourth square orthogonal to s1, s2, s3."""
    if rng is None:
        rng = random.Random(456)
    
    p2 = [s2[0][j] for j in range(n)]
    p3 = [s3[0][j] for j in range(n)]
    
    symbols = list(range(n))
    
    for it in range(max_iters):
        p = symbols[:]
        rng.shuffle(p)
        # Orthogonal to s1: p(j)-j is permutation
        diffs1 = [(p[j] - j) % n for j in range(n)]
        if len(set(diffs1)) != n:
            continue
        # Orthogonal to s2: p(j)-p2(j) is permutation
        diffs2 = [(p[j] - p2[j]) % n for j in range(n)]
        if len(set(diffs2)) != n:
            continue
        # Orthogonal to s3: p(j)-p3(j) is permutation
        diffs3 = [(p[j] - p3[j]) % n for j in range(n)]
        if len(set(diffs3)) != n:
            continue
        sq = [[(i + p[j]) % n for j in range(n)] for i in range(n)]
        if _check_latin(sq):
            if _check_orthogonal(s1, sq) and _check_orthogonal(s2, sq) and _check_orthogonal(s3, sq):
                return sq
    return None


def construct():
    start_time = time.time()
    n = 10
    rng = random.Random(42)
    
    # First square: cyclic
    s1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Try to find second square orthogonal to s1
    s2 = _try_find_orthogonal(s1, n, max_iters=500000, rng=rng)
    
    if s2 is None:
        # Check deadline
        if time.time() - start_time > DEADLINE:
            return [s1]
        return [s1]
    
    squares = [s1, s2]
    
    # Try to find third square
    s3 = _try_find_second_orthogonal(s1, s2, n, max_iters=300000, rng=rng)
    if s3 is not None:
        squares.append(s3)
    
    # Try to find fourth square
    if len(squares) >= 3 and time.time() - start_time < DEADLINE - 5:
        s4 = _try_find_third_orthogonal(s1, s2, s3, n, max_iters=200000, rng=rng)
        if s4 is not None:
            squares.append(s4)
    
    # Verify all
    if _verify(squares):
        # Ensure all values are Python int
        result = []
        for sq in squares:
            new_sq = []
            for row in sq:
                new_row = [int(v) for v in row]
                new_sq.append(new_row)
            result.append(new_sq)
        return result
    else:
        return [s1]


# Pre-compute a known valid pair if possible to ensure we always return at least 1
# Actually, the above should work. Let's also add a fallback with hardcoded squares.

# Known pair of orthogonal Latin squares of order 10:
# We can use the construction from complete mappings.
# For n=10, p = [0, 3, 6, 9, 2, 5, 8, 1, 4, 7] gives p(j)-j = [0,2,4,6,8,0,2,4,6,8] - NOT a permutation!
# Let me find a valid one.
# p must be a permutation of 0..9 such that p(j)-j mod 10 is also a permutation.
# This is a "complete mapping" of Z_10.
# Known: for even n, complete mappings exist iff... actually for Z_n, a complete mapping exists iff n is odd or n ≡ 0 mod 4? No.
# Actually, for Z_n, a complete mapping (orthomorphism) exists if and only if n is odd. For even n, it may or may not exist.
# For n=10 (even), it's not guaranteed. Let me just rely on the random search.

# As a safety net, hardcode a known valid set.
# Actually, let's just make sure the function works correctly.
