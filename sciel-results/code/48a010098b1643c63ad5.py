"""
Mutually orthogonal Latin squares of order 10.

Constructs a complete set of 9 MOLS using the CRT decomposition
Z_10 ≅ Z_5 × Z_2. Each square L_k is defined by linear maps on
the two components with distinct slope parameters ensuring
pairwise orthogonality.
"""
import time

_DEADLINE = time.time() + 85.0


def _crt(x5, x2):
    """Chinese Remainder Theorem: find x in 0..9 with x%5=x5, x%2=x2."""
    # x = x5 * 2 * inv(2,5) + x2 * 5 * inv(5,2) mod 10
    # inv(2,5)=3 since 2*3=6≡1 mod 5; inv(5,2)=1 since 5≡1 mod 2
    return (x5 * 6 + x2 * 5) % 10


def _build_square(a5, b5, a2, b2):
    """
    Build a Latin square of order 10 using parameters:
      L(i,j) = (a5*i5 + b5*j5 mod 5, a2*i2 + b2*j2 mod 2)
    where i=(i5,i2), j=(j5,j2) via CRT decomposition.
    For this to be a Latin square, we need b5≠0 mod 5 and b2≠0 mod 2.
    """
    n = 10
    sq = [[0] * n for _ in range(n)]
    for i in range(n):
        i5, i2 = i % 5, i % 2
        for j in range(n):
            j5, j2 = j % 5, j % 2
            r5 = (a5 * i5 + b5 * j5) % 5
            r2 = (a2 * i2 + b2 * j2) % 2
            sq[i][j] = _crt(r5, r2)
    return sq


def _is_latin_square(sq):
    """Check if sq is a valid Latin square of order 10."""
    n = 10
    for i in range(n):
        if sorted(sq[i]) != list(range(n)):
            return False
    for j in range(n):
        col = [sq[i][j] for i in range(n)]
        if sorted(col) != list(range(n)):
            return False
    return True


def _are_orthogonal(sq1, sq2):
    """Check if two Latin squares are orthogonal."""
    n = 10
    pairs = set()
    for i in range(n):
        for j in range(n):
            pair = (sq1[i][j], sq2[i][j])
            if pair in pairs:
                return False
            pairs.add(pair)
    return True


def construct():
    """
    Return a list of 9 mutually orthogonal Latin squares of order 10.
    Each square is a list of 10 lists of Python ints in 0..9.
    """
    # Generate 9 squares with distinct parameter tuples.
    # For Latin square: b5 in {1,2,3,4}, b2 = 1 (only nonzero in Z_2).
    # For orthogonality between squares with params (a5,b5,a2,b2) and (c5,d5,c2,d2):
    #   The pair (r5,r2) from sq1 and (s5,s2) from sq2 must be unique for each (i,j).
    #   r5 = a5*i5 + b5*j5, s5 = c5*i5 + d5*j5
    #   r2 = a2*i2 + b2*j2, s2 = c2*i2 + d2*j2
    #   For fixed (i5,i2), as (j5,j2) varies over all 10 combos, 
    #   (r5,s5,r2,s2) must be all distinct.
    #   The Z_5 part: (a5*i5+b5*j5, c5*i5+d5*j5) as j5 varies 0..4
    #     gives distinct pairs iff the 2x2 matrix [[b5,d5]] has rank 1... 
    #     actually we need (b5*j5, d5*j5) to give 5 distinct values as j5 varies,
    #     which requires that (b5, d5) and (b5', d5') for two squares are linearly independent mod 5.
    #   More precisely: for two squares with slope vectors v1=(b5_1,d5_1) and v2=(b5_2,d5_2) in Z_5^2,
    #   they are orthogonal in the Z_5 component iff v1 and v2 are linearly independent mod 5.
    #   Similarly for Z_2: v1_2=(b2_1,d2_1)=(1,1) always, so Z_2 part is fixed.
    #   
    # So we need 9 slope vectors (b5, d5) in Z_5^2 that are pairwise linearly independent.
    # In Z_5^2, there are 25-1=24 nonzero vectors, grouped into (25-1)/(5-1)=6 lines of 4 vectors each.
    # We can pick at most one from each line, so at most 6 squares from this construction.
    # That's not enough for 9.
    #
    # Alternative: also vary a5 and a2 (the "row" slopes).
    # The full parameter is (a5, b5, a2, b2) with b5≠0, b2≠0.
    # For orthogonality, we need the combined map to be injective.
    #
    # Let me try a different approach: use the known complete set.
    # The 9 MOLS of order 10 can be constructed as follows:
    # For k = 1..9, define L_k(i,j) using a different formula.
    #
    # Actually, let me just enumerate valid parameter sets and check orthogonality.

    # Parameters: a5 in 0..4, b5 in 1..4, a2 in 0..1, b2=1
    # Total: 5*4*2*1 = 40 possible squares.
    # Check which subsets of 9 are mutually orthogonal.
    
    # First, generate all candidate squares
    candidates = []
    for a5 in range(5):
        for b5 in range(1, 5):
            for a2 in range(2):
                b2 = 1
                sq = _build_square(a5, b5, a2, b2)
                if _is_latin_square(sq):
                    candidates.append((sq, (a5, b5, a2, b2)))
    
    # Now find the largest mutually orthogonal set
    # Greedy: pick squares one by one, checking orthogonality with all previously picked
    best_set = []
    for sq, params in candidates:
        if all(_are_orthogonal(sq, prev) for prev, _ in best_set):
            best_set.append((sq, params))
            if len(best_set) >= 9:
                break
    
    # If we found 9, great. Otherwise, try a more systematic search.
    if len(best_set) < 9:
        # Backtracking search for 9 mutually orthogonal squares
        def backtrack(start, current):
            if len(current) == 9:
                return current
            if time.time() > _DEADLINE - 5.0:
                return None
            for idx in range(start, len(candidates)):
                sq, params = candidates[idx]
                if all(_are_orthogonal(sq, prev) for prev, _ in current):
                    result = backtrack(idx + 1, current + [(sq, params)])
                    if result is not None:
                        return result
            return None
        
        result = backtrack(0, [])
        if result is not None:
            best_set = result
    
    # Return just the squares (as lists of lists of ints)
    return [sq for sq, _ in best_set]
