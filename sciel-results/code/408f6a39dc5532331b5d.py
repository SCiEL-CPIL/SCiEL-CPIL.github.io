"""Construct mutually orthogonal Latin squares of order 10."""
import time
import random

# EVOLVE-BLOCK-START

def _is_latin(square, n=10):
    """Check if square is a valid Latin square of order n."""
    for row in square:
        if len(row) != n:
            return False
        if sorted(row) != list(range(n)):
            return False
    for j in range(n):
        col = [square[i][j] for i in range(n)]
        if sorted(col) != list(range(n)):
            return False
    return True

def _are_orthogonal(s1, s2, n=10):
    """Check if two Latin squares are orthogonal."""
    pairs = set()
    for i in range(n):
        for j in range(n):
            pairs.add((s1[i][j], s2[i][j]))
    return len(pairs) == n * n

def _verify_all(squares, n=10):
    """Verify all squares are Latin and pairwise orthogonal."""
    for sq in squares:
        if not _is_latin(sq, n):
            return False
    for i in range(len(squares)):
        for j in range(i + 1, len(squares)):
            if not _are_orthogonal(squares[i], squares[j], n):
                return False
    return True

def _hardcoded_squares():
    """
    Return a set of mutually orthogonal Latin squares of order 10.
    
    We use the known construction: the maximum number of MOLS(10) is 9.
    We construct them using the affine geometry approach over GF(9) with a careful completion.
    
    Actually, let's use a well-known explicit construction. The standard approach for order 10
    uses the fact that 10 = 2 * 5, but MOLS(10) max is 9.
    
    Let me use a direct construction based on the group Z_5 x Z_2 approach.
    
    Actually, the simplest reliable approach: use the known fact that we can construct
    up to 9 MOLS(10). Let me hardcode a verified set.
    
    I'll construct squares using the method from the literature. For order 10, one approach:
    Use the affine plane of order 9 (which gives 8 MOLS of order 9), then extend to order 10.
    
    A simpler approach: construct squares directly and verify.
    
    Let me use a construction based on the group Z_5 x Z_2.
    
    Actually, let me just hardcode a known valid set. The maximum is 9, and they exist.
    
    Here's a construction using the fact that MOLS(10) can be built from the 
    orthogonal array OA(100, 11, 10, 2).
    
    Let me use a computational approach with a deadline guard.
    """
    n = 10
    # Start with the cyclic square
    base = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [base]
    
    # Try to add more squares using various constructions
    # Method: for each multiplier a in 1..n-1, try square (i*a + j) % n
    # But this only works for prime n. For n=10, we need a different approach.
    
    # Use the approach: represent elements as pairs (a, b) where a in Z_5, b in Z_2
    # Element value = 5*a + b
    
    def idx(a, b):
        return 5 * a + b
    
    def decompose(v):
        return (v // 5, v % 5)  # Wait, this is wrong. Let me fix.
    
    # Actually, let me use a different decomposition.
    # Represent elements 0..9 as pairs (a, b) with a in {0,...,4}, b in {0,1}
    # value = a * 2 + b  ... no, that gives 0..9 but not uniquely in a nice way.
    # Let's use: value = 5*b + a where a in 0..4, b in 0,1. So value = 5*b + a.
    
    def to_pair(v):
        b = v // 5
        a = v % 5
        return (a, b)
    
    def from_pair(a, b):
        return 5 * b + a
    
    # Construct squares using linear maps over the "group" Z_5 x Z_2
    # Square k: entry at position (i, j) where i,j are decomposed as (ai, bi), (aj, bj)
    # We need a function f_k: (ai,bi,aj,bj) -> (ak, bk)
    
    # For the cyclic-like square: entry = (i+j) mod 10
    # In pair form: i = (ai, bi), j = (aj, bj), i+j mod 10
    # This is complex. Let me just try to find MOLS computationally.
    
    # Alternative: use a known published set of 9 MOLS(10).
    # Since I can't look it up, let me try a constructive approach.
    
    # Try: for each a in range(1, n), construct square S_a[i][j] = (i*a + j) % n
    # Check which are Latin and orthogonal to each other.
    
    candidates = []
    for a in range(1, n):
        sq = [[(i * a + j) % n for j in range(n)] for i in range(n)]
        if _is_latin(sq, n):
            candidates.append((a, sq))
    
    # Now find the maximum set of mutually orthogonal candidates
    best_set = []
    for a, sq in candidates:
        if all(_are_orthogonal(sq, bsq, n) for bsq in best_set):
            best_set.append(sq)
    
    # Also try other constructions
    # Square: S[i][j] = (i + j*a) % n
    candidates2 = []
    for a in range(1, n):
        sq = [[(i + j * a) % n for j in range(n)] for i in range(n)]
        if _is_latin(sq, n):
            candidates2.append((a, sq))
    
    for a, sq in candidates2:
        if all(_are_orthogonal(sq, bsq, n) for bsq in best_set):
            best_set.append(sq)
    
    # Try: S[i][j] = (i*a + j*b) % n for various a, b
    candidates3 = []
    for a in range(1, n):
        for b in range(1, n):
            sq = [[(i * a + j * b) % n for j in range(n)] for i in range(n)]
            if _is_latin(sq, n):
                candidates3.append((a, b, sq))
    
    for a, b, sq in candidates3:
        if all(_are_orthogonal(sq, bsq, n) for bsq in best_set):
            best_set.append(sq)
    
    # Try: S[i][j] = (i*b + j*a) % n
    candidates4 = []
    for a in range(1, n):
        for b in range(1, n):
            sq = [[(i * b + j * a) % n for j in range(n)] for i in range(n)]
            if _is_latin(sq, n):
                candidates4.append((a, b, sq))
    
    for a, b, sq in candidates4:
        if all(_are_orthogonal(sq, bsq, n) for bsq in best_set):
            best_set.append(sq)
    
    # Remove duplicates
    unique_best = []
    seen = set()
    for sq in best_set:
        key = tuple(tuple(row) for row in sq)
        if key not in seen:
            seen.add(key)
            unique_best.append(sq)
    
    # Verify
    if _verify_all(unique_best, n):
        return unique_best
    else:
        return [base]

def construct():
    n = 10
    deadline = time.time() + 80  # Safety margin before 90s limit
    
    # Try the hardcoded/computational construction
    squares = _hardcoded_squares()
    
    # Ensure we have at least 1
    if not squares:
        squares = [[[(i + j) % n for j in range(n)] for i in range(n)]]
    
    # Verify the result
    if not _verify_all(squares, n):
        # Fallback to just the base square
        squares = [[[(i + j) % n for j in range(n)] for i in range(n)]]
    
    # Convert to ensure pure Python ints and lists
    result = []
    for sq in squares:
        result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])
    
    return result

# EVOLVE-BLOCK-END
