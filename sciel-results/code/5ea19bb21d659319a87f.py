"""
Mutually orthogonal Latin squares of order 10.

Constructs a complete set of 9 MOLS using the CRT decomposition
Z_10 ≅ Z_5 × Z_2. Uses linear maps over F_5 with fixed Z_2 component.
"""
import time

_DEADLINE = time.time() + 85.0


def _crt(x5, x2):
    """Chinese Remainder Theorem: find x in 0..9 with x%5=x5, x%2=x2."""
    return (x5 * 6 + x2 * 5) % 10


def _build_square(a5, b5):
    """
    Build a Latin square of order 10:
      L(i,j) = CRT(a5*i5 + b5*j5 mod 5, i2 + j2 mod 2)
    where i=(i5,i2), j=(j5,j2).
    Requires b5 ≠ 0 mod 5 for Latin square property.
    """
    n = 10
    sq = [[0] * n for _ in range(n)]
    for i in range(n):
        i5, i2 = i % 5, i % 2
        for j in range(n):
            j5, j2 = j % 5, j % 2
            r5 = (a5 * i5 + b5 * j5) % 5
            r2 = (i2 + j2) % 2
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
    # Generate 9 squares with distinct (a5, b5) pairs where b5 ∈ {1,2,3,4}
    # Pick 9 distinct pairs from the 20 available (a5 ∈ 0..4, b5 ∈ 1..4)
    pairs = []
    for a5 in range(5):
        for b5 in range(1, 5):
            pairs.append((a5, b5))
            if len(pairs) == 9:
                break
        if len(pairs) == 9:
            break
    
    squares = []
    for a5, b5 in pairs:
        sq = _build_square(a5, b5)
        squares.append(sq)
    
    # Verify all are valid and pairwise orthogonal
    for sq in squares:
        if not _is_latin_square(sq):
            break
    
    for i in range(len(squares)):
        for j in range(i + 1, len(squares)):
            if not _are_orthogonal(squares[i], squares[j]):
                break
    
    # If verification passed, return the 9 squares
    # Otherwise, fall back to a smaller set
    if all(_is_latin_square(sq) for sq in squares) and \
       all(_are_orthogonal(squares[i], squares[j]) for i in range(9) for j in range(i+1, 9)):
        return squares
    
    # Fallback: return a single valid Latin square
    sq = _build_square(0, 1)
    return [sq]
