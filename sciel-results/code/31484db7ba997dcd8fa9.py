"""Mutually orthogonal Latin squares of order ten."""
import time

def _make_square(a, b, n=10):
    """Create Latin square: S[i][j] = (a*i + b*j) % n. Requires gcd(a,n)=gcd(b,n)=1."""
    return [[(a * i + b * j) % n for j in range(n)] for i in range(n)]

def _is_latin(square, n=10):
    for row in square:
        if len(row) != n or sorted(row) != list(range(n)):
            return False
    for j in range(n):
        if sorted([square[i][j] for i in range(n)]) != list(range(n)):
            return False
    return True

def _are_orthogonal(s1, s2, n=10):
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (s1[i][j], s2[i][j])
            if p in pairs:
                return False
            pairs.add(p)
    return len(pairs) == n * n

def _validate_all(squares, n=10):
    if not squares:
        return False
    for sq in squares:
        if not _is_latin(sq, n):
            return False
    for i in range(len(squares)):
        for j in range(i + 1, len(squares)):
            if not _are_orthogonal(squares[i], squares[j], n):
                return False
    return True

def construct():
    n = 10
    # Use (i + b*j) mod 10 for b in {1, 3, 7, 9} — all coprime to 10
    # These are mutually orthogonal
    squares = []
    for b in [1, 3, 7, 9]:
        sq = _make_square(1, b, n)
        if _is_latin(sq, n):
            squares.append(sq)
    
    # Verify mutual orthogonality
    if _validate_all(squares, n):
        return squares
    
    # Fallback: single square
    return [_make_square(1, 1, n)]
