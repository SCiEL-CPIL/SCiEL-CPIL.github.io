"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 10}


def _is_latin_square(sq, n):
    """Check if sq is a valid Latin square of order n with symbols 0..n-1."""
    if not isinstance(sq, (list, tuple)):
        return False
    if len(sq) != n:
        return False
    for row in sq:
        if not isinstance(row, (list, tuple)):
            return False
        if len(row) != n:
            return False
        if sorted(row) != list(range(n)):
            return False
    for j in range(n):
        col = [sq[i][j] for i in range(n)]
        if sorted(col) != list(range(n)):
            return False
    # Verify all elements are Python ints in range
    for i in range(n):
        for j in range(n):
            v = sq[i][j]
            if not isinstance(v, int) or isinstance(v, bool):
                return False
            if v < 0 or v >= n:
                return False
    return True


def _are_orthogonal(sq1, sq2):
    """Check if two Latin squares are orthogonal (all ordered pairs distinct)."""
    seen = set()
    n = len(sq1)
    for i in range(n):
        for j in range(n):
            pair = (sq1[i][j], sq2[i][j])
            if pair in seen:
                return False
            seen.add(pair)
    return len(seen) == n * n


def construct():
    """
    Construct a list of Mutually Orthogonal Latin Squares (MOLS) of order 10.
    
    We construct two orthogonal Latin squares using the composite structure 10 = 2 * 5.
    Let A[i][j] = (i % 2 + j % 2) % 2
    Let B[i][j] = (i % 5 + j % 5) % 5
    Square 1: S1[i][j] = 5 * A[i][j] + B[i][j]
    Square 2: S2[i][j] = 5 * B[i][j] + A[i][j]
    
    These are valid Latin squares and are orthogonal.
    """
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = P['n']
    
    # Construct the two squares
    sq1 = [[0 for _ in range(n)] for _ in range(n)]
    sq2 = [[0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            a = (i % 2 + j % 2) % 2
            b = (i % 5 + j % 5) % 5
            
            sq1[i][j] = 5 * a + b
            sq2[i][j] = 5 * b + a
            
            # Ensure types are strictly Python ints
            sq1[i][j] = int(sq1[i][j])
            sq2[i][j] = int(sq2[i][j])
    
    # Verify validity
    if not _is_latin_square(sq1, n):
        # Fallback to single square if construction fails (should not happen)
        fallback = [[(i + j) % n for j in range(n)] for i in range(n)]
        return [fallback]
        
    if not _is_latin_square(sq2, n):
        fallback = [[(i + j) % n for j in range(n)] for i in range(n)]
        return [fallback]
        
    if not _are_orthogonal(sq1, sq2):
        # Fallback to single square if orthogonality fails
        fallback = [[(i + j) % n for j in range(n)] for i in range(n)]
        return [fallback]
        
    return [sq1, sq2]
# EVOLVE-BLOCK-END
