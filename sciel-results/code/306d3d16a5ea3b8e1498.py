"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _is_latin(sq):
    n = len(sq)
    for i in range(n):
        if sorted(sq[i]) != list(range(n)):
            return False
    for j in range(n):
        col = [sq[i][j] for i in range(n)]
        if sorted(col) != list(range(n)):
            return False
    return True

def _orthogonal(sq1, sq2):
    n = len(sq1)
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (sq1[i][j], sq2[i][j])
            if p in pairs:
                return False
            pairs.add(p)
    return True

def construct():
    deadline = time.time() + 55  # safety margin before 60s kill
    n = 10
    
    # Construct two MOLS of order 10 using GF(25) structure
    # GF(25) = GF(5)[x]/(x^2 + 2), alpha^2 = 3 mod 5
    # Elements: a + b*alpha, a,b in GF(5), total 25 elements
    # We'll use 10 carefully chosen elements to form order-10 squares
    
    # Simpler: use known construction for order 10 = 2*5
    # L1[i][j] = (i + j) % 10
    # L2[i][j] = (i + 3*j) % 10
    # Verify these are orthogonal
    
    sq1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    sq2 = [[(i + 3 * j) % n for j in range(n)] for i in range(n)]
    
    # Verify both are Latin
    if not _is_latin(sq1) or not _is_latin(sq2):
        # Fallback to single square
        return [sq1]
    
    # Verify orthogonality
    if _orthogonal(sq1, sq2):
        return [sq1, sq2]
    else:
        # Try other multipliers
        for m in [2, 3, 4, 6, 7, 8, 9]:
            if m == 1:
                continue
            sq_m = [[(i + m * j) % n for j in range(n)] for i in range(n)]
            if _is_latin(sq_m) and _orthogonal(sq1, sq_m):
                return [sq1, sq_m]
        return [sq1]
# EVOLVE-BLOCK-END
