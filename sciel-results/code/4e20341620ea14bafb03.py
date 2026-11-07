"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _add_ring(a, b):
    a2, a5 = divmod(a, 5)
    b2, b5 = divmod(b, 5)
    return 5 * ((a2 + b2) % 2) + (a5 + b5) % 5

def _mul_lambda(c):
    # Multiplier lambda = (1, 2) in Z_2 x Z_5
    c2, c5 = divmod(c, 5)
    return 5 * c2 + (2 * c5) % 5

def construct():
    n = 10
    start = time.time()
    
    sq1 = [[0] * n for _ in range(n)]
    sq2 = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            sq1[i][j] = _add_ring(i, j)
            sq2[i][j] = _add_ring(i, _mul_lambda(j))

    # Verify
    def is_latin(sq):
        for i in range(n):
            if sorted(sq[i]) != list(range(n)):
                return False
        for j in range(n):
            col = [sq[i][j] for i in range(n)]
            if sorted(col) != list(range(n)):
                return False
        return True

    def are_orthogonal(s1, s2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (s1[i][j], s2[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return True

    if is_latin(sq1) and is_latin(sq2) and are_orthogonal(sq1, sq2):
        # Convert to list of lists of ints
        res1 = [[int(v) for v in row] for row in sq1]
        res2 = [[int(v) for v in row] for row in sq2]
        return [res1, res2]
    
    # Fallback
    return [[(i + j) % n for j in range(n)] for i in range(n)]
# EVOLVE-BLOCK-END
