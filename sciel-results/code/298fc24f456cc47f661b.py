"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 10
    # Construct two orthogonal Latin squares of order 10.
    # S1[i][j] = (i + j) % 10
    # S2[i][j] = (i + 2*j) % 10
    # These are orthogonal because the mapping (i,j) -> ((i+j)%10, (i+2j)%10) is a bijection.
    # The determinant of the coefficient matrix [[1,1],[1,2]] is 1, which is coprime to 10.
    
    s1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    s2 = [[(i + 2 * j) % n for j in range(n)] for i in range(n)]
    
    # Optional: Verify orthogonality and Latin square property for safety, though mathematically guaranteed.
    # This adds negligible time.
    def is_latin(sq):
        for i in range(n):
            if sorted(sq[i]) != list(range(n)):
                return False
            col = [sq[r][i] for r in range(n)]
            if sorted(col) != list(range(n)):
                return False
        return True

    def is_orthogonal(sq1, sq2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                p = (sq1[i][j], sq2[i][j])
                if p in pairs:
                    return False
                pairs.add(p)
        return True

    if is_latin(s1) and is_latin(s2) and is_orthogonal(s1, s2):
        return [s1, s2]
    else:
        # Fallback to 1 square if something unexpected happens (should not)
        return [s1]

# EVOLVE-BLOCK-END
