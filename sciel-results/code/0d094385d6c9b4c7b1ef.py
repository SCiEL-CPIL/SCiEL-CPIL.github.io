"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 10}

def construct():
    n = P['n']
    # Generate two mutually orthogonal Latin squares of order 10
    # S1[i][j] = (i + j) % n
    # S2[i][j] = (i + 2*j) % n
    # They are orthogonal because gcd(2-1, 10) = 1
    
    s1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    s2 = [[(i + 2 * j) % n for j in range(n)] for i in range(n)]
    
    return [s1, s2]
# EVOLVE-BLOCK-END
