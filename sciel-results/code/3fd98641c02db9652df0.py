"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 53 decomposition: 7 blocks of 2x2x2 (Strassen) + 1 block of 2x2x1 (Schoolbook)
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 7 Strassen blocks
    for b in range(7):
        # Strassen terms 1-7
        # M1
        U[0][col] += 1; U[1][col] += 1
        V[0][col] += 1; V[1][col + 1] += 1  # Wait, V indices are j*p+k. j in 0,1. k in 2b, 2b+1.
        # Correction: V[j*p+k]
        # M1: (a+d)(e+h) -> A00+A11, B0,2b + B1,2b+1
        V[0 * p + 2 * b][col] += 1
        V[1 * p + (2 * b + 1)][col] += 1
        W[(2 * b) * n + 0][col] += 1  # C0, 2b
        W[(2 * b + 1) * n + 1][col] += 1 # C1, 2b+1
        
        # M2: (c+d)e -> A10+A11, B0,2b
        U[1][col + 1] += 1; U[1][col + 1] += 1 # Wait, U[1,0] and U[1,1]
        U[1 * m + 0][col + 1] += 1
        U[1 * m + 1][col + 1] += 1
        V[0 * p + 2 * b][col + 1] += 1
        W[(2 * b) * n + 1][col + 1] += 1 # C1, 2b
        
        # M3: a(f-h) -> A00, B1,2b+1 - B1,2b
        U[0 * m + 0][col + 2] += 1
        V[1 * p + (2 * b + 1)][col + 2] += 1
        V[1 * p + 2 * b][col + 2] -= 1
        W[(2 * b + 1) * n + 0][col + 2] += 1 # C0, 2b+1
        
        # M4: d(g-e) -> A11, B1,2b - B0,2b
        U[1 * m + 1][col + 3] += 1
        V[1 * p + 2 * b][col + 3] += 1
        V[0 * p + 2 * b][col + 3] -= 1
        W[(2 * b) * n + 1][col + 3] += 1 # C1, 2b
        
        # M5: (a+b)h -> A00+A01, B1,2b+1
        U[0 * m + 0][col + 4] += 1
        U[0 * m + 1][col + 4] += 1
        V[1 * p + (2 * b + 1)][col + 4] += 1
        W[(2 * b) * n + 0][col + 4] -= 1 # C0, 2b
        W[(2 * b + 1) * n + 0][col + 4] += 1 # C0, 2b+1
        
        # M6: (c-a)(e+f) -> A10-A00, B0,2b + B0,2b+1
        U[1 * m + 0][col + 5] += 1
        U[0 * m + 0][col + 5] -= 1
        V[0 * p + 2 * b][col + 5] += 1
        V[0 * p + (2 * b + 1)][col + 5] += 1
        W[(2 * b + 1) * n + 1][col + 5] += 1 # C1, 2b+1
        
        # M7: (b-d)(g+h) -> A01-A11, B1,2b + B1,2b+1
        U[0 * m + 1][col + 6] += 1
        U[1 * m + 1][col + 6] -= 1
        V[1 * p + 2 * b][col + 6] += 1
        V[1 * p + (2 * b + 1)][col + 6] += 1
        W[(2 * b + 1) * n + 0][col + 6] += 1 # C0, 2b+1
        
        col += 7
    
    # 1 Schoolbook block for column 14
    # Term 1: A00 B0,14 C0,14
    U[0 * m + 0][col] += 1
    V[0 * p + 14][col] += 1
    W[14 * n + 0][col] += 1
    col += 1
    
    # Term 2: A01 B1,14 C0,14
    U[0 * m + 1][col] += 1
    V[1 * p + 14][col] += 1
    W[14 * n + 0][col] += 1
    col += 1
    
    # Term 3: A10 B0,14 C1,14
    U[1 * m + 0][col] += 1
    V[0 * p + 14][col] += 1
    W[14 * n + 1][col] += 1
    col += 1
    
    # Term 4: A11 B1,14 C1,14
    U[1 * m + 1][col] += 1
    V[1 * p + 14][col] += 1
    W[14 * n + 1][col] += 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
