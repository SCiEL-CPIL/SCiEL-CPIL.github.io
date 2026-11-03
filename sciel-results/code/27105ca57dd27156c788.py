"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    rank = 54
    
    # U: (n*m, rank) = (10, 54)
    # V: (m*p, rank) = (30, 54)
    # W: (p*n, rank) = (12, 54)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen blocks for J in {0,1} (indices 0,1 and 2,3)
    # and K in {0,1}, {2,3}, {4,5}
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j_a, j_b in j_pairs:
        for k_a, k_b in k_pairs:
            # 7 Strassen multiplications
            # M1 = (A_0ja + A_1jb) * (B_ja,ka + B_jb,kb)
            # M2 = (A_1ja + A_1jb) * B_ja,ka
            # M3 = A_0ja * (B_ja,kb - B_jb,kb)
            # M4 = A_1jb * (B_jb,ka - B_ja,ka)
            # M5 = (A_0ja + A_0jb) * B_jb,kb
            # M6 = (A_1ja - A_0ja) * (B_ja,ka + B_ja,kb)
            # M7 = (A_0jb - A_1jb) * (B_jb,ka + B_jb,kb)
            
            # Define U, V, W contributions for each M
            
            # M1
            U[0*m + j_a][col] += 1
            U[1*m + j_b][col] += 1
            V[j_a*p + k_a][col] += 1
            V[j_b*p + k_b][col] += 1
            # W contributions: C_0,ka += 1, C_1,kb += 1
            W[k_a*n + 0][col] += 1
            W[k_b*n + 1][col] += 1
            col += 1
            
            # M2
            U[1*m + j_a][col] += 1
            U[1*m + j_b][col] += 1
            V[j_a*p + k_a][col] += 1
            # W: C_1,ka += 1, C_1,kb -= 1
            W[k_a*n + 1][col] += 1
            W[k_b*n + 1][col] -= 1
            col += 1
            
            # M3
            U[0*m + j_a][col] += 1
            V[j_a*p + k_b][col] += 1
            V[j_b*p + k_b][col] -= 1
            # W: C_0,kb += 1, C_1,kb += 1
            W[k_b*n + 0][col] += 1
            W[k_b*n + 1][col] += 1
            col += 1
            
            # M4
            U[1*m + j_b][col] += 1
            V[j_b*p + k_a][col] += 1
            V[j_a*p + k_a][col] -= 1
            # W: C_0,ka += 1, C_1,ka += 1
            W[k_a*n + 0][col] += 1
            W[k_a*n + 1][col] += 1
            col += 1
            
            # M5
            U[0*m + j_a][col] += 1
            U[0*m + j_b][col] += 1
            V[j_b*p + k_b][col] += 1
            # W: C_0,ka -= 1, C_0,kb += 1
            W[k_a*n + 0][col] -= 1
            W[k_b*n + 0][col] += 1
            col += 1
            
            # M6
            U[1*m + j_a][col] += 1
            U[0*m + j_a][col] -= 1
            V[j_a*p + k_a][col] += 1
            V[j_a*p + k_b][col] += 1
            # W: C_1,kb += 1
            W[k_b*n + 1][col] += 1
            col += 1
            
            # M7
            U[0*m + j_b][col] += 1
            U[1*m + j_b][col] -= 1
            V[j_b*p + k_a][col] += 1
            V[j_b*p + k_b][col] += 1
            # W: C_0,ka += 1
            W[k_a*n + 0][col] += 1
            col += 1
            
    # Scalar blocks for J=4 (index 4)
    j = 4
    for k in range(p):
        # C_0,k += A_0,4 * B_4,k
        U[0*m + j][col] += 1
        V[j*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        
        # C_1,k += A_1,4 * B_4,k
        U[1*m + j][col] += 1
        V[j*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
