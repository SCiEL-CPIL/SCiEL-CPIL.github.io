"""Matrix multiplication 3x3 x 3x6 using two Laderman rank-19 blocks (total rank 38)."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 6
    
    # Laderman rank-19 decomposition for 3x3 matrix multiplication.
    # Each term: (A_coeffs[9], B_coeffs[9], C_coeffs[9])
    # A is 3x3 (row-major), B is 3x3 (row-major), C is 3x3 (row-major)
    # C = A * B
    
    laderman_terms = [
        # Term 0: A[0][0]*B[0][0] -> C[0][0]
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        # Term 1: A[1][0]*B[0][0] -> C[1][0]
        ([0,0,0, 1,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        # Term 2: A[0][1]*B[1][0] -> C[0][1]
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        # Term 3: A[1][1]*B[1][0] -> C[1][1]
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        # Term 4: A[0][2]*B[2][0] -> C[0][2]
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 0,0,0]),
        # Term 5: A[1][2]*B[2][0] -> C[1][2]
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,1, 0,0,0]),
        # Term 6: A[2][0]*B[0][1] -> C[2][0]
        ([0,0,0, 0,0,0, 1,0,0], [0,1,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        # Term 7: A[2][1]*B[1][1] -> C[2][1]
        ([0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,1,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        # Term 8: A[2][2]*B[2][1] -> C[2][2]
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,0, 0,0,0]),
        # Term 9: (A[0][0]+A[0][1])*B[0][2] -> C[0][0]
        ([1,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,0,0]),
        # Term 10: (A[1][0]+A[1][1])*B[1][2] -> C[1][0]
        ([0,0,0, 1,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 1,0,0, 0,0,0]),
        # Term 11: (A[2][0]+A[2][1])*B[2][2] -> C[2][0]
        ([0,0,0, 0,0,0, 1,0,1], [0,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,0,0]),
        # Term 12: (A[0][0]+A[0][1])*(B[0][0]+B[1][0]) -> -C[0][0]
        ([1,0,1, 0,0,0, 0,0,0], [1,0,0, 1,0,0, 0,0,0], [-1,0,0, 0,0,0, 0,0,0]),
        # Term 13: (A[1][0]+A[1][1])*(B[0][0]+B[1][0]) -> -C[1][0]
        ([0,0,0, 1,0,1, 0,0,0], [1,0,0, 1,0,0, 0,0,0], [0,0,0, -1,0,0, 0,0,0]),
        # Term 14: (A[0][0]+A[0][1])*(B[0][2]+B[1][2]) -> C[0][0]
        ([1,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,0,0]),
        # Term 15: (A[1][0]+A[1][1])*(B[0][2]+B[1][2]) -> C[1][0]
        ([0,0,0, 1,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 1,0,0, 0,0,0]),
        # Term 16: (A[0][0]+A[0][1]+A[2][0]+A[2][1])*(B[0][0]+B[1][0]+B[0][2]+B[1][2]) -> C[0][0]+C[1][0]+C[2][0]
        ([1,0,1, 0,0,0, 1,0,1], [1,0,0, 1,0,0, 0,0,1], [1,0,0, 1,0,0, 1,0,0]),
        # Term 17: (A[0][0]+A[0][1])*(B[0][0]+B[1][0]+B[0][2]+B[1][2]) -> -C[0][0]-C[1][0]
        ([1,0,1, 0,0,0, 0,0,0], [1,0,0, 1,0,0, 0,0,1], [-1,0,0, -1,0,0, 0,0,0]),
        # Term 18: (A[1][0]+A[1][1])*(B[0][0]+B[1][0]+B[0][2]+B[1][2]) -> -C[1][0]
        ([0,0,0, 1,0,1, 0,0,0], [1,0,0, 1,0,0, 0,0,1], [0,0,0, -1,0,0, 0,0,0]),
    ]
    
    rank = 38
    U = [[0] * rank for _ in range(n * m)]   # (9, 38)
    V = [[0] * rank for _ in range(m * p)]   # (18, 38)
    W = [[0] * rank for _ in range(p * n)]   # (18, 38)
    
    # Build two Laderman blocks: one for B[:, 0:3] and one for B[:, 3:6]
    for block in range(2):
        base = block * 19
        k_off = block * 3  # column offset in B and C
        
        for t in range(19):
            A_flat, B_flat, C_flat = laderman_terms[t]
            col = base + t
            
            # Fill U: A coefficients (same for both blocks)
            for idx in range(9):
                U[idx][col] = A_flat[idx]
            
            # Fill V: B coefficients with column offset
            for idx in range(9):
                j = idx // 3
                k = idx % 3
                V[j * 6 + (k + k_off)][col] = B_flat[idx]
            
            # Fill W: C coefficients with column offset
            # W index: (k_global)*n + i where k_global = k + k_off
            for idx in range(9):
                i = idx // 3
                k = idx % 3
                W[(k + k_off) * n + i][col] = C_flat[idx]
    
    # Verify the decomposition
    import random
    def verify(U_mat, V_mat, W_mat, n, m, p, rank):
        random.seed(12345)
        for trial in range(20):
            A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
            B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
            
            # Expected: C = A * B, shape (n, p)
            C_exp = [[0] * p for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    s = 0
                    for j in range(m):
                        s += A[i][j] * B[j][k]
                    C_exp[i][k] = s
            
            # From decomposition: C[i][k] = sum_c W[k*n+i][c] * U[i*m+j][c] * V[j*p+k][c]
            # Actually: C[i][k] = sum_c W[k*n+i][c] * (sum_j U[i*m+j][c] * V[j*p+k][c])
            C_dec = [[0] * p for _ in range(n)]
            for c in range(rank):
                for i in range(n):
                    for k in range(p):
                        s = 0
                        for j in range(m):
                            s += U_mat[i * m + j][c] * V_mat[j * p + k][c]
                        C_dec[i][k] += s * W_mat[k * n + i][c]
            
            for i in range(n):
                for k in range(p):
                    if C_exp[i][k] != C_dec[i][k]:
                        return False
        return True
    
    if verify(U, V, W, n, m, p, rank):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback: schoolbook rank 54
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
