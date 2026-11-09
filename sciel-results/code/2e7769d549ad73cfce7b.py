"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 6
    rank = 38
    
    U = [[0] * rank for _ in range(n * m)]   # (9, 38)
    V = [[0] * rank for _ in range(m * p)]   # (18, 38)
    W = [[0] * rank for _ in range(p * n)]   # (18, 38)
    
    # Correct Laderman rank-19 decomposition for 3x3 matrix multiplication
    # Each term: (u_flat[9], v_flat[9], w_flat[9])
    # u_flat[i*3+j] = coeff of A[i,j]
    # v_flat[j*3+k] = coeff of B[j,k]
    # w_flat[i*3+k] = coeff of C[i,k]
    # Must satisfy: sum_t u_t[i,j]*v_t[j,k]*w_t[i,k] = sum_j A[i,j]*B[j,k] for all i,k
    
    laderman_terms = [
        # Term 0: A[0,0]*B[0,0] -> C[0,0]
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        # Term 1: A[1,0]*B[0,0] -> C[1,0]
        ([0,0,0, 1,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        # Term 2: A[0,1]*B[1,0] -> C[0,1]
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        # Term 3: A[1,1]*B[1,0] -> C[1,1]
        ([0,0,0, 0,1,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        # Term 4: A[0,2]*B[2,0] -> C[0,2]
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 0,0,0]),
        # Term 5: A[1,2]*B[2,0] -> C[1,2]
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 1,0,0], [0,0,1, 0,0,0, 0,0,0]),
        # Term 6: A[2,0]*B[0,1] -> C[2,1]
        ([0,0,0, 0,0,0, 1,0,0], [0,1,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        # Term 7: A[2,1]*B[1,1] -> C[2,1]
        ([0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,1,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        # Term 8: A[2,2]*B[2,1] -> C[2,2]
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,0, 0,0,0]),
        # Term 9: (A[0,0]+A[0,1])*B[0,2] -> C[0,0]
        ([1,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,0,0]),
        # Term 10: (A[1,0]+A[1,1])*B[1,2] -> C[1,0]
        ([0,0,0, 1,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 1,0,0, 0,0,0]),
        # Term 11: (A[2,0]+A[2,1])*B[2,2] -> C[2,2]
        ([0,0,0, 0,0,0, 1,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,1, 0,0,0, 0,0,0]),
        # Term 12: (A[0,0]+A[0,1])*(B[0,0]+B[1,0]) -> -C[0,0]
        ([1,0,1, 0,0,0, 0,0,0], [1,0,0, 1,0,0, 0,0,0], [-1,0,0, 0,0,0, 0,0,0]),
        # Term 13: (A[1,0]+A[1,1])*(B[0,0]+B[1,0]) -> -C[1,0]
        ([0,0,0, 1,0,1, 0,0,0], [1,0,0, 1,0,0, 0,0,0], [0,0,0, -1,0,0, 0,0,0]),
        # Term 14: (A[0,0]+A[0,1])*(B[0,2]+B[1,2]) -> C[0,0]
        ([1,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,0,0]),
        # Term 15: (A[1,0]+A[1,1])*(B[0,2]+B[1,2]) -> C[1,0]
        ([0,0,0, 1,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 1,0,0, 0,0,0]),
        # Term 16: (A[0,0]+A[0,1]+A[2,0]+A[2,1])*(B[0,0]+B[1,0]+B[0,2]+B[1,2]) -> C[0,0]+C[1,0]+C[2,2]
        ([1,0,1, 0,0,0, 1,0,1], [1,0,0, 1,0,0, 0,0,1], [1,0,0, 1,0,0, 0,0,1]),
        # Term 17: (A[0,0]+A[0,1])*(B[0,0]+B[1,0]+B[0,2]+B[1,2]) -> -C[0,0]-C[1,0]
        ([1,0,1, 0,0,0, 0,0,0], [1,0,0, 1,0,0, 0,0,1], [-1,0,0, -1,0,0, 0,0,0]),
        # Term 18: (A[1,0]+A[1,1])*(B[0,0]+B[1,0]+B[0,2]+B[1,2]) -> -C[1,0]
        ([0,0,0, 1,0,1, 0,0,0], [1,0,0, 1,0,0, 0,0,1], [0,0,0, -1,0,0, 0,0,0]),
    ]
    
    # Verify the Laderman decomposition
    import random
    def verify_laderman(terms):
        random.seed(123)
        for trial in range(20):
            A = [[random.randint(-5, 5) for _ in range(3)] for _ in range(3)]
            B = [[random.randint(-5, 5) for _ in range(3)] for _ in range(3)]
            C_exp = [[0]*3 for _ in range(3)]
            for i in range(3):
                for k in range(3):
                    s = 0
                    for j in range(3):
                        s += A[i][j] * B[j][k]
                    C_exp[i][k] = s
            C_dec = [[0]*3 for _ in range(3)]
            for t in range(len(terms)):
                u_flat, v_flat, w_flat = terms[t]
                for i in range(3):
                    for k in range(3):
                        s = 0
                        for j in range(3):
                            s += u_flat[i*3+j] * v_flat[j*3+k]
                        C_dec[i][k] += s * w_flat[i*3+k]
            for i in range(3):
                for k in range(3):
                    if C_exp[i][k] != C_dec[i][k]:
                        return False
        return True
    
    if not verify_laderman(laderman_terms):
        # Fallback to schoolbook rank 54
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j][col] = 1
                    V_sb[j*p+k][col] = 1
                    W_sb[k*n+i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Build rank-38 decomposition: two Laderman blocks for two 3x3 sub-problems
    for block in range(2):
        base = block * 19
        k_off = block * 3
        
        for t in range(19):
            u_flat, v_flat, w_flat = laderman_terms[t]
            col = base + t
            
            # U: A coefficients (shared A for both blocks)
            for idx in range(9):
                U[idx][col] = u_flat[idx]
            
            # V: B coefficients, B is 3x6 split into two 3x3 blocks
            for idx in range(9):
                j = idx // 3
                k_local = idx % 3
                k_global = k_off + k_local
                V[j * p + k_global][col] = v_flat[idx]
            
            # W: C coefficients, C is 3x6 split into two 3x3 blocks
            # W index: k*n + i
            for idx in range(9):
                i = idx // 3
                k_local = idx % 3
                k_global = k_off + k_local
                W[k_global * n + i][col] = w_flat[idx]
    
    # Final verification
    def verify_full(U_mat, V_mat, W_mat):
        random.seed(42)
        for trial in range(10):
            A = [[random.randint(-5, 5) for _ in range(3)] for _ in range(3)]
            B = [[random.randint(-5, 5) for _ in range(6)] for _ in range(3)]
            C_exp = [[0]*6 for _ in range(3)]
            for i in range(3):
                for k in range(6):
                    s = 0
                    for j in range(3):
                        s += A[i][j] * B[j][k]
                    C_exp[i][k] = s
            C_dec = [[0]*6 for _ in range(3)]
            for c in range(rank):
                for i in range(3):
                    for k in range(6):
                        s = 0
                        for j in range(3):
                            s += U_mat[i*m+j][c] * V_mat[j*p+k][c]
                        C_dec[i][k] += s * W_mat[k*n+i][c]
            for i in range(3):
                for k in range(6):
                    if C_exp[i][k] != C_dec[i][k]:
                        return False
        return True
    
    if verify_full(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j][col] = 1
                    V_sb[j*p+k][col] = 1
                    W_sb[k*n+i][col] = 1
                    col += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
