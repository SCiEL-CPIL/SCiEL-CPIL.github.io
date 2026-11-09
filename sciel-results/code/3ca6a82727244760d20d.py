import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 100
    n, m, p = 3, 3, 7

    def verify_bilinear(U, V, W, n, m, p):
        """
        Verify that the bilinear decomposition correctly computes matrix multiplication.
        C = A * B where A is n x m, B is m x p, C is n x p.
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        Bilinear decomposition of rank r:
        C[i,k] = sum_s (sum_j U[i*m+j, s] * A[i,j]) * (sum_j V[j*p+k, s] * B[j,k]) * W[k*n+i, s]
        
        Actually, the standard form is:
        C[i,k] = sum_s M_s[i] * N_s[k]
        where M_s[i] = sum_j U_s[i,j] * A[i,j] and N_s[k] = sum_j V_s[j,k] * B[j,k]
        So C[i,k] = sum_s (sum_j U[i*m+j,s]*A[i,j]) * (sum_j V[j*p+k,s]*B[j,k])
        and W is not directly used in this form, OR W[k*n+i,s] is an additional factor.
        
        Given the interface has W, the decomposition must be:
        C[i,k] = sum_s W[k*n+i, s] * (sum_j U[i*m+j, s] * A[i,j]) * (sum_j V[j*p+k, s] * B[j,k])
        """
        import random
        random.seed(12345)
        A = [[random.randint(0, 9) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(0, 9) for _ in range(p)] for _ in range(m)]
        
        C_true = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        rank = len(U[0])
        for i in range(n):
            for k in range(p):
                val = 0
                for s in range(rank):
                    left = sum(U[i * m + j][s] * A[i][j] for j in range(m))
                    right = sum(V[j * p + k][s] * B[j][k] for j in range(m))
                    val += W[k * n + i][s] * left * right
                if val != C_true[i][k]:
                    return False
        return True

    def verify_tensor(U, V, W, n, m, p):
        """
        Alternative: verify the tensor identity directly.
        T[i,j,k] = 1 for all i,j,k.
        T[i,j,k] = sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
        """
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for s in range(rank):
                        val += U[i * m + j][s] * V[j * p + k][s] * W[k * n + i][s]
                    if val != 1:
                        return False
        return True

    # Try rank-3 solution based on tensor identity
    U3 = [[0]*3 for _ in range(9)]
    V3 = [[0]*3 for _ in range(21)]
    W3 = [[0]*3 for _ in range(21)]
    for j in range(3):
        for i in range(3):
            U3[i*3+j][j] = 1
        for k in range(7):
            V3[j*7+k][j] = 1
        for i in range(3):
            for k in range(7):
                W3[k*3+i][j] = 1

    # Test both verification methods
    if verify_tensor(U3, V3, W3, n, m, p):
        if verify_bilinear(U3, V3, W3, n, m, p):
            return {"U": U3, "V": V3, "W": W3}
    
    # Try schoolbook
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

    if verify_tensor(U_sb, V_sb, W_sb, n, m, p):
        if verify_bilinear(U_sb, V_sb, W_sb, n, m, p):
            return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Fallback: return schoolbook anyway
    return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
