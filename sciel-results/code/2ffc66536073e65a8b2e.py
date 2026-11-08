"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 3x3 * 3x8 multiplication using Strassen 2x2 for the top-left
    2x2 block of A times the 2x8 block of B (as two 2x2*2x4 Strassen),
    plus schoolbook for remaining terms. Total rank = 68.
    """
    import numpy as np
    import time
    
    start_time = time.time()
    deadline = start_time + 100
    
    n, m, p = 3, 3, 8
    
    # Build Strassen 2x2*2x2 decomposition helper
    # For A (2x2) * B (2x2) = C (2x2):
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    #
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b10-b11)
    # M4 = a11*(b00-b10)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # c00 = M1+M4-M2+M7
    # c01 = M3+M1
    # c10 = M2+M4
    # c11 = M1-M3+M5-M6
    
    def strassen_2x2(a_idx, b_idx, c_idx, U, V, W, col):
        """
        a_idx: 4 indices into U for A[0,0], A[0,1], A[1,0], A[1,1]
        b_idx: 4 indices into V for B[0,0], B[0,1], B[1,0], B[1,1]
        c_idx: 4 indices into W for C[0,0], C[0,1], C[1,0], C[1,1]
        Returns new col after adding 7 terms.
        """
        a00, a01, a10, a11 = a_idx
        b00, b01, b10, b11 = b_idx
        c00, c01, c10, c11 = c_idx
        
        # M1 = (a00+a11)*(b00+b11)
        U[a00][col] += 1; U[a11][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[c00][col] += 1; W[c01][col] += 1; W[c10][col] -= 1; W[c11][col] += 1
        col += 1
        
        # M2 = (a10+a11)*b00
        U[a10][col] += 1; U[a11][col] += 1
        V[b00][col] += 1
        W[c00][col] -= 1; W[c10][col] += 1
        col += 1
        
        # M3 = a00*(b10-b11)
        U[a00][col] += 1
        V[b10][col] += 1; V[b11][col] -= 1
        W[c01][col] += 1; W[c11][col] -= 1
        col += 1
        
        # M4 = a11*(b00-b10)
        U[a11][col] += 1
        V[b00][col] += 1; V[b10][col] -= 1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1
        
        # M5 = (a00+a01)*b11
        U[a00][col] += 1; U[a01][col] += 1
        V[b11][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M6 = (a10-a00)*(b00+b01)
        U[a10][col] += 1; U[a00][col] -= 1
        V[b00][col] += 1; V[b01][col] += 1
        W[c11][col] -= 1
        col += 1
        
        # M7 = (a01-a11)*(b10+b11)
        U[a01][col] += 1; U[a11][col] -= 1
        V[b10][col] += 1; V[b11][col] += 1
        W[c00][col] += 1
        col += 1
        
        return col
    
    # U index: i*m + j for A[i,j]
    # V index: j*p + k for B[j,k]  
    # W index: k*n + i for C[i,k]
    
    # A11 = A[0:2, 0:2]: U[0], U[1], U[2], U[3]
    # A12 = A[0:2, 2:3]: U[4], U[5]
    # A21 = A[2:3, 0:2]: U[6], U[7]
    # A22 = A[2:3, 2:3]: U[8]
    
    # B[0:2, 0:2]: V[0], V[1], V[8], V[9]
    # B[0:2, 2:4]: V[2], V[3], V[10], V[11]
    # B[0:2, 4:6]: V[4], V[5], V[12], V[13]
    # B[0:2, 6:8]: V[6], V[7], V[14], V[15]
    # B[2:3, 0:8]: V[16..23]
    
    # C[0:2, 0:2]: W[0], W[3], W[1], W[4]
    # C[0:2, 2:4]: W[6], W[9], W[7], W[10]
    # C[0:2, 4:6]: W[12], W[15], W[13], W[16]
    # C[0:2, 6:8]: W[18], W[21], W[19], W[22]
    # C[2:3, 0:8]: W[2], W[5], W[8], W[11], W[14], W[17], W[20], W[23]
    
    # Total rank: 4 Strassen (28) + schoolbook (40) = 68
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Strassen for A11 * B[0:2, 0:2]
    col = strassen_2x2([0, 1, 2, 3], [0, 1, 8, 9], [0, 3, 1, 4], U, V, W, col)
    
    # Strassen for A11 * B[0:2, 2:4]
    col = strassen_2x2([0, 1, 2, 3], [2, 3, 10, 11], [6, 9, 7, 10], U, V, W, col)
    
    # Strassen for A11 * B[0:2, 4:6]
    col = strassen_2x2([0, 1, 2, 3], [4, 5, 12, 13], [12, 15, 13, 16], U, V, W, col)
    
    # Strassen for A11 * B[0:2, 6:8]
    col = strassen_2x2([0, 1, 2, 3], [6, 7, 14, 15], [18, 21, 19, 22], U, V, W, col)
    
    # Schoolbook for A12 * B[2:3, 0:8]: 2*1*8 = 16 terms
    for i in range(2):
        for k in range(8):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Schoolbook for A21 * B[0:2, 0:8]: 1*2*8 = 16 terms
    for j in range(2):
        for k in range(8):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
    
    # Schoolbook for A22 * B[2:3, 0:8]: 1*1*8 = 8 terms
    for k in range(8):
        U[2 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 2][col] = 1
        col += 1
    
    # Verify correctness
    if time.time() < deadline:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Test with random matrices
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        # Compute C from decomposition: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is not straightforward. The tensor decomposition means:
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # where T is the matrix multiplication tensor.
        
        # Actually, the decomposition is:
        # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
        # = sum_j,k' A[i,j] * B[j,k'] * delta[k,k']  (schoolbook)
        
        # The tensor form: T_{ijk} = sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)}
        # where T_{ijk} = delta_{j,k} (the matrix multiplication tensor)
        
        # Wait, I need to be more careful. The matrix multiplication tensor is:
        # T(i,j,k) = 1 if this term appears in (A*B)[i,k] = sum_j A[i,j]*B[j,k]
        # So T(i,j,k) = delta_{j,j} = 1 for all valid i,j,k? No.
        
        # Actually, the tensor is defined as:
        # T(i,j,k) such that (A*B)[i,k] = sum_{j} A[i,j] * B[j,k]
        # In tensor form: (A*B)[i,k] = T(i,j,k) * A[i,j] * B[j,k] (Einstein summation)
        # So T(i,j,k) = 1 for all i,j,k (since each term A[i,j]*B[j,k] appears exactly once)
        
        # The decomposition: T(i,j,k) = sum_r U(i,j,r) * V(j,k,r) * W(k,i,r)
        
        # Let's verify:
        C_from_decomp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s = 0.0
                    for j in range(m):
                        s += U_np[i*m + j, r] * V_np[j*p + k, r]
                    C_from_decomp[i, k] += s * W_np[k*n + i, r]
        
        if np.allclose(C_from_decomp, C_expected, atol=1e-10):
            pass  # Correct!
        else:
            # Fall back to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
