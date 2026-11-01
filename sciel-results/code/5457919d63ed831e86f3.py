"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    start_time = time.time()
    deadline = 120  # seconds
    
    n, m, p = 4, 4, 4
    rank = n * m * p
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # We want to find rank-r decomposition: T = sum_c U[:,c] outer V[:,c] outer W[:,c]
    
    # Start with schoolbook rank 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try Strassen-like recursive decomposition for 4x4
    # 4x4 can be split into 2x2 blocks
    # Use Strassen for 2x2 (rank 7) recursively
    
    def strassen_2x2():
        """Return U, V, W for 2x2 matrix multiplication with rank 7."""
        # Indices: A is 2x2, B is 2x2, C is 2x2
        # A = [[a00, a01], [a10, a11]]
        # B = [[b00, b01], [b10, b11]]
        # C = [[c00, c01], [c10, c11]]
        #
        # Strassen's 7 products:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        #
        # c00 = M1+M4-M5+M7
        # c01 = M3+M5
        # c10 = M2+M4
        # c11 = M1-M2+M3+M6
        #
        # U indices: a00=0, a01=1, a10=2, a11=3
        # V indices: b00=0, b01=1, b10=2, b11=3
        # W indices: c00=0, c01=1, c10=2, c11=3 (W[k*n+i] with n=2, so W[0*2+0]=c00, W[0*2+1]=c01, W[1*2+0]=c10, W[1*2+1]=c11)
        
        U2 = [[0]*7 for _ in range(4)]
        V2 = [[0]*7 for _ in range(4)]
        W2 = [[0]*7 for _ in range(4)]
        
        # M1 = (a00+a11)(b00+b11)
        U2[0][0] = 1; U2[3][0] = 1
        V2[0][0] = 1; V2[3][0] = 1
        W2[0][0] = 1; W2[3][0] = 1
        
        # M2 = (a10+a11)b00
        U2[2][1] = 1; U2[3][1] = 1
        V2[0][1] = 1
        W2[2][1] = 1; W2[3][1] = -1
        
        # M3 = a00(b01-b11)
        U2[0][2] = 1
        V2[1][2] = 1; V2[3][2] = -1
        W2[1][2] = 1; W2[3][2] = 1
        
        # M4 = a11(b10-b00)
        U2[3][3] = 1
        V2[2][3] = 1; V2[0][3] = -1
        W2[0][3] = 1; W2[2][3] = 1
        
        # M5 = (a00+a01)b11
        U2[0][4] = 1; U2[1][4] = 1
        V2[3][4] = 1
        W2[0][4] = -1; W2[1][4] = 1
        
        # M6 = (a10-a00)(b00+b01)
        U2[2][5] = 1; U2[0][5] = -1
        V2[0][5] = 1; V2[1][5] = 1
        W2[3][5] = 1
        
        # M7 = (a01-a11)(b10+b11)
        U2[1][6] = 1; U2[3][6] = -1
        V2[2][6] = 1; V2[3][6] = 1
        W2[0][6] = 1; W2[3][6] = 1
        
        return U2, V2, W2
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is correct."""
        import numpy as np
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        r = U_arr.shape[1]
        
        # Build tensor from decomposition
        # T_decomp[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Check against schoolbook
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1
                    actual = 0.0
                    for c in range(r):
                        actual += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(actual - expected) > 1e-9:
                        return False
        # Also check that non-multiplication entries are zero
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # For each pair (I, J) where I = i*m+j, J = j*p+k
                    # The only K that should give nonzero is K = k*n+i
                    I = i * m + j
                    J = j * p + k
                    K = k * n + i
                    for K2 in range(p * n):
                        if K2 == K:
                            continue
                        val = 0.0
                        for c in range(r):
                            val += U_arr[I, c] * V_arr[J, c] * W_arr[K2, c]
                        if abs(val) > 1e-9:
                            return False
        return True
    
    # Build 4x4 using 2x2 Strassen blocks
    # 4x4 matrix A split into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # Similarly B and C
    # 
    # We can use a hierarchical approach:
    # Level 1: 4x4 = 2x2 blocks of 2x2 matrices
    # Use 7 "products" at the top level, each product is a 2x2*2x2 multiplication
    # Each 2x2*2x2 uses rank 7, so total rank = 7*7 = 49
    
    def build_4x4_strassen():
        """Build 4x4 matrix multiplication using hierarchical Strassen, rank 49."""
        U4 = [[0]*49 for _ in range(16)]
        V4 = [[0]*49 for _ in range(16)]
        W4 = [[0]*49 for _ in range(16)]
        
        # Top-level Strassen for 2x2 blocks
        # A blocks: A00 (indices 0-3), A01 (4-7), A10 (8-11), A11 (12-15)
        # B blocks: B00 (0-3), B01 (4-7), B10 (8-11), B11 (12-15)
        # C blocks: C00 (0-3), C01 (4-7), C10 (8-11), C11 (12-15)
        
        # Top-level Strassen combinations (same as 2x2 but for blocks)
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1+M4-M5+M7
        # C01 = M3+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        # For each top-level product Mt, we need to compute a 2x2*2x2 product
        # using the 2x2 Strassen decomposition (rank 7)
        # The 49 rank = 7 top-level * 7 bottom-level
        
        # Define the top-level structure
        # For each top-level term, we have:
        # - A factors: list of (block, sign) pairs
        # - B factors: list of (block, sign) pairs
        # - C factors: list of (block, sign) pairs
        
        top_terms = [
            # (A_factors, B_factors, C_factors)
            # M1 = (A00+A11)(B00+B11), contributes to C00(+1), C11(+1)
            ([((0, 1), (12, 1))],  # A00 + A11
             [((0, 1), (12, 1))],  # B00 + B11
             [((0, 1), (12, 1))]), # C00 + C11
            # M2 = (A10+A11)B00, contributes to C10(+1), C11(-1)
            ([((8, 1), (12, 1))],
             [((0, 1),)],
             [((8, 1), (12, -1))]),
            # M3 = A00(B01-B11), contributes to C01(+1), C11(+1)
            ([((0, 1),)],
             [((4, 1), (12, -1))],
             [((4, 1), (12, 1))]),
            # M4 = A11(B10-B00), contributes to C00(+1), C10(+1)
            ([((12, 1),)],
             [((8, 1), (0, -1))],
             [((0, 1), (8, 1))]),
            # M5 = (A00+A01)B11, contributes to C00(-1), C01(+1)
            ([((0, 1), (4, 1))],
             [((12, 1),)],
             [((0, -1), (4, 1))]),
            # M6 = (A10-A00)(B00+B01), contributes to C11(+1)
            ([((8, 1), (0, -1))],
             [((0, 1), (4, 1))],
             [((12, 1),)]),
            # M7 = (A01-A11)(B10+B11), contributes to C00(+1), C11(+1)
            ([((4, 1), (12, -1))],
             [((8, 1), (12, 1))],
             [((0, 1), (12, 1))]),
        ]
        
        # 2x2 Strassen decomposition
        U2, V2, W2 = strassen_2x2()
        
        for t_idx, (A_factors, B_factors, C_factors) in enumerate(top_terms):
            # For this top-level term, we need to compute:
            # (sum of A factors) * (sum of B factors) -> contributes to (sum of C factors)
            # Using 2x2 Strassen for the inner 2x2*2x2
            #
            # The inner product computes X*Y where X and Y are 2x2 matrices
            # X = sum of A factors (each is a 2x2 block with a sign)
            # Y = sum of B factors
            # Result Z = X*Y, which contributes to sum of C factors
            #
            # Using 2x2 Strassen: Z = sum_c U2[:,c] outer V2[:,c] outer W2[:,c]
            # where U2[i,c] * V2[j,c] * W2[k,c] gives the (i,j) -> k contribution
            #
            # But we need to account for the signs in the factors
            # X[i,j] = sum_{(block, sign) in A_factors} sign * A_block[i,j]
            # Y[i,j] = sum_{(block, sign) in B_factors} sign * B_block[i,j]
            # Z[i,j] = sum_{(block, sign) in C_factors} sign * C_block[i,j]
            
            for c2 in range(7):
                col = t_idx * 7 + c2
                # U2[i2, c2] gives the coefficient for A[i2, j2] in the inner product
                # V2[j2, c2] gives the coefficient for B[i2, j2]
                # W2[k2, c2] gives the coefficient for C[i2, j2]
                
                for i2 in range(2):
                    for j2 in range(2):
                        # A contribution: U2[i2, c2] * (sum of A factors at position (i2,j2))
                        # A block (block_idx) has elements at global indices:
                        # block_idx*4 + i2*2 + j2
                        for (block, sign) in A_factors:
                            global_idx = block * 4 + i2 * 2 + j2
                            U4[global_idx][col] += U2[i2 * 2 + j2, c2] * sign
                
                for i2 in range(2):
                    for j2 in range(2):
                        for (block, sign) in B_factors:
                            global_idx = block * 4 + i2 * 2 + j2
                            V4[global_idx][col] += V2[i2 * 2 + j2, c2] * sign
                
                for i2 in range(2):
                    for j2 in range(2):
                        for (block, sign) in C_factors:
                            global_idx = block * 4 + i2 * 2 + j2
                            W4[global_idx][col] += W2[i2 * 2 + j2, c2] * sign
        
        return U4, V4, W4
    
    U4, V4, W4 = build_4x4_strassen()
    
    # Verify the 4x4 Strassen decomposition
    if verify_decomposition(U4, V4, W4, 4, 4, 4):
        best_U, best_V, best_W = U4, V4, W4
        best_rank = 49
    else:
        # Fall back to schoolbook
        pass
    
    # Try to further reduce rank by finding linear dependencies
    # Check if we can reduce the rank by finding zero columns or linear dependencies
    
    def try_reduce_rank(U, V, W, n, m, p):
        """Try to reduce rank by removing linearly dependent columns."""
        import numpy as np
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        r = U_arr.shape[1]
        
        # Try removing each column and checking validity
        for c in range(r):
            if time.time() - start_time > deadline - 30:
                break
            U_new = np.delete(U_arr, c, axis=1)
            V_new = np.delete(V_arr, c, axis=1)
            W_new = np.delete(W_arr, c, axis=1)
            if verify_decomposition(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p):
                return U_new.tolist(), V_new.tolist(), W_new.tolist()
        return None
    
    # Try to reduce rank
    reduced = try_reduce_rank(best_U, best_V, best_W, 4, 4, 4)
    if reduced is not None:
        best_U, best_V, best_W = reduced
        best_rank = best_U[0].__len__() if best_U else 0
        # Try again
        reduced2 = try_reduce_rank(best_U, best_V, best_W, 4, 4, 4)
        if reduced2 is not None:
            best_U, best_V, best_W = reduced2
            best_rank = best_U[0].__len__() if best_U else 0
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
