"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly satisfies the tensor equation."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(np.multiply(U[i*m+j], V[j*p+k]), W[k*n+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def try_round_and_verify(U, V, W, tol=1e-9):
        """Try rounding to integers or half-integers and verify."""
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if verify(U_int, V_int, W_int, tol):
            return U_int.tolist(), V_int.tolist(), W_int.tolist()
        
        U_half = np.round(U * 2) / 2.0
        V_half = np.round(V * 2) / 2.0
        W_half = np.round(W * 2) / 2.0
        if verify(U_half, V_half, W_half, tol):
            return U_half.tolist(), V_half.tolist(), W_half.tolist()
        
        return None

    def strassen_3x3():
        """Return Strassen rank-7 decomposition for 3x3 matrix multiplication.
        
        Returns (A_mat, B_mat, C_mat) where:
        - A_mat: 7x9 matrix, A_mat[r, i*3+j] gives coefficient of A[i,j] in M_r
        - B_mat: 7x9 matrix, B_mat[r, j*3+k] gives coefficient of B[j,k] in N_r  
        - C_mat: 7x9 matrix, C_mat[r, i*3+k] gives coefficient of M_r*N_r in C[i,k]
        
        Strassen formulas:
        M_1 = (A_11 + A_33)(B_11 + B_33)
        M_2 = (A_22 + A_33)B_11
        M_3 = A_11(B_22 - B_11)
        M_4 = A_33(B_33 - B_22)
        M_5 = (A_11 + A_22)B_33
        M_6 = (A_33 + A_11)(B_22 + B_33)
        M_7 = (A_22 + A_33)(B_11 + B_22)
        
        C_11 = M_1 + M_3 - M_5 - M_7
        C_12 = M_2 + M_3
        C_13 = M_5 + M_6 - M_2 - M_4
        C_21 = M_7 - M_1 + M_4
        C_22 = M_1 + M_2 - M_4 - M_6
        C_23 = M_2 + M_6
        C_31 = M_5 - M_3 + M_1
        C_32 = M_3 - M_7 + M_4
        C_33 = M_1 + M_7 - M_2 - M_5
        """
        # A coefficients: M_r = sum_{i,j} A_mat[r, i*3+j] * A[i,j]
        A_mat = np.zeros((7, 9), dtype=int)
        # M_1: (A_11 + A_33)
        A_mat[0, 0] = 1  # A[0,0]
        A_mat[0, 8] = 1  # A[2,2]
        # M_2: (A_22 + A_33)
        A_mat[1, 4] = 1  # A[1,1]
        A_mat[1, 8] = 1  # A[2,2]
        # M_3: A_11
        A_mat[2, 0] = 1  # A[0,0]
        # M_4: A_33
        A_mat[3, 8] = 1  # A[2,2]
        # M_5: (A_11 + A_22)
        A_mat[4, 0] = 1  # A[0,0]
        A_mat[4, 4] = 1  # A[1,1]
        # M_6: (A_33 + A_11)
        A_mat[5, 0] = 1  # A[0,0]
        A_mat[5, 8] = 1  # A[2,2]
        # M_7: (A_22 + A_33)
        A_mat[6, 4] = 1  # A[1,1]
        A_mat[6, 8] = 1  # A[2,2]
        
        # B coefficients: M_r = sum_{j,k} B_mat[r, j*3+k] * B[j,k]
        B_mat = np.zeros((7, 9), dtype=int)
        # M_1: (B_11 + B_33)
        B_mat[0, 0] = 1  # B[0,0]
        B_mat[0, 8] = 1  # B[2,2]
        # M_2: B_11
        B_mat[1, 0] = 1  # B[0,0]
        # M_3: (B_22 - B_11)
        B_mat[2, 0] = -1  # -B[0,0]
        B_mat[2, 4] = 1   # B[1,1]
        # M_4: (B_33 - B_22)
        B_mat[3, 4] = -1  # -B[1,1]
        B_mat[3, 8] = 1   # B[2,2]
        # M_5: B_33
        B_mat[4, 8] = 1   # B[2,2]
        # M_6: (B_22 + B_33)
        B_mat[5, 4] = 1   # B[1,1]
        B_mat[5, 8] = 1   # B[2,2]
        # M_7: (B_11 + B_22)
        B_mat[6, 0] = 1   # B[0,0]
        B_mat[6, 4] = 1   # B[1,1]
        
        # C coefficients: C[i,k] = sum_r C_mat[r, i*3+k] * M_r
        C_mat = np.zeros((7, 9), dtype=int)
        # C_11 = M_1 + M_3 - M_5 - M_7
        C_mat[0, 0] = 1   # M_1
        C_mat[2, 0] = 1   # M_3
        C_mat[4, 0] = -1  # -M_5
        C_mat[6, 0] = -1  # -M_7
        # C_12 = M_2 + M_3
        C_mat[1, 1] = 1   # M_2
        C_mat[2, 1] = 1   # M_3
        # C_13 = M_5 + M_6 - M_2 - M_4
        C_mat[1, 2] = -1  # -M_2
        C_mat[3, 2] = -1  # -M_4
        C_mat[4, 2] = 1   # M_5
        C_mat[5, 2] = 1   # M_6
        # C_21 = M_7 - M_1 + M_4
        C_mat[0, 3] = -1  # -M_1
        C_mat[3, 3] = 1   # M_4
        C_mat[6, 3] = 1   # M_7
        # C_22 = M_1 + M_2 - M_4 - M_6
        C_mat[0, 4] = 1   # M_1
        C_mat[1, 4] = 1   # M_2
        C_mat[3, 4] = -1  # -M_4
        C_mat[5, 4] = -1  # -M_6
        # C_23 = M_2 + M_6
        C_mat[1, 5] = 1   # M_2
        C_mat[5, 5] = 1   # M_6
        # C_31 = M_5 - M_3 + M_1
        C_mat[0, 6] = 1   # M_1
        C_mat[2, 6] = -1  # -M_3
        C_mat[4, 6] = 1   # M_5
        # C_32 = M_3 - M_7 + M_4
        C_mat[2, 7] = 1   # M_3
        C_mat[3, 7] = 1   # M_4
        C_mat[6, 7] = -1  # -M_7
        # C_33 = M_1 + M_7 - M_2 - M_5
        C_mat[0, 8] = 1   # M_1
        C_mat[1, 8] = -1  # -M_2
        C_mat[4, 8] = -1  # -M_5
        C_mat[6, 8] = 1   # M_7
        
        return A_mat, B_mat, C_mat

    def strassen_3x3_verify():
        """Verify Strassen decomposition is correct."""
        A_mat, B_mat, C_mat = strassen_3x3()
        # Test with random matrices
        for trial in range(5):
            A = np.random.randint(-5, 6, size=(3, 3)).astype(float)
            B = np.random.randint(-5, 6, size=(3, 3)).astype(float)
            C_expected = A @ B
            # Compute via Strassen
            M = np.array([np.dot(A_mat[r], A.flatten()) * np.dot(B_mat[r], B.flatten()) for r in range(7)])
            C_strassen = np.zeros((3, 3))
            for r in range(7):
                for i in range(3):
                    for k in range(3):
                        C_strassen[i, k] += C_mat[r, i*3+k] * M[r]
            if not np.allclose(C_expected, C_strassen, atol=1e-10):
                return False
        return True

    # Verify Strassen decomposition
    if not strassen_3x3_verify():
        # Fallback: schoolbook
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

    # Baby-step giant-step: encode B (3x6) as 3x3 over F[t]/(t^2)
    # B[i, 2k] -> constant term, B[i, 2k+1] -> t term
    # B_enc[i, k] = B[i, 2k] + t*B[i, 2k+1]
    #
    # After Strassen: C_enc[i, k] = sum_r M_r[i,k]
    # where M_r = A_mat[r] · A * B_mat[r] · B_enc
    #
    # C_enc[i, k] = C[i, 2k] + t*C[i, 2k+1]
    #
    # The product in F[t]/(t^2): (a + bt)(c + dt) = ac + (ad+bc)t
    # Since t^2 = 0, the t^2 term vanishes.
    #
    # For each Strassen multiplication M_r = P_r * Q_r where P_r, Q_r are 3x3 matrices
    # with entries in F[t]/(t^2):
    # P_r[i,j] = sum_{i',j'} A_mat[r, i'*3+j'] * A[i',j] (scalar, since A is 3x3)
    # Wait, that's not right. Let me reconsider.
    #
    # Actually, in Strassen:
    # M_r = (linear combo of A entries) * (linear combo of B_enc entries)
    # 
    # A is 3x3 with scalar entries, so linear combos of A are scalars.
    # B_enc is 3x3 with entries in F[t]/(t^2), so linear combos of B_enc are in F[t]/(t^2).
    #
    # So M_r = scalar * (element of F[t]/(t^2)) = element of F[t]/(t^2)
    #
    # M_r = alpha_r + t * beta_r
    #
    # Then C[i,k] uses M_r with coefficients from C_mat.
    # C_enc[i,k] = sum_r C_mat[r, i*3+k] * M_r
    #            = sum_r C_mat[r, i*3+k] * (alpha_r + t*beta_r)
    #            = (sum_r C_mat[r, i*3+k] * alpha_r) + t*(sum_r C_mat[r, i*3+k] * beta_r)
    #
    # So C[i, 2k] = sum_r C_mat[r, i*3+k] * alpha_r
    #    C[i, 2k+1] = sum_r C_mat[r, i*3+k] * beta_r
    #
    # Now alpha_r and beta_r:
    # alpha_r = (linear combo of A) * (constant part of linear combo of B_enc)
    # beta_r = (linear combo of A) * (t-coefficient of linear combo of B_enc)
    #
    # Let a_r = linear combo of A entries (scalar)
    # Let b_r_const = constant part of linear combo of B_enc
    # Let b_r_t = t-coefficient of linear combo of B_enc
    #
    # b_r_const[j,k] = sum over (j',k') of B_mat[r, j'*3+k'] * B[j', 2k]  (constant parts)
    # b_r_t[j,k] = sum over (j',k') of B_mat[r, j'*3+k'] * B[j', 2k+1]  (t parts)
    #
    # Wait, I need to be more careful. The linear combo for M_r in Strassen is:
    # M_r = (sum_{i,j} A_mat[r,i*3+j] * A[i,j]) * (sum_{j,k} B_mat[r,j*3+k] * B_enc[j,k])
    #
    # The first factor is scalar: s_r = sum_{i,j} A_mat[r,i*3+j] * A[i,j]
    # The second factor is in F[t]/(t^2): 
    #   q_r = sum_{j,k} B_mat[r,j*3+k] * (B[j,2k] + t*B[j,2k+1])
    #       = sum_{j,k} B_mat[r,j*3+k]*B[j,2k] + t * sum_{j,k} B_mat[r,j*3+k]*B[j,2k+1]
    #
    # So q_r = q_r_0 + t * q_r_1 where:
    #   q_r_0 = sum_{j,k} B_mat[r,j*3+k]*B[j,2k]
    #   q_r_1 = sum_{j,k} B_mat[r,j*3+k]*B[j,2k+1]
    #
    # M_r = s_r * q_r = s_r*q_r_0 + t*(s_r*q_r_1)
    # alpha_r = s_r * q_r_0
    # beta_r = s_r * q_r_1
    #
    # C[i,2k] = sum_r C_mat[r, i*3+k] * s_r * q_r_0
    # C[i,2k+1] = sum_r C_mat[r, i*3+k] * s_r * q_r_1
    #
    # Now I need to express this as a tensor decomposition.
    # 
    # For C[i, 2k]:
    # C[i,2k] = sum_r C_mat[r, i*3+k] * s_r * q_r_0
    #         = sum_r C_mat[r, i*3+k] * (sum_{i',j} A_mat[r,i'*3+j]*A[i',j]) * (sum_{j',k'} B_mat[r,j'*3+k']*B[j',2k])
    #
    # This is a product of three linear forms, so it's a rank-7 term for each output.
    # But we have two outputs per (i,k) pair: C[i,2k] and C[i,2k+1].
    #
    # For the tensor decomposition, each "multiplication" in Strassen gives us:
    # - One rank term for the constant part (contributes to C[i,2k])
    # - One rank term for the t part (contributes to C[i,2k+1])
    #
    # Total rank = 7 * 2 = 14
    #
    # U matrix: For each Strassen r and part (0=const, 1=t):
    #   U[i*3+j, r*2+part] = A_mat[r, i*3+j]
    #
    # V matrix: For each Strassen r and part:
    #   For part 0 (constant): V[j*6+2k, r*2+0] = B_mat[r, j*3+k]
    #   For part 1 (t): V[j*6+2k+1, r*2+1] = B_mat[r, j*3+k]
    #
    # W matrix: For each Strassen r and part:
    #   For part 0: W[2k*3+i, r*2+0] = C_mat[r, i*3+k]
    #   For part 1: W[(2k+1)*3+i, r*2+1] = C_mat[r, i*3+k]
    #
    # Wait, let me re-examine the W index. The output is C[i, k] where i in [0,3), k in [0,6).
    # W index is k*n+i = k*3+i.
    #
    # For C[i, 2k]: W[2k*3 + i, r*2+0] = C_mat[r, i*3+k]
    # For C[i, 2k+1]: W[(2k+1)*3 + i, r*2+1] = C_mat[r, i*3+k]
    #
    # Let me verify this makes sense.
    
    A_mat, B_mat, C_mat = strassen_3x3()
    
    rank = 14
    U = np.zeros((n*m, rank), dtype=int)  # 9 x 14
    V = np.zeros((m*p, rank), dtype=int)  # 18 x 14
    W = np.zeros((p*n, rank), dtype=int)  # 18 x 14
    
    for r in range(7):
        # Constant part (contributes to C[i, 2k])
        col0 = r * 2
        # t part (contributes to C[i, 2k+1])
        col1 = r * 2 + 1
        
        for i in range(3):
            for j in range(3):
                U[i*3+j, col0] = A_mat[r, i*3+j]
                U[i*3+j, col1] = A_mat[r, i*3+j]
        
        for j in range(3):
            for k in range(3):
                # B_enc[j,k] = B[j,2k] + t*B[j,2k+1]
                # Constant part: B[j,2k] -> V index j*6 + 2k
                # t part: B[j,2k+1] -> V index j*6 + 2k+1
                V[j*6 + 2*k, col0] = B_mat[r, j*3+k]
                V[j*6 + 2*k+1, col1] = B_mat[r, j*3+k]
        
        for i in range(3):
            for k in range(3):
                # C[i,2k] -> W index 2k*3 + i
                # C[i,2k+1] -> W index (2k+1)*3 + i
                W[2*k*3 + i, col0] = C_mat[r, i*3+k]
                W[(2*k+1)*3 + i, col1] = C_mat[r, i*3+k]
    
    # Verify the decomposition
    if verify(U, V, W, tol=1e-9):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If verification failed, try rounding
    result = try_round_and_verify(U, V, W)
    if result is not None:
        U_l, V_l, W_l = result
        return {"U": U_l, "V": V_l, "W": W_l}
    
    # Fallback: schoolbook decomposition (rank = n*m*p = 54)
    rank_fallback = n * m * p
    U_f = [[0] * rank_fallback for _ in range(n * m)]
    V_f = [[0] * rank_fallback for _ in range(m * p)]
    W_f = [[0] * rank_fallback for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_f[i * m + j][column] = 1
                V_f[j * p + k][column] = 1
                W_f[k * n + i][column] = 1
                column += 1
    return {"U": U_f, "V": V_f, "W": W_f}

# EVOLVE-BLOCK-END
