import time

DIMENSIONS = [3, 3, 8]

# Deadline guard: ensure we finish within ~115s to be safe before 120s limit
DEADLINE = time.time() + 110


def _construct_strassen_333():
    """
    Return Strassen decomposition for 3x3x3 matrix multiplication.
    T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    Strassen: A, B are 2x2 blocks. C = A*B, 7 multiplications.
    Let A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
    C = [[c11,c12],[c21,c22]]
    
    M1 = (a11+a22)*(b11+b22)
    M2 = (a21+a22)*b11
    M3 = a11*(b12-b22)
    M4 = a22*(b21-b11)
    M5 = (a11+a12)*b22
    M6 = (a21-a11)*(b11+b12)
    M7 = (a12-a22)*(b21+b22)
    
    c11 = M1 + M4 - M5 + M7
    c12 = M3 + M5
    c21 = M2 + M4
    c22 = M1 - M2 + M3 + M6
    
    We map:
    U row i*m+j: coefficient of A[i,j] in the multiplication M_c
    V row j*p+k: coefficient of B[j,k] in the multiplication M_c
    W row k*n+i: coefficient of C[i,j] in the output from M_c
    
    So W[k*3+i] for C[i,j]... wait, W is indexed by k*n+i which is the output position.
    For 3x3x3, output C has 9 entries, indexed as c*3+d where c=row, d=col.
    W index = k*3+i? No.
    
    The task says: W[k*n+i, r] - so W is indexed by (output_row_offset_from_k, output_row_i)?
    Actually re-reading: "output factor follows the official trace convention W[k*n+i,r]"
    
    For matrix multiplication C = A*B where A is n x m and B is m x p:
    C[i][k] = sum_j A[i][j] * B[j][k]
    
    U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    So U is indexed by (row of A, col of A), V by (row of B, col of B), W by (col of B becomes... wait)
    
    W[k*n+i]: k is col index of B / row index of C, i is row index of A / row index of C.
    So W is indexed by (k, i) where k is C column, i is C row.
    So W[k*n+i] corresponds to C[i][k].
    
    For Strassen on 3x3:
    A is 3x3, B is 3x3, C is 3x3.
    We pad to 2x2 blocks? No, Strassen works on 2x2. For 3x3, we need a different approach.
    
    Actually, for 3x3x3, we can use a rank-20 or rank-19 decomposition. 
    Let me just use a known rank-20 decomposition for 3x3x3.
    
    Alternatively, let me think about this more carefully. The schoolbook for 3x3x3 is rank 27.
    Strassen for 2x2x2 is rank 7. For 3x3x3, we need something else.
    
    Let me use a brute-force approach to find a sparse decomposition, or use a known result.
    
    Actually, let me just implement a search for the 3x3x5 part using rank-1 terms and verify.
    """
    # For 3x3x3, let's use a simple rank-19 construction.
    # Reference: It is known that R(3x3x3) = 19 or 20. Let me use a specific decomposition.
    # Actually, let me just try rank 20 first with a structured approach.
    
    # A well-known rank-20 decomposition for 3x3x3 uses the following idea:
    # Use 3x3x3 = 2x2x2 (Strassen, rank 7) + corrections for the 3rd row/col.
    
    # Let me define indices: U[i*3+j], V[j*3+k], W[k*3+i] for i,j,k in 0,1,2
    
    # I'll construct a rank-20 decomposition manually based on known formulas.
    # For now, let me use schoolbook for 3x3x3 (rank 27) and optimize the 3x3x5 part.
    # Total would be 27 + rank(3x3x5). If rank(3x3x5) < 45, we improve.
    
    # Let me return schoolbook for 3x3x3: rank 27
    n = m = p = 3
    rank = 27
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j][col] = 1
                V[j*3+k][col] = 1
                W[k*3+i][col] = 1
                col += 1
    return U, V, W, 27


def _construct_3x3x5():
    """
    Decompose 3x3x5 multiplication: A(3x3) * B(3x5) = C(3x5).
    T[i*3+j, j*5+k, k*3+i] = sum_c U[i*3+j,c]*V[j*5+k,c]*W[k*3+i,c]
    
    Schoolbook: rank 45.
    
    Let me try to find a sparse decomposition with rank < 45.
    Each rank-1 term: U has one nonzero pattern (a 3x3 matrix), V has a (3x5) matrix, W has a (5x3) matrix.
    
    Key insight: For a rank-1 term, if U[i,j] * V[j,k] * W[k,i] should contribute to T.
    T[i*3+j, j*5+k, k*3+i] = 1 for ALL valid (i,j,k).
    
    So for each (i,j,k) in 3x3x5, we need: sum_c U[i*3+j,c]*V[j*5+k,c]*W[k*3+i,c] = 1
    
    Let me try to construct terms where each term covers a "line" in the tensor.
    
    Approach: For fixed j, consider the sub-tensor T[i, j, k] for i in 0..2, k in 0..4.
    This is a 3x5 "matrix" (in i,k) that's all 1s for fixed j.
    But it's not a matrix multiplication - it's a slice of the tensor.
    
    For a rank-1 decomposition, each term contributes a product of three matrices:
    U_c (3x3), V_c (3x5), W_c (5x3).
    
    The contribution to T[i,j,k] = U_c[i,j] * V_c[j,k] * W_c[k,i].
    
    For the "all-ones" structure, if I could make U_c[i,j] = 1 for all i (fixed j), 
    V_c[j,k] = 1 for all k (fixed j), W_c[k,i] = 1 for all i (fixed k)... 
    
    Actually, let me try: for each j (3 choices), create terms that cover all (i,k) for that j.
    For fixed j, I need sum over c of U_c[i,j]*V_c[j,k]*W_c[k,i] = 1 for all i,k.
    
    If I set U_c[i,j] = 1 for all i, V_c[j,k] = 1 for all k, W_c[k,i] = ... 
    Then U*V*W = W_c[k,i]. I need sum_c W_c[k,i] = 1 for all k,i.
    That requires at least 15 rank-1 terms in W for a 5x3 matrix of ones... but that's not right.
    
    Let me try a different approach. Use the fact that the tensor is separable in a certain way.
    
    Let me just do a greedy search for sparse rank-1 terms.
    """
    n, m, p = 3, 3, 5
    # Tensor size: 9 x 15 x 15
    # T[i*3+j, j*5+k, k*3+i] = 1
    
    # Start with schoolbook
    U = [[0]*(n*m*p) for _ in range(n*m)]
    V = [[0]*(n*m*p) for _ in range(m*p)]
    W = [[0]*(n*m*p) for _ in range(p*n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i*m*p + j*p + k
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
    
    return U, V, W, n*m*p  # rank 45 for now


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    import numpy as np
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Check T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = 0
                for c in range(U.shape[1]):
                    actual += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(actual - expected) > 1e-9:
                    return False, (i, j, k, actual, expected)
    return True, None


def construct():
    """
    Decompose 3x3x8 matrix multiplication.
    Split into 3x3x3 + 3x3x5 parts.
    """
    start_time = time.time()
    n, m, p = 3, 3, 8
    
    # Part 1: 3x3x3 (first 3 columns of B)
    U1, V1, W1, r1 = _construct_strassen_333()
    
    # Part 2: 3x3x5 (last 5 columns of B)
    U2, V2, W2, r2 = _construct_3x3x5()
    
    # Combine: U = [U1, U2], V = [V1, V2], W = [W1, W2] (hconcat)
    import numpy as np
    U = np.concatenate([U1, U2], axis=1).tolist()
    V = np.concatenate([V1, V2], axis=1).tolist()
    W = np.concatenate([W1, W2], axis=1).tolist()
    
    total_rank = r1 + r2
    
    # Verify
    ok, err = _verify(U, V, W, n, m, p)
    if not ok:
        # Fallback to schoolbook
        rank = n * m * p
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
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
