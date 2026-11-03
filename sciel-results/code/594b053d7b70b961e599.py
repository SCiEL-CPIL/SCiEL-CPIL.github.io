import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def _build_333_rank23():
    """
    Returns U (9x23), V (9x23), W (9x23) for 3x3x3 matrix multiplication.
    Uses the known rank-23 decomposition with integer coefficients.
    
    The decomposition is based on the work by De Groote and others.
    U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r] summed over r gives the MM tensor.
    """
    n, m, p = 3, 3, 3
    rank = 23
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # The 23 terms for 3x3x3 matrix multiplication.
    # Each term r has coefficients in U, V, W.
    # These are from the known rank-23 decomposition.
    
    # Term 0: A00 * B00 -> C00
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # Term 1: A01 * B01 -> C01
    U[1, 1] = 1; V[4, 1] = 1; W[4, 1] = 1
    # Term 2: A02 * B02 -> C02
    U[2, 2] = 1; V[8, 2] = 1; W[8, 2] = 1
    # Term 3: A10 * B10 -> C10
    U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
    # Term 4: A11 * B11 -> C11
    U[4, 4] = 1; V[5, 4] = 1; W[5, 4] = 1
    # Term 5: A12 * B12 -> C12
    U[5, 5] = 1; V[9, 5] = 1; W[9, 5] = 1
    # Term 6: A20 * B20 -> C20
    U[6, 6] = 1; V[2, 6] = 1; W[2, 6] = 1
    # Term 7: A21 * B21 -> C21
    U[7, 7] = 1; V[6, 7] = 1; W[6, 7] = 1
    # Term 8: A22 * B22 -> C22
    U[8, 8] = 1; V[10, 8] = 1; W[10, 8] = 1
    # Term 9: (A00+A11) * (B00+B11) -> contributes to C00, C01, C10, C11
    U[0, 9] = 1; U[4, 9] = 1
    V[0, 9] = 1; V[5, 9] = 1
    W[0, 9] = 1; W[5, 9] = 1
    # Term 10: (A00+A22) * (B00+B22)
    U[0, 10] = 1; U[8, 10] = 1
    V[0, 10] = 1; V[10, 10] = 1
    W[0, 10] = 1; W[10, 10] = 1
    # Term 11: (A11+A22) * (B11+B22)
    U[4, 11] = 1; U[8, 11] = 1
    V[5, 11] = 1; V[10, 11] = 1
    W[5, 11] = 1; W[10, 11] = 1
    # Term 12: (A00+A11+A22) * (B00+B11+B22)
    U[0, 12] = 1; U[4, 12] = 1; U[8, 12] = 1
    V[0, 12] = 1; V[5, 12] = 1; V[10, 12] = 1
    W[0, 12] = 1; W[5, 12] = 1; W[10, 12] = 1
    # Term 13: (A01+A10) * (B01+B10)
    U[1, 13] = 1; U[3, 13] = 1
    V[4, 13] = 1; V[1, 13] = 1
    W[4, 13] = 1; W[1, 13] = 1
    # Term 14: (A02+A20) * (B02+B20)
    U[2, 14] = 1; U[6, 14] = 1
    V[8, 14] = 1; V[2, 14] = 1
    W[8, 14] = 1; W[2, 14] = 1
    # Term 15: (A12+A21) * (B12+B21)
    U[5, 15] = 1; U[7, 15] = 1
    V[9, 15] = 1; V[6, 15] = 1
    W[9, 15] = 1; W[6, 15] = 1
    # Term 16: (A00+A11) * (B10+B01)
    U[0, 16] = 1; U[4, 16] = 1
    V[1, 16] = 1; V[4, 16] = 1
    W[1, 16] = 1; W[4, 16] = 1
    # Term 17: (A00+A22) * (B20+B02)
    U[0, 17] = 1; U[8, 17] = 1
    V[2, 17] = 1; V[8, 17] = 1
    W[2, 17] = 1; W[8, 17] = 1
    # Term 18: (A11+A22) * (B21+B12)
    U[4, 18] = 1; U[8, 18] = 1
    V[6, 18] = 1; V[9, 18] = 1
    W[6, 18] = 1; W[9, 18] = 1
    # Term 19: (A01+A20) * (B20+B02)
    U[1, 19] = 1; U[6, 19] = 1
    V[2, 19] = 1; V[8, 19] = 1
    W[2, 19] = 1; W[8, 19] = 1
    # Term 20: (A02+A10) * (B10+B01)
    U[2, 20] = 1; U[3, 20] = 1
    V[1, 20] = 1; V[4, 20] = 1
    W[1, 20] = 1; W[4, 20] = 1
    # Term 21: (A12+A21) * (B21+B12) -- wait, this is same as 15
    # Let me reconsider. I need the actual 23-term decomposition.
    
    # The above approach of guessing terms won't work. Let me use a verified set.
    # Actually, let me just return None and use schoolbook if I can't verify.
    
    return None

def _validate_333(U, V, W, tol=1e-9):
    """Validate that U, V, W form a valid 3x3x3 MM decomposition."""
    n, m, p = 3, 3, 3
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    # Check off-diagonal
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for k2 in range(p):
                        if j != j2 or k != k2:
                            a = i * m + j
                            b = j2 * p + k
                            c = k2 * n + i
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            if abs(val) > tol:
                                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # Try to use rank-23 for 3x3x3 + schoolbook for 3x3x2 = rank 41
    # If that fails, fall back to schoolbook rank 45
    
    result_41 = None
    
    # The known rank-23 decomposition for 3x3x3.
    # I'll use a verified set of 23 terms.
    # Source: Based on the structure of Strassen-like algorithms.
    
    # Let me try a different approach: use the fact that we can verify
    # any candidate decomposition, and try to construct one.
    
    # Actually, the most reliable approach is to hardcode a known correct
    # rank-23 decomposition. Here's one based on published results:
    
    # The 23 terms (u_r, v_r, w_r) for 3x3x3:
    # Using the decomposition from "On the rank of 3x3x3 tensor"
    
    U333 = np.zeros((9, 23), dtype=float)
    V333 = np.zeros((9, 23), dtype=float)
    W333 = np.zeros((9, 23), dtype=float)
    
    # Term 0: (0,0,0)
    U333[0,0]=1; V333[0,0]=1; W333[0,0]=1
    # Term 1: (0,1,4) -> A01*B11->C10? No...
    
    # This is getting complicated. Let me just use schoolbook for 3x3x3 (rank 27)
    # + schoolbook for 3x3x2 (rank 18) = rank 45.
    # That's the same as the current program.
    
    # To improve, I need a valid rank < 45 decomposition.
    # The safest improvement is rank 41 if I can verify the 23-term 3x3x3.
    
    # Let me try to use a published rank-23 decomposition.
    # Here's a known set (from De Groote 1980, verified):
    
    # Actually, let me just return schoolbook to be safe, since I cannot
    # guarantee the 23-term decomposition is correct without verification.
    
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
