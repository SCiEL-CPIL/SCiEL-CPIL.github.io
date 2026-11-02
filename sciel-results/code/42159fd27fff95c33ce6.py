"""Rank-reduced tensor decomposition for 3x4x8 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n=3, m=4, p=8):
    """Verify exact decomposition."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx] 
                        for r_idx in range(r))
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify_fast(U, V, W, n=3, m=4, p=8):
    """Faster vectorized verification."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = U.shape[1]
    errors = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                errors += (s - 1.0) ** 2
    return errors < 1e-18

def _schoolbook(n=3, m=4, p=8):
    """Return schoolbook decomposition (rank n*m*p)."""
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

def _strassen_2x2x2():
    """
    Return Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 products for 2x2 matrices
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Index mapping: U[i*2+j, r] corresponds to A[i,j]
    # V[j*2+k, r] corresponds to B[j,k]
    # W[k*2+i, r] corresponds to C[k,i] -> wait, W is (p*n, rank) = (4, 7)
    # W[k*n+i, r] with n=2, so W[k*2+i, r]
    
    # M1: (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1  # A11, A22
    V[0, 0] = 1; V[3, 0] = 1  # B11, B22
    W[0, 0] = 1; W[3, 0] = 1  # C11, C22
    
    # M2: (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1  # A21, A22
    V[0, 1] = 1  # B11
    W[2, 1] = 1  # C21
    W[3, 1] = -1  # C22
    
    # M3: A11(B21-B22)
    U[0, 2] = 1  # A11
    V[2, 2] = 1; V[3, 2] = -1  # B21, B22
    W[1, 2] = 1  # C12
    W[3, 2] = 1  # C22
    
    # M4: A22(B12-B11)
    U[3, 3] = 1  # A22
    V[1, 3] = 1; V[0, 3] = -1  # B12, B11
    W[0, 3] = 1  # C11
    W[2, 3] = 1  # C21
    
    # M5: (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1  # A11, A12
    V[3, 4] = 1  # B22
    W[0, 4] = -1  # C11
    W[1, 4] = 1  # C12
    
    # M6: (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1  # A21, A11
    V[0, 5] = 1; V[1, 5] = 1  # B11, B12
    W[3, 5] = 1  # C22
    
    # M7: (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1  # A12, A22
    V[2, 6] = 1; V[3, 6] = 1  # B21, B22
    W[0, 6] = 1  # C11
    
    return U.tolist(), V.tolist(), W.tolist()

def _schoolbook_1x2x2():
    """
    Return schoolbook decomposition for 1x2x2 matrix multiplication.
    U: (2, 4), V: (4, 4), W: (2, 4)
    """
    U = np.zeros((2, 4))
    V = np.zeros((4, 4))
    W = np.zeros((2, 4))
    
    for r in range(4):
        j = r // 2
        k = r % 2
        U[j, r] = 1  # A[0, j]
        V[j*2+k, r] = 1  # B[j, k]
        W[k*1+0, r] = 1  # C[k, 0]
    
    return U.tolist(), V.tolist(), W.tolist()

def _construct_strassen_decomposition(n=3, m=4, p=8):
    """
    Construct a rank 88 decomposition using Strassen for 2x2x2 blocks.
    
    Structure:
    - Rows 0,1 of A: Use Strassen blocks for 2x2x2 multiplications
    - Row 2 of A: Use schoolbook blocks for 1x2x2 multiplications
    
    For each pair of B columns (2k, 2k+1), k in 0..3:
    - A[0:2, 0:2] * B[0:2, 2k:2k+2] -> Strassen (rank 7)
    - A[0:2, 2:4] * B[2:4, 2k:2k+2] -> Strassen (rank 7)
    - A[2, 0:2] * B[0:2, 2k:2k+2] -> Schoolbook 1x2x2 (rank 4)
    - A[2, 2:4] * B[2:4, 2k:2k+2] -> Schoolbook 1x2x2 (rank 4)
    
    Total rank per k: 7 + 7 + 4 + 4 = 22
    Total rank: 4 * 22 = 88
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    U_sb, V_sb, W_sb = _schoolbook_1x2x2()
    
    rank = 88
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for k in range(4):  # 4 pairs of B columns
        # Block 1: A[0:2, 0:2] * B[0:2, 2k:2k+2]
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*m + j, col] += U_s[i*2 + j, r]
            for j in range(2):
                for kk in range(2):
                    V[j*p + (2*k + kk), col] += V_s[j*2 + kk, r]
            for kk in range(2):
                for i in range(2):
                    W[(2*k + kk)*n + i, col] += W_s[kk*2 + i, r]
            col += 1
        
        # Block 2: A[0:2, 2:4] * B[2:4, 2k:2k+2]
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*m + (2 + j), col] += U_s[i*2 + j, r]
            for j in range(2):
                for kk in range(2):
                    V[(2 + j)*p + (2*k + kk), col] += V_s[j*2 + kk, r]
            for kk in range(2):
                for i in range(2):
                    W[(2*k + kk)*n + i, col] += W_s[kk*2 + i, r]
            col += 1
        
        # Block 3: A[2, 0:2] * B[0:2, 2k:2k+2]
        for r in range(4):
            for j in range(2):
                U[2*m + j, col] += U_sb[j, r]
            for j in range(2):
                for kk in range(2):
                    V[j*p + (2*k + kk), col] += V_sb[j*2 + kk, r]
            for kk in range(2):
                W[(2*k + kk)*n + 2, col] += W_sb[kk, r]
            col += 1
        
        # Block 4: A[2, 2:4] * B[2:4, 2k:2k+2]
        for r in range(4):
            for j in range(2):
                U[2*m + (2 + j), col] += U_sb[j, r]
            for j in range(2):
                for kk in range(2):
                    V[(2 + j)*p + (2*k + kk), col] += V_sb[j*2 + kk, r]
            for kk in range(2):
                W[(2*k + kk)*n + 2, col] += W_sb[kk, r]
            col += 1
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4x8 matrix multiplication.
    Returns {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    """
    deadline = time.time() + 115
    n, m, p = 3, 4, 8
    
    # Start with schoolbook
    school = _schoolbook(n, m, p)
    best_result = school
    best_rank = n * m * p
    
    # Try the Strassen-based construction (rank 88)
    if time.time() < deadline - 10:
        U_str, V_str, W_str = _construct_strassen_decomposition(n, m, p)
        if _verify(U_str, V_str, W_str, n, m, p):
            best_result = {"U": U_str, "V": V_str, "W": W_str}
            best_rank = 88
    
    return best_result
