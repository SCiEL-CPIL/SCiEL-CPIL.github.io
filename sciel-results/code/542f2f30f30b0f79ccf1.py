import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def _schoolbook(n, m, p):
    """Return (U, V, W) for the schoolbook decomposition of M_{n,m,p} with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_222():
    """
    Return (U, V, W) for the Strassen decomposition of M_{2,2,2} with rank 7.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen's 7 products:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A11+A22)(B21-B11)
    # P3 = (A11-A21)B11
    # P4 = (A22-A12)B22
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # U coefficients for A entries (row index = i*2+j)
    # V coefficients for B entries (row index = j*2+k)
    # W coefficients for C entries (row index = k*2+i)
    
    # P1: A11+A22, B11+B22
    U[0, 0] = 1; U[3, 0] = 1  # A11, A22
    V[0, 0] = 1; V[3, 0] = 1  # B11, B22
    W[0, 0] = 1; W[3, 0] = 1  # C11 (k=0,i=0), C22 (k=1,i=1)
    
    # P2: A11+A22, B21-B11
    U[0, 1] = 1; U[3, 1] = 1
    V[1, 1] = 1; V[0, 1] = -1  # B21, B11
    W[1, 1] = 1  # C21 (k=1,i=0)
    
    # P3: A11-A21, B11
    U[0, 2] = 1; U[2, 2] = -1  # A11, A21
    V[0, 2] = 1  # B11
    W[0, 2] = 1; W[1, 2] = 1  # C11 (k=0,i=0), C12 (k=1,i=0)
    
    # P4: A22-A12, B22
    U[3, 3] = 1; U[1, 3] = -1  # A22, A12
    V[3, 3] = 1  # B22
    W[1, 3] = 1; W[3, 3] = 1  # C21 (k=1,i=0), C22 (k=1,i=1)
    
    # P5: A11+A12, B22
    U[0, 4] = 1; U[1, 4] = 1  # A11, A12
    V[3, 4] = 1  # B22
    W[3, 4] = -1; W[1, 4] = 1  # C22 (k=1,i=1) with -1, C12 (k=1,i=0) with +1
    
    # P6: A21-A11, B11+B12
    U[2, 5] = 1; U[0, 5] = -1  # A21, A11
    V[0, 5] = 1; V[1, 5] = 1  # B11, B12
    W[3, 5] = 1  # C22 (k=1,i=1)
    
    # P7: A12-A22, B21+B22
    U[1, 6] = 1; U[3, 6] = -1  # A12, A22
    V[1, 6] = 1; V[3, 6] = 1  # B21, B22
    W[0, 6] = 1  # C11 (k=0,i=0)
    
    return U, V, W

def _combine(U1, V1, W1, U2, V2, W2):
    """
    Combine two decompositions by direct sum (block diagonal in the rank dimension).
    This is valid when the two decompositions operate on disjoint sets of output entries
    or can be combined linearly. For our purpose, we'll just concatenate the rank dimensions.
    """
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    rank = rank1 + rank2
    
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    return U, V, W

def _build_232():
    """
    Build M_{2,3,2} with rank <= 11.
    Split the 3 columns of A into 2+1.
    M_{2,2,2} uses Strassen (rank 7).
    M_{2,1,2} uses schoolbook (rank 4).
    Total rank = 7 + 4 = 11.
    """
    # M_{2,2,2}: Strassen
    U_s, V_s, W_s = _strassen_222()
    
    # M_{2,1,2}: schoolbook, rank = 2*1*2 = 4
    U_b, V_b, W_b = _schoolbook(2, 1, 2)
    
    # Now we need to embed these into M_{2,3,2}.
    # A is 2x3, B is 3x2, C is 2x2.
    # Split A's columns: A = [A_left (2x2) | A_right (2x1)]
    # Split B's rows: B = [B_top (2x2); B_bottom (1x2)]
    # C = A_left * B_top + A_right * B_bottom
    
    # For the left part (2x2 x 2x2), we use Strassen.
    # U_s has shape (4, 7), rows correspond to (i,j) for i in {0,1}, j in {0,1}
    # V_s has shape (4, 7), rows correspond to (j,k) for j in {0,1}, k in {0,1}
    # W_s has shape (4, 7), rows correspond to (k,i) for k in {0,1}, i in {0,1}
    
    # For the right part (2x1 x 1x2), we use schoolbook.
    # U_b has shape (2, 4), rows correspond to (i,j) for i in {0,1}, j in {0}
    # V_b has shape (2, 4), rows correspond to (j,k) for j in {0}, k in {0,1}
    # W_b has shape (4, 4), rows correspond to (k,i) for k in {0,1}, i in {0,1}
    
    # We need to create U_full (6, 11), V_full (6, 11), W_full (4, 11)
    # U_full rows: (i,j) for i in {0,1}, j in {0,1,2}
    # V_full rows: (j,k) for j in {0,1,2}, k in {0,1}
    # W_full rows: (k,i) for k in {0,1}, i in {0,1}
    
    rank = 11
    U_full = np.zeros((6, rank))
    V_full = np.zeros((6, rank))
    W_full = np.zeros((4, rank))
    
    # Left part: j in {0,1}
    # U_s row (i,j) maps to U_full row (i,j)
    # V_s row (j,k) maps to V_full row (j,k)
    # W_s row (k,i) maps to W_full row (k,i)
    for i in range(2):
        for j in range(2):
            U_full[i*3 + j, :7] = U_s[i*2 + j, :]
    for j in range(2):
        for k in range(2):
            V_full[j*2 + k, :7] = V_s[j*2 + k, :]
    for k in range(2):
        for i in range(2):
            W_full[k*2 + i, :7] = W_s[k*2 + i, :]
    
    # Right part: j = 2
    # U_b row (i,0) maps to U_full row (i,2)
    # V_b row (0,k) maps to V_full row (2,k)
    # W_b row (k,i) maps to W_full row (k,i)
    for i in range(2):
        U_full[i*3 + 2, 7:11] = U_b[i*1 + 0, :]
    for k in range(2):
        V_full[2*2 + k, 7:11] = V_b[0*2 + k, :]
    for k in range(2):
        for i in range(2):
            W_full[k*2 + i, 7:11] = W_b[k*2 + i, :]
    
    return U_full, V_full, W_full

def _build_233():
    """
    Build M_{2,3,3} with rank <= 17.
    Split the 3 columns of B into 1+2.
    M_{2,3,1} uses schoolbook (rank 6).
    M_{2,3,2} uses the construction from _build_232 (rank 11).
    Total rank = 6 + 11 = 17.
    """
    # M_{2,3,1}: schoolbook, rank = 2*3*1 = 6
    U_b1, V_b1, W_b1 = _schoolbook(2, 3, 1)
    
    # M_{2,3,2}: rank 11
    U_232, V_232, W_232 = _build_232()
    
    # A is 2x3, B is 3x3, C is 2x3.
    # Split B's columns: B = [B_left (3x1) | B_right (3x2)]
    # C = A * B_left + A * B_right
    # C is 2x3, so C = [C_left (2x1) | C_right (2x2)]
    
    rank = 17
    U_full = np.zeros((6, rank))
    V_full = np.zeros((9, rank))
    W_full = np.zeros((6, rank))
    
    # Left part: k = 0
    # U_b1 row (i,j) maps to U_full row (i,j)
    # V_b1 row (j,0) maps to V_full row (j,0)
    # W_b1 row (0,i) maps to W_full row (0,i)
    for i in range(2):
        for j in range(3):
            U_full[i*3 + j, :6] = U_b1[i*3 + j, :]
    for j in range(3):
        V_full[j*3 + 0, :6] = V_b1[j*1 + 0, :]
    for i in range(2):
        W_full[0*2 + i, :6] = W_b1[0*2 + i, :]
    
    # Right part: k in {1,2}
    # U_232 row (i,j) maps to U_full row (i,j)
    # V_232 row (j,k) maps to V_full row (j, k+1)
    # W_232 row (k,i) maps to W_full row (k+1, i)
    for i in range(2):
        for j in range(3):
            U_full[i*3 + j, 6:17] = U_232[i*3 + j, :]
    for j in range(3):
        for k in range(2):
            V_full[j*3 + (k+1), 6:17] = V_232[j*2 + k, :]
    for k in range(2):
        for i in range(2):
            W_full[(k+1)*2 + i, 6:17] = W_232[k*2 + i, :]
    
    return U_full, V_full, W_full

def _build_235():
    """
    Build M_{2,3,5} with rank <= 28.
    Split the 5 columns of B into 2+3.
    M_{2,3,2} uses the construction from _build_232 (rank 11).
    M_{2,3,3} uses the construction from _build_233 (rank 17).
    Total rank = 11 + 17 = 28.
    """
    U_232, V_232, W_232 = _build_232()
    U_233, V_233, W_233 = _build_233()
    
    # A is 2x3, B is 3x5, C is 2x5.
    # Split B's columns: B = [B_left (3x2) | B_right (3x3)]
    # C = A * B_left + A * B_right
    # C is 2x5, so C = [C_left (2x2) | C_right (2x3)]
    
    rank = 28
    U_full = np.zeros((6, rank))
    V_full = np.zeros((15, rank))
    W_full = np.zeros((10, rank))
    
    # Left part: k in {0,1}
    # U_232 row (i,j) maps to U_full row (i,j)
    # V_232 row (j,k) maps to V_full row (j,k)
    # W_232 row (k,i) maps to W_full row (k,i)
    for i in range(2):
        for j in range(3):
            U_full[i*3 + j, :11] = U_232[i*3 + j, :]
    for j in range(3):
        for k in range(2):
            V_full[j*5 + k, :11] = V_232[j*2 + k, :]
    for k in range(2):
        for i in range(2):
            W_full[k*2 + i, :11] = W_232[k*2 + i, :]
    
    # Right part: k in {2,3,4}
    # U_233 row (i,j) maps to U_full row (i,j)
    # V_233 row (j,k) maps to V_full row (j, k+2)
    # W_233 row (k,i) maps to W_full row (k+2, i)
    for i in range(2):
        for j in range(3):
            U_full[i*3 + j, 11:28] = U_233[i*3 + j, :]
    for j in range(3):
        for k in range(3):
            V_full[j*5 + (k+2), 11:28] = V_233[j*3 + k, :]
    for k in range(3):
        for i in range(2):
            W_full[(k+2)*2 + i, 11:28] = W_233[k*2 + i, :]
    
    return U_full, V_full, W_full

def _build_237():
    """
    Build M_{2,3,7} with rank <= 39.
    Split the 7 columns of B into 2+5.
    M_{2,3,2} uses the construction from _build_232 (rank 11).
    M_{2,3,5} uses the construction from _build_235 (rank 28).
    Total rank = 11 + 28 = 39.
    """
    U_232, V_232, W_232 = _build_232()
    U_235, V_235, W_235 = _build_235()
    
    # A is 2x3, B is 3x7, C is 2x7.
    # Split B's columns: B = [B_left (3x2) | B_right (3x5)]
    # C = A * B_left + A * B_right
    # C is 2x7, so C = [C_left (2x2) | C_right (2x5)]
    
    rank = 39
    U_full = np.zeros((6, rank))
    V_full = np.zeros((21, rank))
    W_full = np.zeros((14, rank))
    
    # Left part: k in {0,1}
    # U_232 row (i,j) maps to U_full row (i,j)
    # V_232 row (j,k) maps to V_full row (j,k)
    # W_232 row (k,i) maps to W_full row (k,i)
    for i in range(2):
        for j in range(3):
            U_full[i*3 + j, :11] = U_232[i*3 + j, :]
    for j in range(3):
        for k in range(2):
            V_full[j*7 + k, :11] = V_232[j*2 + k, :]
    for k in range(2):
        for i in range(2):
            W_full[k*2 + i, :11] = W_232[k*2 + i, :]
    
    # Right part: k in {2,3,4,5,6}
    # U_235 row (i,j) maps to U_full row (i,j)
    # V_235 row (j,k) maps to V_full row (j, k+2)
    # W_235 row (k,i) maps to W_full row (k+2, i)
    for i in range(2):
        for j in range(3):
            U_full[i*3 + j, 11:39] = U_235[i*3 + j, :]
    for j in range(3):
        for k in range(5):
            V_full[j*7 + (k+2), 11:39] = V_235[j*5 + k, :]
    for k in range(5):
        for i in range(2):
            W_full[(k+2)*2 + i, 11:39] = W_235[k*2 + i, :]
    
    return U_full, V_full, W_full

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is correct.
    Check that for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    And for all other (alpha, beta, gamma), the sum is 0.
    """
    rank = U.shape[1]
    
    # Check the 1-entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    
    # Check the 0-entries
    # This is expensive, but for small dimensions it's fine.
    # We'll check a subset for speed, or all if rank is small.
    for alpha in range(n*m):
        for beta in range(m*p):
            for gamma in range(p*n):
                # Skip the 1-entries
                i1 = alpha // m
                j1 = alpha % m
                j2 = beta // p
                k2 = beta % p
                k1 = gamma // n
                i2 = gamma % n
                if i1 == i2 and j1 == j2 and k1 == k2:
                    continue
                s = 0.0
                for r in range(rank):
                    s += U[alpha, r] * V[beta, r] * W[gamma, r]
                if abs(s) > 1e-9:
                    return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Try the constructive approach
    try:
        U, V, W = _build_237()
        
        # Verify
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
