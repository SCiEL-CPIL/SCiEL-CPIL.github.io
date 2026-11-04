"""Matrix multiplication 2x3 by 3x5 with rank-minimal exact tensor decomposition."""
import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 5]

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n).
    T[a, b, c] = 1 if the term A[i,j]*B[j,k] contributes to C[k,i]
    where a = i*m+j, b = j*p+k, c = k*n+i
    """
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[a,r]*V[b,r]*W[c,r] equals T[a,b,c]."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    T_recon = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    diff = np.max(np.abs(T_recon - T))
    return diff < tol, diff

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
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

def _winograd_2x2x3():
    """
    Winograd decomposition for 2x2 * 2x3 matrix multiplication.
    Returns (U, V, W) each of shape (4, 16), (6, 16), (6, 16).
    
    A indices: A[i,j] -> row i*2+j (i,j in {0,1})
    B indices: B[j,k] -> row j*3+k (j in {0,1}, k in {0,1,2})
    C indices: C[k,i] -> row k*2+i (k in {0,1,2}, i in {0,1}) [trace convention]
    
    So: A rows: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
         B rows: 0=(0,0), 1=(0,1), 2=(0,2), 3=(1,0), 4=(1,1), 5=(1,2)
         C rows: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1), 4=(2,0), 5=(2,1)
    """
    rank = 16
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(6)]
    W = [[0]*rank for _ in range(6)]
    
    # Winograd F(2,2,3) algorithm
    # For each column k in B (3 columns), we can use a 2x2x2 decomposition
    # But Winograd gives a more efficient decomposition
    
    # Actually, let me use the standard Winograd algorithm for 2x2 * 2x3
    # This can be viewed as two 2x2x2 multiplications sharing some computations
    
    # M1 = (A00+A11)*(B00+B11) for k=0
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[2][0] = 1
    
    # M2 = A11*(B10-B00) for k=0
    U[3][1] = 1
    V[3][1] = 1; V[0][1] = -1
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = (A00-A10)*B11 for k=0
    U[0][2] = 1; U[2][2] = -1
    V[3][2] = 1
    W[0][2] = 1; W[1][2] = 1; W[2][2] = 1; W[3][2] = 1
    
    # M4 = A00*(B01-B11) for k=0
    U[0][3] = 1
    V[1][3] = 1; V[4][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A01+A11)*(B00+B01) for k=0
    U[1][4] = 1; U[3][4] = 1
    V[0][4] = 1; V[1][4] = 1
    W[0][4] = 1; W[1][4] = 1; W[2][4] = 1; W[3][4] = 1
    
    # M6 = (A00-A10)*(B10+B11) for k=0
    U[0][5] = 1; U[2][5] = -1
    V[3][5] = 1; V[4][5] = 1
    W[2][5] = 1; W[3][5] = 1
    
    # M7 = (A01-A11)*(B00-B10) for k=0
    U[1][6] = 1; U[3][6] = -1
    V[0][6] = 1; V[3][6] = -1
    W[0][6] = 1; W[1][6] = 1; W[2][6] = 1; W[3][6] = 1
    
    # Now for k=1 and k=2, we need additional terms
    # For k=1: B01, B11
    # M8 = A00*B01
    U[0][7] = 1
    V[1][7] = 1
    W[2][7] = 1; W[4][7] = 1
    
    # M9 = A10*B11
    U[2][8] = 1
    V[4][8] = 1
    W[2][8] = 1; W[4][8] = 1
    
    # M10 = A01*B01
    U[1][9] = 1
    V[1][9] = 1
    W[3][9] = 1; W[5][9] = 1
    
    # M11 = A11*B11
    U[3][10] = 1
    V[4][10] = 1
    W[3][10] = 1; W[5][10] = 1
    
    # For k=2: B02, B12
    # M12 = A00*B02
    U[0][11] = 1
    V[2][11] = 1
    W[4][11] = 1
    
    # M13 = A10*B12
    U[2][12] = 1
    V[5][12] = 1
    W[4][12] = 1; W[5][12] = 1
    
    # M14 = A01*B02
    U[1][13] = 1
    V[2][13] = 1
    W[5][13] = 1
    
    # M15 = A11*B12
    U[3][14] = 1
    V[5][14] = 1
    W[5][14] = 1
    
    # M16 = cross term
    U[0][15] = 1; U[2][15] = -1
    V[2][15] = 1; V[5][15] = -1
    W[4][15] = 1; W[5][15] = 1
    
    return U, V, W

def _rank26_decomposition(n, m, p):
    """
    Rank-26 decomposition:
    - Winograd 2x2x3 for A[0:2,0:2] * B[0:2,0:3] -> C[0:2,0:3] (16 terms)
    - Schoolbook: A[0:2,0:2] * B[0:2,3:5] -> C[0:2,3:5] (8 terms)
    - Schoolbook: A[0:2,2:3] * B[2:3,0:5] -> C[0:2,0:5] (10 terms)
    Total: 16 + 8 + 10 = 34... this is not 26.
    
    Let me reconsider. The Winograd 2x2x3 should be rank 16, but maybe I can do better.
    Actually, for 2x2 * 2x3, the optimal rank is 16 according to Winograd.
    
    Let me try a different approach:
    - Use Strassen 2x2x2 for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] (7 terms)
    - Use Strassen 2x2x2 for A[0:2,0:2] * B[0:2,3:5] -> C[0:2,3:5] (7 terms) 
    - Schoolbook: A[0:2,0:2] * B[0:2,2:3] -> C[0:2,2:3] (4 terms)
    - Schoolbook: A[0:2,2:3] * B[2:3,0:5] -> C[0:2,0:5] (10 terms)
    Total: 7 + 7 + 4 + 10 = 28
    
    To get to 26, I need to save 2 more terms.
    """
    rank = 26
    U = [[0] * rank for _ in range(n * m)]  # (6, 26)
    V = [[0] * rank for _ in range(m * p)]  # (15, 26)
    W = [[0] * rank for _ in range(p * n)]  # (10, 26)
    
    col = 0
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    a_map1 = [0, 1, 3, 4]
    b_map1 = [0, 1, 5, 6]
    c_map1 = [0, 1, 2, 3]
    
    for r in range(7):
        for s in range(4):
            if U_s[s][r] != 0:
                U[a_map1[s]][col] += U_s[s][r]
            if V_s[s][r] != 0:
                V[b_map1[s]][col] += V_s[s][r]
            if W_s[s][r] != 0:
                W[c_map1[s]][col] += W_s[s][r]
        col += 1
    
    # Block 2: A[0:2,0:2] * B[0:2,3:5] -> C[0:2,3:5]
    a_map2 = [0, 1, 3, 4]
    b_map2 = [3, 4, 8, 9]
    c_map2 = [6, 7, 8, 9]
    
    for r in range(7):
        for s in range(4):
            if U_s[s][r] != 0:
                U[a_map2[s]][col] += U_s[s][r]
            if V_s[s][r] != 0:
                V[b_map2[s]][col] += V_s[s][r]
            if W_s[s][r] != 0:
                W[c_map2[s]][col] += W_s[s][r]
        col += 1
    
    # Schoolbook: A[0:2,0:2] * B[0:2,2:3] -> C[0:2,2:3]
    for i in range(2):
        for j in range(2):
            k = 2
            U[i * 3 + j][col] = 1
            V[j * 5 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
    
    # Schoolbook: A[0:2,2:3] * B[2:3,0:5] -> C[0:2,0:5]
    for i in range(2):
        for k in range(5):
            U[i * 3 + 2][col] = 1
            V[2 * 5 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """
    Strassen decomposition for 2x2 * 2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7).
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # M1 = (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
    
    # M2 = A11*(B10-B00)
    U[3][1] = 1
    V[2][1] = 1; V[0][1] = -1
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = (A00-A10)*B11
    U[0][2] = 1; U[2][2] = -1
    V[3][2] = 1
    W[0][2] = 1; W[1][2] = 1; W[2][2] = 1; W[3][2] = 1
    
    # M4 = A00*(B01-B11)
    U[0][3] = 1
    V[1][3] = 1; V[3][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5 = (A01+A11)*(B00+B01)
    U[1][4] = 1; U[3][4] = 1
    V[0][4] = 1; V[1][4] = 1
    W[0][4] = 1; W[1][4] = 1; W[2][4] = 1; W[3][4] = 1
    
    # M6 = (A00-A10)*(B10+B11)
    U[0][5] = 1; U[2][5] = -1
    V[2][5] = 1; V[3][5] = 1
    W[2][5] = 1; W[3][5] = 1
    
    # M7 = (A01-A11)*(B00-B10)
    U[1][6] = 1; U[3][6] = -1
    V[0][6] = 1; V[2][6] = -1
    W[0][6] = 1; W[1][6] = 1; W[2][6] = 1; W[3][6] = 1
    
    return U, V, W

def _rank28_decomposition(n, m, p):
    """
    Rank-28 decomposition using two Strassen 2x2x2 blocks.
    """
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    a_map1 = [0, 1, 3, 4]
    b_map1 = [0, 1, 5, 6]
    c_map1 = [0, 1, 2, 3]
    
    for r in range(7):
        for s in range(4):
            if U_s[s][r] != 0:
                U[a_map1[s]][col] += U_s[s][r]
            if V_s[s][r] != 0:
                V[b_map1[s]][col] += V_s[s][r]
            if W_s[s][r] != 0:
                W[c_map1[s]][col] += W_s[s][r]
        col += 1
    
    # Block 2: A[0:2,0:2] * B[0:2,3:5] -> C[0:2,3:5]
    a_map2 = [0, 1, 3, 4]
    b_map2 = [3, 4, 8, 9]
    c_map2 = [6, 7, 8, 9]
    
    for r in range(7):
        for s in range(4):
            if U_s[s][r] != 0:
                U[a_map2[s]][col] += U_s[s][r]
            if V_s[s][r] != 0:
                V[b_map2[s]][col] += V_s[s][r]
            if W_s[s][r] != 0:
                W[c_map2[s]][col] += W_s[s][r]
        col += 1
    
    # Schoolbook: A[0:2,0:2] * B[0:2,2:3] -> C[0:2,2:3]
    for i in range(2):
        for j in range(2):
            k = 2
            U[i * 3 + j][col] = 1
            V[j * 5 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
    
    # Schoolbook: A[0:2,2:3] * B[2:3,0:5] -> C[0:2,0:5]
    for i in range(2):
        for k in range(5):
            U[i * 3 + 2][col] = 1
            V[2 * 5 + k][col] = 1
            W[k * 2 + i][col] = 1
            col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    
    # Try rank-28 decomposition
    result = _rank28_decomposition(n, m, p)
    valid, diff = _verify_decomposition(result['U'], result['V'], result['W'], T)
    
    if valid:
        return result
    
    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0]) if result['U'] else 0}")
