import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    U, V, W = get_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def get_decomposition(n, m, p):
    # Base cases
    if m == 1:
        return schoolbook(n, 1, p)
    if n == 1:
        return schoolbook(1, m, p)
    if p == 1:
        return schoolbook(n, m, 1)
    
    # Strassen for 2x2x2
    if n == 2 and m == 2 and p == 2:
        return strassen_2x2()

    # Recursive split of m
    # Try splits m1, m2
    best_rank = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for m1 in range(1, m):
        m2 = m - m1
        U1, V1, W1 = get_decomposition(n, m1, p)
        U2, V2, W2 = get_decomposition(n, m2, p)
        
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        r_total = r1 + r2
        
        if r_total < best_rank:
            best_rank = r_total
            # Construct total U, V, W
            # U: (n*m, r_total)
            U_total = np.zeros((n*m, r_total))
            V_total = np.zeros((m*p, r_total))
            W_total = np.zeros((p*n, r_total))
            
            # Fill U1
            # U1 is for A1 (n x m1). Indices i, j in [0, m1)
            # In U_total, row index is i*m + j
            for i in range(n):
                for j in range(m1):
                    U_total[i*m + j, :r1] = U1[i*m1 + j, :]
            
            # Fill U2
            # U2 is for A2 (n x m2). Indices i, j in [0, m2)
            # In U_total, row index is i*m + (m1 + j)
            for i in range(n):
                for j in range(m2):
                    U_total[i*m + (m1 + j), r1:r1+r2] = U2[i*m2 + j, :]
            
            # Fill V1
            # V1 is for B1 (m1 x p). Indices j, k in [0, m1), [0, p)
            # In V_total, row index is j*p + k
            for j in range(m1):
                for k in range(p):
                    V_total[j*p + k, :r1] = V1[j*m1 + k, :] # Wait, V1 index is j*m1 + k? No, V1 is (m1*p, r1). Row index j*m1+k? No, V1 shape is (m1*p, r1).
                    # V1 row index is j*p + k? No, V1 corresponds to B1 which is m1 x p.
                    # The index in V1 is j*p + k? No, the index in V1 is j*p + k is wrong.
                    # V1 is (m1*p, r1). The row index is j*p + k?
                    # Let's check schoolbook. V is (m*p, rank). Row index j*p + k.
                    # So V1 row index is j*p + k? No, V1 is for a sub-matrix of size m1 x p.
                    # The local index is j*p + k? No, if the matrix is m1 x p, the index is j*p + k.
                    # But in the global V, the index is j*p + k.
                    # So V1[j*p + k] maps to V_total[j*p + k].
                    
            # Correction:
            # V1 shape is (m1*p, r1). Row index in V1 is j*p + k?
            # No, if V1 is for a matrix of size m1 x p, the row index is j*p + k?
            # Yes, if we flatten row-major.
            # But in the global V, the row index is j*p + k.
            # So we can just copy directly if the indices align.
            # V1 row j*p+k corresponds to global row j*p+k.
            
            # Let's re-verify V indexing.
            # V is (m*p, rank). Index (j,k) -> j*p + k.
            # V1 is (m1*p, rank). Index (j,k) -> j*p + k.
            # So V_total[j*p + k, :r1] = V1[j*p + k, :].
            
            # V2 is (m2*p, rank). Index (j,k) -> j*p + k.
            # But V2 corresponds to B2 which starts at row m1.
            # So global row is (m1 + j)*p + k.
            # V2 row is j*p + k.
            
            for j in range(m1):
                for k in range(p):
                    V_total[j*p + k, :r1] = V1[j*p + k, :]
            
            for j in range(m2):
                for k in range(p):
                    V_total[(m1 + j)*p + k, r1:r1+r2] = V2[j*p + k, :]
            
            # Fill W
            # W is (p*n, rank). Index (k,i) -> k*n + i.
            # W1 is (p*n, r1). Index (k,i) -> k*n + i.
            # W2 is (p*n, r2). Index (k,i) -> k*n + i.
            # They don't overlap in rows?
            # W1 and W2 both have shape (p*n, r).
            # The rows are indexed by (k,i).
            # Since C1 and C2 are both n x p, the indices (k,i) are the same.
            # So we just stack columns.
            
            W_total[:, :r1] = W1
            W_total[:, r1:r1+r2] = W2
            
            best_U = U_total
            best_V = V_total
            best_W = W_total
            
    return best_U, best_V, best_W

def schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W

def strassen_2x2():
    # Returns U, V, W for 2x2x2
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Coefficients for M1..M7
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Indices:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3 (k*n+i)
    # W index: k*2+i
    
    # Term 1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = -1; W[3, 0] = 1
    
    # Term 2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # Term 3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # Term 4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # Term 5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # Term 6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Term 7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = -1
    
    return U, V, W
