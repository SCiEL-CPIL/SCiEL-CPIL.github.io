import numpy as np
import time

DIMENSIONS = [4, 4, 5]
_START_TIME = time.time()
_TIME_LIMIT = 140  # seconds

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _validate_np(U, V, W, n, m, p):
    """
    Validate decomposition using NumPy.
    """
    R = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Compute the decomposition tensor D
    D = np.zeros((n*m, m*p, p*n), dtype=int)
    
    for r in range(R):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        D += np.einsum('i,j,k->ijk', u, v, w)
        
    # Construct Target Tensor T
    T = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1
                
    if not np.array_equal(D, T):
        return False
    return True

def _get_strassen_2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    Shapes: (4, 7)
    """
    # Strassen formulas:
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-f)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # Indices for A (2x2): a=0, b=1, c=2, d=3
    # Indices for B (2x2): e=0, f=1, g=2, h=3
    # Indices for C (2x2): c00=0, c01=1, c10=2, c11=3
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # P1: (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1 # C00 += P1, C11 += P1
    
    # P2: (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1 # C10 += P2, C11 -= P2
    
    # P3: a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1 # C01 += P3, C11 += P3
    
    # P4: d(g-f)
    U[3, 3] = 1
    V[2, 3] = 1; V[1, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1 # C00 += P4, C10 += P4
    
    # P5: (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1 # C00 -= P5, C01 += P5
    
    # P6: (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1 # C11 += P6
    
    # P7: (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1 # C00 += P7
    
    return U, V, W

def _kron_strassen_4x4():
    """
    Returns U, V, W for 4x4x4 decomposition using Kronecker product of Strassen.
    Shapes: (16, 49)
    """
    U2, V2, W2 = _get_strassen_2x2()
    R2 = 7
    R4 = R2 * R2
    
    U4 = np.zeros((16, R4), dtype=int)
    V4 = np.zeros((16, R4), dtype=int)
    W4 = np.zeros((16, R4), dtype=int)
    
    for r1 in range(R2):
        for r2 in range(R2):
            r4 = r1 * R2 + r2
            # U4[u, r4] = U2[u1, r1] * U2[u2, r2]
            # u = u1 * 2 + u2
            for u1 in range(4):
                for u2 in range(4):
                    u = u1 * 2 + u2
                    U4[u, r4] = U2[u1, r1] * U2[u2, r2]
                    V4[u, r4] = V2[u1, r1] * V2[u2, r2]
                    W4[u, r4] = W2[u1, r1] * W2[u2, r2]
                    
    return U4, V4, W4

def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                U[idx_u, r] = 1
                V[idx_v, r] = 1
                W[idx_w, r] = 1
                r += 1
                
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    
    # Fallback to schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = (U_school, V_school, W_school)
    best_rank = n * m * p
    
    # Try the structured 4x4x4 + 4x4x1 decomposition
    if n == 4 and m == 4 and p == 5:
        U4, V4, W4 = _kron_strassen_4x4()
        
        # Add the 4x4x1 part (k=4)
        # This part handles C[i, 4] = sum_j A[i, j] B[j, 4]
        # There are 16 such terms.
        # We append them to U, V, W
        
        R4 = U4.shape[1]
        R_new = R4 + 16
        
        U_new = np.zeros((16, R_new), dtype=int)
        V_new = np.zeros((20, R_new), dtype=int)
        W_new = np.zeros((20, R_new), dtype=int)
        
        # Copy the 4x4x4 part
        # Note: The 4x4x4 part uses indices k in 0..3.
        # W index for 4x4x4 is k*4 + i.
        # In the 4x4x5 tensor, W index is k*4 + i.
        # So indices 0..15 in W_new correspond to k=0..3.
        # This matches the 4x4x4 decomposition directly.
        
        U_new[:, :R4] = U4
        V_new[:16, :R4] = V4
        W_new[:16, :R4] = W4
        
        # Add the 4x4x1 part
        # For each i in 0..3, j in 0..3:
        # Term: A[i, j] * B[j, 4]
        # U index: i*4 + j
        # V index: j*5 + 4
        # W index: 4*4 + i = 16 + i
        
        r_offset = R4
        for i in range(4):
            for j in range(4):
                idx_u = i * 4 + j
                idx_v = j * 5 + 4
                idx_w = 16 + i
                
                U_new[idx_u, r_offset] = 1
                V_new[idx_v, r_offset] = 1
                W_new[idx_w, r_offset] = 1
                r_offset += 1
        
        # Validate
        if _validate_np(U_new, V_new, W_new, n, m, p):
            best_result = (U_new.tolist(), V_new.tolist(), W_new.tolist())
            best_rank = R_new
            
    return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
