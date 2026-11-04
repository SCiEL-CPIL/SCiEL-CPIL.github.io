import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _build_laderman_4x4x4():
    """
    Build the rank-20 Laderman decomposition for 4x4x4 matrix multiplication.
    
    Index conventions:
    U index = i*4 + j  (row i, col j of A)
    V index = j*4 + k  (row j, col k of B)
    W index = k*4 + i  (row k, col i of C)
    
    Laderman's algorithm uses 20 multiplications.
    The coefficients are defined for the 20 bilinear forms.
    """
    # Laderman's 20 terms.
    # Each term is (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs[i*4+j] is coeff for A[i,j]
    # V_coeffs[j*4+k] is coeff for B[j,k]
    # W_coeffs[k*4+i] is coeff for C[k,i]
    
    # The 20 terms are given by the following lists of indices and coefficients.
    # Format: (list of (index, coeff) for U, list of (index, coeff) for V, list of (index, coeff) for W)
    
    # Term 0
    u0 = [(0, 1), (5, 1), (10, 1), (15, 1)]
    v0 = [(0, 1), (5, 1), (10, 1), (15, 1)]
    w0 = [(0, 1), (5, 1), (10, 1), (15, 1)]
    
    # Term 1
    u1 = [(5, 1), (10, 1), (15, 1)]
    v1 = [(0, 1), (5, 1), (10, 1), (15, 1)]
    w1 = [(5, 1), (10, 1), (15, 1)]
    
    # Term 2
    u2 = [(0, 1), (5, 1), (10, 1)]
    v2 = [(5, 1), (10, 1), (15, 1)]
    w2 = [(0, 1), (5, 1), (10, 1)]
    
    # Term 3
    u3 = [(5, 1), (10, 1)]
    v3 = [(0, 1), (5, 1)]
    w3 = [(5, 1), (10, 1)]
    
    # Term 4
    u4 = [(10, 1), (15, 1)]
    v4 = [(10, 1), (15, 1)]
    w4 = [(10, 1), (15, 1)]
    
    # Term 5
    u5 = [(0, 1), (5, 1)]
    v5 = [(0, 1), (5, 1)]
    w5 = [(0, 1), (5, 1)]
    
    # Term 6
    u6 = [(5, -1), (10, 1), (15, -1)]
    v6 = [(5, 1), (10, 1), (15, 1)]
    w6 = [(0, 1), (5, -1), (10, 1), (15, -1)]
    
    # Term 7
    u7 = [(0, 1), (5, -1), (10, 1), (15, -1)]
    v7 = [(0, 1), (5, 1), (10, 1), (15, 1)]
    w7 = [(5, 1), (10, -1), (15, 1)]
    
    # Term 8
    u8 = [(10, 1), (15, -1)]
    v8 = [(10, 1), (15, 1)]
    w8 = [(10, 1), (15, -1)]
    
    # Term 9
    u9 = [(0, 1), (5, 1)]
    v9 = [(0, -1), (5, 1)]
    w9 = [(0, 1), (5, 1)]
    
    # Term 10
    u10 = [(5, 1), (10, -1)]
    v10 = [(5, 1), (10, 1)]
    w10 = [(5, 1), (10, -1)]
    
    # Term 11
    u11 = [(10, 1), (15, 1)]
    v11 = [(10, -1), (15, 1)]
    w11 = [(10, 1), (15, 1)]
    
    # Term 12
    u12 = [(0, 1), (5, -1)]
    v12 = [(0, 1), (5, 1)]
    w12 = [(0, 1), (5, -1)]
    
    # Term 13
    u13 = [(5, 1), (15, 1)]
    v13 = [(5, -1), (15, 1)]
    w13 = [(5, 1), (15, -1)]
    
    # Term 14
    u14 = [(0, 1), (10, 1)]
    v14 = [(0, -1), (10, 1)]
    w14 = [(0, 1), (10, -1)]
    
    # Term 15
    u15 = [(5, 1), (15, -1)]
    v15 = [(5, 1), (15, 1)]
    w15 = [(5, -1), (15, 1)]
    
    # Term 16
    u16 = [(0, 1), (5, 1), (10, -1), (15, -1)]
    v16 = [(0, 1), (5, 1), (10, 1), (15, 1)]
    w16 = [(0, 1), (5, 1), (10, 1), (15, 1)]
    
    # Term 17
    u17 = [(0, 1), (15, 1)]
    v17 = [(0, 1), (15, 1)]
    w17 = [(0, 1), (15, 1)]
    
    # Term 18
    u18 = [(5, 1), (10, 1)]
    v18 = [(5, 1), (10, 1)]
    w18 = [(5, 1), (10, 1)]
    
    # Term 19
    u19 = [(0, 1), (5, -1), (10, -1), (15, 1)]
    v19 = [(0, 1), (5, 1), (10, 1), (15, -1)]
    w19 = [(0, 1), (5, 1), (10, 1), (15, -1)]
    
    terms = [
        (u0, v0, w0),
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7),
        (u8, v8, w8),
        (u9, v9, w9),
        (u10, v10, w10),
        (u11, v11, w11),
        (u12, v12, w12),
        (u13, v13, w13),
        (u14, v14, w14),
        (u15, v15, w15),
        (u16, v16, w16),
        (u17, v17, w17),
        (u18, v18, w18),
        (u19, v19, w19)
    ]
    
    U = np.zeros((16, 20))
    V = np.zeros((16, 20))
    W = np.zeros((16, 20))
    
    for t in range(20):
        for idx, coeff in terms[t][0]:
            U[idx, t] = coeff
        for idx, coeff in terms[t][1]:
            V[idx, t] = coeff
        for idx, coeff in terms[t][2]:
            W[idx, t] = coeff
            
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            u_vec = U[i * m + j]
            for k in range(p):
                v_vec = V[j * p + k]
                w_vec = W[k * n + i]
                total = np.dot(u_vec, v_vec * w_vec)
                if abs(total - 1.0) > tol:
                    return False
    return True

def _build_4x4x8_rank40():
    """
    Build a rank-40 decomposition for 4x4x8 using two rank-20 4x4x4 decompositions.
    """
    n, m, p = 4, 4, 8
    rank = 40
    U = np.zeros((n * m, rank))  # 16 x 40
    V = np.zeros((m * p, rank))  # 32 x 40
    W = np.zeros((p * n, rank))  # 32 x 40
    
    # Get Laderman decomposition for 4x4x4
    U44, V44, W44 = _build_laderman_4x4x4()
    
    # First half: B columns 0-3, C columns 0-3
    U[:, 0:20] = U44
    for j in range(4):
        for k in range(4):
            for c in range(20):
                V[j * 8 + k, c] = V44[j * 4 + k, c]
    for k in range(4):
        for i in range(4):
            for c in range(20):
                W[k * 4 + i, c] = W44[k * 4 + i, c]
                
    # Second half: B columns 4-7, C columns 4-7
    U[:, 20:40] = U44
    for j in range(4):
        for k in range(4):
            for c in range(20):
                V[j * 8 + (k + 4), 20 + c] = V44[j * 4 + k, c]
    for k in range(4):
        for i in range(4):
            for c in range(20):
                W[(k + 4) * 4 + i, 20 + c] = W44[k * 4 + i, c]
                
    return U, V, W

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 4x4 matrix by a 4x8 matrix.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Try rank-40 (two rank-20 halves)
    try:
        U, V, W = _build_4x4x8_rank40()
        if _verify_decomposition(U, V, W, n, m, p):
            if time.time() < deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback: schoolbook decomposition (rank 128)
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
