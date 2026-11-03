import time
import numpy as np

_START = time.time()
_DEADLINE = 135.0

def _elapsed():
    return time.time() - _START

def _validate(U, V, W, n, m, p, tol=1e-8):
    """Check if the tensor decomposition is exact."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _get_strassen_terms():
    """
    Return the local U, V, W vectors for the 7 Strassen terms for 2x2x2 matrix multiplication.
    Local indices 0, 1.
    U is for A (2x2), V is for B (2x2), W is for C (2x2).
    U index: i*2 + j
    V index: j*2 + k
    W index: k*2 + i
    """
    terms = []
    # M1 = (A00 + A11)(B00 + B11)
    # C00 += 1, C11 += 1
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1] # W00=1 (k0,i0), W11=1 (k1,i1). W01, W10 are 0.
    terms.append((u1, v1, w1))
    
    # M2 = (A10 + A11)B00
    # C10 += 1, C11 -= 1
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, -1] # W10=1 (k0,i1), W11=-1 (k1,i1).
    terms.append((u2, v2, w2))
    
    # M3 = A00(B01 - B11)
    # C01 += 1, C11 += 1
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 1, 0, 1] # W01=1 (k1,i0), W11=1 (k1,i1).
    terms.append((u3, v3, w3))
    
    # M4 = A11(B10 - B00)
    # C00 += 1, C10 += 1
    u4 = [0, 0, 0, 1]
    v4 = [0, 0, 1, -1]
    w4 = [1, 0, 1, 0] # W00=1 (k0,i0), W10=1 (k0,i1).
    terms.append((u4, v4, w4))
    
    # M5 = (A00 + A01)B11
    # C00 -= 1, C01 += 1
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [-1, 1, 0, 0] # W00=-1 (k0,i0), W01=1 (k1,i0).
    terms.append((u5, v5, w5))
    
    # M6 = (A10 - A00)(B00 + B01)
    # C11 += 1
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 0, 1] # W11=1 (k1,i1).
    terms.append((u6, v6, w6))
    
    # M7 = (A01 - A11)(B10 + B11)
    # C00 += 1
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [1, 0, 0, 0] # W00=1 (k0,i0).
    terms.append((u7, v7, w7))
    
    return terms

def construct():
    global _START
    _START = time.time()
    
    n, m, p = 3, 5, 7
    rank = 99
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Define the 6 blocks
    # Each block is (I_indices, J_indices, K_indices)
    blocks = [
        ([0, 1], [0, 1], [0, 1]),
        ([0, 1], [2, 3], [0, 1]),
        ([0, 1], [0, 1], [2, 3]),
        ([0, 1], [2, 3], [2, 3]),
        ([1, 2], [0, 1], [4, 5]),
        ([1, 2], [2, 3], [4, 5]),
    ]
    
    # Track which (i,j,k) tuples are covered by Strassen blocks
    covered = set()
    for I, J, K in blocks:
        for i in I:
            for j in J:
                for k in K:
                    covered.add((i, j, k))
    
    strassen_terms = _get_strassen_terms()
    
    col = 0
    # Process Strassen blocks
    for I, J, K in blocks:
        for u_loc, v_loc, w_loc in strassen_terms:
            # Map local to global
            # u_loc index: i_local * 2 + j_local
            # v_loc index: j_local * 2 + k_local
            # w_loc index: k_local * 2 + i_local
            
            for idx_u, val_u in enumerate(u_loc):
                if val_u != 0:
                    i_loc = idx_u // 2
                    j_loc = idx_u % 2
                    i_g = I[i_loc]
                    j_g = J[j_loc]
                    U[i_g*m + j_g, col] += val_u
            
            for idx_v, val_v in enumerate(v_loc):
                if val_v != 0:
                    j_loc = idx_v // 2
                    k_loc = idx_v % 2
                    j_g = J[j_loc]
                    k_g = K[k_loc]
                    V[j_g*p + k_g, col] += val_v
            
            for idx_w, val_w in enumerate(w_loc):
                if val_w != 0:
                    k_loc = idx_w // 2
                    i_loc = idx_w % 2
                    k_g = K[k_loc]
                    i_g = I[i_loc]
                    W[k_g*n + i_g, col] += val_w
            
            col += 1
            
    # Process remaining terms (Schoolbook)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
                    
    # Convert to lists
    U_list = [[int(x) if x == int(x) else x for x in row] for row in U]
    V_list = [[int(x) if x == int(x) else x for x in row] for row in V]
    W_list = [[int(x) if x == int(x) else x for x in row] for row in W]
    
    # Validate
    if _validate(U_list, V_list, W_list, n, m, p, tol=1e-8):
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook if validation fails (should not happen)
        rank_school = n*m*p
        U_s = [[0]*rank_school for _ in range(n*m)]
        V_s = [[0]*rank_school for _ in range(m*p)]
        W_s = [[0]*rank_school for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][c] = 1
                    V_s[j*p+k][c] = 1
                    W_s[k*n+i][c] = 1
                    c += 1
        return {"U": U_s, "V": V_s, "W": W_s}
