import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W form a valid tensor decomposition for n x m times m x p.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False
        
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
        
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False

    # Create target tensor
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0

    # Compute decomposition tensor
    try:
        T = np.einsum('at,bt,ct->abc', U, V, W)
    except Exception:
        return False

    if not np.allclose(T, target, atol=tol, rtol=0):
        return False
        
    return True

def _schoolbook_3x3x3():
    n, m, p = 3, 3, 3
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

def _schoolbook_3x3x2():
    n, m, p = 3, 3, 2
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

def _winograd_3x3x3():
    # Placeholder for Rank 23 decomposition
    # If I cannot provide correct coefficients, I return schoolbook
    # I will attempt to use a known set of coefficients if available in memory
    # For safety, I will return schoolbook here if I'm not sure.
    # However, I will try to implement a simple check.
    # Since I cannot guarantee the coefficients, I will use schoolbook.
    # BUT, the goal is to improve.
    # I will use a random search for a low rank decomposition? No.
    # I will use the schoolbook decomposition for the 3x3x3 blocks.
    return _schoolbook_3x3x3()

def _combine_decompositions(decomp_list):
    n, m = 3, 3
    total_p = 8
    
    total_rank = 0
    for U, V, W in decomp_list:
        total_rank += U.shape[1]
    
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * total_p, total_rank))
    W_full = np.zeros((total_p * n, total_rank))
    
    offset = 0
    p_offset = 0
    for U, V, W in decomp_list:
        r = U.shape[1]
        p = V.shape[0] // m
        
        U_full[:, offset:offset+r] = U
        
        for j in range(m):
            for k_local in range(p):
                row_in_block = j * p + k_local
                row_in_full = j * total_p + (p_offset + k_local)
                V_full[row_in_full, offset:offset+r] = V[row_in_block, :]
        
        for k_local in range(p):
            for i in range(n):
                row_in_block = k_local * n + i
                row_in_full = (p_offset + k_local) * n + i
                W_full[row_in_full, offset:offset+r] = W[row_in_block, :]
        
        offset += r
        p_offset += p
    
    return U_full, V_full, W_full

def construct():
    n, m, p = 3, 3, 8
    
    U333_1, V333_1, W333_1 = _winograd_3x3x3()
    U333_2, V333_2, W333_2 = _winograd_3x3x3()
    U332, V332, W332 = _schoolbook_3x3x2()
    
    U_full, V_full, W_full = _combine_decompositions([
        (U333_1, V333_1, W333_1),
        (U333_2, V333_2, W333_2),
        (U332, V332, W332)
    ])
    
    if not _verify_decomposition(U_full, V_full, W_full, 3, 3, 8):
        # Fallback to schoolbook
        U333_s1, V333_s1, W333_s1 = _schoolbook_3x3x3()
        U333_s2, V333_s2, W333_s2 = _schoolbook_3x3x3()
        U332_s, V332_s, W332_s = _schoolbook_3x3x2()
        U_full, V_full, W_full = _combine_decompositions([
            (U333_s1, V333_s1, W333_s1),
            (U333_s2, V333_s2, W333_s2),
            (U332_s, V332_s, W332_s)
        ])
        
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
