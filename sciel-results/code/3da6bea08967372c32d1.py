import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0
    
    dim_u = n * m  # 16
    dim_v = m * p  # 28
    dim_w = p * n  # 28
    
    # Build target tensor for verification
    T_target = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1
    
    # Part 1: 4x4x4 using Strassen (rank 49)
    U_4, V_4, W_4 = _strassen_4x4x4()
    
    if time.time() > deadline:
        return _schoolbook(n, m, p)
    
    # Part 2: 4x4x3 for k=4,5,6 using schoolbook (rank 48)
    U_3, V_3, W_3 = _schoolbook_4x4x3(n, m, p, k_start=4)
    
    if time.time() > deadline:
        return _schoolbook(n, m, p)
    
    # Combine
    U_all = np.hstack([U_4, U_3])
    V_all = np.hstack([V_4, V_3])
    W_all = np.hstack([W_4, W_3])
    
    rank = U_all.shape[1]
    
    # Verify
    T_decomp = np.einsum('ar,br,cr->abc', U_all, V_all, W_all)
    
    if not np.allclose(T_decomp, T_target, atol=1e-10):
        return _schoolbook(n, m, p)
    
    return {
        "U": U_all.tolist(),
        "V": V_all.tolist(),
        "W": W_all.tolist()
    }


def _strassen_2x2x2():
    """Strassen decomposition of 2x2x2 matrix multiplication.
    
    Returns U, V, W each of shape (4, 7) such that
    sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    where T[i*2+j, j*2+k, k*2+i] = 1
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (a11+a22)(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (a21+a22)b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a11(b12-b22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = a22(b21-b11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a11+a12)b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (a21-a11)(b11+b12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (a12-a22)(b21+b22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _strassen_4x4x4():
    """Build 4x4x4 decomposition via Kronecker of Strassen 2x2x2.
    
    Returns U (16,49), V (28,49), W (28,49) for the 4x4x4 sub-tensor
    where k ranges 0..3.
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    U_list = []
    V_list = []
    W_list = []
    
    for r in range(7):
        for s in range(7):
            # U: length 16 (4x4 matrix A)
            u = np.kron(U2[:, r], U2[:, s])
            
            # V: length 28 (4x7 matrix B), but we only use first 4 columns (k=0..3)
            # Kron gives length 16 with index j*4+k. Need to map to j*7+k.
            v_kron = np.kron(V2[:, r], V2[:, s])
            v_full = np.zeros(28)
            for j in range(4):
                for k in range(4):
                    v_full[j * 7 + k] = v_kron[j * 4 + k]
            
            # W: length 28 (7x4 matrix C), index k*4+i. For k=0..3, indices 0..15.
            w_kron = np.kron(W2[:, r], W2[:, s])
            w_full = np.zeros(28)
            w_full[:16] = w_kron
            
            U_list.append(u)
            V_list.append(v_full)
            W_list.append(w_full)
    
    U = np.array(U_list).T  # (16, 49)
    V = np.array(V_list).T  # (28, 49)
    W = np.array(W_list).T  # (28, 49)
    
    return U, V, W


def _schoolbook_4x4x3(n, m, p, k_start=4):
    """Schoolbook decomposition for columns k_start..p-1 of the 4x4xp multiplication.
    
    Returns U (16, rank), V (28, rank), W (28, rank)
    """
    U_list = []
    V_list = []
    W_list = []
    
    for i in range(n):
        for j in range(m):
            for k in range(k_start, p):
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(p * n)
                
                a = i * m + j
                b = j * p + k
                c = k * n + i
                
                u[a] = 1
                v[b] = 1
                w[c] = 1
                
                U_list.append(u)
                V_list.append(v)
                W_list.append(w)
    
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return U, V, W


def _schoolbook(n, m, p):
    """Return schoolbook decomposition of n x m times m x p."""
    rank = n * m * p
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
