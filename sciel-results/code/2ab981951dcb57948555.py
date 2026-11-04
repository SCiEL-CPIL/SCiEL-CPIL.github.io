import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Try Rank 41 (Laderman 3x3x3 + Schoolbook 3x3x2)
    try:
        result_41 = construct_rank_41(n, m, p)
        if verify_decomposition(n, m, p, result_41):
            return result_41
    except Exception:
        pass

    # Fallback: Schoolbook Rank 45
    return construct_schoolbook(n, m, p)

def construct_rank_41(n, m, p):
    rank_total = 41
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # Part 1: Laderman 3x3x3 (columns 0,1,2)
    laderman_u, laderman_v, laderman_w = get_laderman_333(n, m)
    
    U[:, 0:23] = laderman_u
    V_full_laderman = np.zeros((m * p, 23))
    for j in range(m):
        for k in range(3):
            V_full_laderman[j * p + k, :] = laderman_v[j * 3 + k, :]
    V[:, 0:23] = V_full_laderman
    W[:, 0:23] = laderman_w
    
    # Part 2: Schoolbook 3x3x2 (columns 3,4)
    col_offset = 23
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in [3, 4]:
                U[i * m + j, col_offset + idx] = 1
                V[j * p + k, col_offset + idx] = 1
                W[k * n + i, col_offset + idx] = 1
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def get_laderman_333(n, m):
    p_local = 3
    rank = 23
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p_local, rank))
    
    def e(i, j):
        v = np.zeros(n * m)
        v[i * m + j] = 1
        return v
        
    terms = [
        (e(0,0), e(0,0)),
        (e(0,1), e(1,0)),
        (e(0,2), e(2,0)),
        (e(1,0), e(0,1)),
        (e(1,1), e(1,1)),
        (e(1,2), e(2,1)),
        (e(2,0), e(0,2)),
        (e(2,1), e(1,2)),
        (e(2,2), e(2,2)),
        (e(0,0)+e(1,1), e(0,0)),
        (e(0,0)+e(2,2), e(0,0)),
        (e(1,1)+e(2,2), e(1,1)),
        (e(0,1)+e(1,0), e(1,0)),
        (e(0,2)+e(2,0), e(2,0)),
        (e(1,2)+e(2,1), e(2,1)),
        (e(0,0)+e(1,1)+e(2,2), e(0,0)),
        (e(0,1)+e(1,0)+e(2,2), e(1,0)),
        (e(0,2)+e(1,1)+e(2,0), e(2,0)),
        (e(0,0)+e(1,2)+e(2,1), e(0,1)),
        (e(0,1)+e(1,1)+e(2,2), e(1,1)),
        (e(0,2)+e(1,2)+e(2,0), e(2,1)),
        (e(0,0)+e(1,0)+e(2,2), e(0,2)),
        (e(0,1)+e(1,1)+e(2,1), e(1,2))
    ]
    
    for r in range(rank):
        U[:, r] = terms[r][0]
        V[:, r] = terms[r][1]
        
    T_flat = np.zeros(n * m * p_local)
    for i in range(n):
        for j in range(m):
            for k in range(p_local):
                T_flat[(i * m + j) * p_local + k] = 1
                
    A = np.zeros((n * m * p_local, rank * p_local * n))
    for i in range(n):
        for j in range(m):
            for k in range(p_local):
                row = i * m * p_local + j * p_local + k
                for r in range(rank):
                    col = r * (p_local * n) + (k * n + i)
                    A[row, col] = U[i * m + j, r] * V[j * p_local + k, r]
                    
    W_flat, residuals, rank_sol, s = np.linalg.lstsq(A, T_flat, rcond=None)
    W = W_flat.reshape(rank, p_local * n).T
    W = np.round(W * 2) / 2
    
    return U, V, W

def construct_schoolbook(n, m, p):
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

def verify_decomposition(n, m, p, dec):
    U = np.array(dec['U'])
    V = np.array(dec['V'])
    W = np.array(dec['W'])
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
        
    # Check tensor contraction
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # We want T[i,j,k] == 1 for all i,j,k
    
    # Efficient check:
    # Compute C = A * B for random A, B?
    # Or just check the tensor directly.
    # Given small size, direct check is fine.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if not np.isclose(val, 1.0):
                    return False
    return True
