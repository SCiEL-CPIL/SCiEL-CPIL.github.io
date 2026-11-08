"""Search for a rank-minimal exact tensor decomposition of 2x5 x 5x5 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 5]
N, M, P = DIMENSIONS

# Deadline guard
_DEADLINE = time.time() + 140.0
_RNG = random.Random(12345)

def _tensor_indices():
    idx = []
    for i in range(N):
        for j in range(M):
            for k in range(P):
                idx.append((i * M + j, j * P + k, k * N + i))
    return idx

_TENSOR = _tensor_indices()

def _verify(U, V, W, tol=1e-6):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for (u, v, w) in _TENSOR:
        s = np.dot(U[u, :], V[v, :] * W[w, :])
        if abs(s - 1.0) > tol:
            return False
    tensor_set = set(_TENSOR)
    for u in range(N * M):
        for v in range(M * P):
            for w in range(P * N):
                if (u, v, w) in tensor_set:
                    continue
                s = np.dot(U[u, :], V[v, :] * W[w, :])
                if abs(s) > tol:
                    return False
    return True

def _get_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen terms
    # 1: (A00+A11)(B00+B11) -> C00, C11
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    
    # 2: (A10+A11)B00 -> C10
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1 # C10 is k=0, i=1 -> idx 2
    
    # 3: A00(B10-B11) -> C00, C01
    U[0,2]=1
    V[2,2]=1; V[3,2]=-1
    W[0,2]=1; W[1,2]=-1 # C00 (k=0,i=0) idx 0, C01 (k=1,i=0) idx 1
    
    # 4: A11(B01-B00) -> C11, C10
    U[3,3]=1
    V[1,3]=1; V[0,3]=-1
    W[3,3]=1; W[2,3]=-1 # C11 (k=1,i=1) idx 3, C10 (k=0,i=1) idx 2
    
    # 5: (A00+A01)B11 -> C01, C11
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[1,4]=1; W[3,4]=1
    
    # 6: (A10-A00)(B00+B10) -> C10, C11
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[2,5]=1
    W[2,5]=1; W[3,5]=1
    
    # 7: (A01-A11)(B01+B11) -> C00, C01
    U[1,6]=1; U[3,6]=-1
    V[1,6]=1; V[3,6]=1
    W[0,6]=1; W[1,6]=-1
    
    return U, V, W

def _build_2x4x4_rank18():
    """
    Returns U, V, W for 2x4x4 rank 18 decomposition.
    This is a known result.
    """
    # Placeholder: If I don't have the exact 18 terms, I will return None
    # to trigger fallback.
    return None

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    start = time.time()
    
    # Strategy 1: Try Rank 28 (2x4x4 Rank 18 + Boundary 10)
    # Since I don't have the 18 terms hardcoded reliably in this thought process,
    # I will implement Strategy 2: Rank 49 (Strassen on one 2x2x2 block)
    # and Strategy 3: Rank 50 (Schoolbook).
    
    # Actually, I can implement Rank 49 safely.
    # Partition 5 into 2 + 3.
    # Block 1: j in {0,1}, k in {0,1} -> 2x2x2 (Strassen, Rank 7)
    # Block 2: j in {0,1}, k in {2,3,4} -> 2x2x3 (Schoolbook, Rank 12)
    # Block 3: j in {2,3,4}, k in {0,1} -> 2x3x2 (Schoolbook, Rank 12)
    # Block 4: j in {2,3,4}, k in {2,3,4} -> 2x3x3 (Schoolbook, Rank 18)
    # Total: 7 + 12 + 12 + 18 = 49.
    
    total_rank = 49
    U = np.zeros((N * M, total_rank))
    V = np.zeros((M * P, total_rank))
    W = np.zeros((P * N, total_rank))
    
    col = 0
    
    # Block 1: Strassen 2x2x2
    U_s, V_s, W_s = _get_strassen_2x2x2()
    for c in range(7):
        for i in range(2):
            for j in range(2):
                u_val = U_s[i*2+j, c]
                if u_val != 0:
                    U[i*5+j, col] = u_val
        for j in range(2):
            for k in range(2):
                v_val = V_s[j*2+k, c]
                if v_val != 0:
                    V[j*5+k, col] = v_val
        for k in range(2):
            for i in range(2):
                w_val = W_s[k*2+i, c]
                if w_val != 0:
                    W[k*2+i, col] = w_val
        col += 1
        
    # Block 2: Schoolbook 2x2x3 (j in {0,1}, k in {2,3,4})
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                U[i*5+j, col] = 1.0
                V[j*5+k, col] = 1.0
                W[k*2+i, col] = 1.0
                col += 1
                
    # Block 3: Schoolbook 2x3x2 (j in {2,3,4}, k in {0,1})
    for i in range(2):
        for j in range(2, 5):
            for k in range(2):
                U[i*5+j, col] = 1.0
                V[j*5+k, col] = 1.0
                W[k*2+i, col] = 1.0
                col += 1
                
    # Block 4: Schoolbook 2x3x3 (j in {2,3,4}, k in {2,3,4})
    for i in range(2):
        for j in range(2, 5):
            for k in range(2, 5):
                U[i*5+j, col] = 1.0
                V[j*5+k, col] = 1.0
                W[k*2+i, col] = 1.0
                col += 1
                
    # Verify
    if _verify(U, V, W, 1e-6):
        def to_list(M):
            return [[float(x) for x in row] for row in M]
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    else:
        # Fallback to Schoolbook Rank 50
        U_fb, V_fb, W_fb = _build_schoolbook(N, M, P)
        assert _verify(U_fb, V_fb, W_fb, 1e-6), "Schoolbook failed"
        def to_list(M):
            return [[float(x) for x in row] for row in M]
        return {"U": to_list(U_fb), "V": to_list(V_fb), "W": to_list(W_fb)}
