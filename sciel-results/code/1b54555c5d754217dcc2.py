import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def _build_strassen_2x2():
    """Strassen 2x2x2 decomposition, rank 7."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1 = (A11+A22)(B11+B22)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (A21+A22)B11
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    # M3 = A11(B21-B22)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    # M4 = A22(B12-B11)
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[1,3]=1
    # M5 = (A11+A12)B22
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # M6 = (A21-A11)(B11+B12)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7 = (A12-A22)(B21+B22)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _build_4x4x4_strassen():
    """4x4x4 Strassen decomposition, rank 49."""
    U2, V2, W2 = _build_strassen_2x2()
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    big_mults = [
        {'A': [((0,0),1),((1,1),1)], 'B': [((0,0),1),((1,1),1)], 'C': [((0,0),1),((1,1),1)]},
        {'A': [((1,0),1),((1,1),1)], 'B': [((0,0),1)], 'C': [((0,1),1),((1,1),-1)]},
        {'A': [((0,0),1)], 'B': [((0,1),1),((1,1),-1)], 'C': [((1,0),1),((1,1),1)]},
        {'A': [((1,1),1)], 'B': [((1,0),1),((0,0),-1)], 'C': [((0,0),1),((0,1),1)]},
        {'A': [((0,0),1),((0,1),1)], 'B': [((1,1),1)], 'C': [((0,0),-1),((1,0),1)]},
        {'A': [((1,0),1),((0,0),-1)], 'B': [((0,0),1),((0,1),1)], 'C': [((1,1),1)]},
        {'A': [((0,1),1),((1,1),-1)], 'B': [((1,0),1),((1,1),1)], 'C': [((0,0),1)]},
    ]
    for big_idx, big in enumerate(big_mults):
        base_r = big_idx * 7
        for small_idx in range(7):
            r = base_r + small_idx
            for (bi,bk), sA in big['A']:
                for i2 in range(2):
                    for k2 in range(2):
                        i=2*bi+i2; k=2*bk+k2
                        U[i*4+k, r] += sA * U2[i2*2+k2, small_idx]
            for (bk,bj), sB in big['B']:
                for k2 in range(2):
                    for j2 in range(2):
                        k=2*bk+k2; j=2*bj+j2
                        V[k*4+j, r] += sB * V2[k2*2+j2, small_idx]
            for (bj,bi), sC in big['C']:
                for j2 in range(2):
                    for i2 in range(2):
                        j=2*bj+j2; i=2*bi+i2
                        W[j*4+i, r] += sC * W2[j2*2+i2, small_idx]
    return U, V, W

def _build_4x4x9():
    """
    Build 4x4x9 decomposition:
    - Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
    - C1 = A*B1 uses Strassen 4x4x4 (rank 49)
    - C2 = A*B2 uses Strassen 4x4x4 (rank 49)
    - C3 = A*B3 uses schoolbook 4x4x1 (rank 16)
    Total rank: 114
    """
    U44, V44, W44 = _build_4x4x4_strassen()
    total_rank = 114
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    # Block 1: columns 0-3 of B
    U[:, 0:49] = U44
    for k in range(4):
        for j in range(4):
            V[k*9+j, 0:49] = V44[k*4+j, :]
    for j in range(4):
        for i in range(4):
            W[j*4+i, 0:49] = W44[j*4+i, :]
    
    # Block 2: columns 4-7 of B
    U[:, 49:98] = U44
    for k in range(4):
        for j in range(4):
            V[k*9+(j+4), 49:98] = V44[k*4+j, :]
    for j in range(4):
        for i in range(4):
            W[(j+4)*4+i, 49:98] = W44[j*4+i, :]
    
    # Block 3: column 8 of B (4x4x1 schoolbook)
    for i in range(4):
        for k in range(4):
            r = 98 + i*4 + k
            U[i*4+k, r] = 1
            V[k*9+8, r] = 1
            W[8*4+i, r] = 1
    
    return U, V, W, total_rank

def _verify_fast(U, V, W, rank, n, m, p, num_trials=3):
    """Vectorized verification using matrix operations."""
    # Reshape factor matrices to tensor form for einsum
    # U: (n*m, rank) -> (n, m, rank)
    U_3d = U.reshape(n, m, rank)
    # V: (m*p, rank) -> (m, p, rank)
    V_3d = V.reshape(m, p, rank)
    # W: (p*n, rank) -> (p, n, rank)
    W_3d = W.reshape(p, n, rank)
    
    for _ in range(num_trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_exp = A @ B
        
        # a_all[i, r] = sum_k U[i, k, r] * A[i, k]
        a_all = np.einsum('ikr,ik->ir', U_3d, A)  # (n, rank)
        # b_all[j, r] = sum_k V[k, j, r] * B[k, j]
        b_all = np.einsum('kjr,kj->jr', V_3d, B)  # (p, rank)
        # C_comp[i, j] = sum_r W[j, i, r] * a_all[i, r] * b_all[j, r]
        C_comp = np.einsum('jir,ir,jr->ij', W_3d, a_all, b_all)
        
        if not np.allclose(C_comp, C_exp, atol=1e-10):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140
    n, m, p = DIMENSIONS
    U, V, W, total_rank = _build_4x4x9()
    
    if time.time() > deadline - 10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if _verify_fast(U, V, W, total_rank, n, m, p, num_trials=3):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    idx = 0
    for i in range(n):
        for j in range(p):
            for k in range(m):
                U[i*m+k, idx] = 1
                V[k*p+j, idx] = 1
                W[j*n+i, idx] = 1
                idx += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
