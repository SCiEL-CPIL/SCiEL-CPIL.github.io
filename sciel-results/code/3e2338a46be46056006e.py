import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def _verify(U, V, W, tol=1e-8):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    """
    n, m, p = DIMENSIONS
    if not isinstance(U, np.ndarray):
        U = np.array(U)
    if not isinstance(V, np.ndarray):
        V = np.array(V)
    if not isinstance(W, np.ndarray):
        W = np.array(W)
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2():
    """
    Correct Strassen 2x2 decomposition.
    Returns U, V, W of shape (4, 7).
    U corresponds to A (2x2), V to B (2x2), W to C (2x2).
    Indices in U/V/W are row-major: 0=00, 1=01, 2=10, 3=11.
    W index: k*n+i. For 2x2, n=2.
    W[0,0] is C[0,0] (k=0, i=0)
    W[1,0] is C[1,0] (k=1, i=0) -> Wait.
    Trace convention: W[k*n+i, r].
    For C[0,0]: k=0, i=0 -> idx 0.
    For C[0,1]: k=0, i=1 -> idx 1.
    For C[1,0]: k=1, i=0 -> idx 2.
    For C[1,1]: k=1, i=1 -> idx 3.
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # P1 = (A00+A11)(B00+B11)
    # A: 0,3. B: 0,3. C: 0,3
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # P2 = (A10+A11)B00
    # A: 2,3. B: 0. C: 2,3 (C[1,0] and C[1,1]?)
    # C[1,0] is idx 2. C[1,1] is idx 3.
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # P3 = A00(B01-B11)
    # A: 0. B: 1,3. C: 1,3 (C[0,1] and C[1,1]?)
    # C[0,1] is idx 1. C[1,1] is idx 3.
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # P4 = A11(B10-B00)
    # A: 3. B: 2,0. C: 0,2 (C[0,0] and C[1,0]?)
    # C[0,0] is idx 0. C[1,0] is idx 2.
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # P5 = (A00+A01)B11
    # A: 0,1. B: 3. C: 0,1 (C[0,0] and C[0,1]?)
    # C[0,0] is idx 0. C[0,1] is idx 1.
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # P6 = (A10-A00)(B00+B01)
    # A: 2,0. B: 0,1. C: 3 (C[1,1]?)
    # C[1,1] is idx 3.
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # P7 = (A01-A11)(B10+B11)
    # A: 1,3. B: 2,3. C: 0 (C[0,0]?)
    # C[0,0] is idx 0.
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_schoolbook():
    n, m, p = DIMENSIONS
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

def _build_block_strassen_189():
    """
    Rank 189 decomposition: 3x3 blocks of 2x2 Strassen.
    Partition 6x6 into 3x3 blocks of size 2x2.
    Use Schoolbook for 3x3 block multiplication (27 products).
    Each 2x2 block product uses Strassen (Rank 7).
    Total Rank = 27 * 7 = 189.
    """
    n, m, p = 6, 6, 6
    nblocks = 3
    block_size = 2
    
    Us, Vs, Ws = _strassen_2x2()
    srank = 7
    
    total_rank = nblocks * nblocks * nblocks * srank
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col = 0
    for bi in range(nblocks):
        for bj in range(nblocks):
            for bk in range(nblocks):
                # This corresponds to the product A_block[bi,bj] * B_block[bj,bk]
                # contributing to C_block[bk,bi]
                for r in range(srank):
                    # Fill U for A_block[bi,bj]
                    # A is 6x6. Block bi,bj is rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
                    for di in range(block_size):
                        for dj in range(block_size):
                            i = 2*bi + di
                            j = 2*bj + dj
                            U[i*m+j, col] = Us[di*2+dj, r]
                            
                    # Fill V for B_block[bj,bk]
                    # B is 6x6. Block bj,bk is rows 2*bj..2*bj+1, cols 2*bk..2*bk+1
                    for dj in range(block_size):
                        for dk in range(block_size):
                            j = 2*bj + dj
                            k = 2*bk + dk
                            V[j*p+k, col] = Vs[dj*2+dk, r]
                            
                    # Fill W for C_block[bk,bi]
                    # C is 6x6. Block bk,bi is rows 2*bk..2*bk+1, cols 2*bi..2*bi+1
                    # W index is k*n+i.
                    for dk in range(block_size):
                        for di in range(block_size):
                            k = 2*bk + dk
                            i = 2*bi + di
                            W[k*n+i, col] = Ws[dk*2+di, r]
                    col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 100
    
    best_rank = 216
    best_result = None
    
    # 1. Try Block Strassen 189 (3x3 blocks of 2x2 Strassen)
    if time.time() < deadline - 10:
        try:
            U_bs, V_bs, W_bs = _build_block_strassen_189()
            if _verify(U_bs, V_bs, W_bs):
                rank = U_bs.shape[1]
                if rank < best_rank:
                    best_rank = rank
                    best_result = (U_bs, V_bs, W_bs)
        except Exception as e:
            pass
            
    # 2. Fallback to Schoolbook (Rank 216)
    if best_result is None:
        U_s, V_s, W_s = _build_schoolbook()
        best_result = (U_s, V_s, W_s)
        best_rank = 216
        
    U, V, W = best_result
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
