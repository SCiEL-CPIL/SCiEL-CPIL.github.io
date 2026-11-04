import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def _build_2x2x4():
    """Rank-4 decomposition for 2x2x4 matrix multiplication."""
    U = np.zeros((4, 4))  # (n*m, rank) = (4, 4)
    V = np.zeros((8, 4))  # (m*p, rank) = (8, 4)
    W = np.zeros((8, 4))  # (p*n, rank) = (8, 4)
    
    # Term 1: A[0,0]*B[0,j] contributes to C[0,j] for j=0..3
    # Term 2: A[0,1]*B[1,j] contributes to C[0,j] for j=0..3
    # Term 3: A[1,0]*B[0,j] contributes to C[1,j] for j=0..3
    # Term 4: A[1,1]*B[1,j] contributes to C[1,j] for j=0..3
    
    # U index: i*2+k, V index: k*4+j, W index: j*2+i
    
    # Term 0: A[0,0] * B[0,j] -> C[0,j]
    U[0*2+0, 0] = 1  # i=0, k=0
    for j in range(4):
        V[0*4+j, 0] = 1  # k=0, j
        W[j*2+0, 0] = 1  # j, i=0
    
    # Term 1: A[0,1] * B[1,j] -> C[0,j]
    U[0*2+1, 1] = 1  # i=0, k=1
    for j in range(4):
        V[1*4+j, 1] = 1  # k=1, j
        W[j*2+0, 1] = 1  # j, i=0
    
    # Term 2: A[1,0] * B[0,j] -> C[1,j]
    U[1*2+0, 2] = 1  # i=1, k=0
    for j in range(4):
        V[0*4+j, 2] = 1  # k=0, j
        W[j*2+1, 2] = 1  # j, i=1
    
    # Term 3: A[1,1] * B[1,j] -> C[1,j]
    U[1*2+1, 3] = 1  # i=1, k=1
    for j in range(4):
        V[1*4+j, 3] = 1  # k=1, j
        W[j*2+1, 3] = 1  # j, i=1
    
    return U, V, W

def _build_2x2x1():
    """Rank-2 decomposition for 2x2x1 matrix multiplication."""
    U = np.zeros((4, 2))  # (n*m, rank) = (4, 2)
    V = np.zeros((2, 2))  # (m*p, rank) = (2, 2)
    W = np.zeros((2, 2))  # (p*n, rank) = (2, 2)
    
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    
    # Term 0: A[*,0] * B[0,0] -> C[*,0]
    U[0*2+0, 0] = 1  # i=0, k=0
    U[1*2+0, 0] = 1  # i=1, k=0
    V[0*1+0, 0] = 1  # k=0, j=0
    W[0*2+0, 0] = 1  # j=0, i=0
    W[0*2+1, 0] = 1  # j=0, i=1
    
    # Term 1: A[*,1] * B[1,0] -> C[*,0]
    U[0*2+1, 1] = 1  # i=0, k=1
    U[1*2+1, 1] = 1  # i=1, k=1
    V[1*1+0, 1] = 1  # k=1, j=0
    W[0*2+0, 1] = 1  # j=0, i=0
    W[0*2+1, 1] = 1  # j=0, i=1
    
    return U, V, W

def _build_4x4x9_block():
    """
    Build 4x4x9 decomposition using block structure.
    Split k=4 into two halves of 2.
    Split i=4 into two halves of 2.
    Split j=9 into 4+4+1.
    
    Total rank: 2 (k-blocks) * 2 (i-blocks) * (4+4+2) = 40
    """
    total_rank = 40
    U = np.zeros((16, total_rank))  # (n*m, rank) = (16, 40)
    V = np.zeros((36, total_rank))  # (m*p, rank) = (36, 40)
    W = np.zeros((36, total_rank))  # (p*n, rank) = (36, 40)
    
    U22x4, V22x4, W22x4 = _build_2x2x4()
    U22x1, V22x1, W22x1 = _build_2x2x1()
    
    # For each k-block (0 or 1), i-block (0 or 1), j-block (0, 1, or 2)
    # k-block 0: k=0,1; k-block 1: k=2,3
    # i-block 0: i=0,1; i-block 1: i=2,3
    # j-block 0: j=0,1,2,3; j-block 1: j=4,5,6,7; j-block 2: j=8
    
    r = 0
    for k_block in range(2):
        for i_block in range(2):
            # j-block 0: 2x2x4, rank 4
            for sub_r in range(4):
                # U: index (i*4+k), where i = 2*i_block + i2, k = 2*k_block + k2
                for i2 in range(2):
                    for k2 in range(2):
                        i = 2 * i_block + i2
                        k = 2 * k_block + k2
                        U[i * 4 + k, r] += U22x4[i2 * 2 + k2, sub_r]
                # V: index (k*9+j), where k = 2*k_block + k2, j = 0 + j2
                for k2 in range(2):
                    for j2 in range(4):
                        k = 2 * k_block + k2
                        j = j2
                        V[k * 9 + j, r] += V22x4[k2 * 4 + j2, sub_r]
                # W: index (j*4+i), where j = j2, i = 2*i_block + i2
                for j2 in range(4):
                    for i2 in range(2):
                        j = j2
                        i = 2 * i_block + i2
                        W[j * 4 + i, r] += W22x4[j2 * 2 + i2, sub_r]
                r += 1
            
            # j-block 1: 2x2x4, rank 4
            for sub_r in range(4):
                for i2 in range(2):
                    for k2 in range(2):
                        i = 2 * i_block + i2
                        k = 2 * k_block + k2
                        U[i * 4 + k, r] += U22x4[i2 * 2 + k2, sub_r]
                for k2 in range(2):
                    for j2 in range(4):
                        k = 2 * k_block + k2
                        j = 4 + j2
                        V[k * 9 + j, r] += V22x4[k2 * 4 + j2, sub_r]
                for j2 in range(4):
                    for i2 in range(2):
                        j = 4 + j2
                        i = 2 * i_block + i2
                        W[j * 4 + i, r] += W22x4[j2 * 2 + i2, sub_r]
                r += 1
            
            # j-block 2: 2x2x1, rank 2
            for sub_r in range(2):
                for i2 in range(2):
                    for k2 in range(2):
                        i = 2 * i_block + i2
                        k = 2 * k_block + k2
                        U[i * 4 + k, r] += U22x1[i2 * 2 + k2, sub_r]
                for k2 in range(2):
                    V[k * 9 + 8, r] += V22x1[k2 * 1 + 0, sub_r]
                for i2 in range(2):
                    i = 2 * i_block + i2
                    W[8 * 4 + i, r] += W22x1[0 * 2 + i2, sub_r]
                r += 1
    
    assert r == total_rank, f"Rank mismatch: {r} != {total_rank}"
    return U, V, W, total_rank

def construct():
    start_time = time.time()
    deadline = start_time + 140
    n, m, p = DIMENSIONS
    U, V, W, total_rank = _build_4x4x9_block()
    
    # Verify with a test
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 9)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((4, 9))
    for r in range(total_rank):
        a_vec = np.zeros(4)
        b_vec = np.zeros(9)
        for i in range(4):
            for k in range(4):
                a_vec[i] += U[i * 4 + k, r] * A_test[i, k]
        for j in range(9):
            for k in range(4):
                b_vec[j] += V[k * 9 + j, r] * B_test[k, j]
        for i in range(4):
            for j in range(9):
                C_computed[i, j] += W[j * 4 + i, r] * a_vec[i] * b_vec[j]
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to naive
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        idx = 0
        for i in range(n):
            for j in range(p):
                for k in range(m):
                    U[i * m + k, idx] = 1
                    V[k * p + j, idx] = 1
                    W[j * n + i, idx] = 1
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
