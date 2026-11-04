import time
import numpy as np

START = time.time()
DEADLINE = START + 130.0

def construct():
    """
    Returns a dictionary with 'U', 'V', 'W' matrices for the rank-98 decomposition
    of 4x4x8 matrix multiplication.
    """
    if time.time() > DEADLINE:
        # Fallback: schoolbook (rank 128) just in case, though construction is fast
        return _schoolbook(4, 4, 8)
    
    return _build_rank98()

def _build_strassen_2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen).
    U: (4, 7) for A (2x2)
    V: (4, 7) for B (2x2)
    W: (4, 7) for C (2x2)
    Indices: A_00, A_01, A_10, A_11 -> 0, 1, 2, 3
    """
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)

    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1

    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1

    # M4 = A11(B01-B00)
    U[3, 3] = 1
    V[0, 3] = -1; V[1, 3] = 1
    W[0, 3] = 1; W[1, 3] = 1

    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1

    return U, V, W

def _build_4x4x4_strassen():
    """
    Constructs rank-49 factors for 4x4x4 matrix multiplication.
    Uses the Kronecker product of the 2x2x2 Strassen decomposition.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank_444 = 49
    U4 = np.zeros((16, rank_444), dtype=np.int16)
    V4 = np.zeros((16, rank_444), dtype=np.int16)
    W4 = np.zeros((16, rank_444), dtype=np.int16)
    
    # The 4x4x4 tensor is the Kronecker product of two 2x2x2 tensors.
    # Index mapping for U (A part): row index is (i, j) in 4x4? No, U is (n*m, rank).
    # For 4x4 A, n=4, m=4. U is 16x49.
    # A_{ij} maps to row i*m + j.
    # In 2x2, A_{uv} maps to row u*2 + v.
    # For 4x4 formed by block 2x2s, A is partitioned into A00, A01, A10, A11 (each 2x2).
    # Row (i,j) where i,j in 0..3.
    # i = 2*i0 + i1, j = 2*j0 + j1.
    # In 2x2 decomposition, U2 has rows for A elements.
    # U4[i*4+j, c0*7+c1] should be U2[i0*2+j0, c0] * U2[i1*2+j1, c1]?
    # Let's check the indices.
    # Strassen recursion: M4 = M2x2 \otimes M2x2.
    # Rank = 7 * 7 = 49.
    # Column index in rank-49: c = c0 * 7 + c1.
    # U4 row (i,j) for A.
    # A is a 4x4 matrix. We view it as 2x2 blocks of 2x2 matrices.
    # A[i,j] corresponds to A_block[i//2, j//2][i%2, j%2].
    # The 2x2 decomposition handles the block indices and the internal indices separately.
    # Actually, the Kronecker product of decompositions works as:
    # U4[i,j; c0,c1] = U2[i//2, j//2; c0] * U2[i%2, j%2; c1] ? 
    # No, U2 rows are indexed by (u,v) for 2x2.
    # Row index in U2 is u*2+v.
    # So for A4[i,j], i=2*i0+i1, j=2*j0+j1.
    # Block index is (i0, j0). Element index is (i1, j1).
    # U4[i*4+j, c0*7+c1] = U2[i0*2+j0, c0] * U2[i1*2+j1, c1] is INCORRECT.
    # U2[u*2+v, c] extracts the u,v element.
    # We need to map the 4x4 index to the 2x2 "super" indices and 2x2 "sub" indices.
    # The decomposition is separable: Tensor_4x4x4 = Tensor_2x2x2 (blocks) \otimes Tensor_2x2x2 (elements).
    # So U4[row_4, col_4] = U2[row_2_super, col_2_super] * U2[row_2_sub, col_2_sub].
    # row_4 = i*4+j. i=2*i0+i1, j=2*j0+j1.
    # Super index pair: (i0, j0) -> row_2_super = i0*2 + j0.
    # Sub index pair: (i1, j1) -> row_2_sub = i1*2 + j1.
    
    for c0 in range(7):
        for c1 in range(7):
            c4 = c0 * 7 + c1
            for i in range(4):
                i0, i1 = i // 2, i % 2
                for j in range(4):
                    j0, j1 = j // 2, j % 2
                    r4 = i * 4 + j
                    r2s = i0 * 2 + j0
                    r2sub = i1 * 2 + j1
                    U4[r4, c4] = U2[r2s, c0] * U2[r2sub, c1]
                    
            for j in range(4):
                j0, j1 = j // 2, j % 2
                for k in range(4):
                    k0, k1 = k // 2, k % 2
                    r4 = j * 4 + k
                    r2s = j0 * 2 + k0
                    r2sub = j1 * 2 + k1
                    V4[r4, c4] = V2[r2s, c0] * V2[r2sub, c1]
                    
            for k in range(4):
                k0, k1 = k // 2, k % 2
                for i in range(4):
                    i0, i1 = i // 2, i % 2
                    r4 = k * 4 + i
                    r2s = k0 * 2 + i0
                    r2sub = k1 * 2 + i1
                    W4[r4, c4] = W2[r2s, c0] * W2[r2sub, c1]
                    
    return U4, V4, W4

def _build_rank98():
    """
    Constructs rank-98 factors for 4x4x8 matrix multiplication.
    U: (16, 98)
    V: (32, 98)
    W: (32, 98)
    """
    U4, V4, W4 = _build_4x4x4_strassen()
    
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    # Block 0: k in 0..3
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, :]
            W[k * 4 + i_offset(4), 0:49] = 0 # Placeholder, fix loop below
    
    # Correct V and W assignment for Block 0
    # V is (m*p, rank) = (4*8, 98). Row j*8+k.
    # For block 0 (k 0..3), V row j*8+k gets V4 row j*4+k.
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, :]
            
    # W is (p*n, rank) = (8*4, 98). Row k*4+i.
    # For block 0 (k 0..3), W row k*4+i gets W4 row k*4+i.
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W4[k * 4 + i, :]
            
    # Block 1: k in 4..7
    # U is same as block 0 because A is shared.
    U[:, 49:98] = U4
    
    # V for block 1: k_global = 4 + k_local.
    # V row j*8 + (4+k_local) gets V4 row j*4+k_local.
    for j in range(4):
        for k_local in range(4):
            k_global = 4 + k_local
            V[j * 8 + k_global, 49:98] = V4[j * 4 + k_local, :]
            
    # W for block 1: k_global = 4 + k_local.
    # W row k_global*4 + i gets W4 row k_local*4+i.
    for k_local in range(4):
        k_global = 4 + k_local
        for i in range(4):
            W[k_global * 4 + i, 49:98] = W4[k_local * 4 + i, :]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def i_offset(n):
    return n

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
