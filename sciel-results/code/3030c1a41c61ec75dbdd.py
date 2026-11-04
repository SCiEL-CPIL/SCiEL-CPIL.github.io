import time
import numpy as np

DIMENSIONS = [4, 4, 8]
START = time.time()
DEADLINE = START + 140.0  # Safety margin under 150s kill

def _build_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition (rank 7).
    Indices:
    U[i*2+j, r]
    V[j*2+k, r]
    W[k*2+i, r]
    """
    U = np.zeros((4, 7), dtype=np.int8)
    V = np.zeros((4, 7), dtype=np.int8)
    W = np.zeros((4, 7), dtype=np.int8)

    # r=0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # r=1: M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # r=2: M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # r=3: M4 = A11(B10-B00)
    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[1, 3] = 1

    # r=4: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # r=5: M6 = (A10-A00)(B00+B01)
    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # r=6: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _build_rank98():
    """
    Builds a rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    Uses the tensor product of Strassen's 2x2x2 decomposition to create a
    rank-49 decomposition for 4x4x4, then duplicates it for the two 4x4 blocks
    of the 4x8 output.
    """
    if time.time() > DEADLINE:
        return None

    # Step 1: Get 2x2x2 Strassen tensors
    U2, V2, W2 = _build_strassen_2x2x2()
    
    # Step 2: Construct 4x4x4 tensors via tensor product (Kronecker product)
    # Rank is 7 * 7 = 49
    rank_444 = 49
    U4 = np.zeros((16, rank_444), dtype=np.int16)
    V4 = np.zeros((16, rank_444), dtype=np.int16)
    W4 = np.zeros((16, rank_444), dtype=np.int16)
    
    # Vectorized construction using broadcasting and reshaping
    # U4[i*4+j, c0*7+c1] = U2[i0*2+j0, c0] * U2[i1*2+j1, c1]
    # This is the Kronecker product of the columns of U2.
    # We can construct the full tensor product of the matrices.
    # Let U2 be reshaped to (2, 2, 7).
    U2_3d = U2.reshape(2, 2, 7)
    V2_3d = V2.reshape(2, 2, 7)
    W2_3d = W2.reshape(2, 2, 7)
    
    # The tensor product of two decompositions is the decomposition of the tensor product of the tensors.
    # The rank-1 tensors are U_i \otimes U_j, etc.
    # We can generate the columns by iterating over the 49 combinations.
    # For speed, we use a loop over the 49 ranks, but vectorize the row assignments.
    
    for c in range(49):
        c0 = c // 7
        c1 = c % 7
        
        # U4 column c is the Kronecker product of U2[:, c0] and U2[:, c1]
        # U2[:, c0] is a vector of length 4 (representing 2x2 matrix)
        # np.kron returns a vector of length 16.
        U4[:, c] = np.kron(U2[:, c0], U2[:, c1])
        V4[:, c] = np.kron(V2[:, c0], V2[:, c1])
        W4[:, c] = np.kron(W2[:, c0], W2[:, c1])
        
        if time.time() > DEADLINE:
            return None

    # Step 3: Construct 4x4x8 tensors
    total_rank = 98
    U = np.zeros((16, total_rank), dtype=np.int16)
    V = np.zeros((32, total_rank), dtype=np.int16)
    W = np.zeros((32, total_rank), dtype=np.int16)
    
    # Block 0: k=0..3
    U[:, 0:49] = U4
    # V rows are j*8+k. For k=0..3, V[j*8+k, 0:49] = V4[j*4+k, 0:49]
    # We can reshape V4 to (4, 4, 49) and then place it into V.
    # V4 is 16x49. Rows are j*4+k.
    # We need to map row index j*4+k to j*8+k.
    # This is equivalent to inserting zeros? No, just re-indexing.
    # V[j*8+k, c] = V4[j*4+k, c]
    # Let's use a loop for clarity and safety, as it's only 16 rows.
    for j in range(4):
        for k in range(4):
            V[j*8 + k, 0:49] = V4[j*4 + k, :]
            W[k*4 + j, 0:49] = W4[k*4 + j, :] # W index is k*n+i. n=4. W4 index is k*4+i.
            # Wait, W4[k*4+i, c] corresponds to output k, input i.
            # In the 4x4x8 case, W[k*4+i, c] for k=0..3.
            # So W[k*4+i, 0:49] = W4[k*4+i, 0:49].
            # My loop above was W[k*4+j...]. That is wrong.
            # It should be W[k*4+i, 0:49] = W4[k*4+i, 0:49].
            # So I can just copy the rows.
            
    # Correction:
    # W4 has rows k*4+i.
    # W (4x4x8) has rows k*4+i for k=0..3.
    # So W[0:16, 0:49] = W4.
    W[0:16, 0:49] = W4
    
    # Block 1: k=4..7
    U[:, 49:98] = U4
    for j in range(4):
        for k_local in range(4):
            k_global = 4 + k_local
            V[j*8 + k_global, 49:98] = V4[j*4 + k_local, :]
    
    # W rows for k=4..7 are k*4+i = (4+k_local)*4 + i = 16 + k_local*4 + i.
    # W4 rows are k_local*4 + i.
    # So W[16:32, 49:98] = W4.
    W[16:32, 49:98] = W4

    if time.time() > DEADLINE:
        return None

    # Step 4: Vectorized verification
    n, m, p = 4, 4, 8
    result = np.zeros((n, m, p), dtype=np.float64)
    for r in range(total_rank):
        u_col = U[:, r].reshape(n, m)
        v_col = V[:, r].reshape(m, p)
        w_col = W[:, r].reshape(p, n)
        result += np.einsum('ij,jk,ki->ijk', u_col, v_col, w_col)
    
    if not np.allclose(result, 1.0, atol=1e-9):
        return None
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _schoolbook(n=4, m=4, p=8):
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

def construct():
    try:
        if time.time() < DEADLINE:
            res = _build_rank98()
            if res is not None:
                return res
    except Exception:
        pass
    return _schoolbook()

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
