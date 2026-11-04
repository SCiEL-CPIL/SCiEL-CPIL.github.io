import time
import numpy as np

DIMENSIONS = [4, 4, 8]
START = time.time()
DEADLINE = START + 135.0

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

def _build_strassen_2x2x4():
    """
    Returns U, V, W for 2x2x4 decomposition (rank 7).
    This is the merged Strassen decomposition.
    U is 2x2 (4 entries), V is 2x4 (8 entries), W is 4x2 (8 entries).
    """
    U222, V222, W222 = _build_strassen_2x2x2()
    
    rank = 7
    U = np.zeros((4, rank), dtype=np.int8)
    V = np.zeros((8, rank), dtype=np.int8)
    W = np.zeros((8, rank), dtype=np.int8)
    
    for r in range(rank):
        U[:, r] = U222[:, r]
        
        # V is 2x4. Block 1 is cols 0-1, Block 2 is cols 2-3.
        # V_block1 = V222[:, r] (2x2)
        # V_block2 = V222[:, r] (2x2)
        # So V[:, r] is [V222[:, r]; V222[:, r]]?
        # No, V is flattened 2x4.
        # V[j*4+k, r].
        # j=0, k=0,1,2,3.
        # V[0, r] = V222[0, r] (B00)
        # V[1, r] = V222[1, r] (B01)
        # V[2, r] = V222[0, r] (B00 of 2nd block)
        # V[3, r] = V222[1, r] (B01 of 2nd block)
        # V[4, r] = V222[2, r] (B10)
        # V[5, r] = V222[3, r] (B11)
        # V[6, r] = V222[2, r] (B10 of 2nd block)
        # V[7, r] = V222[3, r] (B11 of 2nd block)
        
        V[0, r] = V222[0, r]
        V[1, r] = V222[1, r]
        V[2, r] = V222[0, r]
        V[3, r] = V222[1, r]
        V[4, r] = V222[2, r]
        V[5, r] = V222[3, r]
        V[6, r] = V222[2, r]
        V[7, r] = V222[3, r]
        
        # W is 4x2. Block 1 is rows 0-1, Block 2 is rows 2-3.
        # W[k*2+i, r].
        # k=0, i=0,1. W[0, r], W[1, r].
        # k=1, i=0,1. W[2, r], W[3, r].
        # k=2, i=0,1. W[4, r], W[5, r].
        # k=3, i=0,1. W[6, r], W[7, r].
        
        # W_block1 = W222[:, r] (2x2)
        # W_block2 = W222[:, r] (2x2)
        
        W[0, r] = W222[0, r]
        W[1, r] = W222[1, r]
        W[2, r] = W222[2, r]
        W[3, r] = W222[3, r]
        
        W[4, r] = W222[0, r]
        W[5, r] = W222[1, r]
        W[6, r] = W222[2, r]
        W[7, r] = W222[3, r]
        
    return U, V, W

def _build_rank49():
    """
    Builds a rank-49 decomposition for 4x4x8 using tensor product of
    2x2x4 (rank 7) and 2x2x2 (rank 7).
    """
    U224, V224, W224 = _build_strassen_2x2x4()
    U222, V222, W222 = _build_strassen_2x2x2()
    
    rank = 49
    U = np.zeros((16, rank), dtype=np.int16)
    V = np.zeros((32, rank), dtype=np.int16)
    W = np.zeros((32, rank), dtype=np.int16)
    
    for c0 in range(7):
        for c1 in range(7):
            c = c0 * 7 + c1
            # U is 4x4. Tensor product of U224 (2x2) and U222 (2x2).
            # U[i*4+j, c] = U224[i0*2+j0, c0] * U222[i1*2+j1, c1]
            for i in range(4):
                i0, i1 = i // 2, i % 2
                for j in range(4):
                    j0, j1 = j // 2, j % 2
                    U[i*4+j, c] = U224[i0*2+j0, c0] * U222[i1*2+j1, c1]
            
            # V is 4x8. Tensor product of V224 (2x4) and V222 (2x2).
            # V[j*8+k, c] = V224[j0*4+k0, c0] * V222[j1*2+k1, c1]
            for j in range(4):
                j0, j1 = j // 2, j % 2
                for k in range(8):
                    k0, k1 = k // 2, k % 2
                    V[j*8+k, c] = V224[j0*4+k0, c0] * V222[j1*2+k1, c1]
            
            # W is 8x4. Tensor product of W224 (4x2) and W222 (2x2).
            # W[k*4+i, c] = W224[k0*2+i0, c0] * W222[k1*2+i1, c1]
            for k in range(8):
                k0, k1 = k // 2, k % 2
                for i in range(4):
                    i0, i1 = i // 2, i % 2
                    W[k*4+i, c] = W224[k0*2+i0, c0] * W222[k1*2+i1, c1]
    
    # Verify
    n, m, p = 4, 4, 8
    result = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
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
            res = _build_rank49()
            if res is not None:
                return res
    except Exception:
        pass
    return _schoolbook()

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
