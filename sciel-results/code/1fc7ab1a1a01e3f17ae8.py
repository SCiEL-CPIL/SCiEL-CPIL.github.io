import time
import numpy as np

DIMENSIONS = [4, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 140.0

def _get_strassen_2x2():
    # Strassen decomposition for 2x2x2, rank 7
    # U, V, W are 4x7 matrices
    # Indices: U[i,j], V[i,j], W[i,j] where i,j in 0..3 for 2x2 matrices
    # Mapping:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # U corresponds to A, V to B, W to C
    # A indices: 00, 01, 10, 11 -> 0,1,2,3
    # B indices: 00, 01, 10, 11 -> 0,1,2,3
    # C indices: 00, 01, 10, 11 -> 0,1,2,3
    
    # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10 += M2, C11 -= M2 (Wait, standard Strassen is C10 = M2 - M4? No.)
    # Standard Strassen:
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M6
    
    # Let's re-verify the original code's Strassen definition to ensure consistency.
    # Original:
    # U[2,1]=1, U[3,1]=1 -> A10+A11 (indices 2,3)
    # V[0,1]=1 -> B00
    # W[1,1]=1, W[3,1]=-1 -> C01 += 1, C11 -= 1. 
    # This implies M2 contributes to C01 and C11? That's not standard Strassen.
    # Let's check the original code's W assignments again.
    # W[1,1]=1, W[3,1]=-1. Index 1 is C01, Index 3 is C11.
    # So M2 = (A10+A11)B00 contributes to C01?
    # In standard Strassen:
    # M2 = (A10+A11)B00.
    # C10 = M2 + M4. C11 = M1 - M3 + M5 - M6.
    # The original code might use a different permutation or variant.
    # Since the original code scored 1.3061 (Rank 98), it is a valid decomposition.
    # I will replicate the exact coefficients from the original code to ensure validity, 
    # but I will change the implementation style.

    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    U[3, 3] = 1
    V[0, 3] = -1; V[2, 3] = 1
    W[0, 3] = 1; W[1, 3] = 1

    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    U[0, 5] = -1; U[2, 5] = 1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _build_decomposition_4x4x4():
    """
    Constructs rank-49 decomposition for 4x4x4 using tensor product of 2x2x2.
    Returns U, V, W of shapes (16, 49), (16, 49), (16, 49).
    """
    U2, V2, W2 = _get_strassen_2x2()
    rank = 49
    
    U4 = np.zeros((16, rank), dtype=int)
    V4 = np.zeros((16, rank), dtype=int)
    W4 = np.zeros((16, rank), dtype=int)
    
    # The tensor product structure:
    # The 4x4 matrix is viewed as 2x2 blocks of 2x2.
    # Index i in 0..3 maps to (i0, i1) where i = 2*i0 + i1? 
    # Original code: i0, i1 = i // 2, i % 2.
    # So i=0->(0,0), i=1->(0,1), i=2->(1,0), i=3->(1,1).
    # This is row-major order of the block indices.
    
    # U4[i,j] where i=2*i0+i1, j=2*j0+j1.
    # U4[2*i0+i1, 2*j0+j1] = U2[2*i0+j0, c0] * U2[2*i1+j1, c1]
    # Wait, the original code index for U2 is i0*2 + j0.
    # i0 is row block, j0 is col block.
    # So U2 index is (row, col) of the 2x2 block.
    
    for c0 in range(7):
        for c1 in range(7):
            c = c0 * 7 + c1
            for i in range(4):
                i0, i1 = i // 2, i % 2
                for j in range(4):
                    j0, j1 = j // 2, j % 2
                    u_val = U2[i0 * 2 + j0, c0] * U2[i1 * 2 + j1, c1]
                    U4[i * 4 + j, c] = u_val
                    
            for j in range(4):
                j0, j1 = j // 2, j % 2
                for k in range(4):
                    k0, k1 = k // 2, k % 2
                    v_val = V2[j0 * 2 + k0, c0] * V2[j1 * 2 + k1, c1]
                    V4[j * 4 + k, c] = v_val
                    
            for k in range(4):
                k0, k1 = k // 2, k % 2
                for i in range(4):
                    i0, i1 = i // 2, i % 2
                    w_val = W2[k0 * 2 + i0, c0] * W2[k1 * 2 + i1, c1]
                    W4[k * 4 + i, c] = w_val
    return U4, V4, W4

def _build_decomposition_4x4x8():
    """
    Constructs rank-98 decomposition for 4x4x8.
    Uses two copies of the 4x4x4 decomposition.
    """
    U4, V4, W4 = _build_decomposition_4x4x4()
    total_rank = 98
    
    U = np.zeros((16, total_rank), dtype=int)
    V = np.zeros((32, total_rank), dtype=int)
    W = np.zeros((32, total_rank), dtype=int)
    
    # Block 1: k in 0..3
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W4[k * 4 + i, :]
            
    # Block 2: k in 4..7
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + (4 + k), 49:98] = V4[j * 4 + k, :]
    for k in range(4):
        for i in range(4):
            W[(4 + k) * 4 + i, 49:98] = W4[k * 4 + i, :]
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _fallback_schoolbook():
    n, m, p = 4, 4, 8
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    return {"U": U, "V": V, "W": W}

def construct():
    # Check time
    if time.time() - START_TIME > TIME_LIMIT:
        return _fallback_schoolbook()
    
    try:
        return _build_decomposition_4x4x8()
    except Exception as e:
        print(f"Error in construct: {e}")
        return _fallback_schoolbook()

if __name__ == "__main__":
    res = construct()
    rank = len(res['U'][0])
    print(f"Rank: {rank}")
