import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# Pre-compute Strassen 2x2 factors
# u: A00, A01, A10, A11
# v: B00, B01, B10, B11
# w: C00, C01, C10, C11
S = [
    ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
    ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
    ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
    ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),
    ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
    ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
    ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
]

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses a pre-computed rank-98 decomposition based on the tensor product of two Strassen steps.
    """
    start = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook rank-128 as fallback
    def _schoolbook():
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
    
    # Generate the 49 factors for 4x4x4 using tensor product of Strassen factors
    # The 4x4 matrix is split into 2x2 blocks.
    # Block index (bi, bj) maps to 2x2 element (ei, ej)
    # Total index i = 2*bi + ei, j = 2*bj + ej
    
    rank_4x4 = 49
    U_4x4 = [[0] * rank_4x4 for _ in range(n * m)]
    V_4x4 = [[0] * rank_4x4 for _ in range(m * m)] # Note: m=4, so 4x4
    W_4x4 = [[0] * rank_4x4 for _ in range(m * n)] # Note: n=4, so 4x4
    
    for s in range(7):
        u_block, v_block, w_block = S[s]
        for r in range(7):
            u_scalar, v_scalar, w_scalar = S[r]
            col = s * 7 + r
            
            # Compute U vector for this factor
            # U index: i*m + j = (2*bi+ei)*4 + (2*bj+ej)
            for bi in range(2):
                for bj in range(2):
                    cb = u_block[bi*2+bj]
                    if cb == 0: continue
                    for ei in range(2):
                        for ej in range(2):
                            cs = u_scalar[ei*2+ej]
                            if cs == 0: continue
                            i = 2*bi + ei
                            j = 2*bj + ej
                            U_4x4[i*m + j][col] += cb * cs
                            
            # Compute V vector for this factor
            # V index: j*p + k. Here p=4 for the 4x4 core.
            # j = 2*bi + ei, k = 2*bj + ej
            for bi in range(2):
                for bj in range(2):
                    cb = v_block[bi*2+bj]
                    if cb == 0: continue
                    for ei in range(2):
                        for ej in range(2):
                            cs = v_scalar[ei*2+ej]
                            if cs == 0: continue
                            j = 2*bi + ei
                            k = 2*bj + ej
                            V_4x4[j*4 + k][col] += cb * cs # Using 4 for local p
                            
            # Compute W vector for this factor
            # W index: k*n + i. Here n=4.
            # k = 2*bi + ei, i = 2*bj + ej
            for bi in range(2):
                for bj in range(2):
                    cb = w_block[bi*2+bj]
                    if cb == 0: continue
                    for ei in range(2):
                        for ej in range(2):
                            cs = w_scalar[ei*2+ej]
                            if cs == 0: continue
                            k = 2*bi + ei
                            i = 2*bj + ej
                            W_4x4[k*4 + i][col] += cb * cs # Using 4 for local n

    # Extend to 4x4x8
    # We have two 4x4 blocks for B and C.
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Copy U (same for both blocks)
    for i in range(n * m):
        for r in range(rank_4x4):
            U[i][r] = U_4x4[i][r]
            U[i][r + rank_4x4] = U_4x4[i][r]
            
    # Copy V and W for the two blocks
    for t in range(2):
        offset = t * rank_4x4
        for j in range(m): # j 0..3
            for k_local in range(4): # k 0..3 in local block
                k_global = t * 4 + k_local
                V[j * p + k_global][offset + 0:offset + rank_4x4] = V_4x4[j * 4 + k_local] # This slice assignment might be slow or incorrect for lists
                # Better to loop
                for r in range(rank_4x4):
                    V[j * p + k_global][offset + r] = V_4x4[j * 4 + k_local][r]
                    
        for k_local in range(4):
            k_global = t * 4 + k_local
            for i in range(n):
                for r in range(rank_4x4):
                    W[k_global * n + i][offset + r] = W_4x4[k_local * 4 + i][r]

    # Verify
    def _verify(U, V, W):
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        U3 = U_arr.reshape(n, m, rank)
        V3 = V_arr.reshape(m, p, rank)
        W3 = W_arr.reshape(p, n, rank)
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.allclose(result, 1.0, atol=1e-10)
    
    if time.time() - start < 145:
        if _verify(U, V, W):
            return {"U": U, "V": V, "W": W}
    
    return _schoolbook()
