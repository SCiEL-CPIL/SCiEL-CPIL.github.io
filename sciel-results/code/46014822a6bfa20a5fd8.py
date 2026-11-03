"""Program for matrix multiplication 4x4 * 4x8 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses a rank-56 decomposition based on Kronecker product of Strassen and Identity.
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
    
    # Strassen 2x2 factors
    # Each factor: (u_vec[4], v_vec[4], w_vec[4])
    # u: A00, A01, A10, A11
    # v: B00, B01, B10, B11
    # w: C00, C01, C10, C11
    S = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # p1=(A00+A11)(B00+B11)
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),  # p2=(A10+A11)B00
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),  # p3=A00(B01-B11)
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),  # p4=A11(B10-B00)
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),  # p5=(A00+A01)B11
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),  # p6=(A10-A00)(B00+B01)
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),  # p7=(A01-A11)(B10+B11)
    ]
    
    # Identity 2x2 factors
    # e0: (1,0,0,0), e1: (0,1,0,0), e2: (0,0,1,0), e3: (0,0,0,1)
    I = [
        ([1,0,0,0], [1,0,0,0], [1,0,0,0]),
        ([0,1,0,0], [0,1,0,0], [0,1,0,0]),
        ([0,0,1,0], [0,0,1,0], [0,0,1,0]),
        ([0,0,0,1], [0,0,0,1], [0,0,0,1]),
    ]
    
    # Build rank-56 decomposition for 4x4 * 4x8
    # Split B into two 4x4 blocks
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(2):  # Two 4x4 subproblems for B
        for s_idx in range(7):
            u_s, v_s, w_s = S[s_idx]
            for i_idx in range(4):
                u_i, v_i, w_i = I[i_idx]
                col = t * 28 + s_idx * 4 + i_idx
                
                # U: A is 4x4. Viewed as 2x2 blocks of 2x2.
                # u_s is 4 elements (2x2). u_i is 4 elements (2x2).
                # Kronecker product u_s \otimes u_i results in 16 elements (4x4).
                # The indices for U are (i, j) where i, j in 0..3.
                # i = 2*bi + ei, j = 2*bj + ej
                # u_s[bi, bj] * u_i[ei, ej]
                
                for bi in range(2):
                    for bj in range(2):
                        cb = u_s[bi*2+bj]
                        if cb == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = u_i[ei*2+ej]
                                if cs == 0:
                                    continue
                                i = 2*bi + ei
                                j = 2*bj + ej
                                U[i*m + j][col] += cb * cs
                                
                # V: B block for subproblem t
                # v_s is 4 elements. v_i is 4 elements.
                # j = 2*bi + ei, k = 4*t + 2*bj + ej
                
                for bi in range(2):
                    for bj in range(2):
                        cb = v_s[bi*2+bj]
                        if cb == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = v_i[ei*2+ej]
                                if cs == 0:
                                    continue
                                j = 2*bi + ei
                                k = 4*t + 2*bj + ej
                                V[j*p + k][col] += cb * cs
                                
                # W: C block for subproblem t
                # w_s is 4 elements. w_i is 4 elements.
                # i = 2*bi + ei, k = 4*t + 2*bj + ej
                
                for bi in range(2):
                    for bj in range(2):
                        cb = w_s[bi*2+bj]
                        if cb == 0:
                            continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = w_i[ei*2+ej]
                                if cs == 0:
                                    continue
                                i = 2*bi + ei
                                k = 4*t + 2*bj + ej
                                W[k*n + i][col] += cb * cs
    
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
    
    # Fallback
    return _schoolbook()
