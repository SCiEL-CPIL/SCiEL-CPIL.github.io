"""Matrix multiplication 3x3x6 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 140.0

    n, m, p = DIMENSIONS  # 3, 3, 6

    def make_rank7_331():
        """
        Return U, V, W for rank-7 decomposition of 3x3x1 matrix-vector multiplication.
        U: (9, 7) indexed by (i,j) -> i*3+j
        V: (3, 7) indexed by j -> j
        W: (3, 7) indexed by i -> i
        """
        U = np.zeros((9, 7), dtype=float)
        V = np.zeros((3, 7), dtype=float)
        
        # Rank 7 decomposition for 3x3x1
        # Terms:
        # 0: (a00+a11+a22) * (b0+b1+b2)
        U[0, 0] = 1; U[4, 0] = 1; U[8, 0] = 1
        V[0, 0] = 1; V[1, 0] = 1; V[2, 0] = 1
        # 1: (a01+a12+a20) * (b1+b2+b0)
        U[1, 1] = 1; U[5, 1] = 1; U[6, 1] = 1
        V[1, 1] = 1; V[2, 1] = 1; V[0, 1] = 1
        # 2: (a02+a10+a21) * (b2+b0+b1)
        U[2, 2] = 1; U[3, 2] = 1; U[7, 2] = 1
        V[2, 2] = 1; V[0, 2] = 1; V[1, 2] = 1
        # 3: (a00+a10+a20) * (b0+b1+b2)
        U[0, 3] = 1; U[3, 3] = 1; U[6, 3] = 1
        V[0, 3] = 1; V[1, 3] = 1; V[2, 3] = 1
        # 4: (a01+a11+a21) * (b1+b2+b0)
        U[1, 4] = 1; U[4, 4] = 1; U[7, 4] = 1
        V[1, 4] = 1; V[2, 4] = 1; V[0, 4] = 1
        # 5: (a02+a12+a22) * (b2+b0+b1)
        U[2, 5] = 1; U[5, 5] = 1; U[8, 5] = 1
        V[2, 5] = 1; V[0, 5] = 1; V[1, 5] = 1
        # 6: (a00-a11+a22) * (b0-b1+b2)
        U[0, 6] = 1; U[4, 6] = -1; U[8, 6] = 1
        V[0, 6] = 1; V[1, 6] = -1; V[2, 6] = 1
        
        # Solve for W
        W = np.zeros((3, 7), dtype=float)
        for i in range(3):
            A_sys = np.zeros((3, 7), dtype=float)
            b_sys = np.ones(3, dtype=float)
            for j in range(3):
                for r in range(7):
                    A_sys[j, r] = U[i*3+j, r] * V[j, r]
            try:
                w, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b_sys, rcond=None)
                W[i, :] = w
            except:
                return None
        
        # Verify
        for i in range(3):
            for j in range(3):
                s = sum(U[i*3+j, r] * V[j, r] * W[i, r] for r in range(7))
                if abs(s - 1) > 1e-9:
                    return None
        
        return U, V, W

    r7 = make_rank7_331()
    
    if r7 is not None and time.time() - t0 < DEADLINE - 5:
        U1, V1, W1 = r7
        rank = 42
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        
        for k in range(p):
            offset = k * 7
            U[:, offset:offset+7] = U1
            # V indices: j*p + k
            for j in range(m):
                V[j*p+k, offset:offset+7] = V1[j, :]
            # W indices: k*n + i
            for i in range(n):
                W[k*n+i, offset:offset+7] = W1[i, :]
        
        # Vectorized verification
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        if np.allclose(T, 1.0, atol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank 54
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
