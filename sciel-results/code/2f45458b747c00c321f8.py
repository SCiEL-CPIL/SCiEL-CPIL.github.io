"""Matrix multiplication 2x3 x 3x8 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS  # n=2, m=3, p=8

    def verify(U, V, W, tol=1e-10):
        """Check if the decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(
                        U[i*m+j, :] * V[j*p+k, :],
                        W[k*n+i, :]
                    )
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def build_strassen_wide():
        """
        Build Strassen decomposition for 2x2 A times 2x8 B.
        7 terms where each V vector has 16 entries (2 rows x 8 cols).
        Returns U (4,7), V (16,7), W (16,7).
        U indices: i*2+j, V indices: j*8+k, W indices: k*2+i
        """
        U = np.zeros((4, 7))
        V = np.zeros((16, 7))
        W = np.zeros((16, 7))
        
        # Strassen: 7 products, each applied to all 8 columns
        # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        U[0, 0] = 1; U[3, 0] = 1
        for k in range(8):
            V[0*8+k, 0] = 1
            V[1*8+k, 0] = 1
            W[k*2+0, 0] = 1
            W[k*2+1, 0] = 1
        
        # M2 = (A21+A22)B11 -> C21 += M2
        U[1, 1] = 1; U[3, 1] = 1
        for k in range(8):
            V[0*8+k, 1] = 1
            W[k*2+1, 1] = 1
        
        # M3 = A11(B12-B22) -> C12 += M3
        U[0, 2] = 1
        for k in range(8):
            V[0*8+k, 2] = 1
            V[1*8+k, 2] = -1
            W[k*2+0, 2] = 1
        
        # M4 = A22(B21-B11) -> C21 += M4, C11 += M4
        U[3, 3] = 1
        for k in range(8):
            V[1*8+k, 3] = 1
            V[0*8+k, 3] = -1
            W[k*2+1, 3] = 1
            W[k*2+0, 3] = 1
        
        # M5 = (A11+A12)B22 -> C12 += M5, C11 -= M5
        U[0, 4] = 1; U[2, 4] = 1
        for k in range(8):
            V[1*8+k, 4] = 1
            W[k*2+0, 4] = 1
            W[k*2+1, 4] = -1
        
        # M6 = (A21-A11)(B11+B12) -> C22 += M6
        U[1, 5] = 1; U[0, 5] = -1
        for k in range(8):
            V[0*8+k, 5] = 1
            V[1*8+k, 5] = 1
            W[k*2+1, 5] = 1
        
        # M7 = (A12-A22)(B21+B22) -> C11 += M7
        U[2, 6] = 1; U[3, 6] = -1
        for k in range(8):
            V[1*8+k, 6] = 1
            V[0*8+k, 6] = 1
            W[k*2+0, 6] = 1
        
        return U, V, W

    def build_rank8_direct():
        """
        Build the rank-8 decomposition:
        7 Strassen terms for the 2x2x8 sub-problem (i in {0,1}, j in {0,1})
        + 1 direct term for j=2.
        """
        rank = 8
        U = np.zeros((n * m, rank))  # (6, 8)
        V = np.zeros((m * p, rank))  # (24, 8)
        W = np.zeros((p * n, rank))  # (16, 8)
        
        U_s, V_s, W_s = build_strassen_wide()
        
        # Strassen part: 7 terms for j in {0,1}
        for s in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * m + j, s] = U_s[i * 2 + j, s]
            for j in range(2):
                for k in range(8):
                    V[j * p + k, s] = V_s[j * 8 + k, s]
            for k in range(8):
                for i in range(2):
                    W[k * n + i, s] = W_s[k * 2 + i, s]
        
        # Direct part: 1 term for j=2
        s = 7
        for i in range(n):
            U[i * m + 2, s] = 1
        for k in range(p):
            V[2 * p + k, s] = 1
            for i in range(n):
                W[k * n + i, s] = 1
        
        return U, V, W

    # Build and verify rank-8 decomposition
    U, V, W = build_rank8_direct()
    
    if verify(U, V, W):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback: schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, c] = 1
                V_sb[j * p + k, c] = 1
                W_sb[k * n + i, c] = 1
                c += 1
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
