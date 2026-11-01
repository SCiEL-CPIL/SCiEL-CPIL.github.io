import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    def verify_decomposition(U, V, W):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c] for c in range(R))
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def build_grouped_strassen():
        """7 Strassen blocks (14 k's) + 1 schoolbook column = rank 53."""
        R = 7 * 7 + 4
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        def strassen_block(k0, k1, offset):
            # P1 = (A00+A11)(B00+B11)
            U[0, offset] = 1; U[3, offset] = 1
            V[0*p+k0, offset] = 1; V[1*p+k1, offset] = 1
            W[k0*n+0, offset] = 1; W[k1*n+0, offset] = 1
            W[k0*n+1, offset] = 1; W[k1*n+1, offset] = 1
            # P2 = (A10+A11)B00
            U[2, offset+1] = 1; U[3, offset+1] = 1
            V[0*p+k0, offset+1] = 1
            W[k1*n+0, offset+1] = 1; W[k1*n+1, offset+1] = 1
            # P3 = A00(B10-B11)
            U[0, offset+2] = 1
            V[1*p+k0, offset+2] = 1; V[1*p+k1, offset+2] = -1
            W[k0*n+1, offset+2] = -1; W[k1*n+1, offset+2] = -1
            # P4 = A11(B01-B00)
            U[3, offset+3] = 1
            V[0*p+k1, offset+3] = 1; V[0*p+k0, offset+3] = -1
            W[k0*n+0, offset+3] = -1; W[k1*n+0, offset+3] = 1
            # P5 = (A00+A01)B11
            U[0, offset+4] = 1; U[1, offset+4] = 1
            V[1*p+k1, offset+4] = 1
            W[k0*n+0, offset+4] = 1; W[k0*n+1, offset+4] = 1
            # P6 = (A10-A00)(B00+B01)
            U[2, offset+5] = 1; U[0, offset+5] = -1
            V[0*p+k0, offset+5] = 1; V[0*p+k1, offset+5] = 1
            W[k1*n+1, offset+5] = 1
            # P7 = (A01-A11)(B10+B11)
            U[1, offset+6] = 1; U[3, offset+6] = -1
            V[1*p+k0, offset+6] = 1; V[1*p+k1, offset+6] = 1
            W[k0*n+0, offset+6] = 1; W[k1*n+0, offset+6] = 1
        
        col = 0
        for block in range(7):
            strassen_block(block*2, block*2+1, col)
            col += 7
        # Schoolbook for k=14
        for i in range(n):
            for j in range(m):
                U[i*m+j, col] = 1
                V[j*p+14, col] = 1
                W[14*n+i, col] = 1
                col += 1
        return U, V, W

    # Try rank 53 construction
    U, V, W = build_grouped_strassen()
    if verify_decomposition(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook rank 60
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
