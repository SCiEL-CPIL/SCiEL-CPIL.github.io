"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return Strassen 2x2x2 decomposition (U, V, W) each of shape (4, 7).
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # r=0: (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # r=1: (A10+A11)B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    # r=2: A00(B01-B11)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    # r=3: A11(B10-B00)
    U[3,3]=1; V[0,3]=-1; V[2,3]=1; W[0,3]=1; W[1,3]=1
    # r=4: (A00+A01)B11
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # r=5: (A10-A00)(B00+B01)
    U[0,5]=-1; U[2,5]=1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # r=6: (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W


def _build_4x4x4_strassen():
    """Build 4x4x4 decomposition via tensor product of two Strassen 2x2x2.
    Rank = 7*7 = 49.
    U shape (16,49), V shape (16,49), W shape (16,49).
    Indexing: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]."""
    U2, V2, W2 = _strassen_2x2x2()
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    for c in range(7):
        for d in range(7):
            r = c * 7 + d
            for i0 in range(2):
                for i1 in range(2):
                    for j0 in range(2):
                        for j1 in range(2):
                            idx = (i0*2+i1)*4 + (j0*2+j1)
                            U[idx, r] = U2[i0*2+j0, c] * U2[i1*2+j1, d]
            for j0 in range(2):
                for j1 in range(2):
                    for k0 in range(2):
                        for k1 in range(2):
                            idx = (j0*2+j1)*4 + (k0*2+k1)
                            V[idx, r] = V2[j0*2+k0, c] * V2[j1*2+k1, d]
            for k0 in range(2):
                for k1 in range(2):
                    for i0 in range(2):
                        for i1 in range(2):
                            idx = (k0*2+i0)*4 + (k1*2+i1)
                            W[idx, r] = W2[k0*2+i0, c] * W2[k1*2+i1, d]
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    deadline = time.time() + 140  # safety margin

    # Build 4x4x4 Strassen decomposition (rank 49)
    U444, V444, W444 = _build_4x4x4_strassen()

    # Build 4x4x8 by splitting output k into two blocks of 4
    # Total rank = 49 * 2 = 98
    rank = 98
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    for b in range(2):
        col_start = b * 49
        col_end = (b + 1) * 49
        # U is the same
        U[:, col_start:col_end] = U444
        # V: V[j*8 + (b*4+k), col] = V444[j*4+k, col-col_start]
        for j in range(m):
            for k in range(4):
                k_actual = b * 4 + k
                V[j*p + k_actual, col_start:col_end] = V444[j*4 + k, :]
        # W: W[(b*4+k)*n + i, col] = W444[k*4+i, col-col_start]
        for k in range(4):
            k_actual = b * 4 + k
            for i in range(n):
                W[k_actual*n + i, col_start:col_end] = W444[k*4 + i, :]

    # Quick verification (sparse check for speed)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(total - 1.0) > 1e-9:
                    break

    # Convert to lists
    U_list = [[int(x) if x == int(x) else x for x in row] for row in U]
    V_list = [[int(x) if x == int(x) else x for x in row] for row in V]
    W_list = [[int(x) if x == int(x) else x for x in row] for row in W]

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
