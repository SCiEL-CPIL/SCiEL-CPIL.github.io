import numpy as np

DIMENSIONS = [3, 4, 7]


def _strassen_m222():
    """Return (U, V, W) for 2x2*2x2, rank 7, indices:
       U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # Strassen terms M1..M7
    U[0, 0] = 1; U[0, 1] = 0; U[0, 2] = 1; U[0, 3] = 0; U[0, 4] = 1; U[0, 5] = 1; U[0, 6] = 1
    U[1, 0] = 0; U[1, 1] = 0; U[1, 2] = 1; U[1, 3] = 0; U[1, 4] = 0; U[1, 5] = 1; U[1, 6] = -1
    U[2, 0] = 0; U[2, 1] = 0; U[2, 2] = 0; U[2, 3] = 1; U[2, 4] = -1; U[2, 5] = 0; U[2, 6] = 0
    U[3, 0] = 1; U[3, 1] = 1; U[3, 2] = 0; U[3, 3] = 1; U[3, 4] = 0; U[3, 5] = -1; U[3, 6] = -1
    V[0, 0] = 1; V[0, 1] = -1; V[0, 2] = 0; V[0, 3] = 1; V[0, 4] = 1; V[0, 5] = 1; V[0, 6] = -1
    V[1, 0] = 1; V[1, 1] = 1; V[1, 2] = 0; V[1, 3] = 0; V[1, 4] = 1; V[1, 5] = 0; V[1, 6] = 1
    V[2, 0] = 0; V[2, 1] = 0; V[2, 2] = 1; V[2, 3] = 0; V[2, 4] = 1; V[2, 5] = 1; V[2, 6] = 0
    V[3, 0] = 1; V[3, 1] = 1; V[3, 2] = 1; V[3, 3] = 1; V[3, 4] = 0; V[3, 5] = 0; V[3, 6] = -1
    W[0, 0] = 1; W[0, 1] = 0; W[0, 2] = 0; W[0, 3] = 1; W[0, 4] = -1; W[0, 5] = 0; W[0, 6] = -1
    W[1, 0] = 0; W[1, 1] = 0; W[1, 2] = 1; W[1, 3] = 0; W[1, 4] = 1; W[1, 5] = 0; W[1, 6] = 0
    W[2, 0] = 0; W[2, 1] = 1; W[2, 2] = 0; W[2, 3] = 1; W[2, 4] = 0; W[2, 5] = 0; W[2, 6] = 0
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 0; W[3, 3] = 0; W[3, 4] = 0; W[3, 5] = 1; W[3, 6] = 1
    return U, V, W


def _m1mp_decomp(m, p):
    """Decomposition for M_{1,m,p}: 1xm * mxp -> 1xp.
       rank = m (using the fact that C[0,k] = sum_j A[0,j]*B[j,k])
       U has shape (m, m), V has shape (m*p, m), W has shape (p, m)
    """
    r = m
    U = np.zeros((m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p, r))
    for j in range(m):
        U[j, j] = 1
        for k in range(p):
            V[j * p + k, j] = 1
        W[:, j] = 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    total_U_rows = n * m  # 12
    total_V_rows = m * p  # 28
    total_W_rows = p * n  # 21
    
    # We'll use a block decomposition:
    # Split n=3 into 2+1, m=4 into 2+2, p=7 into 4+3
    # 
    # C[0:2, 0:4] = A[0:2, 0:4] * B[0:4, 0:4]
    #   = A[0:2,0:2]*B[0:2,0:4] + A[0:2,2:4]*B[2:4,0:4]
    #   = (A[0:2,0:2]*B[0:2,0:2] + A[0:2,0:2]*B[0:2,2:4]) + (A[0:2,2:4]*B[2:4,0:2] + A[0:2,2:4]*B[2:4,2:4])
    #   Each (2,2,2) uses Strassen rank 7: 4 * 7 = 28
    #
    # C[0:2, 4:7] = A[0:2, 0:4] * B[0:4, 4:7]
    #   = A[0:2,0:2]*B[0:2,4:7] + A[0:2,2:4]*B[2:4,4:7]
    #   = (A[0:2,0:2]*B[0:2,4:6] + A[0:2,0:2]*B[0:2,6:7]) + (A[0:2,2:4]*B[2:4,4:6] + A[0:2,2:4]*B[2:4,6:7])
    #   (2,2,2) Strassen: 2 * 7 = 14
    #   (2,2,1) schoolbook: 2 * 4 = 8
    #   Total: 22
    #
    # C[2:3, 0:4] = A[2:3, 0:4] * B[0:4, 0:4] -> M_{1,4,4} rank 4
    # C[2:3, 4:7] = A[2:3, 0:4] * B[0:4, 4:7] -> M_{1,4,3} rank 4
    #
    # Total rank: 28 + 22 + 4 + 4 = 58
    
    rank = 58
    U_full = np.zeros((total_U_rows, rank))
    V_full = np.zeros((total_V_rows, rank))
    W_full = np.zeros((total_W_rows, rank))
    
    offset = 0
    
    def add_strassen(i0, j0, k0):
        """Add Strassen decomposition for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]"""
        nonlocal offset
        sU, sV, sW = _strassen_m222()
        for c in range(7):
            for ii in range(2):
                for jj in range(2):
                    U_full[(i0 + ii) * m + (j0 + jj), offset + c] += sU[ii * 2 + jj, c]
            for jj in range(2):
                for kk in range(2):
                    V_full[(j0 + jj) * p + (k0 + kk), offset + c] += sV[jj * 2 + kk, c]
            for kk in range(2):
                for ii in range(2):
                    W_full[(k0 + kk) * n + (i0 + ii), offset + c] += sW[kk * 2 + ii, c]
        offset += 7
    
    def add_schoolbook_221(i0, j0, k0):
        """Add schoolbook for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+1] -> C[i0:i0+2, k0:k0+1]"""
        nonlocal offset
        for ii in range(2):
            for jj in range(2):
                for kk in range(1):
                    U_full[(i0 + ii) * m + (j0 + jj), offset] = 1
                    V_full[(j0 + jj) * p + (k0 + kk), offset] = 1
                    W_full[(k0 + kk) * n + (i0 + ii), offset] = 1
                    offset += 1
    
    def add_m1mp(i0, j0, k0, mj, pk):
        """Add M_{1,mj,pk} decomposition for A[i0:i0+1, j0:j0+mj] * B[j0:j0+mj, k0:k0+pk] -> C[i0:i0+1, k0:k0+pk]"""
        nonlocal offset
        sU, sV, sW = _m1mp_decomp(mj, pk)
        r = sU.shape[1]
        for c in range(r):
            for jj in range(mj):
                U_full[i0 * m + (j0 + jj), offset + c] += sU[jj, c]
            for jj in range(mj):
                for kk in range(pk):
                    V_full[(j0 + jj) * p + (k0 + kk), offset + c] += sV[jj * pk + kk, c]
            for kk in range(pk):
                W_full[(k0 + kk) * n + i0, offset + c] += sW[kk, c]
        offset += r
    
    # Block 1: C[0:2, 0:4] = A[0:2, 0:4] * B[0:4, 0:4]
    add_strassen(0, 0, 0)   # A[0:2,0:2]*B[0:2,0:2] -> C[0:2,0:2]
    add_strassen(0, 0, 2)   # A[0:2,0:2]*B[0:2,2:4] -> C[0:2,2:4]
    add_strassen(0, 2, 0)   # A[0:2,2:4]*B[2:4,0:2] -> C[0:2,0:2]
    add_strassen(0, 2, 2)   # A[0:2,2:4]*B[2:4,2:4] -> C[0:2,2:4]
    # offset = 28
    
    # Block 2: C[0:2, 4:7] = A[0:2, 0:4] * B[0:4, 4:7]
    add_strassen(0, 0, 4)   # A[0:2,0:2]*B[0:2,4:6] -> C[0:2,4:6]
    add_schoolbook_221(0, 0, 6)  # A[0:2,0:2]*B[0:2,6:7] -> C[0:2,6:7]
    add_strassen(0, 2, 4)   # A[0:2,2:4]*B[2:4,4:6] -> C[0:2,4:6]
    add_schoolbook_221(0, 2, 6)  # A[0:2,2:4]*B[2:4,6:7] -> C[0:2,6:7]
    # offset = 28 + 7 + 4 + 7 + 4 = 50
    
    # Block 3: C[2:3, 0:4] = A[2:3, 0:4] * B[0:4, 0:4]
    add_m1mp(2, 0, 0, 4, 4)
    # offset = 50 + 4 = 54
    
    # Block 4: C[2:3, 4:7] = A[2:3, 0:4] * B[0:4, 4:7]
    add_m1mp(2, 0, 4, 4, 3)
    # offset = 54 + 4 = 58
    
    assert offset == rank, f"Offset {offset} != rank {rank}"
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
