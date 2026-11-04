import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def _strassen_2x2x2(U, V, W, i0, j0, k0, offset):
    """
    Add Strassen 2x2x2 block at global positions i0, j0, k0 starting at column offset.
    """
    n, m, p = 4, 5, 6
    
    # M1 = (A00+A11)*(B00+B11)
    # Contributes to: C00(+1), C01(+1), C10(+1), C11(+1)
    U[i0*m + j0, offset+0] = 1
    U[(i0+1)*m + (j0+1), offset+0] = 1
    V[j0*p + k0, offset+0] = 1
    V[(j0+1)*p + (k0+1), offset+0] = 1
    W[k0*n + i0, offset+0] = 1
    W[(k0+1)*n + i0, offset+0] = 1
    W[k0*n + (i0+1), offset+0] = 1
    W[(k0+1)*n + (i0+1), offset+0] = 1
    
    # M2 = (A10+A11)*B00
    # Contributes to: C10(+1), C11(-1)
    U[(i0+1)*m + j0, offset+1] = 1
    U[(i0+1)*m + (j0+1), offset+1] = 1
    V[j0*p + k0, offset+1] = 1
    W[k0*n + (i0+1), offset+1] = 1
    W[(k0+1)*n + (i0+1), offset+1] = -1
    
    # M3 = A00*(B01-B11)
    # Contributes to: C11(+1)
    U[i0*m + j0, offset+2] = 1
    V[j0*p + (k0+1), offset+2] = 1
    V[(j0+1)*p + (k0+1), offset+2] = -1
    W[(k0+1)*n + (i0+1), offset+2] = 1
    
    # M4 = A11*(B10-B00)
    # Contributes to: C00(-1), C01(+1)
    U[(i0+1)*m + (j0+1), offset+3] = 1
    V[(j0+1)*p + k0, offset+3] = 1
    V[j0*p + k0, offset+3] = -1
    W[k0*n + i0, offset+3] = -1
    W[(k0+1)*n + i0, offset+3] = 1
    
    # M5 = (A00+A01)*B11
    # Contributes to: C01(+1), C11(+1)
    U[i0*m + j0, offset+4] = 1
    U[i0*m + (j0+1), offset+4] = 1
    V[(j0+1)*p + (k0+1), offset+4] = 1
    W[(k0+1)*n + i0, offset+4] = 1
    W[(k0+1)*n + (i0+1), offset+4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    # Contributes to: C00(+1), C10(+1)
    U[(i0+1)*m + j0, offset+5] = 1
    U[i0*m + j0, offset+5] = -1
    V[j0*p + k0, offset+5] = 1
    V[j0*p + (k0+1), offset+5] = 1
    W[k0*n + i0, offset+5] = 1
    W[k0*n + (i0+1), offset+5] = 1
    
    # M7 = (A01-A11)*(B10+B11)
    # Contributes to: C00(+1), C11(+1)
    U[i0*m + (j0+1), offset+6] = 1
    U[(i0+1)*m + (j0+1), offset+6] = -1
    V[(j0+1)*p + k0, offset+6] = 1
    V[(j0+1)*p + (k0+1), offset+6] = 1
    W[k0*n + i0, offset+6] = 1
    W[(k0+1)*n + (i0+1), offset+6] = 1
    
    return offset + 7

def _schoolbook_block(U, V, W, i0, j0, k0, ni, nj, nk, offset):
    """
    Add schoolbook block for sub-block starting at (i0,j0,k0) with sizes ni,nj,nk.
    """
    n, m, p = 4, 5, 6
    for ii in range(ni):
        for jj in range(nj):
            for kk in range(nk):
                i = i0 + ii
                j = j0 + jj
                k = k0 + kk
                U[i*m + j, offset] = 1
                V[j*p + k, offset] = 1
                W[k*n + i, offset] = 1
                offset += 1
    return offset

def _build_block_strassen(n, m, p):
    """
    Build decomposition using Strassen for 2x2x2 blocks and schoolbook elsewhere.
    Partition:
    - i: [0,1], [2,3]  (sizes 2, 2)
    - j: [0,1], [2,3], [4]  (sizes 2, 2, 1)
    - k: [0,1], [2,3], [4,5]  (sizes 2, 2, 2)
    """
    i_blocks = [(0, 2), (2, 2)]  # (start, size)
    j_blocks = [(0, 2), (2, 2), (4, 1)]
    k_blocks = [(0, 2), (2, 2), (4, 2)]
    
    # Calculate total rank
    total_rank = 0
    for i0, ni in i_blocks:
        for j0, nj in j_blocks:
            for k0, nk in k_blocks:
                if ni == 2 and nj == 2 and nk == 2:
                    total_rank += 7  # Strassen
                else:
                    total_rank += ni * nj * nk  # schoolbook
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    offset = 0
    for i0, ni in i_blocks:
        for j0, nj in j_blocks:
            for k0, nk in k_blocks:
                if ni == 2 and nj == 2 and nk == 2:
                    offset = _strassen_2x2x2(U, V, W, i0, j0, k0, offset)
                else:
                    offset = _schoolbook_block(U, V, W, i0, j0, k0, ni, nj, nk, offset)
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify the tensor decomposition reconstructs matrix multiplication exactly."""
    rank = U.shape[1]
    # Check positive constraints
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    # Check zero constraints (sample)
    import random
    random.seed(42)
    for _ in range(3000):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        g = random.randint(0, p*n-1)
        i = a // m; j = a % m
        jp = b // p; k = b % p
        kn = g // n; ip = g % n
        if i != ip or j != jp or k != kn:
            val = 0.0
            for r in range(rank):
                val += U[a, r] * V[b, r] * W[g, r]
            if abs(val) > tol:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Build block-Strassen decomposition
    U, V, W = _build_block_strassen(n, m, p)
    rank = U.shape[1]
    
    # Verify
    if time.time() < deadline - 10:
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook if verification fails
    schoolbook_rank = n * m * p
    U2 = np.zeros((n * m, schoolbook_rank))
    V2 = np.zeros((m * p, schoolbook_rank))
    W2 = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U2[i * m + j, col] = 1
                V2[j * p + k, col] = 1
                W2[k * n + i, col] = 1
                col += 1
    
    return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
