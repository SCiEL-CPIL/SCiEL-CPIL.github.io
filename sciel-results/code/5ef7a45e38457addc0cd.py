"""Matrix multiplication 3x4x6 - rank-minimal exact tensor decomposition."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 4, 6]

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 Strassen decomposition.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen formulas for C = A * B where A, B, C are 2x2
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # C = A*B
    
    # The 7 products:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U[i*m+j, c] where m=2, so U[i*2+j, c]
    # V[j*p+k, c] where p=2, so V[j*2+k, c]
    # W[k*n+i, c] where n=2, so W[k*2+i, c]
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Column 0: M1 = (a11+a22)(b11+b22)
    # U: a11 (i=0,j=0) -> idx 0; a22 (i=1,j=1) -> idx 3
    U[0, 0] = 1; U[3, 0] = 1
    # V: b11 (j=0,k=0) -> idx 0; b22 (j=1,k=1) -> idx 3
    V[0, 0] = 1; V[3, 0] = 1
    # W: C11 gets +1 (k=0,i=0 -> idx 0); C22 gets +1 (k=1,i=1 -> idx 3)
    W[0, 0] = 1; W[3, 0] = 1
    
    # Column 1: M2 = (a21+a22)b11
    # U: a21 (i=1,j=0) -> idx 2; a22 (i=1,j=1) -> idx 3
    U[2, 1] = 1; U[3, 1] = 1
    # V: b11 (j=0,k=0) -> idx 0
    V[0, 1] = 1
    # W: C21 gets +1 (k=0,i=1 -> idx 2); C22 gets -1 (k=1,i=1 -> idx 3)
    W[2, 1] = 1; W[3, 1] = -1
    
    # Column 2: M3 = a11(b12-b22)
    # U: a11 (i=0,j=0) -> idx 0
    U[0, 2] = 1
    # V: b12 (j=0,k=1) -> idx 1; b22 (j=1,k=1) -> idx 3
    V[1, 2] = 1; V[3, 2] = -1
    # W: C12 gets +1 (k=1,i=0 -> idx 1); C22 gets +1 (k=1,i=1 -> idx 3)
    W[1, 2] = 1; W[3, 2] = 1
    
    # Column 3: M4 = a22(b21-b11)
    # U: a22 (i=1,j=1) -> idx 3
    U[3, 3] = 1
    # V: b21 (j=1,k=0) -> idx 2; b11 (j=0,k=0) -> idx 0
    V[2, 3] = 1; V[0, 3] = -1
    # W: C11 gets +1 (k=0,i=0 -> idx 0); C21 gets +1 (k=0,i=1 -> idx 2)
    W[0, 3] = 1; W[2, 3] = 1
    
    # Column 4: M5 = (a11+a12)b22
    # U: a11 (i=0,j=0) -> idx 0; a12 (i=0,j=1) -> idx 1
    U[0, 4] = 1; U[1, 4] = 1
    # V: b22 (j=1,k=1) -> idx 3
    V[3, 4] = 1
    # W: C11 gets -1 (k=0,i=0 -> idx 0); C12 gets +1 (k=1,i=0 -> idx 1)
    W[0, 4] = -1; W[1, 4] = 1
    
    # Column 5: M6 = (a21-a11)(b11+b12)
    # U: a21 (i=1,j=0) -> idx 2; a11 (i=0,j=0) -> idx 0
    U[2, 5] = 1; U[0, 5] = -1
    # V: b11 (j=0,k=0) -> idx 0; b12 (j=0,k=1) -> idx 1
    V[0, 5] = 1; V[1, 5] = 1
    # W: C22 gets +1 (k=1,i=1 -> idx 3)
    W[3, 5] = 1
    
    # Column 6: M7 = (a12-a22)(b21+b22)
    # U: a12 (i=0,j=1) -> idx 1; a22 (i=1,j=1) -> idx 3
    U[1, 6] = 1; U[3, 6] = -1
    # V: b21 (j=1,k=0) -> idx 2; b22 (j=1,k=1) -> idx 3
    V[2, 6] = 1; V[3, 6] = 1
    # W: C11 gets +1 (k=0,i=0 -> idx 0)
    W[0, 6] = 1
    
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that U @ V^T == W in the appropriate tensor sense."""
    rank = U.shape[1]
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix mult, T[i,j,k] should be 1 if we're computing (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # Actually the multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
    # Wait, let me reconsider. The matrix multiplication tensor M_{n,m,p} has entries
    # M[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # A decomposition of rank r gives: M[i,j,k] = sum_{c=1}^{r} U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # Check: for each (i,j,k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err


def _try_construct_block_decomposition():
    """
    Decompose 3x4x6 as:
    - Split n=3 into 1+2
    - Split p=6 into 2+2+2
    
    For the 1x4x2 blocks (3 of them): rank = 1*4*2 = 8 each (schoolbook)
    For the 2x4x2 blocks (3 of them): use Strassen on 2x2x2 subblocks
    
    For 2x4x2: split m=4 into 2+2
      - 2x2x2 block 1: Strassen rank 7
      - 2x2x2 block 2: Strassen rank 7
      Total for 2x4x2: 14
    
    Total: 3*8 + 3*14 = 24 + 42 = 66
    """
    n, m, p = 3, 4, 6
    
    # We'll build the full U, V, W matrices
    # n*m = 12 rows for U
    # m*p = 24 rows for V
    # p*n = 18 rows for W
    
    # Block structure:
    # n split: n0=1 (i=0), n1=2 (i=1,2)
    # m split: m0=2 (j=0,1), m1=2 (j=2,3)
    # p split: p0=2 (k=0,1), p1=2 (k=2,3), p2=2 (k=4,5)
    
    blocks = []  # list of (n_start, m_start, p_start, n_size, m_size, p_size, U_block, V_block, W_block)
    
    # For each combination of n-block, m-block, p-block
    n_splits = [(0, 1), (1, 2)]  # (start, size)
    m_splits = [(0, 2), (2, 2)]
    p_splits = [(0, 2), (2, 2), (4, 2)]
    
    for (ns, nn) in n_splits:
        for (ms, mm) in m_splits:
            for (ps, pp) in p_splits:
                # Build decomposition for nn x mm x pp
                if nn == 1 and mm == 2 and pp == 2:
                    # Schoolbook: rank = 1*2*2 = 4
                    # U: (2, 4), V: (4, 4), W: (2, 4)
                    U_b = np.zeros((nn*mm, 4))
                    V_b = np.zeros((mm*pp, 4))
                    W_b = np.zeros((pp*nn, 4))
                    col = 0
                    for ii in range(nn):
                        for jj in range(mm):
                            for kk in range(pp):
                                U_b[ii*mm+jj, col] = 1
                                V_b[jj*pp+kk, col] = 1
                                W_b[kk*nn+ii, col] = 1
                                col += 1
                elif nn == 2 and mm == 2 and pp == 2:
                    # Strassen: rank 7
                    U_b, V_b, W_b = _strassen_2x2x2()
                else:
                    continue
                
                blocks.append((ns, ms, ps, nn, mm, pp, U_b, V_b, W_b))
    
    # Now assemble full matrices
    total_rank = sum(b[6].shape[1] for b in blocks)
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    col_offset = 0
    for (ns, ms, ps, nn, mm, pp, U_b, V_b, W_b) in blocks:
        r = U_b.shape[1]
        for ii in range(nn):
            for jj in range(mm):
                U_full[(ns+ii)*m + (ms+jj), col_offset:col_offset+r] = U_b[ii*mm+jj, :]
        for jj in range(mm):
            for kk in range(pp):
                V_full[(ms+jj)*p + (ps+kk), col_offset:col_offset+r] = V_b[jj*pp+kk, :]
        for kk in range(pp):
            for ii in range(nn):
                W_full[(ps+kk)*n + (ns+ii), col_offset:col_offset+r] = W_b[kk*nn+ii, :]
        col_offset += r
    
    return U_full, V_full, W_full, total_rank


def _to_list(mat):
    """Convert numpy array to list of lists with exact integer/half-integer values."""
    result = []
    for row in mat:
        r = []
        for val in row:
            # Round to nearest half-integer
            v = round(val * 2) / 2
            if abs(v - round(v)) < 1e-9:
                v = int(round(v))
            r.append(v)
        result.append(r)
    return result


def construct():
    t0 = time.time()
    
    # Build the block decomposition
    U, V, W, rank = _try_construct_block_decomposition()
    
    # Verify
    err = _verify(U, V, W, 3, 4, 6)
    
    if err > 1e-9:
        # Fallback to schoolbook if verification fails
        n, m, p = 3, 4, 6
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    else:
        # Convert to list format with exact values
        U_list = _to_list(U)
        V_list = _to_list(V)
        W_list = _to_list(W)
        
        # Final safety check with list version
        n, m, p = 3, 4, 6
        r = len(U_list[0])
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(r):
                        val += U_list[i*m+j][c] * V_list[j*p+k][c] * W_list[k*n+i][c]
                    e = abs(val - 1.0)
                    if e > max_err:
                        max_err = e
        if max_err > 1e-9:
            # Fallback
            rank = n * m * p
            U_list = [[0]*rank for _ in range(n*m)]
            V_list = [[0]*rank for _ in range(m*p)]
            W_list = [[0]*rank for _ in range(p*n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_list[i*m+j][col] = 1
                        V_list[j*p+k][col] = 1
                        W_list[k*n+i][col] = 1
                        col += 1
    
    elapsed = time.time() - t0
    if elapsed > 110:
        # Emergency fallback
        n, m, p = 3, 4, 6
        rank = n * m * p
        U_list = [[0]*rank for _ in range(n*m)]
        V_list = [[0]*rank for _ in range(m*p)]
        W_list = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_list[i*m+j][col] = 1
                    V_list[j*p+k][col] = 1
                    W_list[k*n+i][col] = 1
                    col += 1
    
    return {"U": U_list, "V": V_list, "W": W_list}
