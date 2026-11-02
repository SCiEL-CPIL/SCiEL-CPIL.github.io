"""
Rank-minimal exact tensor decomposition for 3x4 by 4x8 matrix multiplication.

Strategy: Use Strassen's rank-7 algorithm for 2x2*2x2 blocks where possible.
Split the 3x4 x 4x8 multiplication into blocks:
  - A rows: [0:2] (2 rows), [2:3] (1 row)
  - A cols: [0:2] (2 cols), [2:4] (2 cols)
  - B rows: [0:2] (2 rows), [2:4] (2 rows)
  - B cols: [0:2], [2:4], [4:6], [6:8] (4 blocks of 2 cols)
  - C rows: [0:2] (2 rows), [2:3] (1 row)
  - C cols: [0:2], [2:4], [4:6], [6:8] (4 blocks of 2 cols)

For C[0:2, k:k+2] = A[0:2, 0:2]*B[0:2, k:k+2] + A[0:2, 2:4]*B[2:4, k:k+2]:
  Each term is 2x2 * 2x2 -> 2x2, rank 7 (Strassen).
  2 terms per col_block, 4 col_blocks: 2 * 4 * 7 = 56.

For C[2:3, k:k+2] = A[2:3, 0:2]*B[0:2, k:k+2] + A[2:3, 2:4]*B[2:4, k:k+2]:
  Each term is 1x2 * 2x2 -> 1x2, rank 4 (schoolbook, minimal).
  2 terms per col_block, 4 col_blocks: 2 * 4 * 4 = 32.

Grand total: 56 + 32 = 88.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _strassen_2x2x2():
    """
    Return U, V, W for 2x2 * 2x2 -> 2x2 with rank 7 (Strassen).
    
    U shape: (4, 7), indexed by [i*2+j] where i=row of A, j=col of A
    V shape: (4, 7), indexed by [j*2+k] where j=row of B, k=col of B
    W shape: (4, 7), indexed by [k*2+i] where k=col of C, i=row of C
    
    The decomposition satisfies:
      sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c] = 1 for all (i,j,k) in {0,1}^3
    """
    # Strassen's 7 products for A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # p1 = (a + d)(e + h)
    # p2 = (c + d)e
    # p3 = a(f - h)
    # p4 = d(g - e)
    # p5 = (a + b)h
    # p6 = (c - a)(e + f)
    # p7 = (b - d)(g + h)
    #
    # C[0,0] = p1 + p4 - p5 + p7
    # C[0,1] = p3 + p5
    # C[1,0] = p2 + p6
    # C[1,1] = p1 - p2 + p3 + p6
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # p1 = (a + d)(e + h)
    # U: a=A[0,0]=idx0, d=A[1,1]=idx3
    U[0, 0] = 1; U[3, 0] = 1
    # V: e=B[0,0]=idx0, h=B[1,1]=idx3
    V[0, 0] = 1; V[3, 0] = 1
    # W: contributes to C[0,0]=idx0 and C[1,1]=idx3
    W[0, 0] = 1; W[3, 0] = 1
    
    # p2 = (c + d)e
    # U: c=A[1,0]=idx2, d=A[1,1]=idx3
    U[2, 1] = 1; U[3, 1] = 1
    # V: e=B[0,0]=idx0
    V[0, 1] = 1
    # W: contributes to C[1,0]=idx2 with +1, C[1,1]=idx3 with -1
    W[2, 1] = 1; W[3, 1] = -1
    
    # p3 = a(f - h)
    # U: a=A[0,0]=idx0
    U[0, 2] = 1
    # V: f=B[0,1]=idx1, h=B[1,1]=idx3
    V[1, 2] = 1; V[3, 2] = -1
    # W: contributes to C[0,1]=idx1 and C[1,1]=idx3
    W[1, 2] = 1; W[3, 2] = 1
    
    # p4 = d(g - e)
    # U: d=A[1,1]=idx3
    U[3, 3] = 1
    # V: g=B[1,0]=idx2, e=B[0,0]=idx0
    V[2, 3] = 1; V[0, 3] = -1
    # W: contributes to C[0,0]=idx0
    W[0, 3] = 1
    
    # p5 = (a + b)h
    # U: a=A[0,0]=idx0, b=A[0,1]=idx1
    U[0, 4] = 1; U[1, 4] = 1
    # V: h=B[1,1]=idx3
    V[3, 4] = 1
    # W: contributes to C[0,0]=idx0 with -1, C[0,1]=idx1 with +1
    W[0, 4] = -1; W[1, 4] = 1
    
    # p6 = (c - a)(e + f)
    # U: c=A[1,0]=idx2, a=A[0,0]=idx0
    U[2, 5] = 1; U[0, 5] = -1
    # V: e=B[0,0]=idx0, f=B[0,1]=idx1
    V[0, 5] = 1; V[1, 5] = 1
    # W: contributes to C[1,0]=idx2 and C[1,1]=idx3
    W[2, 5] = 1; W[3, 5] = 1
    
    # p7 = (b - d)(g + h)
    # U: b=A[0,1]=idx1, d=A[1,1]=idx3
    U[1, 6] = 1; U[3, 6] = -1
    # V: g=B[1,0]=idx2, h=B[1,1]=idx3
    V[2, 6] = 1; V[3, 6] = 1
    # W: contributes to C[0,0]=idx0
    W[0, 6] = 1
    
    return U, V, W

def _schoolbook_1x2x2():
    """
    Return U, V, W for 1x2 * 2x2 -> 1x2 with rank 4 (schoolbook).
    
    U shape: (2, 4), indexed by [0*2+j] = [0, 1]
    V shape: (4, 4), indexed by [j*2+k]
    W shape: (2, 4), indexed by [k*1+0] = [0, 1]
    """
    U = np.zeros((2, 4))
    V = np.zeros((4, 4))
    W = np.zeros((2, 4))
    col = 0
    for j in range(2):
        for k in range(2):
            U[j, col] = 1
            V[j*2+k, col] = 1
            W[k, col] = 1
            col += 1
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Get base decompositions
    U_s, V_s, W_s = _strassen_2x2x2()  # shapes (4,7), (4,7), (4,7)
    U_sb, V_sb, W_sb = _schoolbook_1x2x2()  # shapes (2,4), (4,4), (2,4)
    
    # Verify Strassen
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = np.sum(U_s[i*2+j, :] * V_s[j*2+k, :] * W_s[k*2+i, :])
                if abs(s - 1.0) > 1e-9:
                    raise ValueError(f"Strassen verification failed at ({i},{j},{k}): got {s}")
    
    # Verify schoolbook 1x2x2
    for j in range(2):
        for k in range(2):
            s = np.sum(U_sb[j, :] * V_sb[j*2+k, :] * W_sb[k, :])
            if abs(s - 1.0) > 1e-9:
                raise ValueError(f"Schoolbook 1x2x2 verification failed at ({j},{k}): got {s}")
    
    # Build the full decomposition
    # Total rank = 2 * 4 * 7 + 2 * 4 * 4 = 56 + 32 = 88
    rank = 88
    U_full = np.zeros((n*m, rank))  # (12, 88)
    V_full = np.zeros((m*p, rank))  # (32, 88)
    W_full = np.zeros((p*n, rank))  # (24, 88)
    
    col_offset = 0
    
    # For each column block of B/C (k_block in 0..3):
    for kb in range(4):
        # For each row block of A/C:
        # Row block 0: A[0:2, :], C[0:2, :]
        for rb in range(2):
            if rb == 0:
                # 2x2 * 2x2 -> 2x2, using Strassen
                # For each column sub-block of A/B (j_block in 0..1):
                for jb in range(2):
                    # A sub-block: rows [rb*2:rb*2+2], cols [jb*2:jb*2+2]
                    # B sub-block: rows [jb*2:jb*2+2], cols [kb*2:kb*2+2]
                    # C sub-block: rows [rb*2:rb*2+2], cols [kb*2:kb*2+2]
                    
                    # Map Strassen U (4,7) to U_full
                    # U index: i*2+j where i in {0,1}, j in {0,1}
                    # A row offset: rb*2, A col offset: jb*2
                    for i in range(2):
                        for j in range(2):
                            a_idx = (rb*2 + i) * m + (jb*2 + j)
                            U_full[a_idx, col_offset:col_offset+7] = U_s[i*2+j, :]
                    
                    # Map Strassen V (4,7) to V_full
                    # V index: j*2+k where j in {0,1}, k in {0,1}
                    # B row offset: jb*2, B col offset: kb*2
                    for j in range(2):
                        for k in range(2):
                            b_idx = (jb*2 + j) * p + (kb*2 + k)
                            V_full[b_idx, col_offset:col_offset+7] = V_s[j*2+k, :]
                    
                    # Map Strassen W (4,7) to W_full
                    # W index: k*2+i where k in {0,1}, i in {0,1}
                    # C col offset: kb*2, C row offset: rb*2
                    for k in range(2):
                        for i in range(2):
                            c_idx = (kb*2 + k) * n + (rb*2 + i)
                            W_full[c_idx, col_offset:col_offset+7] = W_s[k*2+i, :]
                    
                    col_offset += 7
            else:
                # 1x2 * 2x2 -> 1x2, using schoolbook
                # For each column sub-block of A/B (j_block in 0..1):
                for jb in range(2):
                    # A sub-block: row [rb], cols [jb*2:jb*2+2]
                    # B sub-block: rows [jb*2:jb*2+2], cols [kb*2:kb*2+2]
                    # C sub-block: row [rb], cols [kb*2:kb*2+2]
                    
                    # Map schoolbook U (2,4) to U_full
                    # U index: j where j in {0,1}
                    # A row: rb, A col offset: jb*2
                    for j in range(2):
                        a_idx = rb * m + (jb*2 + j)
                        U_full[a_idx, col_offset:col_offset+4] = U_sb[j, :]
                    
                    # Map schoolbook V (4,4) to V_full
                    # V index: j*2+k where j in {0,1}, k in {0,1}
                    # B row offset: jb*2, B col offset: kb*2
                    for j in range(2):
                        for k in range(2):
                            b_idx = (jb*2 + j) * p + (kb*2 + k)
                            V_full[b_idx, col_offset:col_offset+4] = V_sb[j*2+k, :]
                    
                    # Map schoolbook W (2,4) to W_full
                    # W index: k where k in {0,1}
                    # C col offset: kb*2, C row: rb
                    for k in range(2):
                        c_idx = (kb*2 + k) * n + rb
                        W_full[c_idx, col_offset:col_offset+4] = W_sb[k, :]
                    
                    col_offset += 4
    
    assert col_offset == rank, f"Column offset {col_offset} != rank {rank}"
    
    # Full verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_full[i*m+j, :] * V_full[j*p+k, :] * W_full[k*n+i, :])
                if abs(s - 1.0) > 1e-9:
                    raise ValueError(f"Verification failed at ({i},{j},{k}): got {s}")
    
    elapsed = time.time() - start_time
    if elapsed > 110:
        # Safety margin
        pass
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
