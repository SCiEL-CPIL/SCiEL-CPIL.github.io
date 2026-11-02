"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2():
    """Return U, V, W for 2x2 x 2x2 -> 2x2 using Strassen (rank 7).
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen's 7 products for 2x2 matmul
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U: A indices (i,j) -> row i*2+j
    # V: B indices (j,k) -> row j*2+k
    # W: C indices (k,i) -> row k*2+i
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1: (a00+a11)(b00+b11)
    U[0, 0] = 1  # a00
    U[3, 0] = 1  # a11
    V[0, 0] = 1  # b00
    V[3, 0] = 1  # b11
    W[0, 0] = 1  # C00 (k=0,i=0)
    W[3, 0] = 1  # C11 (k=1,i=1)
    
    # M2: (a10+a11)b00
    U[2, 1] = 1  # a10
    U[3, 1] = 1  # a11
    V[0, 1] = 1  # b00
    W[2, 1] = -1 # C10 (k=0,i=1)
    W[3, 1] = 1  # C11 (k=1,i=1)
    
    # M3: a00(b01-b11)
    U[0, 2] = 1  # a00
    V[1, 2] = 1  # b01
    V[3, 2] = -1 # b11
    W[1, 2] = 1  # C01 (k=1,i=0)
    W[3, 2] = 1  # C11 (k=1,i=1)
    
    # M4: a11(b10-b00)
    U[3, 3] = 1  # a11
    V[2, 3] = 1  # b10
    V[0, 3] = -1 # b00
    W[0, 3] = 1  # C00 (k=0,i=0)
    W[2, 3] = 1  # C10 (k=0,i=1)
    
    # M5: (a00+a01)b11
    U[0, 4] = 1  # a00
    U[1, 4] = 1  # a01
    V[3, 4] = 1  # b11
    W[0, 4] = -1 # C00 (k=0,i=0)
    W[1, 4] = 1  # C01 (k=1,i=0)
    
    # M6: (a10-a00)(b00+b01)
    U[2, 5] = 1  # a10
    U[0, 5] = -1 # a00
    V[0, 5] = 1  # b00
    V[1, 5] = 1  # b01
    W[3, 5] = 1  # C11 (k=1,i=1)
    
    # M7: (a01-a11)(b10+b11)
    U[1, 6] = 1  # a01
    U[3, 6] = -1 # a11
    V[2, 6] = 1  # b10
    V[3, 6] = 1  # b11
    W[0, 6] = 1  # C00 (k=0,i=0)
    
    return U, V, W


def _strassen_4x4():
    """Return U, V, W for 4x4 x 4x4 -> 4x4 using Strassen on 2x2 blocks (rank 49).
    U shape: (16, 49), V shape: (16, 49), W shape: (16, 49)
    Indexing: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    U2, V2, W2 = _strassen_2x2()
    
    # 4x4 matrix as 2x2 blocks of 2x2 matrices
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bj is 2x2
    # C = [[C00, C01], [C10, C11]] where each Ci is 2x2
    
    # Strassen at block level:
    # S1 = (A00+A11)(B00+B11)
    # S2 = (A10+A11)B00
    # S3 = A00(B01-B11)
    # S4 = A11(B10-B00)
    # S5 = (A00+A01)B11
    # S6 = (A10-A00)(B00+B01)
    # S7 = (A01-A11)(B10+B11)
    
    # Each Si is a 2x2 matrix product, so each uses rank 7
    # Total rank = 7 * 7 = 49
    
    U = np.zeros((16, 49), dtype=float)
    V = np.zeros((16, 49), dtype=float)
    W = np.zeros((16, 49), dtype=float)
    
    # Block index mapping for 4x4:
    # A block (bi, bj) where bi in {0,1}, bj in {0,1}
    # A element (i,j) where i = bi*2 + ii, j = bj*2 + jj
    # U row index: i*4 + j = (bi*2+ii)*4 + (bj*2+jj) = bi*8 + ii*4 + bj*2 + jj
    
    # B block (bj, bk) where bj in {0,1}, bk in {0,1}
    # B element (j,k) where j = bj*2 + jj, k = bk*2 + kk
    # V row index: j*4 + k = (bj*2+jj)*4 + (bk*2+kk) = bj*8 + jj*4 + bk*2 + kk
    
    # C block (bk, bi) where bk in {0,1}, bi in {0,1}
    # C element (k,i) where k = bk*2 + kk, i = bi*2 + ii
    # W row index: k*4 + i = (bk*2+kk)*4 + (bi*2+ii) = bk*8 + kk*4 + bi*2 + ii
    
    # For each Strassen term s (0..6) and each 2x2 Strassen term t (0..6):
    # s determines which block-level combination
    # t determines which 2x2-level combination
    
    # Block-level Strassen coefficients:
    # s=0: A blocks {00,11}, B blocks {00,11} -> C blocks {00,11}
    # s=1: A blocks {10,11}, B block {00} -> C blocks {10,11}
    # s=2: A block {00}, B blocks {01,11} -> C blocks {01,11}
    # s=3: A block {11}, B blocks {10,00} -> C blocks {00,10}
    # s=4: A blocks {00,01}, B block {11} -> C blocks {00,01}
    # s=5: A blocks {10,00}, B blocks {00,01} -> C blocks {11}
    # s=6: A blocks {01,11}, B blocks {10,11} -> C blocks {00}
    
    # Define block-level coefficients:
    # For each s, we have (A_blocks, B_blocks, C_blocks) with signs
    block_A = [
        [(0,0,1), (1,1,1)],   # s=0
        [(1,0,1), (1,1,1)],   # s=1
        [(0,0,1)],            # s=2
        [(1,1,1)],            # s=3
        [(0,0,1), (0,1,1)],   # s=4
        [(1,0,1), (0,0,-1)],  # s=5
        [(0,1,1), (1,1,-1)],  # s=6
    ]
    block_B = [
        [(0,0,1), (1,1,1)],   # s=0
        [(0,0,1)],            # s=1
        [(0,1,1), (1,1,-1)],  # s=2
        [(1,0,1), (0,0,-1)],  # s=3
        [(1,1,1)],            # s=4
        [(0,0,1), (0,1,1)],   # s=5
        [(1,0,1), (1,1,1)],   # s=6
    ]
    block_C = [
        [(0,0,1), (1,1,1)],   # s=0
        [(1,0,-1), (1,1,1)],  # s=1
        [(0,1,1), (1,1,1)],   # s=2
        [(0,0,1), (1,0,1)],   # s=3
        [(0,0,-1), (0,1,1)],  # s=4
        [(1,1,1)],            # s=5
        [(0,0,1)],            # s=6
    ]
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            
            # U: for each (bi, bj, sign) in block_A[s]
            for (bi, bj, sign) in block_A[s]:
                for (ii, jj) in [(0,0),(0,1),(1,0),(1,1)]:
                    # 2x2 Strassen U contribution
                    u_val = U2[ii*2+jj, t] * sign
                    if u_val != 0:
                        i = bi * 2 + ii
                        j = bj * 2 + jj
                        U[i*4+j, r] += u_val
            
            # V: for each (bj, bk, sign) in block_B[s]
            for (bj, bk, sign) in block_B[s]:
                for (jj, kk) in [(0,0),(0,1),(1,0),(1,1)]:
                    v_val = V2[jj*2+kk, t] * sign
                    if v_val != 0:
                        j = bj * 2 + jj
                        k = bk * 2 + kk
                        V[j*4+k, r] += v_val
            
            # W: for each (bk, bi, sign) in block_C[s]
            for (bk, bi, sign) in block_C[s]:
                for (kk, ii) in [(0,0),(0,1),(1,0),(1,1)]:
                    w_val = W2[kk*2+ii, t] * sign
                    if w_val != 0:
                        k = bk * 2 + kk
                        i = bi * 2 + ii
                        W[k*4+i, r] += w_val
    
    return U, V, W


def construct():
    start_time = time.time()
    deadline = 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p  # 144
    U_fb = np.zeros((nm, rank_schoolbook))
    V_fb = np.zeros((mp, rank_schoolbook))
    W_fb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_fb.copy(), V_fb.copy(), W_fb.copy()
    
    # Try Strassen-based decomposition
    # 4x4 x 4x9 = 4x4 x [4x4 | 4x4 | 4x1]
    # Two 4x4x4 Strassen blocks (rank 49 each) + one 4x4x1 schoolbook block (rank 16)
    # Total rank = 49 + 49 + 16 = 114
    
    try:
        U4, V4, W4 = _strassen_4x4()
        
        rank_total = 49 + 49 + 16
        U = np.zeros((nm, rank_total))
        V = np.zeros((mp, rank_total))
        W = np.zeros((pn, rank_total))
        
        # Block 1: B columns 0-3 (k=0,1,2,3)
        # U: same as U4 (16 x 49)
        # V: V4 but with k offset by 0 (k=0..3), so V[j*9+k, r] = V4[j*4+k, r]
        # W: W4 but with k offset by 0, so W[k*4+i, r] -> W[k*4+i, r] in full W (k=0..3)
        U[:, 0:49] = U4
        for j in range(4):
            for k in range(4):
                V[j*9+k, 0:49] = V4[j*4+k, :]
        for k in range(4):
            for i in range(4):
                W[k*4+i, 0:49] = W4[k*4+i, :]
        
        # Block 2: B columns 4-7 (k=4,5,6,7)
        # U: same as U4
        # V: V4 but with k offset by 4, so V[j*9+(k+4), r] = V4[j*4+k, r]
        # W: W4 but with k offset by 4, so W[(k+4)*4+i, r] = W4[k*4+i, r]
        U[:, 49:98] = U4
        for j in range(4):
            for k in range(4):
                V[j*9+(k+4), 49:98] = V4[j*4+k, :]
        for k in range(4):
            for i in range(4):
                W[(k+4)*4+i, 49:98] = W4[k*4+i, :]
        
        # Block 3: B column 8 (k=8)
        # Schoolbook 4x4x1: rank 16
        # U[i*4+j, r] = 1 for all i,j
        # V[j*9+8, r] = 1 for all j
        # W[8*4+i, r] = 1 for all i
        # Each rank term corresponds to one (i,j) pair
        for i in range(4):
            for j in range(4):
                r = 98 + i*4 + j
                U[i*4+j, r] = 1
                V[j*9+8, r] = 1
                W[8*4+i, r] = 1
        
        # Verify the decomposition
        max_err = 0.0
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for r in range(rank_total):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                max_err = max(max_err, abs(s - 1.0))
        
        if max_err < 1e-10:
            if rank_total < best_rank:
                best_rank = rank_total
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
        else:
            # If Strassen verification fails, fall back to schoolbook
            pass
    
    except Exception:
        # Fall back to schoolbook if Strassen construction fails
        pass
    
    # Convert to list format
    return {
        "U": [[float(x) for x in row] for row in best_U],
        "V": [[float(x) for x in row] for row in best_V],
        "W": [[float(x) for x in row] for row in best_W]
    }
# EVOLVE-BLOCK-END
