import time
import numpy as np

_START = time.time()
_DEADLINE = _START + 140.0

def _time_left():
    return _DEADLINE - time.time()

def _strassen_2x2x2():
    """
    Strassen 2x2x2 decomposition.
    U: (4,7), V: (4,7), W: (4,7)
    U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = (a00+a11)*(b00+b11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # P2 = (a10+a11)*b00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[1,1] = 1; W[3,1] = -1
    
    # P3 = a00*(b01-b11)
    U[0,2] = 1
    V[1,2] = 1; V[3,2] = -1
    W[2,2] = 1; W[3,2] = 1
    
    # P4 = a11*(b10-b00)
    U[3,3] = 1
    V[2,3] = 1; V[0,3] = -1
    W[0,3] = 1; W[1,3] = 1
    
    # P5 = (a00+a01)*b11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = -1; W[2,4] = 1
    
    # P6 = (a10-a00)*(b00+b01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # P7 = (a01-a11)*(b10+b11)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1
    
    return U, V, W

def _build_4x4x2():
    """
    Build 4x4x2 decomposition using 4 Strassen 2x2x2 blocks.
    A: 4x4, B: 4x2, C: 4x2
    U: (16, 28), V: (8, 28), W: (8, 28)
    U[i*4+j, r], V[j*2+k, r], W[k*4+i, r]
    """
    U2, V2, W2 = _strassen_2x2x2()
    rank = 28
    U = np.zeros((16, rank))
    V = np.zeros((8, rank))
    W = np.zeros((8, rank))
    
    blocks = [(0,0,0,0,0), (0,2,2,0,0), (2,0,0,2,0), (2,2,2,2,0)]
    
    for bi, (ai, aj, bj, ci, ck) in enumerate(blocks):
        base = bi * 7
        for t in range(7):
            col = base + t
            for li in range(2):
                for lj in range(2):
                    la = li * 2 + lj
                    if U2[la, t] != 0:
                        gi, gj = ai + li, aj + lj
                        U[gi * 4 + gj, col] += U2[la, t]
            for lj in range(2):
                for lk in range(2):
                    lb = lj * 2 + lk
                    if V2[lb, t] != 0:
                        gj, gk = bj + lj, ck + lk
                        V[gj * 2 + gk, col] += V2[lb, t]
            for lk in range(2):
                for li in range(2):
                    lc = lk * 2 + li
                    if W2[lc, t] != 0:
                        gk, gi = ck + lk, ci + li
                        W[gk * 4 + gi, col] += W2[lc, t]
    
    return U, V, W

def _build_4x1x2():
    """
    Build 4x1x2 decomposition (schoolbook, rank=8).
    A: 4x1, B: 1x2, C: 4x2
    U: (4, 8), V: (2, 8), W: (8, 8)
    U[i*1+j, r] where j=0, V[j*2+k, r] where j=0, W[k*4+i, r]
    """
    n, m, p = 4, 1, 2
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1
    return U, V, W

def _build_4x5x6():
    """
    Build 4x5x6 decomposition.
    Split m=5 into m1=4, m2=1.
    C = A[:, 0:4] @ B[0:4, :] + A[:, 4:5] @ B[4:5, :]
    
    Part 1: 4x4x6 = 3 blocks of 4x4x2 (rank 28 each) = 84
    Part 2: 4x1x6 = 3 blocks of 4x1x2 (rank 8 each) = 24
    Total rank: 108
    """
    U_4x4x2, V_4x4x2, W_4x4x2 = _build_4x4x2()
    U_4x1x2, V_4x1x2, W_4x1x2 = _build_4x1x2()
    
    total_rank = 3 * 28 + 3 * 8  # 84 + 24 = 108
    U = np.zeros((4 * 5, total_rank))
    V = np.zeros((5 * 6, total_rank))
    W = np.zeros((6 * 4, total_rank))
    
    # Part 1: A_left (4x4) @ B_top (4x6)
    # Split B_top into 3 blocks of 4x2
    for block_idx in range(3):
        base = block_idx * 28
        k_start = block_idx * 2
        
        # U part: rows are i*5+j for i in 0..3, j in 0..3
        for r in range(16):
            i, j = divmod(r, 4)
            U[i * 5 + j, base:base+28] += U_4x4x2[r, :]
        
        # V part: rows are j*6+k for j in 0..3, k in k_start..k_start+1
        for r in range(8):
            j, k_local = divmod(r, 2)
            k = k_start + k_local
            V[j * 6 + k, base:base+28] += V_4x4x2[r, :]
        
        # W part: rows are k*4+i for k in k_start..k_start+1, i in 0..3
        for r in range(8):
            k_local, i = divmod(r, 4)
            k = k_start + k_local
            W[k * 4 + i, base:base+28] += W_4x4x2[r, :]
    
    # Part 2: A_right (4x1) @ B_bottom (1x6)
    # Split B_bottom into 3 blocks of 1x2
    for block_idx in range(3):
        base = 84 + block_idx * 8
        k_start = block_idx * 2
        
        # U part: rows are i*5+4 for i in 0..3
        for i in range(4):
            U[i * 5 + 4, base:base+8] += U_4x1x2[i, :]
        
        # V part: rows are 4*6+k for k in k_start..k_start+1
        for k_local in range(2):
            k = k_start + k_local
            V[4 * 6 + k, base:base+8] += V_4x1x2[k_local, :]
        
        # W part: rows are k*4+i for k in k_start..k_start+1, i in 0..3
        for r in range(8):
            k_local, i = divmod(r, 4)
            k = k_start + k_local
            W[k * 4 + i, base:base+8] += W_4x1x2[r, :]
    
    return U, V, W

def construct():
    if time.time() > _DEADLINE:
        # Fallback to schoolbook
        n, m, p = 4, 5, 6
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
    
    U, V, W = _build_4x5x6()
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
