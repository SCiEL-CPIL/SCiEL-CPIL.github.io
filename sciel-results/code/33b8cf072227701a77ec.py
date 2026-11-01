"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _verify(U, V, W, n, m, p, rank):
    """Verify the tensor decomposition: T[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]"""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a][d] * V[b][d] * W[c][d] for d in range(rank))
                if val != 1:
                    return False
    # Also verify that non-matching entries are 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    for jp in range(m):
                        for kp in range(p):
                            if (i, j, k) != (ip, jp, kp):
                                a = i * m + j
                                b = jp * p + kp
                                c = k * n + ip
                                val = sum(U[a][d] * V[b][d] * W[c][d] for d in range(rank))
                                if val != 0:
                                    return False
    return True


def _build_rank11_2x3x2():
    """
    Build a rank-11 decomposition for 2x3 * 3x2 matrix multiplication.
    
    A = [[a00, a01, a02],
         [a10, a11, a12]]  (2x3)
    B = [[b00, b01],
         [b10, b11],
         [b20, b21]]  (3x2)
    C = [[c00, c01],
         [c10, c11]]  (2x2)
    
    Strategy: Split into 2x2 core (Strassen, rank 7) + outer product (rank 4) = 11.
    
    A1 = [[a00, a01], [a10, a11]] (2x2)
    A2 = [a02, a12] (2x1)
    B1 = [[b00, b01], [b10, b11]] (2x2)
    B2 = [b20, b21] (1x2)
    
    C = A1*B1 + A2*B2
    
    Strassen for A1*B1:
    m1 = (a00+a11)*(b00+b11)
    m2 = (a10+a11)*b00
    m3 = a00*(b10-b11)
    m4 = a11*(b01-b00)
    m5 = (a00+a01)*b11
    m6 = (a10-a00)*(b00+b10)
    m7 = (a01-a11)*(b01+b11)
    
    c00 = m1 + m4 - m5 + m7
    c01 = m3 + m5
    c10 = m2 + m4
    c11 = m1 - m2 + m3 - m7
    
    Outer product A2*B2:
    t0 = a02*b20
    t1 = a02*b21
    t2 = a12*b20
    t3 = a12*b21
    
    c00 += t0
    c01 += t1
    c10 += t2
    c11 += t3
    
    Total: 11 products.
    
    U matrix (n*m = 2*3 = 6 rows, 11 cols):
    Row index: i*m+j = i*3+j
    U[0][0]=1, U[4][0]=1  (a00, a11 in m1)
    U[3][1]=1, U[4][1]=1  (a10, a11 in m2)
    U[0][2]=1             (a00 in m3)
    U[4][3]=1             (a11 in m4)
    U[0][4]=1, U[1][4]=1  (a00, a01 in m5)
    U[3][5]=1, U[0][5]=-1 (a10, a00 in m6)
    U[1][6]=1, U[4][6]=-1 (a01, a11 in m7)
    U[2][7]=1             (a02 in t0)
    U[2][8]=1             (a02 in t1)
    U[5][9]=1             (a12 in t2)
    U[5][10]=1            (a12 in t3)
    
    V matrix (m*p = 3*2 = 6 rows, 11 cols):
    Row index: j*p+k = j*2+k
    V[0][0]=1, V[1][0]=1  (b00, b11 in m1)
    V[0][1]=1             (b00 in m2)
    V[2][2]=1, V[3][2]=-1 (b10, b11 in m3)
    V[1][3]=1, V[0][3]=-1 (b01, b00 in m4)
    V[3][4]=1             (b11 in m5)
    V[0][5]=1, V[2][5]=1  (b00, b10 in m6)
    V[1][6]=1, V[3][6]=1  (b01, b11 in m7)
    V[4][7]=1             (b20 in t0)
    V[5][8]=1             (b21 in t1)
    V[4][9]=1             (b20 in t2)
    V[5][10]=1            (b21 in t3)
    
    W matrix (p*n = 2*2 = 4 rows, 11 cols):
    Row index: k*n+i = k*2+i
    W[0][0]=1, W[0][3]=1, W[0][4]=-1, W[0][6]=1, W[0][7]=1  (c00)
    W[1][2]=1, W[1][4]=1, W[1][8]=1                          (c01)
    W[2][1]=1, W[2][3]=1, W[2][9]=1                          (c10)
    W[3][0]=1, W[3][1]=-1, W[3][2]=1, W[3][6]=-1, W[3][10]=1 (c11)
    """
    rank = 11
    U = [[0] * rank for _ in range(6)]
    V = [[0] * rank for _ in range(6)]
    W = [[0] * rank for _ in range(4)]
    
    # m1 = (a00+a11)*(b00+b11)
    U[0][0] = 1
    U[4][0] = 1
    V[0][0] = 1
    V[1][0] = 1
    W[0][0] = 1
    W[3][0] = 1
    
    # m2 = (a10+a11)*b00
    U[3][1] = 1
    U[4][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    W[3][1] = -1
    
    # m3 = a00*(b10-b11)
    U[0][2] = 1
    V[2][2] = 1
    V[3][2] = -1
    W[1][2] = 1
    W[3][2] = 1
    
    # m4 = a11*(b01-b00)
    U[4][3] = 1
    V[1][3] = 1
    V[0][3] = -1
    W[0][3] = 1
    W[2][3] = 1
    
    # m5 = (a00+a01)*b11
    U[0][4] = 1
    U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1
    W[1][4] = 1
    
    # m6 = (a10-a00)*(b00+b10)
    U[3][5] = 1
    U[0][5] = -1
    V[0][5] = 1
    V[2][5] = 1
    # m6 doesn't appear in output (Strassen property)
    
    # m7 = (a01-a11)*(b01+b11)
    U[1][6] = 1
    U[4][6] = -1
    V[1][6] = 1
    V[3][6] = 1
    W[0][6] = 1
    W[3][6] = -1
    
    # t0 = a02*b20
    U[2][7] = 1
    V[4][7] = 1
    W[0][7] = 1
    
    # t1 = a02*b21
    U[2][8] = 1
    V[5][8] = 1
    W[1][8] = 1
    
    # t2 = a12*b20
    U[5][9] = 1
    V[4][9] = 1
    W[2][9] = 1
    
    # t3 = a12*b21
    U[5][10] = 1
    V[5][10] = 1
    W[3][10] = 1
    
    return U, V, W


def _verify_2x3x2(U, V, W):
    """Verify rank-11 decomposition for 2x3 * 3x2."""
    n, m, p = 2, 3, 2
    rank = 11
    return _verify(U, V, W, n, m, p, rank)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin

    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # First, verify that our rank-11 2x3x2 construction is correct
    U_block, V_block, W_block = _build_rank11_2x3x2()
    block_valid = _verify_2x3x2(U_block, V_block, W_block)
    
    if not block_valid:
        # Fall back to schoolbook rank 48
        rank = n * m * p  # 48
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
        
        assert _verify(U, V, W, n, m, p, rank)
        return {"U": U, "V": V, "W": W}
    
    # Use the rank-11 block decomposition for 4 blocks of 2x3x2
    # Total rank: 4 * 11 = 44
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 6 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    # The 8 columns of B are split into 4 groups of 2
    # For block t (t=0,1,2,3), we handle columns 2t and 2t+1 of B
    # B rows: j=0,1,2 correspond to b_jk where k=2t or k=2t+1
    
    for t in range(4):
        offset = t * 11
        for d in range(11):
            # U: same for all blocks (depends only on A)
            for a_idx in range(6):
                U[a_idx][offset + d] = U_block[a_idx][d]
            
            # V: depends on which columns of B we're using
            # For block t, B column indices are 2t and 2t+1
            # V_block has rows j*p+k = j*2+k for j=0,1,2 and k=0,1
            # We need to map: in the full problem, B[j][2t+k] corresponds to V_block[j][k]
            for b_idx in range(6):
                # b_idx = j*p + k_full where p=8
                # We want to extract j and k_full, then map to block-local k
                j = b_idx // p
                k_full = b_idx % p
                if k_full == 2 * t or k_full == 2 * t + 1:
                    k_local = k_full - 2 * t  # 0 or 1
                    b_idx_local = j * 2 + k_local
                    V[b_idx][offset + d] = V_block[b_idx_local][d]
                # else: V[b_idx][offset+d] stays 0
            
            # W: depends on which output columns we're using
            # W_block has rows k*n+i = k*2+i for k=0,1 and i=0,1
            # In the full problem, W rows are k_full*n+i where k_full=0,...,7
            # For block t, output columns are 2t and 2t+1
            for c_idx in range(16):
                # c_idx = k_full*n + i
                k_full = c_idx // n
                i = c_idx % n
                if k_full == 2 * t or k_full == 2 * t + 1:
                    k_local = k_full - 2 * t  # 0 or 1
                    c_idx_local = k_local * 2 + i
                    W[c_idx][offset + d] = W_block[c_idx_local][d]
                # else: W[c_idx][offset+d] stays 0
    
    # Verify the full decomposition
    assert _verify(U, V, W, n, m, p, rank)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
