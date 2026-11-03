"""Matrix multiplication 4x5 by 5x6 - rank-minimal exact tensor decomposition."""
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """
    Build the 2x2 Strassen decomposition.
    Returns (U, V, W) where:
      U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_strassen_2x3x3():
    """
    Build a 2x3x3 decomposition using 3 Strassen 2x2x2 blocks + 1 rank-1 block.
    A is 2x3, B is 3x3, C is 2x3.
    Rank = 3*7 + 1 = 22.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 22
    U = [[0]*rank for _ in range(6)]   # 2*3 = 6 rows
    V = [[0]*rank for _ in range(9)]   # 3*3 = 9 rows
    W = [[0]*rank for _ in range(6)]   # 3*2 = 6 rows (p*n = 3*2)
    
    # Block 1: A[:, 0:2] * B[0:2, :] -> C[:, 0:2]
    # This is a 2x2x2 block
    for t in range(7):
        col = t
        for li in range(2):
            for lj in range(2):
                local_a_row = li * 2 + lj
                coeff_a = U2[local_a_row][t]
                if coeff_a != 0:
                    global_i = li
                    global_j = lj
                    u_row = global_i * 3 + global_j
                    U[u_row][col] += coeff_a
        
        for lj in range(2):
            for lk in range(2):
                local_b_row = lj * 2 + lk
                coeff_b = V2[local_b_row][t]
                if coeff_b != 0:
                    global_j = lj
                    global_k = lk
                    v_row = global_j * 3 + global_k
                    V[v_row][col] += coeff_b
        
        for lk in range(2):
            for li in range(2):
                local_c_row = lk * 2 + li
                coeff_c = W2[local_c_row][t]
                if coeff_c != 0:
                    global_k = lk
                    global_i = li
                    w_row = global_k * 2 + global_i
                    W[w_row][col] += coeff_c
    
    # Block 2: A[:, 1:3] * B[1:3, :] -> C[:, 1:3]
    # This is a 2x2x2 block
    for t in range(7):
        col = 7 + t
        for li in range(2):
            for lj in range(2):
                local_a_row = li * 2 + lj
                coeff_a = U2[local_a_row][t]
                if coeff_a != 0:
                    global_i = li
                    global_j = 1 + lj
                    u_row = global_i * 3 + global_j
                    U[u_row][col] += coeff_a
        
        for lj in range(2):
            for lk in range(2):
                local_b_row = lj * 2 + lk
                coeff_b = V2[local_b_row][t]
                if coeff_b != 0:
                    global_j = 1 + lj
                    global_k = 1 + lk
                    v_row = global_j * 3 + global_k
                    V[v_row][col] += coeff_b
        
        for lk in range(2):
            for li in range(2):
                local_c_row = lk * 2 + li
                coeff_c = W2[local_c_row][t]
                if coeff_c != 0:
                    global_k = 1 + lk
                    global_i = li
                    w_row = global_k * 2 + global_i
                    W[w_row][col] += coeff_c
    
    # Block 3: A[:, 2] * B[2, :] -> C[:, 2]
    # This is a 2x1x1 block, rank 1
    col = 14
    for li in range(2):
        U[li * 3 + 2][col] = 1
    V[2 * 3 + 2][col] = 1
    for li in range(2):
        W[2 * 2 + li][col] = 1
    
    return U, V, W

def _build_strassen_4x3x3():
    """
    Build a 4x3x3 decomposition using 6 Strassen 2x2x2 blocks.
    A is 4x3, B is 3x3, C is 4x3.
    Rank = 6*7 = 42.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 42
    U = [[0]*rank for _ in range(12)]  # 4*3 = 12 rows
    V = [[0]*rank for _ in range(9)]   # 3*3 = 9 rows
    W = [[0]*rank for _ in range(12)]  # 3*4 = 12 rows (p*n = 3*4)
    
    # 6 blocks: 2x2x2 each
    # Block (bi, bj): A[2*bi:2*bi+2, 2*bj:2*bj+2] * B[2*bj:2*bj+2, :] -> C[2*bi:2*bi+2, 2*bj:2*bj+2]
    # But B and C have dimension 3, so we need to be careful.
    # Let's use a different approach: split into 2x2x2 blocks where possible.
    # For 4x3x3, we can do:
    # - 2 blocks of 2x2x2 (covering A[0:2,0:2]*B[0:2,0:2] -> C[0:2,0:2])
    # - 2 blocks of 2x2x2 (covering A[2:4,0:2]*B[0:2,0:2] -> C[2:4,0:2])
    # - 2 blocks of 2x2x2 (covering A[0:2,2:3]*B[2:3,2:3] -> C[0:2,2:3]) -- but this is 2x1x1
    # Actually, let's just use schoolbook for this one to keep it simple.
    
    # Use schoolbook for 4x3x3: rank = 4*3*3 = 36
    rank = 36
    U = [[0]*rank for _ in range(12)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(12)]
    
    col = 0
    for i in range(4):
        for j in range(3):
            for k in range(3):
                U[i*3 + j][col] = 1
                V[j*3 + k][col] = 1
                W[k*4 + i][col] = 1
                col += 1
    
    return U, V, W

def _build_strassen_4x2x3():
    """
    Build a 4x2x3 decomposition using 4 Strassen 2x2x2 blocks.
    A is 4x2, B is 2x3, C is 4x3.
    Rank = 4*7 = 28.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 28
    U = [[0]*rank for _ in range(8)]   # 4*2 = 8 rows
    V = [[0]*rank for _ in range(6)]   # 2*3 = 6 rows
    W = [[0]*rank for _ in range(12)]  # 3*4 = 12 rows (p*n = 3*4)
    
    # 4 blocks: 2x2x2 each
    # Block (bi): A[2*bi:2*bi+2, :] * B[:, :] -> C[2*bi:2*bi+2, :]
    for bi in range(2):
        for t in range(7):
            col = bi * 14 + t  # Wait, each block should have rank 7, but we need to handle the k dimension
            
            # Actually, let's think about this differently.
            # A is 4x2, B is 2x3, C is 4x3.
            # We can split into 2 blocks of 2x2x3.
            # Each 2x2x3 can be done with 2 Strassen 2x2x2 blocks + 1 rank-1 block.
            # So total rank = 2 * (2*7 + 1) = 30.
            
            # Let's just use schoolbook for simplicity.
            pass
    
    # Use schoolbook for 4x2x3: rank = 4*2*3 = 24
    rank = 24
    U = [[0]*rank for _ in range(8)]
    V = [[0]*rank for _ in range(6)]
    W = [[0]*rank for _ in range(12)]
    
    col = 0
    for i in range(4):
        for j in range(2):
            for k in range(3):
                U[i*2 + j][col] = 1
                V[j*3 + k][col] = 1
                W[k*4 + i][col] = 1
                col += 1
    
    return U, V, W

def _build_strassen_2x2x3():
    """
    Build a 2x2x3 decomposition using 2 Strassen 2x2x2 blocks + 1 rank-1 block.
    A is 2x2, B is 2x3, C is 2x3.
    Rank = 2*7 + 1 = 15.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 15
    U = [[0]*rank for _ in range(4)]   # 2*2 = 4 rows
    V = [[0]*rank for _ in range(6)]   # 2*3 = 6 rows
    W = [[0]*rank for _ in range(6)]   # 3*2 = 6 rows (p*n = 3*2)
    
    # Block 1: A * B[:, 0:2] -> C[:, 0:2] (2x2x2)
    for t in range(7):
        col = t
        for li in range(2):
            for lj in range(2):
                local_a_row = li * 2 + lj
                coeff_a = U2[local_a_row][t]
                if coeff_a != 0:
                    global_i = li
                    global_j = lj
                    u_row = global_i * 2 + global_j
                    U[u_row][col] += coeff_a
        
        for lj in range(2):
            for lk in range(2):
                local_b_row = lj * 2 + lk
                coeff_b = V2[local_b_row][t]
                if coeff_b != 0:
                    global_j = lj
                    global_k = lk
                    v_row = global_j * 3 + global_k
                    V[v_row][col] += coeff_b
        
        for lk in range(2):
            for li in range(2):
                local_c_row = lk * 2 + li
                coeff_c = W2[local_c_row][t]
                if coeff_c != 0:
                    global_k = lk
                    global_i = li
                    w_row = global_k * 2 + global_i
                    W[w_row][col] += coeff_c
    
    # Block 2: A * B[:, 2] -> C[:, 2] (2x1x1, rank 1)
    col = 14
    for li in range(2):
        for lj in range(2):
            U[li * 2 + lj][col] = 1
    V[2 * 3 + 2][col] = 1
    for li in range(2):
        W[2 * 2 + li][col] = 1
    
    return U, V, W

def _build_strassen_4x2x2():
    """
    Build a 4x2x2 decomposition using 4 Strassen 2x2x2 blocks.
    A is 4x2, B is 2x2, C is 4x2.
    Rank = 4*7 = 28.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 28
    U = [[0]*rank for _ in range(8)]   # 4*2 = 8 rows
    V = [[0]*rank for _ in range(4)]   # 2*2 = 4 rows
    W = [[0]*rank for _ in range(8)]   # 2*4 = 8 rows (p*n = 2*4)
    
    # 4 blocks: 2x2x2 each
    for bi in range(2):
        for t in range(7):
            col = bi * 14 + t
            
            for li in range(2):
                for lj in range(2):
                    local_a_row = li * 2 + lj
                    coeff_a = U2[local_a_row][t]
                    if coeff_a != 0:
                        global_i = 2*bi + li
                        global_j = lj
                        u_row = global_i * 2 + global_j
                        U[u_row][col] += coeff_a
            
            for lj in range(2):
                for lk in range(2):
                    local_b_row = lj * 2 + lk
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        global_j = lj
                        global_k = lk
                        v_row = global_j * 2 + global_k
                        V[v_row][col] += coeff_b
            
            for lk in range(2):
                for li in range(2):
                    local_c_row = lk * 2 + li
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        global_k = lk
                        global_i = 2*bi + li
                        w_row = global_k * 4 + global_i
                        W[w_row][col] += coeff_c
    
    return U, V, W

def _build_strassen_4x3x2():
    """
    Build a 4x3x2 decomposition using 6 Strassen 2x2x2 blocks.
    A is 4x3, B is 3x2, C is 4x2.
    Rank = 6*7 = 42.
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 42
    U = [[0]*rank for _ in range(12)]  # 4*3 = 12 rows
    V = [[0]*rank for _ in range(6)]   # 3*2 = 6 rows
    W = [[0]*rank for _ in range(8)]   # 2*4 = 8 rows (p*n = 2*4)
    
    # 6 blocks: 2x2x2 each
    # Block (bi, bj): A[2*bi:2*bi+2, 2*bj:2*bj+2] * B[2*bj:2*bj+2, :] -> C[2*bi:2*bi+2, 2*bj:2*bj+2]
    for bi in range(2):
        for bj in range(2):
            for t in range(7):
                col = (bi * 2 + bj) * 7
                
                for li in range(2):
                    for lj in range(2):
                        local_a_row = li * 2 + lj
                        coeff_a = U2[local_a_row][t]
                        if coeff_a != 0:
                            global_i = 2*bi + li
                            global_j = 2*bj + lj
                            u_row = global_i * 3 + global_j
                            U[u_row][col] += coeff_a
                
                for lj in range(2):
                    for lk in range(2):
                        local_b_row = lj * 2 + lk
                        coeff_b = V2[local_b_row][t]
                        if coeff_b != 0:
                            global_j = 2*bj + lj
                            global_k = lk
                            v_row = global_j * 2 + global_k
                            V[v_row][col] += coeff_b
                
                for lk in range(2):
                    for li in range(2):
                        local_c_row = lk * 2 + li
                        coeff_c = W2[local_c_row][t]
                        if coeff_c != 0:
                            global_k = lk
                            global_i = 2*bi + li
                            w_row = global_k * 4 + global_i
                            W[w_row][col] += coeff_c
    
    return U, V, W

def _build_strassen_4x5x6():
    """
    Build a 4x5x6 decomposition using a combination of Strassen blocks.
    A is 4x5, B is 5x6, C is 4x6.
    
    Strategy: Split m=5 into {0,1} and {2,3,4}.
    - Block 1: A[:, 0:2] * B[0:2, :] -> C[:, :] (4x2x6)
    - Block 2: A[:, 2:5] * B[2:5, :] -> C[:, :] (4x3x6)
    
    For 4x2x6:
    - Split k=6 into {0,1}, {2,3}, {4,5}
    - 3 blocks of 4x2x2 (rank 28 each) = 84
    
    For 4x3x6:
    - Split k=6 into {0,1}, {2,3}, {4,5}
    - 3 blocks of 4x3x2 (rank 42 each) = 126
    
    Total rank: 84 + 126 = 210. This is worse than schoolbook (120).
    
    Let's try a different approach:
    - Split n=4 into {0,1} and {2,3}
    - Block 1: A[0:2, :] * B[:, 0:3] -> C[0:2, 0:3] (2x5x3)
    - Block 2: A[0:2, :] * B[:, 3:6] -> C[0:2, 3:6] (2x5x3)
    - Block 3: A[2:4, :] * B[:, 0:3] -> C[2:4, 0:3] (2x5x3)
    - Block 4: A[2:4, :] * B[:, 3:6] -> C[2:4, 3:6] (2x5x3)
    
    For 2x5x3:
    - Split m=5 into {0,1}, {2,3}, {4}
    - Block A: A[:, 0:2] * B[0:2, :] -> C[:, :] (2x2x3, rank 15)
    - Block B: A[:, 2:4] * B[2:4, :] -> C[:, :] (2x2x3, rank 15)
    - Block C: A[:, 4] * B[4, :] -> C[:, :] (2x1x3, rank 3)
    Total for 2x5x3: 15 + 15 + 3 = 33
    
    Total rank: 4 * 33 = 132. Still worse than schoolbook (120).
    
    Let's try yet another approach:
    - Use Strassen for the 2x2x2 subblocks where possible.
    - For 4x5x6, we can decompose into:
      - 2 blocks of 2x5x6 (one for rows 0-1, one for rows 2-3)
    
    For 2x5x6:
    - Split m=5 into {0,1}, {2,3}, {4}
    - Block A: A[:, 0:2] * B[0:2, :] -> C[:, :] (2x2x6)
    - Block B: A[:, 2:4] * B[2:4, :] -> C[:, :] (2x2x6)
    - Block C: A[:, 4] * B[4, :] -> C[:, :] (2x1x6)
    
    For 2x2x6:
    - Split k=6 into {0,1}, {2,3}, {4,5}
    - 3 blocks of 2x2x2 (rank 7 each) = 21
    
    For 2x1x6:
    - Split k=6 into {0,1}, {2,3}, {4,5}
    - 3 blocks of 2x1x2 (rank 2 each) = 6
    
    Total for 2x5x6: 21 + 21 + 6 = 48
    Total rank: 2 * 48 = 96
    
    This is better than schoolbook (120)!
    """
    U2, V2, W2 = _build_strassen_2x2()
    
    rank = 96
    U = [[0]*rank for _ in range(20)]  # 4*5 = 20 rows
    V = [[0]*rank for _ in range(30)]  # 5*6 = 30 rows
    W = [[0]*rank for _ in range(24)]  # 6*4 = 24 rows (p*n = 6*4)
    
    def add_2x2x2_block(a_i_start, b_j_start, c_k_start, base_rank):
        for t in range(7):
            col = base_rank + t
            
            for li in range(2):
                for lj in range(2):
                    local_a_row = li * 2 + lj
                    coeff_a = U2[local_a_row][t]
                    if coeff_a != 0:
                        global_i = a_i_start + li
                        global_j = b_j_start + lj
                        u_row = global_i * 5 + global_j
                        U[u_row][col] += coeff_a
            
            for lj in range(2):
                for lk in range(2):
                    local_b_row = lj * 2 + lk
                    coeff_b = V2[local_b_row][t]
                    if coeff_b != 0:
                        global_j = b_j_start + lj
                        global_k = c_k_start + lk
                        v_row = global_j * 6 + global_k
                        V[v_row][col] += coeff_b
            
            for lk in range(2):
                for li in range(2):
                    local_c_row = lk * 2 + li
                    coeff_c = W2[local_c_row][t]
                    if coeff_c != 0:
                        global_k = c_k_start + lk
                        global_i = a_i_start + li
                        w_row = global_k * 4 + global_i
                        W[w_row][col] += coeff_c
    
    def add_2x1x2_block(a_i_start, b_j_start, c_k_start, base_rank):
        col = base_rank
        for li in range(2):
            U[(a_i_start + li) * 5 + b_j_start][col] = 1
        V[b_j_start * 6 + c_k_start][col] = 1
        for li in range(2):
            W[(c_k_start) * 4 + (a_i_start + li)][col] = 1
        col = base_rank + 1
        for li in range(2):
            U[(a_i_start + li) * 5 + b_j_start][col] = 1
        V[b_j_start * 6 + (c_k_start + 1)][col] = 1
        for li in range(2):
            W[(c_k_start + 1) * 4 + (a_i_start + li)][col] = 1
    
    # Row block 0: A[0:2, :] * B[:, :] -> C[0:2, :]
    # Block A: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_2x2x2_block(0, 0, 0, 0)
    # Block B: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_2x2x2_block(0, 0, 2, 7)
    # Block C: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    add_2x2x2_block(0, 0, 4, 14)
    # Block D: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_2x2x2_block(0, 2, 0, 21)
    # Block E: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_2x2x2_block(0, 2, 2, 28)
    # Block F: A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    add_2x2x2_block(0, 2, 4, 35)
    # Block G: A[0:2, 4] * B[4, 0:2] -> C[0:2, 0:2]
    add_2x1x2_block(0, 4, 0, 42)
    # Block H: A[0:2, 4] * B[4, 2:4] -> C[0:2, 2:4]
    add_2x1x2_block(0, 4, 2, 44)
    # Block I: A[0:2, 4] * B[4, 4:6] -> C[0:2, 4:6]
    add_2x1x2_block(0, 4, 4, 46)
    
    # Row block 1: A[2:4, :] * B[:, :] -> C[2:4, :]
    # Block J: A[2:4, 0:2] * B[0:2, 0:2] -> C[2:4, 0:2]
    add_2x2x2_block(2, 0, 0, 48)
    # Block K: A[2:4, 0:2] * B[0:2, 2:4] -> C[2:4, 2:4]
    add_2x2x2_block(2, 0, 2, 55)
    # Block L: A[2:4, 0:2] * B[0:2, 4:6] -> C[2:4, 4:6]
    add_2x2x2_block(2, 0, 4, 62)
    # Block M: A[2:4, 2:4] * B[2:4, 0:2] -> C[2:4, 0:2]
    add_2x2x2_block(2, 2, 0, 69)
    # Block N: A[2:4, 2:4] * B[2:4, 2:4] -> C[2:4, 2:4]
    add_2x2x2_block(2, 2, 2, 76)
    # Block O: A[2:4, 2:4] * B[2:4, 4:6] -> C[2:4, 4:6]
    add_2x2x2_block(2, 2, 4, 83)
    # Block P: A[2:4, 4] * B[4, 0:2] -> C[2:4, 0:2]
    add_2x1x2_block(2, 4, 0, 90)
    # Block Q: A[2:4, 4] * B[4, 2:4] -> C[2:4, 2:4]
    add_2x1x2_block(2, 4, 2, 92)
    # Block R: A[2:4, 4] * B[4, 4:6] -> C[2:4, 4:6]
    add_2x1x2_block(2, 4, 4, 94)
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """
    Build schoolbook decomposition for n x m * m x p -> n x p.
    Returns U, V, W with rank n*m*p.
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    try:
        # Try the Strassen-based decomposition first
        U_strassen, V_strassen, W_strassen = _build_strassen_4x5x6()
        rank_strassen = len(U_strassen[0])
        
        # Verify the decomposition with random matrices
        import numpy as np
        A_test = np.random.randint(-5, 5, (n, m))
        B_test = np.random.randint(-5, 5, (m, p))
        C_expected = A_test @ B_test
        
        U_arr = np.array(U_strassen)
        V_arr = np.array(V_strassen)
        W_arr = np.array(W_strassen)
        
        C_decomp = np.zeros((n, p))
        for r in range(rank_strassen):
            term_A = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    term_A[i] += U_arr[i*m+j, r] * A_test[i, j]
            
            term_B = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    term_B[k] += V_arr[j*p+k, r] * B_test[j, k]
            
            for i in range(n):
                for k in range(p):
                    C_decomp[i, k] += term_A[i] * term_B[k] * W_arr[k*n+i, r]
        
        if np.allclose(C_decomp, C_expected, atol=1e-9):
            return {"U": U_strassen, "V": V_strassen, "W": W_strassen}
        
        # Fallback to schoolbook
        U_fb, V_fb, W_fb = _build_schoolbook(n, m, p)
        return {"U": U_fb, "V": V_fb, "W": W_fb}
        
    except Exception:
        # Fallback to schoolbook if something goes wrong
        U_fb, V_fb, W_fb = _build_schoolbook(n, m, p)
        return {"U": U_fb, "V": V_fb, "W": W_fb}

# EVOLVE-BLOCK-END
