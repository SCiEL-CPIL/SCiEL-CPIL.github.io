"""Matrix multiplication 3,4,8 rank-minimal decomposition."""
DIMENSIONS = [3, 4, 8]

import time

def _strassen_2x2x2():
    """
    Returns (U, V, W) for 2x2x2 matrix multiplication with rank 7.
    """
    U = [[0.0]*7 for _ in range(4)]
    V = [[0.0]*7 for _ in range(4)]
    W = [[0.0]*7 for _ in range(4)]
    
    # M1: (A11 + A22)(B11 + B22)
    U[0][0] = 1.0; U[3][0] = 1.0
    V[0][0] = 1.0; V[3][0] = 1.0
    W[0][0] = 1.0; W[3][0] = 1.0
    
    # M2: (A21 + A22) * B11
    U[2][1] = 1.0; U[3][1] = 1.0
    V[0][1] = 1.0
    W[1][1] = 1.0
    
    # M3: A11 * (B12 - B22)
    U[0][2] = 1.0
    V[1][2] = 1.0; V[3][2] = -1.0
    W[2][2] = 1.0
    
    # M4: A22 * (B21 - B11)
    U[3][3] = 1.0
    V[2][3] = 1.0; V[0][3] = -1.0
    W[0][3] = 1.0; W[1][3] = 1.0
    
    # M5: (A11 + A12) * B22
    U[0][4] = 1.0; U[1][4] = 1.0
    V[3][4] = 1.0
    W[0][4] = -1.0; W[2][4] = 1.0
    
    # M6: (A21 - A11) * (B11 + B12)
    U[2][5] = 1.0; U[0][5] = -1.0
    V[0][5] = 1.0; V[1][5] = 1.0
    W[3][5] = 1.0
    
    # M7: (A12 - A22) * (B21 + B22)
    U[1][6] = 1.0; U[3][6] = -1.0
    V[2][6] = 1.0; V[3][6] = 1.0
    W[0][6] = 1.0; W[3][6] = 1.0
    
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    n, m, p = 3, 4, 8
    rank = 88
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 1x4x8 (Rank 32)
    # A is 1x4, B is 4x8, C is 1x8
    # C_k = sum_j A_j B_jk
    for j in range(4):
        for k in range(8):
            # U row: i=0, j=j -> index j
            U[j][col] = 1.0
            # V row: j=j, k=k -> index j*8+k
            V[j * 8 + k][col] = 1.0
            # W row: k=k, i=0 -> index k*3+0
            W[k * 3][col] = 1.0
            col += 1
    
    # Part 2: 2x4x8 (Rank 56)
    # A is 2x4 (rows 1,2), B is 4x8, C is 2x8 (rows 1,2)
    # Split A into A1 (cols 0,1) and A2 (cols 2,3)
    # Split B into B1 (rows 0,1) and B2 (rows 2,3)
    
    # Subpart 2a: A1 (2x2) * B1 (2x8)
    # A1 is A[1:3, 0:2]
    # B1 is B[0:2, :]
    # Split B1 into 4 blocks of 2x2
    sU, sV, sW = _strassen_2x2x2()
    
    for block_m in range(4):
        # B block is B[0:2, block_m*2 : block_m*2+2]
        for t in range(7):
            # U part: A1 rows i'=0,1; cols j'=0,1
            for ip in range(2):
                for jp in range(2):
                    # Local U index: ip*2+jp
                    val = sU[ip * 2 + jp][t]
                    if val != 0:
                        # Global i = 1 + ip, j = jp
                        U[(1 + ip) * 4 + jp][col] = val
            # V part: B1 rows j'=0,1; cols k'=0,1
            for jp in range(2):
                for kp in range(2):
                    # Local V index: jp*2+kp
                    val = sV[jp * 2 + kp][t]
                    if val != 0:
                        # Global j = jp, k = block_m*2 + kp
                        V[jp * 8 + (block_m * 2 + kp)][col] = val
            # W part: C rows i'=0,1; cols k'=0,1
            for kp in range(2):
                for ip in range(2):
                    # Local W index: kp*2+ip
                    val = sW[kp * 2 + ip][t]
                    if val != 0:
                        # Global i = 1 + ip, k = block_m*2 + kp
                        W[(block_m * 2 + kp) * 3 + (1 + ip)][col] = val
            col += 1
    
    # Subpart 2b: A2 (2x2) * B2 (2x8)
    # A2 is A[1:3, 2:4]
    # B2 is B[2:4, :]
    # Split B2 into 4 blocks of 2x2
    
    for block_m in range(4):
        # B block is B[2:4, block_m*2 : block_m*2+2]
        for t in range(7):
            # U part: A2 rows i'=0,1; cols j'=0,1
            for ip in range(2):
                for jp in range(2):
                    # Local U index: ip*2+jp
                    val = sU[ip * 2 + jp][t]
                    if val != 0:
                        # Global i = 1 + ip, j = 2 + jp
                        U[(1 + ip) * 4 + (2 + jp)][col] = val
            # V part: B2 rows j'=0,1; cols k'=0,1
            for jp in range(2):
                for kp in range(2):
                    # Local V index: jp*2+kp
                    val = sV[jp * 2 + kp][t]
                    if val != 0:
                        # Global j = 2 + jp, k = block_m*2 + kp
                        V[(2 + jp) * 8 + (block_m * 2 + kp)][col] = val
            # W part: C rows i'=0,1; cols k'=0,1
            for kp in range(2):
                for ip in range(2):
                    # Local W index: kp*2+ip
                    val = sW[kp * 2 + ip][t]
                    if val != 0:
                        # Global i = 1 + ip, k = block_m*2 + kp
                        W[(block_m * 2 + kp) * 3 + (1 + ip)][col] = val
            col += 1
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
