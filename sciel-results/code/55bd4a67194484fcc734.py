import time
import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # 4, 4, 6

    # Winograd rank-7 decomposition for 2x2*2x2
    # Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    U2 = [[0.0] * 7 for _ in range(4)]
    V2 = [[0.0] * 7 for _ in range(4)]
    W2 = [[0.0] * 7 for _ in range(4)]
    # term 0: (a00+a11)*(b00+b11) -> contributes to c00, c11
    U2[0][0] = 1.0; U2[3][0] = 1.0
    V2[0][0] = 1.0; V2[3][0] = 1.0
    W2[0][0] = 1.0; W2[3][0] = 1.0
    # term 1: (a10+a11)*b00 -> c10
    U2[2][1] = 1.0; U2[3][1] = 1.0
    V2[0][1] = 1.0
    W2[2][1] = 1.0
    # term 2: a00*(b01-b11) -> c01
    U2[0][2] = 1.0
    V2[1][2] = 1.0; V2[3][2] = -1.0
    W2[1][2] = 1.0
    # term 3: a11*(b10-b00) -> c10
    U2[3][3] = 1.0
    V2[2][3] = 1.0; V2[0][3] = -1.0
    W2[2][3] = 1.0
    # term 4: (a00+a01)*b11 -> c00(-), c01
    U2[0][4] = 1.0; U2[1][4] = 1.0
    V2[3][4] = 1.0
    W2[0][4] = -1.0; W2[1][4] = 1.0
    # term 5: (a10-a00)*(b00+b01) -> c11
    U2[2][5] = 1.0; U2[0][5] = -1.0
    V2[0][5] = 1.0; V2[1][5] = 1.0
    W2[3][5] = 1.0
    # term 6: (a01-a11)*(b10+b11) -> c00, c11
    U2[1][6] = 1.0; U2[3][6] = -1.0
    V2[2][6] = 1.0; V2[3][6] = 1.0
    W2[0][6] = 1.0; W2[3][6] = 1.0

    def verify_2x2(U2, V2, W2):
        """Verify the 2x2x2 decomposition."""
        Ua = np.array(U2, dtype=float)
        Va = np.array(V2, dtype=float)
        Wa = np.array(W2, dtype=float)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0.0
                    for c in range(7):
                        s += Ua[i*2+j, c] * Va[j*2+k, c] * Wa[k*2+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Verify our Winograd kernel
    assert verify_2x2(U2, V2, W2), "Winograd 2x2x2 kernel failed verification"

    # Build 4x4x4 decomposition using 8 Winograd 2x2x2 blocks
    # A is 4x4, B is 4x4, C is 4x4
    # Block structure:
    # C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    # C[0:2,2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
    # C[2:4,0:2] = A[2:4,0:2]*B[0:2,0:2] + A[2:4,2:4]*B[2:4,0:2]
    # C[2:4,2:4] = A[2:4,0:2]*B[0:2,2:4] + A[2:4,2:4]*B[2:4,2:4]
    
    rank_444 = 8 * 7  # 8 blocks, each rank 7
    U444 = [[0.0] * rank_444 for _ in range(16)]  # 4*4
    V444 = [[0.0] * rank_444 for _ in range(16)]  # 4*4
    W444 = [[0.0] * rank_444 for _ in range(16)]  # 4*4

    col = 0
    # For each output block (oi, oj) in {0,1}^2 and each sum block (sj) in {0,1}
    for oi in range(2):  # output row block
        for oj in range(2):  # output col block
            for sj in range(2):  # sum block
                # A block: rows [oi*2:(oi+1)*2], cols [sj*2:(sj+1)*2]
                # B block: rows [sj*2:(sj+1)*2], cols [oj*2:(oj+1)*2]
                # C block: rows [oi*2:(oi+1)*2], cols [oj*2:(oj+1)*2]
                for c in range(7):
                    # U: A[i,j] where i in [oi*2:(oi+1)*2], j in [sj*2:(sj+1)*2]
                    for di in range(2):
                        for dj in range(2):
                            i_global = oi*2 + di
                            j_global = sj*2 + dj
                            U444[i_global*4 + j_global][col] = U2[di*2 + dj][c]
                    
                    # V: B[j,k] where j in [sj*2:(sj+1)*2], k in [oj*2:(oj+1)*2]
                    for dj in range(2):
                        for dk in range(2):
                            j_global = sj*2 + dj
                            k_global = oj*2 + dk
                            V444[j_global*4 + k_global][col] = V2[dj*2 + dk][c]
                    
                    # W: C[i,k] where i in [oi*2:(oi+1)*2], k in [oj*2:(oj+1)*2]
                    for di in range(2):
                        for dk in range(2):
                            i_global = oi*2 + di
                            k_global = oj*2 + dk
                            W444[k_global*4 + i_global][col] = W2[dk*2 + di][c]
                    col += 1
    assert col == rank_444

    # Build 4x4x2 decomposition using 4 Winograd 2x2x2 blocks (adapted)
    # A is 4x4, B is 4x2, C is 4x2
    # C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    # C[2:4,0:2] = A[2:4,0:2]*B[0:2,0:2] + A[2:4,2:4]*B[2:4,0:2]
    # But B is 4x2, so B[0:2,0:2] is 2x2, B[2:4,0:2] is 2x2
    
    rank_442 = 4 * 7  # 4 blocks, each rank 7
    U442 = [[0.0] * rank_442 for _ in range(16)]  # 4*4
    V442 = [[0.0] * rank_442 for _ in range(8)]   # 4*2
    W442 = [[0.0] * rank_442 for _ in range(8)]   # 2*4

    col = 0
    for oi in range(2):  # output row block
        for sj in range(2):  # sum block
            # A block: rows [oi*2:(oi+1)*2], cols [sj*2:(sj+1)*2]
            # B block: rows [sj*2:(sj+1)*2], cols [0:2]
            # C block: rows [oi*2:(oi+1)*2], cols [0:2]
            for c in range(7):
                for di in range(2):
                    for dj in range(2):
                        i_global = oi*2 + di
                        j_global = sj*2 + dj
                        U442[i_global*4 + j_global][col] = U2[di*2 + dj][c]
                
                for dj in range(2):
                    for dk in range(2):
                        j_global = sj*2 + dj
                        k_global = dk  # B has only 2 columns
                        V442[j_global*2 + k_global][col] = V2[dj*2 + dk][c]
                
                for di in range(2):
                    for dk in range(2):
                        i_global = oi*2 + di
                        k_global = dk
                        W442[k_global*4 + i_global][col] = W2[dk*2 + di][c]
                col += 1
    assert col == rank_442

    # Combine 4x4x4 and 4x4x2 into full 4x4x6
    # The first 4 columns of B correspond to 4x4x4
    # The last 2 columns of B correspond to 4x4x2
    total_rank = rank_444 + rank_442  # 56 + 28 = 84
    
    U_full = [[0.0] * total_rank for _ in range(16)]  # 4*4
    V_full = [[0.0] * total_rank for _ in range(24)]  # 4*6
    W_full = [[0.0] * total_rank for _ in range(24)]  # 6*4

    # Copy 4x4x4 part (columns 0-3 of B, columns 0-3 of C)
    for i in range(16):
        for c in range(rank_444):
            U_full[i][c] = U444[i][c]
    
    for j in range(4):  # j index in B (0-3)
        for k in range(4):  # k index in B (0-3)
            for c in range(rank_444):
                V_full[j*6 + k][c] = V444[j*4 + k][c]
    
    for k in range(4):  # k index in C (0-3)
        for i in range(4):  # i index in C (0-3)
            for c in range(rank_444):
                W_full[k*4 + i][c] = W444[k*4 + i][c]

    # Copy 4x4x2 part (columns 4-5 of B, columns 4-5 of C)
    for i in range(16):
        for c in range(rank_442):
            U_full[i][rank_444 + c] = U442[i][c]
    
    for j in range(4):  # j index in B (0-3)
        for k in range(2):  # k index in B (4-5)
            for c in range(rank_442):
                V_full[j*6 + (4 + k)][rank_444 + c] = V442[j*2 + k][c]
    
    for k in range(2):  # k index in C (4-5)
        for i in range(4):  # i index in C (0-3)
            for c in range(rank_442):
                W_full[(4 + k)*4 + i][rank_444 + c] = W442[k*4 + i][c]

    # Verify the full decomposition
    Ua = np.array(U_full, dtype=float)
    Va = np.array(V_full, dtype=float)
    Wa = np.array(W_full, dtype=float)
    
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(6):
                s = 0.0
                for c in range(total_rank):
                    s += Ua[i*4+j, c] * Va[j*6+k, c] * Wa[k*4+i, c]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U_full, "V": V_full, "W": W_full}
