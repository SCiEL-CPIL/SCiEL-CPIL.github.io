"""Matrix multiplication 2,3,8 rank-minimal exact tensor decomposition."""

import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct rank-minimal exact tensor decomposition for 2x3 * 3x8 multiplication.
    Uses Strassen's algorithm (rank 7) for 2x2x2 subproblems and trivial decomposition
    for 2x1x8 blocks, leveraging additive rank property.
    """
    n, m, p = 2, 3, 8
    rank = 44  # 2*22 (two 2x2x2 blocks via Strassen) + 1*8 (one 2x1x8 trivial)
    # Actually: 2 (j=0,1) * 2 (k-pairs) * 7 (Strassen) = 28, plus 1 (j=2) * 8 * 1 = 8, total 36?
    # Wait, let me recount: for j=0 and j=1, we have 2x2x2 blocks (i=0,1; k in {0,1}, {2,3}, {4,5}, {6,7} → 4 blocks each)
    # Actually: i=0,1 (2 rows), j=0,1 (2 cols of A), k=0..7 (8 cols of B)
    # Split k into 4 pairs: (0,1), (2,3), (4,5), (6,7) → 4 blocks of 2x2x2 for j=0 and j=1
    # That's 2 (j values) * 4 (k-pairs) * 7 (Strassen rank) = 56, too many.
    # Let me reconsider: 2x3x8 = split j into j=0,1 (2x2x8 part) and j=2 (2x1x8 part)
    # 2x2x8: split k into 4 pairs of 2, each 2x2x2 uses Strassen rank 7 → 4*7=28
    # 2x1x8: trivial, rank 8
    # Total: 28+8=36
    rank = 36
    
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 36
    V = [[0.0] * rank for _ in range(m * p)]  # 24 x 36
    W = [[0.0] * rank for _ in range(p * n)]  # 16 x 36
    
    # Strassen 2x2x2 decomposition (rank 7)
    # For multiplying X(2x2) * Y(2x2), C = X*Y
    # U matrices: u_r[i,j] for i,j in {0,1}, r in 0..6
    # V matrices: v_r[j,k] for j,k in {0,1}, r in 0..6  
    # W matrices: w_r[k,i] for k,i in {0,1}, r in 0..6
    # Strassen:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # U[r][i,j], V[r][j,k], W[r][k,i]
    U222 = [
        [[1,0],[1,1]],   # r=0: M1, u11=1,u21=1,u22=1
        [[0,0],[1,1]],   # r=1: M2, u21=1,u22=1
        [[1,0],[0,0]],   # r=2: M3, u11=1
        [[0,0],[0,1]],   # r=3: M4, u22=1
        [[1,1],[0,0]],   # r=4: M5, u11=1,u12=1
        [[-1,-1],[1,1]], # r=5: M6, u11=-1,u12=-1,u21=1,u22=1
        [[0,1],[-1,1]],  # r=6: M7, u12=1,u21=-1,u22=1
    ]
    V222 = [
        [[1,1],[1,1]],   # r=0: b11=1,b12=1,b21=1,b22=1
        [[1,0],[0,0]],   # r=1: b11=1
        [[0,1],[0,-1]],  # r=2: b12=1,b22=-1
        [[-1,0],[1,0]],  # r=3: b11=-1,b21=1
        [[0,0],[0,1]],   # r=4: b22=1
        [[1,1],[0,0]],   # r=5: b11=1,b12=1
        [[0,0],[1,1]],   # r=6: b21=1,b22=1
    ]
    W222 = [
        [[1,-1],[1,1]],   # r=0: C11+=M1, C22+=M1 → w00=1,w11=1; C12-=M1? No...
        # Let me redo W: w[k,i] contributes to C[i,k]
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        # w_r[k,i] where C[i,k] += w_r[k,i]*M_r
    ]
    W222 = [
        [[1,1],[0,0]],   # r=0: M1: C[0,0]+=1, C[1,1]+=1
        [[0,0],[1,-1]],  # r=1: M2: C[1,0]+=1, C[1,1]-=1
        [[1,0],[0,1]],   # r=2: M3: C[0,1]+=1, C[1,1]+=1
        [[-1,0],[1,0]],  # r=3: M4: C[0,0]-=1, C[1,0]+=1
        [[0,-1],[0,1]],  # r=4: M5: C[0,0]-=1, C[0,1]+=1
        [[0,0],[0,1]],   # r=5: M6: C[1,1]+=1
        [[1,1],[0,0]],   # r=6: M7: C[0,0]+=1, C[1,0]+=1
    ]
    
    col = 0
    # For j=0 and j=1, split k into 4 pairs
    for j in range(2):
        for kp in range(4):
            k_offset = kp * 2
            for r in range(7):
                # U[i*m+j, col] = U222[r][i][j_local] where j_local is always 0 since this is the j-th column
                # Actually U222[r][i][col_idx] where col_idx represents the position within the 2x2 block
                for i in range(2):
                    U[i*m+j][col] = U222[r][i][0]  # j_local=0 for this sub-block
                for k_local in range(2):
                    for j_local in range(2):
                        pass
                # V[j*p+k, col] = V222[r][j_local][k_local]
                for j_local in range(2):
                    for k_local in range(2):
                        V[j*p+(k_offset+k_local)][col] = V222[r][j_local][k_local]
                # W[k*n+i, col] = W222[r][k_local][i]
                for k_local in range(2):
                    for i in range(2):
                        W[(k_offset+k_local)*n+i][col] = W222[r][k_local][i]
                col += 1
    
    # For j=2, trivial decomposition for 2x1x8
    for k in range(8):
        for i in range(2):
            U[i*m+2][col] = 1  # U[i*3+2, col]
        V[2*p+k][col] = 1  # V[2*8+k, col] = V[16+k, col]
        W[k*n+i][col] = 1  # W[k*2+i, col]
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
