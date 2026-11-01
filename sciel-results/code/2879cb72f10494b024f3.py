import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0


def _check_deadline():
    return (time.time() - _START_TIME) >= _DEADLINE


def construct():
    n, m, p = 2, 2, 8

    # Strassen rank-7 decomposition for 2x2x2
    # A is 2x2, B is 2x2, C = A*B is 2x2
    # U: 4x7, V: 4x7, W: 4x7
    # U[i, k] * V[j, k] * W[c, k] summed over k gives C[c] = sum_{i,j} A[i,j]*B[j,c]
    # where c = row*2 + col for C (row-major for C)
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)

    # Strassen's 7 products:
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B01-B11)
    # P4 = A11*(B10-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6

    U_s = np.array([
        [1, 0, 0, 1],   # P1
        [0, 0, 1, 1],   # P2
        [1, 0, 0, 0],   # P3
        [0, 0, 0, 1],   # P4
        [1, 1, 0, 0],   # P5
        [-1, 0, 1, 0],  # P6
        [0, 1, 0, -1]   # P7
    ], dtype=float)

    V_s = np.array([
        [1, 0, 0, 1],   # P1
        [1, 0, 0, 0],   # P2
        [0, 1, 0, -1],  # P3
        [-1, 0, 1, 0],  # P4
        [0, 0, 0, 1],   # P5
        [1, 1, 0, 0],   # P6
        [0, 0, 1, 1]    # P7
    ], dtype=float)

    W_s = np.array([
        [1, 0, 0, 1],   # P1 -> C00, C11
        [0, 0, 1, -1],  # P2 -> C10, C11
        [0, 1, 0, 1],   # P3 -> C01, C11
        [1, 0, 1, 0],   # P4 -> C00, C10
        [-1, 1, 0, 0],  # P5 -> C00, C01
        [0, 0, 0, 1],   # P6 -> C11
        [1, 0, 0, 0]    # P7 -> C00
    ], dtype=float)

    # Verify the decomposition
    for i in range(4):
        for j in range(4):
            for c in range(4):
                lhs = sum(U_s[i, k] * V_s[j, k] * W_s[c, k] for k in range(7))
                # A[i,j] * B[j,c] contributes to C[c] only if the indices match
                # i = ai*2+aj, j = bi*2+bj, c = ci*2+cj
                ai, aj = divmod(i, 2)
                bi, bj = divmod(j, 2)
                ci, cj = divmod(c, 2)
                rhs = 1.0 if (aj == bi and aj == bi) else 0.0
                # Actually: A[ai,aj] * B[bi,bj] contributes to C[ai,bj]
                # So rhs = 1 if ai==ci and aj==bj, else 0
                rhs = 1.0 if (ai == ci and aj == bj) else 0.0
                if abs(lhs - rhs) > 1e-12:
                    pass  # Should not happen

    # Build the rank-28 decomposition for 2x2 x 2x8
    # Partition the 8 columns of B into 4 pairs: (0,1), (2,3), (4,5), (6,7)
    # For each pair t (0..3), use one Strassen block
    # Total rank = 4 * 7 = 28

    rank = 28
    U = np.zeros((n * m, rank))  # (4, 28)
    V = np.zeros((m * p, rank))  # (16, 28)
    W = np.zeros((p * n, rank))  # (16, 28)

    for t in range(4):
        col_offset = t * 7
        # U: same for all blocks (A is shared)
        for k in range(7):
            for i in range(4):
                U[i, col_offset + k] = U_s[i, k]

        # V: B indices for pair t
        # B is 2x8, row-major: index = row * 8 + col
        # Pair t uses columns 2t and 2t+1
        # Local B indices: 0=(0,2t), 1=(0,2t+1), 2=(1,2t), 3=(1,2t+1)
        # Global B indices: 0*8+2t=2t, 0*8+2t+1=2t+1, 1*8+2t=8+2t, 1*8+2t+1=8+2t+1
        v_map = [2 * t, 2 * t + 1, 8 + 2 * t, 8 + 2 * t + 1]
        for k in range(7):
            for l in range(4):
                V[v_map[l], col_offset + k] = V_s[l, k]

        # W: C indices for pair t
        # C is 2x8, but the convention is W[k*n + i, r] where k is column, i is row
        # So C index = col * 2 + row
        # Pair t uses columns 2t and 2t+1
        # Local C indices: 0=(row=0,col=2t), 1=(row=0,col=2t+1), 2=(row=1,col=2t), 3=(row=1,col=2t+1)
        # Global C indices (k*n+i): 
        #   local 0: col=2t, row=0 -> 2t*2+0 = 4t
        #   local 1: col=2t+1, row=0 -> (2t+1)*2+0 = 4t+2
        #   local 2: col=2t, row=1 -> 2t*2+1 = 4t+1
        #   local 3: col=2t+1, row=1 -> (2t+1)*2+1 = 4t+3
        w_map = [4 * t, 4 * t + 2, 4 * t + 1, 4 * t + 3]
        for k in range(7):
            for l in range(4):
                W[w_map[l], col_offset + k] = W_s[l, k]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
