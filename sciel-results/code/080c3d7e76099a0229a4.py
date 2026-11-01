import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Strassen 2x2x2 decomposition
    # Local indices: A: 0=a00, 1=a01, 2=a10, 3=a11
    #                B: 0=b00, 1=b01, 2=b10, 3=b11
    #                C: 0=c00, 1=c01, 2=c10, 3=c11
    U_loc = np.zeros((4, 7))
    V_loc = np.zeros((4, 7))
    W_loc = np.zeros((4, 7))

    # m1 = (a00+a11)(b00+b11)
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1

    # m2 = (a10+a11)b00
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1

    # m3 = a00(b11-b10)
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1

    # m4 = a11(b00-b10)
    U_loc[3, 3] = 1
    V_loc[0, 3] = -1; V_loc[2, 3] = 1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1

    # m5 = (a00+a01)b11
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1

    # m6 = (a10-a00)(b00+b01)
    U_loc[0, 5] = -1; U_loc[2, 5] = 1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1

    # m7 = (a01-a11)(b10+b11)
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1

    # Verify Strassen locally
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # A local index i, B local index j, C local index k
                # C[i,j] should be sum over l of A[i,l]*B[l,j]
                # Local A: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
                # Local B: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
                # Local C: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
                ai, aj = divmod(i, 2)
                bi, bj = divmod(j, 2)
                ci, cj = divmod(k, 2)
                if ai == bi and aj == cj:
                    expected = 1.0
                else:
                    expected = 0.0
                val = np.sum(U_loc[i, :] * V_loc[j, :] * W_loc[k, :])
                assert abs(val - expected) < 1e-9, f"Strassen local check failed: {i},{j},{k}: {val} vs {expected}"

    # Build global decomposition
    # 4 blocks of 2 columns each
    # For each block t:
    #   Top: A_0 (2x2) x B_block_t_top (2x2) -> C[0, 2t:2t+2] (2x2)
    #   Bot: A_1 (2x2) x B_block_t_bot (2x2) -> C[1, 2t:2t+2] (2x2)
    # Each uses Strassen rank 7
    # Total rank = 4 * 2 * 7 = 56

    rank = 56
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))

    col = 0
    for t in range(4):
        # --- Top part: A_0 x B_block_t_top -> C[0, 2t:2t+2] ---
        # A_0 local (i,j) -> global i*4+j
        # B_block_t_top local (i,j) -> global i*8 + (2t+j)
        # C[0, 2t:2t+2] local (i,j) -> global (2t+j)*2 + 0 = 4t + 2j
        
        for r in range(7):
            c = col + r
            # A_0
            for li in range(2):
                for lj in range(2):
                    loc = li * 2 + lj
                    g = li * 4 + lj
                    U[g, c] = U_loc[loc, r]
            # B_block_t_top
            for li in range(2):
                for lj in range(2):
                    loc = li * 2 + lj
                    g = li * 8 + (2 * t + lj)
                    V[g, c] = V_loc[loc, r]
            # C[0, 2t:2t+2]
            for li in range(2):
                for lj in range(2):
                    loc = li * 2 + lj
                    g = (2 * t + lj) * 2 + 0
                    W[g, c] = W_loc[loc, r]
        col += 7

        # --- Bottom part: A_1 x B_block_t_bot -> C[1, 2t:2t+2] ---
        # A_1 local (i,j) -> global i*4 + (2+j)
        # B_block_t_bot local (i,j) -> global (2+i)*8 + (2t+j)
        # C[1, 2t:2t+2] local (i,j) -> global (2t+j)*2 + 1 = 4t + 2j + 1
        
        for r in range(7):
            c = col + r
            # A_1
            for li in range(2):
                for lj in range(2):
                    loc = li * 2 + lj
                    g = li * 4 + (2 + lj)
                    U[g, c] = U_loc[loc, r]
            # B_block_t_bot
            for li in range(2):
                for lj in range(2):
                    loc = li * 2 + lj
                    g = (2 + li) * 8 + (2 * t + lj)
                    V[g, c] = V_loc[loc, r]
            # C[1, 2t:2t+2]
            for li in range(2):
                for lj in range(2):
                    loc = li * 2 + lj
                    g = (2 * t + lj) * 2 + 1
                    W[g, c] = W_loc[loc, r]
        col += 7

    # Verify the full decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                if not np.isclose(val, 1.0, atol=1e-9):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
