"""Matrix multiplication 2x3x8: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8

    # Build explicit rank-11 decomposition for 2x3x2 using Strassen(7) + 4 rank-1 terms
    # A is 2x3, B is 3x2, C is 2x2
    # U index: i*3+j (i in {0,1}, j in {0,1,2}) -> 6 entries
    # V index: j*2+k (j in {0,1,2}, k in {0,1}) -> 6 entries
    # W index: k*2+i (k in {0,1}, i in {0,1}) -> 4 entries

    # Strassen 7 terms for the 2x2x2 part (A[:, :2] @ B[:2, :]):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    #
    # 4 rank-1 terms for A[:, 2] @ B[2, :]:
    # M8 = A02 * B20 -> contributes to C00
    # M9 = A02 * B21 -> contributes to C01
    # M10 = A12 * B20 -> contributes to C10
    # M11 = A12 * B21 -> contributes to C11

    # Build factor matrices for 2x3x2 (rank 11)
    # U[i*3+j, c], V[j*2+k, c], W[k*2+i, c]
    U_sub = np.zeros((6, 11))
    V_sub = np.zeros((6, 11))
    W_sub = np.zeros((4, 11))

    # Strassen term 1: M1 = (A00+A11)(B00+B11)
    # U: A00 at (0,0), A11 at (1,0)
    U_sub[0*3+0, 0] = 1  # A00
    U_sub[1*3+0, 0] = 1  # A11
    # V: B00 at (0,0), B11 at (1,1)
    V_sub[0*2+0, 0] = 1  # B00
    V_sub[1*2+1, 0] = 1  # B11
    # W: C00 at (0,0), C11 at (1,1)
    W_sub[0*2+0, 0] = 1  # C00
    W_sub[1*2+1, 0] = 1  # C11

    # Strassen term 2: M2 = (A10+A11)B00
    U_sub[1*3+0, 1] = 1  # A10
    U_sub[1*3+1, 1] = 1  # A11
    V_sub[0*2+0, 1] = 1  # B00
    W_sub[1*2+0, 1] = 1  # C10

    # Strassen term 3: M3 = A00(B10-B11)
    U_sub[0*3+0, 2] = 1  # A00
    V_sub[1*2+0, 2] = 1  # B10
    V_sub[1*2+1, 2] = -1  # -B11
    W_sub[0*2+1, 2] = 1  # C01

    # Strassen term 4: M4 = A11(B01-B10)
    U_sub[1*3+1, 3] = 1  # A11
    V_sub[0*2+1, 3] = 1  # B01
    V_sub[1*2+0, 3] = -1  # -B10
    W_sub[0*2+0, 3] = 1  # C00
    W_sub[1*2+0, 3] = 1  # C10

    # Strassen term 5: M5 = (A00+A01)B11
    U_sub[0*3+0, 4] = 1  # A00
    U_sub[0*3+1, 4] = 1  # A01
    V_sub[1*2+1, 4] = 1  # B11
    W_sub[0*2+0, 4] = -1  # -C00
    W_sub[0*2+1, 4] = 1  # C01

    # Strassen term 6: M6 = (A10-A00)(B00+B10)
    U_sub[1*3+0, 5] = 1  # A10
    U_sub[0*3+0, 5] = -1  # -A00
    V_sub[0*2+0, 5] = 1  # B00
    V_sub[1*2+0, 5] = 1  # B10
    W_sub[1*2+1, 5] = 1  # C11

    # Strassen term 7: M7 = (A01-A11)(B01+B11)
    U_sub[0*3+1, 6] = 1  # A01
    U_sub[1*3+1, 6] = -1  # -A11
    V_sub[0*2+1, 6] = 1  # B01
    V_sub[1*2+1, 6] = 1  # B11
    W_sub[0*2+0, 6] = 1  # C00

    # Rank-1 term 8: A02 * B20 -> C00
    U_sub[0*3+2, 7] = 1  # A02
    V_sub[2*2+0, 7] = 1  # B20
    W_sub[0*2+0, 7] = 1  # C00

    # Rank-1 term 9: A02 * B21 -> C01
    U_sub[0*3+2, 8] = 1  # A02
    V_sub[2*2+1, 8] = 1  # B21
    W_sub[0*2+1, 8] = 1  # C01

    # Rank-1 term 10: A12 * B20 -> C10
    U_sub[1*3+2, 9] = 1  # A12
    V_sub[2*2+0, 9] = 1  # B20
    W_sub[1*2+0, 9] = 1  # C10

    # Rank-1 term 11: A12 * B21 -> C11
    U_sub[1*3+2, 10] = 1  # A12
    V_sub[2*2+1, 10] = 1  # B21
    W_sub[1*2+1, 10] = 1  # C11

    # Verify the 2x3x2 decomposition
    T_sub = np.zeros((6, 6, 4))
    for i in range(2):
        for j in range(3):
            for k in range(2):
                T_sub[i*3+j, j*2+k, k*2+i] = 1.0

    recon_sub = np.zeros((6, 6, 4))
    for c in range(11):
        recon_sub += np.einsum('i,j,k->ijk', U_sub[:,c], V_sub[:,c], W_sub[:,c])

    if np.linalg.norm(T_sub - recon_sub) > 1e-10:
        # Fallback: schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Build full 2x3x8 decomposition by tiling 4 copies of 2x3x2
    rank_total = 4 * 11
    U_full = np.zeros((6, rank_total))
    V_full = np.zeros((24, rank_total))
    W_full = np.zeros((16, rank_total))

    for g in range(4):
        col_offset = g * 11
        for c in range(11):
            # U is the same for all groups
            U_full[:, col_offset + c] = U_sub[:, c]
            # V: j*8 + (g*2 + k_sub)
            for j in range(3):
                for k_sub in range(2):
                    V_full[j*8 + g*2 + k_sub, col_offset + c] = V_sub[j*2 + k_sub, c]
            # W: (g*2 + k_sub)*2 + i
            for k_sub in range(2):
                for i in range(2):
                    W_full[(g*2 + k_sub)*2 + i, col_offset + c] = W_sub[k_sub*2 + i, c]

    # Verify full decomposition
    T_full = np.zeros((6, 24, 16))
    for i in range(2):
        for j in range(3):
            for k in range(8):
                T_full[i*3+j, j*8+k, k*2+i] = 1.0

    recon_full = np.zeros((6, 24, 16))
    for c in range(rank_total):
        recon_full += np.einsum('i,j,k->ijk', U_full[:,c], V_full[:,c], W_full[:,c])

    if np.linalg.norm(T_full - recon_full) > 1e-10:
        # Fallback: schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
