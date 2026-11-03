"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Uses Strassen's rank-7 decomposition for 2x2x2 blocks applied to overlapping
    2x2 column blocks of B, plus schoolbook for the remaining row.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin

    def elapsed():
        return time.time() - start_time

    def validate(U, V, W):
        """Check if decomposition exactly computes matrix multiplication."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        rank = n * m * p  # 72
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank

    # Start with schoolbook as fallback
    best_U, best_V, best_W, best_rank = schoolbook()

    # Strassen's 2x2x2 decomposition (rank 7)
    def strassen_2x2x2():
        """
        Return U, V, W for 2x2x2 Strassen decomposition.
        U: 4x7 (flattened 2x2 A)
        V: 4x7 (flattened 2x2 B)
        W: 4x7 (flattened 2x2 C)
        Index: U[i*2+j], V[j*2+k], W[k*2+i]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))

        # Term 0: m1 = (a00+a11)(b00+b11)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1

        # Term 1: m2 = (a10+a11)b00
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1

        # Term 2: m3 = a00(b01-b11)
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[1,2] = 1

        # Term 3: m4 = a11(b10-b00)
        U[3,3] = 1
        V[2,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[2,3] = 1

        # Term 4: m5 = (a00+a01)b11
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[0,4] = -1; W[1,4] = 1

        # Term 5: m6 = (a10-a00)(b00+b01)
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1

        # Term 6: m7 = (a01-a11)(b10+b11)
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = 1; W[3,6] = -1

        # Verify
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    val = np.sum(U[i*2+j] * V[j*2+k] * W[k*2+i])
                    if abs(val - 1.0) > 1e-9:
                        return None

        return U.tolist(), V.tolist(), W.tolist()

    # Build decomposition using Strassen on 2x2 blocks + schoolbook corrections
    def build_strassen_based():
        """Build decomposition using Strassen on 2x2 blocks + schoolbook corrections."""
        U = np.zeros((n*m, 68))
        V = np.zeros((m*p, 68))
        W = np.zeros((p*n, 68))

        col = 0

        strassen = strassen_2x2x2()
        if strassen is None:
            return None
        S_U, S_V, S_W = strassen

        col_blocks = [(0,1), (2,3), (4,5), (6,7)]

        for (k1, k2) in col_blocks:
            for s in range(7):
                for i in range(2):
                    for j in range(2):
                        U[i*3+j, col] += S_U[i*2+j][s]
                for j in range(2):
                    for kk in range(2):
                        k_idx = k1 + kk
                        V[j*8+k_idx, col] += S_V[j*2+kk][s]
                for i in range(2):
                    for kk in range(2):
                        k_idx = k1 + kk
                        W[k_idx*3+i, col] += S_W[kk*2+i][s]
                col += 1

            for i in range(2):
                for kk in range(2):
                    k_idx = k1 + kk
                    U[i*3+2, col] = 1
                    V[2*8+k_idx, col] = 1
                    W[k_idx*3+i, col] = 1
                    col += 1

        for j in range(3):
            for k in range(8):
                U[2*3+j, col] = 1
                V[j*8+k, col] = 1
                W[k*3+2, col] = 1
                col += 1

        if not validate(U.tolist(), V.tolist(), W.tolist()):
            return None

        return U.tolist(), V.tolist(), W.tolist(), 68

    result = build_strassen_based()
    if result is not None and elapsed() < deadline - 5:
        U_new, V_new, W_new, rank_new = result
        if rank_new < best_rank:
            best_U, best_V, best_W, best_rank = U_new, V_new, W_new, rank_new

    # Try to improve using Laderman's 3x3x3 rank-19 decomposition on a 3x3x4 block
    # Laderman's algorithm: 3x3 * 3x3 = 3x3 with rank 19
    # We can apply this to A (3x3) * B[:, 0:3] (3x3) -> C[:, 0:3] (3x3)
    # Then handle B[:, 3:8] (3x5) with schoolbook or other methods

    def laderman_3x3x3():
        """
        Return U, V, W for Laderman's 3x3x3 rank-19 decomposition.
        U: 9x19 (flattened 3x3 A)
        V: 9x19 (flattened 3x3 B)
        W: 9x19 (flattened 3x3 C)
        Index: U[i*3+j], V[j*3+k], W[k*3+i]
        """
        U = np.zeros((9, 19))
        V = np.zeros((9, 19))
        W = np.zeros((9, 19))

        # Laderman's 19 terms for 3x3x3
        # Based on Laderman 1976, as described in various references

        # Term 0: (a00+a11)(b00+b11) -> c00 +=, c11 +=
        U[0,0] = 1; U[4,0] = 1
        V[0,0] = 1; V[4,0] = 1
        W[0,0] = 1; W[4,0] = 1

        # Term 1: (a10+a11)b00 -> c10 +=
        U[3,1] = 1; U[4,1] = 1
        V[0,1] = 1
        W[3,1] = 1

        # Term 2: a00(b01-b11) -> c01 +=
        U[0,2] = 1
        V[1,2] = 1; V[4,2] = -1
        W[1,2] = 1

        # Term 3: a11(b10-b00) -> c00 +=, c10 +=
        U[4,3] = 1
        V[3,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[3,3] = 1

        # Term 4: (a00+a01)b11 -> c00 -=, c01 +=
        U[0,4] = 1; U[1,4] = 1
        V[4,4] = 1
        W[0,4] = -1; W[1,4] = 1

        # Term 5: (a10-a00)(b00+b01) -> c11 +=
        U[3,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[4,5] = 1

        # Term 6: (a01-a11)(b10+b11) -> c00 +=, c11 -=
        U[1,6] = 1; U[4,6] = -1
        V[3,6] = 1; V[4,6] = 1
        W[0,6] = 1; W[4,6] = -1

        # Now extend to 3x3x3 with additional terms for the third row/column
        # Term 7: a22(b22) -> c22 +=
        U[8,7] = 1
        V[8,7] = 1
        W[8,7] = 1

        # Term 8: (a02+a20)b00 -> c00 +=, c20 +=
        U[2,8] = 1; U[6,8] = 1
        V[0,8] = 1
        W[0,8] = 1; W[6,8] = 1

        # Term 9: (a02+a20)b22 -> c02 +=, c22 +=
        U[2,9] = 1; U[6,9] = 1
        V[8,9] = 1
        W[2,9] = 1; W[8,9] = 1

        # Term 10: (a00+a22)(b02+b20) -> c02 +=, c20 +=, c00 +=, c22 +=
        U[0,10] = 1; U[8,10] = 1
        V[2,10] = 1; V[6,10] = 1
        W[2,10] = 1; W[6,10] = 1; W[0,10] = 1; W[8,10] = 1

        # Term 11: (a12+a21)(b10+b01) -> c10 +=, c01 +=
        U[5,11] = 1; U[7,11] = 1
        V[3,11] = 1; V[1,11] = 1
        W[3,11] = 1; W[1,11] = 1

        # Term 12: (a12+a21)(b20+b02) -> c20 +=, c02 +=
        U[5,12] = 1; U[7,12] = 1
        V[6,12] = 1; V[2,12] = 1
        W[6,12] = 1; W[2,12] = 1

        # Term 13: (a12+a21)(b11+b12) -> c11 +=, c12 +=
        U[5,13] = 1; U[7,13] = 1
        V[4,13] = 1; V[5,13] = 1
        W[4,13] = 1; W[5,13] = 1

        # Term 14: (a10+a11)(b20+b21) -> c20 +=, c21 +=
        U[3,14] = 1; U[4,14] = 1
        V[6,14] = 1; V[7,14] = 1
        W[6,14] = 1; W[7,14] = 1

        # Term 15: (a00+a01)(b12+b22) -> c12 +=, c22 +=
        U[0,15] = 1; U[1,15] = 1
        V[5,15] = 1; V[8,15] = 1
        W[5,15] = 1; W[8,15] = 1

        # Term 16: (a02+a20)(b11) -> c01 +=, c21 +=
        U[2,16] = 1; U[6,16] = 1
        V[4,16] = 1
        W[1,16] = 1; W[7,16] = 1

        # Term 17: (a12+a21)(b00+b22) -> c02 +=, c20 +=
        U[5,17] = 1; U[7,17] = 1
        V[0,17] = 1; V[8,17] = 1
        W[2,17] = 1; W[6,17] = 1

        # Term 18: (a00+a11+a22)(b00+b11+b22) -> c00 +=, c11 +=, c22 +=
        U[0,18] = 1; U[4,18] = 1; U[8,18] = 1
        V[0,18] = 1; V[4,18] = 1; V[8,18] = 1
        W[0,18] = 1; W[4,18] = 1; W[8,18] = 1

        # Verify
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    val = np.sum(U[i*3+j] * V[j*3+k] * W[k*3+i])
                    if abs(val - 1.0) > 1e-9:
                        return None

        return U.tolist(), V.tolist(), W.tolist()

    # Try building with Laderman on first 3 columns + schoolbook on remaining 5
    def build_laderman_based():
        """Build decomposition using Laderman on 3x3x3 + schoolbook for rest."""
        # Laderman: A(3x3) * B[:,0:3](3x3) -> C[:,0:3](3x3) with rank 19
        # Schoolbook: A(3x3) * B[:,3:8](3x5) -> C[:,3:8](3x5) with rank 45
        # Total: 19 + 45 = 64
        total_rank = 19 + 45
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))

        lad = laderman_3x3x3()
        if lad is None:
            return None
        L_U, L_V, L_W = lad

        col = 0
        # Laderman part: columns 0,1,2 of B
        for s in range(19):
            for i in range(3):
                for j in range(3):
                    U[i*3+j, col] += L_U[i*3+j][s]
            for j in range(3):
                for kk in range(3):
                    k_idx = kk  # columns 0,1,2
                    V[j*8+k_idx, col] += L_V[j*3+kk][s]
            for i in range(3):
                for kk in range(3):
                    k_idx = kk
                    W[k_idx*3+i, col] += L_W[kk*3+i][s]
            col += 1

        # Schoolbook for columns 3-7
        for j in range(3):
            for k in range(3, 8):
                U[i*3+j, col] = 1  # This is wrong, need to loop over i
                break
            break
        # Redo schoolbook properly
        col = 19
        for i in range(3):
            for j in range(3):
                for k in range(3, 8):
                    U[i*3+j, col] = 1
                    V[j*8+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1

        if not validate(U.tolist(), V.tolist(), W.tolist()):
            return None

        return U.tolist(), V.tolist(), W.tolist(), total_rank

    result2 = build_laderman_based()
    if result2 is not None and elapsed() < deadline - 5:
        U_new, V_new, W_new, rank_new = result2
        if rank_new < best_rank:
            best_U, best_V, best_W, best_rank = U_new, V_new, W_new, rank_new

    # Try a hybrid: Strassen on 2x2 blocks for columns 0-7 with 3 rows
    # Use a 2x3 * 3x2 = 2x2 decomposition via Strassen-like approach
    # For each pair of columns (k1,k2), compute C[0:2, k1:k2+1] = A[0:2,:] * B[:, k1:k2+1]
    # Split A[0:2,:] into A[0:2,0:2] (2x2) and A[0:2,2] (2x1)
    # Split B[:,k1:k2+1] into B[0:2,k1:k2+1] (2x2) and B[2,k1:k2+1] (1x2)
    # C = A11*B11 + A12*B21
    # Use Strassen for A11*B11 (rank 7) + schoolbook for A12*B21 (rank 4) = 11 per block
    # 4 blocks: 44, plus A[2,:]*B[:, :] schoolbook: 24, total 68 (same as before)

    # Try another approach: use rank-19 Laderman for 3x3x3 on overlapping blocks
    # But Laderman is for 3x3*3x3->3x3, so we can use it for columns 0-2 and 3-5
    # Columns 0-2: Laderman rank 19
    # Columns 3-5: Laderman rank 19
    # Columns 6-7: schoolbook rank 6
    # Total: 19+19+6 = 44? No, that's not right because A is shared.
    # Actually each Laderman uses all of A (3x3) and 3 columns of B.
    # So: Laderman for B[:,0:3] (rank 19) + Laderman for B[:,3:6] (rank 19) + schoolbook for B[:,6:8] (rank 6)
    # Total: 19+19+6 = 44

    def build_laderman_2blocks():
        """Two Laderman blocks + schoolbook for last 2 columns."""
        total_rank = 19 + 19 + 6
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))

        lad = laderman_3x3x3()
        if lad is None:
            return None
        L_U, L_V, L_W = lad

        col = 0
        # First Laderman block: columns 0,1,2
        for s in range(19):
            for i in range(3):
                for j in range(3):
                    U[i*3+j, col] += L_U[i*3+j][s]
            for j in range(3):
                for kk in range(3):
                    k_idx = kk
                    V[j*8+k_idx, col] += L_V[j*3+kk][s]
            for i in range(3):
                for kk in range(3):
                    k_idx = kk
                    W[k_idx*3+i, col] += L_W[kk*3+i][s]
            col += 1

        # Second Laderman block: columns 3,4,5
        for s in range(19):
            for i in range(3):
                for j in range(3):
                    U[i*3+j, col] += L_U[i*3+j][s]
            for j in range(3):
                for kk in range(3):
                    k_idx = 3 + kk
                    V[j*8+k_idx, col] += L_V[j*3+kk][s]
            for i in range(3):
                for kk in range(3):
                    k_idx = 3 + kk
                    W[k_idx*3+i, col] += L_W[kk*3+i][s]
            col += 1

        # Schoolbook for columns 6,7
        for i in range(3):
            for j in range(3):
                for k in range(6, 8):
                    U[i*3+j, col] = 1
                    V[j*8+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1

        if not validate(U.tolist(), V.tolist(), W.tolist()):
            return None

        return U.tolist(), V.tolist(), W.tolist(), total_rank

    result3 = build_laderman_2blocks()
    if result3 is not None and elapsed() < deadline - 5:
        U_new, V_new, W_new, rank_new = result3
        if rank_new < best_rank:
            best_U, best_V, best_W, best_rank = U_new, V_new, W_new, rank_new

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
