import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 105  # safety margin before 120s limit

    n, m, p = DIMENSIONS  # 2, 4, 8

    # Tensor equation: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # U: (8, R), V: (32, R), W: (16, R)

    def verify_decomposition(U, V, W):
        """Verify that the decomposition correctly computes matrix multiplication."""
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if s != 1:
                        return False
        return True

    # ---- Standard 56-rank Strassen decomposition (verified baseline) ----
    def build_strassen_56():
        U_local = np.zeros((4, 7), dtype=int)
        V_local = np.zeros((4, 7), dtype=int)
        W_local = np.zeros((4, 7), dtype=int)

        # r=0 (M1)
        U_local[0,0]=1; U_local[3,0]=1
        V_local[0,0]=1; V_local[3,0]=1
        W_local[0,0]=1; W_local[3,0]=1
        # r=1 (M2)
        U_local[2,1]=1; U_local[3,1]=1
        V_local[0,1]=1
        W_local[2,1]=1; W_local[3,1]=-1
        # r=2 (M3)
        U_local[0,2]=1
        V_local[1,2]=1; V_local[3,2]=-1
        W_local[1,2]=1; W_local[3,2]=1
        # r=3 (M4)
        U_local[3,3]=1
        V_local[2,3]=1; V_local[0,3]=-1
        W_local[0,3]=1; W_local[2,3]=1
        # r=4 (M5)
        U_local[0,4]=1; U_local[1,4]=1
        V_local[3,4]=1
        W_local[0,4]=-1; W_local[1,4]=1
        # r=5 (M6)
        U_local[2,5]=1; U_local[0,5]=-1
        V_local[0,5]=1; V_local[1,5]=1
        W_local[3,5]=1
        # r=6 (M7)
        U_local[1,6]=1; U_local[3,6]=-1
        V_local[2,6]=1; V_local[3,6]=1
        W_local[0,6]=1

        total_rank = 56
        U = np.zeros((8, total_rank), dtype=int)
        V = np.zeros((32, total_rank), dtype=int)
        W = np.zeros((16, total_rank), dtype=int)

        for j in range(4):
            for s in range(2):
                base_r = (j * 2 + s) * 7
                a_map = [0, 1, 4, 5] if s == 0 else [2, 3, 6, 7]
                b_map = [0*8+2*j, 0*8+2*j+1, 1*8+2*j, 1*8+2*j+1] if s == 0 else \
                        [2*8+2*j, 2*8+2*j+1, 3*8+2*j, 3*8+2*j+1]
                c_map = [2*j*2+0, (2*j+1)*2+0, 2*j*2+1, (2*j+1)*2+1]
                for l in range(7):
                    r = base_r + l
                    for la in range(4):
                        val = U_local[la, l]
                        if val != 0:
                            U[a_map[la], r] += val
                    for lb in range(4):
                        val = V_local[lb, l]
                        if val != 0:
                            V[b_map[lb], r] += val
                    for lc in range(4):
                        val = W_local[lc, l]
                        if val != 0:
                            W[c_map[lc], r] += val
        return U, V, W

    U56, V56, W56 = build_strassen_56()
    if not verify_decomposition(U56, V56, W56):
        # Fallback: naive 64-rank
        U = np.zeros((8, 64), dtype=int)
        V = np.zeros((32, 64), dtype=int)
        W = np.zeros((16, 64), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i*32 + j*8 + k
                    U[i*4+j, r] = 1
                    V[j*8+k, r] = 1
                    W[k*2+i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    if time.time() > deadline:
        return {"U": U56.tolist(), "V": V56.tolist(), "W": W56.tolist()}

    # ---- New strategy: 4x8 x 8x4 embedding, search for redundant terms ----
    # View the problem as computing only the top-left 2x8 block of (A_padded * B)
    # where A_padded is 4x4 (A in top-left 2x4, zeros elsewhere) and B is 4x8.
    # We build a 4x8 x 8x4 decomposition (i.e., 4x4x4 block structure) and look
    # for rank-1 terms that contribute only to output positions (i>=2 or k>=8),
    # which are irrelevant to our 2x8 output. Those can be dropped.

    # Build a full 4x4x4 Strassen-based decomposition for a 4x8 * 8x4 product,
    # then restrict to the 2x4 * 4x8 sub-problem by zeroing out rows/cols.
    # Actually, let's directly construct a 4x4x8 decomposition using 2x2x2 blocks:
    # 4x4 * 4x8 = 4 2x2x8 blocks = 4 * (4 2x2x2) = 16 2x2x2 blocks, each rank 7 = 112.
    # That's worse. Instead, use the 56-rank and try to find linear dependencies.

    # Approach: try to find a rank-48 or rank-52 decomposition by searching for
    # a set of rank-1 terms. Use the 56-rank as a starting point and try to
    # find 8 (or 4) columns that are linearly dependent in a way that preserves
    # the tensor equation.

    # For each pair of terms (r1, r2), check if their contribution to the tensor
    # can be expressed as a linear combination of other terms. This is expensive,
    # so we limit the search.

    best_rank = 56
    best_U, best_V, best_W = U56, V56, W56

    # Try to find a lower-rank decomposition by removing columns
    # For each column r, check if the tensor contribution of column r is zero
    # when added to a modified version of the other columns.
    # Simpler: for each column r, try to express its contribution as a sum of
    # other columns' contributions.

    # The contribution of column r to the tensor is:
    # T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need sum_r T_r[i,j,k] = 1 for all (i,j,k).

    # Try: for each subset of columns to remove, check if the remaining columns
    # can still sum to 1. This is combinatorial, so we try single-column removals
    # and verify.

    for r_remove in range(56):
        if time.time() > deadline:
            break
        U_test = np.delete(U56, r_remove, axis=1)
        V_test = np.delete(V56, r_remove, axis=1)
        W_test = np.delete(W56, r_remove, axis=1)
        if verify_decomposition(U_test, V_test, W_test):
            best_rank = 55
            best_U, best_V, best_W = U_test, V_test, W_test
            break  # found one, try to find more

    if best_rank < 56:
        # Try to remove more columns from the reduced set
        current_U, current_V, current_W = best_U, best_V, best_W
        current_rank = best_rank
        for r_remove in range(current_rank):
            if time.time() > deadline:
                break
            U_test = np.delete(current_U, r_remove, axis=1)
            V_test = np.delete(current_V, r_remove, axis=1)
            W_test = np.delete(current_W, r_remove, axis=1)
            if verify_decomposition(U_test, V_test, W_test):
                current_U, current_V, current_W = U_test, V_test, W_test
                current_rank -= 1
                best_rank = current_rank
                break

    if best_rank < 56:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # If no single-column removal works, return the verified 56-rank solution
    return {"U": U56.tolist(), "V": V56.tolist(), "W": W56.tolist()}
# EVOLVE-BLOCK-END
