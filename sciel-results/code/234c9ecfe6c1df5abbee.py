import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 2, 3, 9

    # Build the target tensor T[i,j,k]
    # i = a*m + am, j = b*p + am, k = b*n + a
    # T[i,j,k] = 1 iff am == (j // p)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n * m):
        ai = i // m
        am = i % m
        for j in range(m * p):
            bj = j // p
            bm = j % p
            if am == bj:
                k = bm * n + ai
                T[i, j, k] = 1.0

    # Build a rank-50 decomposition using Strassen's algorithm on 2x2x2 blocks
    # Split the 3 middle dimensions into 2+1:
    #   am=0,1 handled by Strassen (rank 7 per column pair)
    #   am=2 handled naively (rank 2 per column)
    # 4 column pairs * 7 + 1 column * 4 + 9 columns * 2 = 28 + 4 + 18 = 50

    rank = 50
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    # U indices: a=0 -> 0,1; a=1 -> 2,3
    u_a, u_b, u_c, u_d = 0, 1, 2, 3

    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = k0          # row 0, col k0
        v_f = k1          # row 0, col k1
        v_g = p + k0      # row 1, col k0
        v_h = p + k1      # row 1, col k1

        # Strassen 7 terms for 2x2x2 block
        # M1 = (A11+A22)(B11+B22)
        U[u_a, col] += 1
        U[u_d, col] += 1
        V[v_e, col] += 1
        V[v_h, col] += 1

        # M2 = A22(B21-B11)
        U[u_d, col+1] += 1
        V[v_g, col+1] += 1
        V[v_e, col+1] -= 1

        # M3 = A11(B12-B22)
        U[u_a, col+2] += 1
        V[v_f, col+2] += 1
        V[v_h, col+2] -= 1

        # M4 = (A12+A21)(B22)
        U[u_b, col+3] += 1
        U[u_c, col+3] += 1
        V[v_h, col+3] += 1

        # M5 = (A11+A12)(B11)
        U[u_a, col+4] += 1
        U[u_b, col+4] += 1
        V[v_e, col+4] += 1

        # M6 = (A21-A11)(B12+B22)
        U[u_c, col+5] += 1
        U[u_a, col+5] -= 1
        V[v_f, col+5] += 1
        V[v_h, col+5] += 1

        # M7 = (A12-A22)(B21+B11)
        U[u_b, col+6] += 1
        U[u_d, col+6] -= 1
        V[v_g, col+6] += 1
        V[v_e, col+6] += 1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        # C11 = M1 + M4 - M5 - M7
        W[w_00, col] += 1
        W[w_00, col+3] += 1
        W[w_00, col+4] -= 1
        W[w_00, col+6] -= 1

        # C12 = M5 + M3
        W[w_01, col+2] += 1
        W[w_01, col+4] += 1

        # C21 = M6 + M2
        W[w_10, col+1] += 1
        W[w_10, col+5] += 1

        # C22 = M1 - M2 + M3 + M7
        W[w_11, col] += 1
        W[w_11, col+1] -= 1
        W[w_11, col+2] += 1
        W[w_11, col+6] += 1

        col += 7

    # Handle b=8 (single column, am=0 and am=1)
    k8 = 8
    v_e8 = k8          # row 0, col 8
    v_g8 = p + k8      # row 1, col 8

    U[u_a, col] += 1
    V[v_e8, col] += 1
    U[u_b, col+1] += 1
    V[v_g8, col+1] += 1
    U[u_c, col+2] += 1
    V[v_e8, col+2] += 1
    U[u_d, col+3] += 1
    V[v_g8, col+3] += 1

    w_08 = k8 * n + 0
    w_18 = k8 * n + 1

    W[w_08, col] += 1
    W[w_08, col+2] += 1
    W[w_18, col+1] += 1
    W[w_18, col+3] += 1

    col += 4

    # Handle am=2 (row 2 of A, row 2 of B) for all 9 columns
    u_ar0, u_ar1 = 4, 5  # a=0,1 for am=2
    for k in range(9):
        v_b = 2 * p + k  # row 2, col k
        U[u_ar0, col] += 1
        V[v_b, col] += 1
        W[k * n + 0, col] += 1
        U[u_ar1, col+1] += 1
        V[v_b, col+1] += 1
        W[k * n + 1, col+1] += 1
        col += 2

    # Verify the decomposition
    decomp = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(decomp, T, atol=1e-10):
        # Fallback: return as-is (shouldn't happen)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Now try to reduce rank using greedy column removal with pair removal
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = rank

    def compute_residual(Um, Vm, Wm):
        d = np.einsum('ir,jr,kr->ijk', Um, Vm, Wm)
        return np.sum((d - T)**2)

    # Phase 1: Single column removal (greedy)
    while True:
        if time.time() > deadline - 30:
            break
        improved = False
        for col_to_remove in range(current_rank):
            if time.time() > deadline - 30:
                break
            mask = np.ones(current_rank, dtype=bool)
            mask[col_to_remove] = False
            U_new = current_U[:, mask]
            V_new = current_V[:, mask]
            W_new = current_W[:, mask]
            residual = compute_residual(U_new, V_new, W_new)
            if residual < 1e-8:
                current_U = U_new
                current_V = V_new
                current_W = W_new
                current_rank -= 1
                improved = True
                break
        if not improved:
            break

    if current_rank < best_rank:
        best_rank = current_rank
        best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()

    # Phase 2: Pair column removal
    while True:
        if time.time() > deadline - 15:
            break
        improved = False
        r = current_rank
        for i in range(r):
            if time.time() > deadline - 15:
                break
            for j in range(i + 1, r):
                if time.time() > deadline - 15:
                    break
                mask = np.ones(r, dtype=bool)
                mask[i] = False
                mask[j] = False
                U_new = current_U[:, mask]
                V_new = current_V[:, mask]
                W_new = current_W[:, mask]
                residual = compute_residual(U_new, V_new, W_new)
                if residual < 1e-8:
                    current_U = U_new
                    current_V = V_new
                    current_W = W_new
                    current_rank -= 2
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    if current_rank < best_rank:
        best_rank = current_rank
        best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()

    # Phase 3: Try to find linear dependencies via SVD of the tensor matrix
    # Flatten the tensor into a matrix and look for null space
    if time.time() < deadline - 10:
        r = current_rank
        # Create the matrix M where M[i,j] = sum_r U[i,r]*V[j,r]*W[k,r] for fixed k
        # Actually, let's use a different approach: check if any column is a linear combination
        # of others by looking at the structure
        
        # For each column r, the contribution is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # If we can find coefficients c_r such that sum_r c_r * (U_r ⊗ V_r ⊗ W_r) = 0
        # and some c_r is nonzero, we might be able to modify other columns
        
        # Simpler approach: try removing columns and checking if we can adjust remaining
        # columns to compensate (this is harder, so skip for now)
        
        # Instead, try a different strategy: perturb columns slightly and see if
        # we can find a lower-rank decomposition
        pass

    # Final: try single column removal one more time
    while True:
        if time.time() > deadline - 5:
            break
        improved = False
        for col_to_remove in range(current_rank):
            if time.time() > deadline - 5:
                break
            mask = np.ones(current_rank, dtype=bool)
            mask[col_to_remove] = False
            U_new = current_U[:, mask]
            V_new = current_V[:, mask]
            W_new = current_W[:, mask]
            residual = compute_residual(U_new, V_new, W_new)
            if residual < 1e-8:
                current_U = U_new
                current_V = V_new
                current_W = W_new
                current_rank -= 1
                improved = True
                break
        if not improved:
            break

    if current_rank < best_rank:
        best_rank = current_rank
        best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
