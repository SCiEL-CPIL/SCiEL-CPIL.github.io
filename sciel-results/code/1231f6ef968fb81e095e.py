import numpy as np
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 2, 3, 9

    # Build the target tensor
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

    # Build rank-50 decomposition using Strassen on 2x2 blocks
    rank = 50
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    u_a, u_b, u_c, u_d = 0, 1, 3, 4

    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1

        U[u_a, col] += 1
        U[u_d, col] += 1
        V[v_e, col] += 1
        V[v_h, col] += 1

        U[u_c, col+1] += 1
        U[u_d, col+1] += 1
        V[v_e, col+1] += 1

        U[u_a, col+2] += 1
        V[v_f, col+2] += 1
        V[v_h, col+2] -= 1

        U[u_d, col+3] += 1
        V[v_g, col+3] += 1
        V[v_e, col+3] -= 1

        U[u_a, col+4] += 1
        U[u_b, col+4] += 1
        V[v_h, col+4] += 1

        U[u_c, col+5] += 1
        U[u_a, col+5] -= 1
        V[v_e, col+5] += 1
        V[v_f, col+5] += 1

        U[u_b, col+6] += 1
        U[u_d, col+6] -= 1
        V[v_g, col+6] += 1
        V[v_h, col+6] += 1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        W[w_00, col] += 1
        W[w_00, col+3] += 1
        W[w_00, col+4] -= 1
        W[w_00, col+6] += 1

        W[w_01, col+2] += 1
        W[w_01, col+4] += 1

        W[w_10, col+1] += 1
        W[w_10, col+3] += 1

        W[w_11, col] += 1
        W[w_11, col+1] -= 1
        W[w_11, col+2] += 1
        W[w_11, col+5] += 1

        col += 7

    k8 = 8
    v_e = k8
    v_g = p + k8

    U[u_a, col] += 1
    V[v_e, col] += 1
    U[u_b, col+1] += 1
    V[v_g, col+1] += 1
    U[u_c, col+2] += 1
    V[v_e, col+2] += 1
    U[u_d, col+3] += 1
    V[v_g, col+3] += 1

    w_08 = k8 * n + 0
    w_18 = k8 * n + 1

    W[w_08, col] += 1
    W[w_08, col+1] += 1
    W[w_18, col+2] += 1
    W[w_18, col+3] += 1

    col += 4

    u_ar0, u_ar1 = 2, 5
    for k in range(9):
        v_b = 2 * p + k
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Greedy rank reduction with efficient residual computation
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank

    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = rank

    # Precompute the current decomposition
    current_decomp = np.einsum('ir,jr,kr->ijk', current_U, current_V, current_W)

    # Iteratively try to remove columns using efficient residual updates
    max_iterations = 20
    for iteration in range(max_iterations):
        if time.time() > deadline - 20:
            break

        best_drop = None
        best_residual = float('inf')

        for drop_col in range(current_rank):
            if time.time() > deadline - 15:
                break

            # The contribution of column drop_col to the tensor is
            # U[:, drop_col] outer V[:, drop_col] outer W[:, drop_col]
            # We can compute the residual as ||current_decomp - T - col_contribution||^2
            # where col_contribution is the rank-1 tensor from this column

            col_contribution = np.einsum('i,j,k->ijk',
                                         current_U[:, drop_col],
                                         current_V[:, drop_col],
                                         current_W[:, drop_col])

            new_decomp = current_decomp - col_contribution
            residual = np.sum((new_decomp - T)**2)

            if residual < best_residual:
                best_residual = residual
                best_drop = drop_col

            if residual < 1e-10:
                break

        if best_drop is None or best_residual > 1e-10:
            break

        # Remove the column
        mask = np.ones(current_rank, dtype=bool)
        mask[best_drop] = False
        current_U = current_U[:, mask]
        current_V = current_V[:, mask]
        current_W = current_W[:, mask]
        current_rank -= 1

        # Update the decomposition
        current_decomp = np.einsum('ir,jr,kr->ijk', current_U, current_V, current_W)

        if current_rank < best_rank:
            best_rank = current_rank
            best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()

    # Also try pairwise removals from the original to find more reductions
    if best_rank >= rank - 2 and time.time() < deadline - 30:
        # Try removing pairs of columns from the original decomposition
        pair_best_rank = best_rank
        pair_best_U, pair_best_V, pair_best_W = best_U.copy(), best_V.copy(), best_W.copy()

        for i in range(rank):
            if time.time() > deadline - 25:
                break
            for j in range(i + 1, rank):
                if time.time() > deadline - 25:
                    break

                mask = np.ones(rank, dtype=bool)
                mask[i] = False
                mask[j] = False
                U_new = U[:, mask]
                V_new = V[:, mask]
                W_new = W[:, mask]

                decomp_new = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
                residual = np.sum((decomp_new - T)**2)

                if residual < 1e-10:
                    pair_best_rank = rank - 2
                    pair_best_U, pair_best_V, pair_best_W = U_new.copy(), V_new.copy(), W_new.copy()
                    break
            if pair_best_rank == rank - 2:
                break

        if pair_best_rank < best_rank:
            best_rank = pair_best_rank
            best_U, best_V, best_W = pair_best_U, pair_best_V, pair_best_W

    if best_rank < rank:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
