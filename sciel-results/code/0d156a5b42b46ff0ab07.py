"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 118.0

    n, m, p = DIMENSIONS  # 2, 4, 8
    dim_a, dim_b, dim_c = n * m, m * p, n * p  # 8, 32, 16

    # Non-zero (a, b, c) indices of the multiplication tensor T[a,b,c]=1
    nz = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nz.append((i * m + j, j * p + k, k * n + i))

    def build_target():
        T = np.zeros((dim_a, dim_b, dim_c))
        for a, b, c in nz:
            T[a, b, c] = 1.0
        return T

    T = build_target()

    def check_exact(U, V, W):
        rec = np.einsum('at,bt,ct->abc', U, V, W)
        return np.max(np.abs(rec - T)) < 1e-8

    # ---- Clean rank-56 Strassen construction (reliable baseline) ----
    def strassen_56():
        rank = 56
        U = np.zeros((dim_a, rank))
        V = np.zeros((dim_b, rank))
        W = np.zeros((dim_c, rank))

        strassen_terms = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
        ]

        col = 0
        # 8 blocks: 4 for Left (A_L, B_L) and 4 for Right (A_R, B_R)
        for block in range(8):
            is_left = block < 4
            block_idx = block % 4

            a_i_offset = 0
            a_j_offset = 0 if is_left else 2
            b_i_offset = 0 if is_left else 2
            b_j_offset = block_idx * 2
            c_i_offset = 0
            c_j_offset = block_idx * 2

            for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
                for (li, lj), val in u_coeffs.items():
                    U[(a_i_offset + li) * m + (a_j_offset + lj), col] = val
                for (li, lj), val in v_coeffs.items():
                    V[(b_i_offset + li) * p + (b_j_offset + lj), col] = val
                for (li, lj), val in w_coeffs.items():
                    # W index convention: W[k*n + i, r]  ->  row = gj*n + gi
                    W[(c_j_offset + lj) * n + (c_i_offset + li), col] = val
                col += 1
        return U, V, W

    U56, V56, W56 = strassen_56()
    if check_exact(U56, V56, W56):
        best_rank = 56
        U_best, V_best, W_best = U56, V56, W56
    else:
        # Schoolbook fallback (rank 64)
        best_rank = 64
        U_best = np.zeros((dim_a, 64))
        V_best = np.zeros((dim_b, 64))
        W_best = np.zeros((dim_c, 64))
        for a, b, c in nz:
            col = a * dim_c + b
            U_best[a, col] = 1.0
            V_best[b, col] = 1.0
            W_best[c, col] = 1.0

    # ---- Structural 4-block partition attempt (target rank 28) ----
    # Split m=4 into two halves (j-group) and p=8 into four k-groups,
    # giving four independent 2x2 x 2x2 multiplications, each rank 7 -> 28.
    def strassen_28():
        rank = 28
        U = np.zeros((dim_a, rank))
        V = np.zeros((dim_b, rank))
        W = np.zeros((dim_c, rank))

        strassen_terms = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
        ]

        col = 0
        for jg in range(2):      # intermediate (m) group: j = 2*jg .. 2*jg+1
            for kg in range(4):  # output (p) group: k = 2*kg .. 2*kg+1
                # A (2x2): rows i=0,1 ; cols j=2*jg, 2*jg+1
                # B (2x2): rows j=2*jg, 2*jg+1 ; cols k=2*kg, 2*kg+1
                # C (2x2): rows i=0,1 ; cols k=2*kg, 2*kg+1
                for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
                    for (li, lj), val in u_coeffs.items():
                        U[li * m + (2 * jg + lj), col] = val
                    for (li, lj), val in v_coeffs.items():
                        V[(2 * jg + li) * p + (2 * kg + lj), col] = val
                    for (li, lj), val in w_coeffs.items():
                        W[(2 * kg + lj) * n + li, col] = val
                    col += 1
        return U, V, W

    U28, V28, W28 = strassen_28()
    if check_exact(U28, V28, W28):
        best_rank = 28
        U_best, V_best, W_best = U28, V28, W28

    # ---- ALS search for a lower exact rank ----
    T_flat = np.zeros((dim_a, dim_b * dim_c))
    T_flat2 = np.zeros((dim_b, dim_a * dim_c))
    T_flat3 = np.zeros((dim_c, dim_a * dim_b))
    for a, b, c in nz:
        T_flat[a, b * dim_c + c] = 1.0
        T_flat2[b, a * dim_c + c] = 1.0
        T_flat3[c, a * dim_b + b] = 1.0

    def als_attempt(rank, seed):
        rng = np.random.RandomState(seed * 1000 + rank * 7)
        U = rng.randn(dim_a, rank)
        V = rng.randn(dim_b, rank)
        W = rng.randn(dim_c, rank)
        for _ in range(30):
            if time.time() > deadline - 15:
                return None
            M = np.einsum('bt,ct->btc', V, W).reshape(rank, dim_b * dim_c)
            U = np.linalg.lstsq(M.T, T_flat.T, rcond=None)[0].T
            M2 = np.einsum('at,ct->atc', U, W).reshape(rank, dim_a * dim_c)
            V = np.linalg.lstsq(M2.T, T_flat2.T, rcond=None)[0].T
            M3 = np.einsum('at,bt->abt', U, V).reshape(rank, dim_a * dim_b)
            W = np.linalg.lstsq(M3.T, T_flat3.T, rcond=None)[0].T
        if check_exact(U, V, W):
            return U, V, W
        return None

    # Search ranks strictly below the current best (skip the structural 28 if it failed)
    for rank in range(best_rank - 1, 15, -1):
        if time.time() > deadline - 20:
            break
        if rank == 28:
            continue  # already handled structurally
        for seed in range(3):
            if time.time() > deadline - 15:
                break
            res = als_attempt(rank, seed)
            if res is not None:
                U_best, V_best, W_best = res
                best_rank = rank
                break
        if best_rank == rank:
            break

    def to_list(mat):
        lst = []
        for row in mat:
            lst.append([round(x * 2) / 2.0 for x in row])
        return lst

    return {"U": to_list(U_best), "V": to_list(V_best), "W": to_list(W_best)}
# EVOLVE-BLOCK-END
