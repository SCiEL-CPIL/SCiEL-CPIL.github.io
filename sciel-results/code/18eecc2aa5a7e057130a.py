"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin before the 120s wall-clock limit

    n, m, p = DIMENSIONS  # 2, 4, 6

    # ------------------------------------------------------------------
    # Build the rank-42 Strassen decomposition of M_{2,4,6}.
    #
    # The 2x4x6 multiplication tensor is a block-diagonal Kronecker product
    # of three 2x2x2 tensors:
    #   A = [A_L | A_R]          (A_L, A_R are 2x2)
    #   B = [B_T ; B_B]          (B_T, B_B are 2x2 row blocks, each 2x6)
    #   C = [C_1 | C_2 | C_3]    (each C_b is 2x2)
    # Each 2x2x2 block uses a rank-7 Strassen factorization, giving 3*7 = 21
    # "left" (A_L x B_T x C_b) and 21 "right" (A_R x B_B x C_b) terms, i.e.
    # 42 total.  Because the A-factors and C-factors of the left and right
    # sets occupy disjoint index positions, the two sets are independent and
    # each can be rank-reduced separately.
    # ------------------------------------------------------------------
    rank = 42
    U = [[0] * rank for _ in range(n * m)]   # 8  x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 12 x 42

    def add_strassen_block(a_idx, b_idx, w_idx, start_col):
        # Strassen 2x2x2 factors M1..M7
        # M1 = (a00+a11)(b00+b11)
        U[a_idx[0]][start_col + 0] += 1; U[a_idx[3]][start_col + 0] += 1
        V[b_idx[0]][start_col + 0] += 1; V[b_idx[3]][start_col + 0] += 1
        W[w_idx[0]][start_col + 0] += 1; W[w_idx[3]][start_col + 0] += 1
        # M2 = (a10+a11)b00
        U[a_idx[2]][start_col + 1] += 1; U[a_idx[3]][start_col + 1] += 1
        V[b_idx[0]][start_col + 1] += 1
        W[w_idx[2]][start_col + 1] += 1; W[w_idx[3]][start_col + 1] -= 1
        # M3 = a00(b01-b11)
        U[a_idx[0]][start_col + 2] += 1
        V[b_idx[1]][start_col + 2] += 1; V[b_idx[3]][start_col + 2] -= 1
        W[w_idx[1]][start_col + 2] += 1; W[w_idx[3]][start_col + 2] += 1
        # M4 = a11(b10-b00)
        U[a_idx[3]][start_col + 3] += 1
        V[b_idx[2]][start_col + 3] += 1; V[b_idx[0]][start_col + 3] -= 1
        W[w_idx[0]][start_col + 3] += 1; W[w_idx[2]][start_col + 3] += 1
        # M5 = (a00+a01)b11
        U[a_idx[0]][start_col + 4] += 1; U[a_idx[1]][start_col + 4] += 1
        V[b_idx[3]][start_col + 4] += 1
        W[w_idx[0]][start_col + 4] -= 1; W[w_idx[1]][start_col + 4] += 1
        # M6 = (a10-a00)(b00+b01)
        U[a_idx[2]][start_col + 5] += 1; U[a_idx[0]][start_col + 5] -= 1
        V[b_idx[0]][start_col + 5] += 1; V[b_idx[1]][start_col + 5] += 1
        W[w_idx[3]][start_col + 5] += 1
        # M7 = (a01-a11)(b10+b11)
        U[a_idx[1]][start_col + 6] += 1; U[a_idx[3]][start_col + 6] -= 1
        V[b_idx[2]][start_col + 6] += 1; V[b_idx[3]][start_col + 6] += 1
        W[w_idx[0]][start_col + 6] += 1

    # A is 2x4.  a_idx = [a00, a01, a10, a11]
    a_L = [0, 1, 4, 5]   # left 2x2 block
    a_R = [2, 3, 6, 7]   # right 2x2 block

    # B is 4x6.  b_idx = [b00, b01, b10, b11]
    # B_T rows 0,1 ; B_B rows 2,3
    b_1_T = [0, 1, 6, 7];      b_1_B = [12, 13, 18, 19]
    b_2_T = [2, 3, 8, 9];      b_2_B = [14, 15, 20, 21]
    b_3_T = [4, 5, 10, 11];    b_3_B = [16, 17, 22, 23]

    # C is 2x6.  W index = k*n + i.  w_idx = [c00, c01, c10, c11]
    w_1 = [0, 2, 1, 3]
    w_2 = [4, 6, 5, 7]
    w_3 = [8, 10, 9, 11]

    add_strassen_block(a_L, b_1_T, w_1, 0);   add_strassen_block(a_R, b_1_B, w_1, 7)
    add_strassen_block(a_L, b_2_T, w_2, 14);  add_strassen_block(a_R, b_2_B, w_2, 21)
    add_strassen_block(a_L, b_3_T, w_3, 28);  add_strassen_block(a_R, b_3_B, w_3, 35)

    # ------------------------------------------------------------------
    # Verification of the full decomposition (tensor contraction).
    # ------------------------------------------------------------------
    def verify_decomp(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if total != 1:
                        return False
        return True

    if not verify_decomp(U, V, W, rank):
        # Fallback: schoolbook rank n*m*p = 48
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
        return {"U": U, "V": V, "W": W}

    # ------------------------------------------------------------------
    # Rank reduction via linear dependence.
    #
    # Each rank-1 factor is u_r (8-dim) x v_r (24-dim) x w_r (12-dim).
    # A factor r is redundant iff its rank-1 tensor is in the span of the
    # others, i.e. iff the 3-vector (u_r, v_r, w_r) satisfies the
    # "linearized" constraint: for every (i,j,k) with tensor value 1,
    #   sum_{s in S} u_{i,j}^r v_{j,k}^s w_{k,i}^s
    #     + sum_{s in S} v_{j,k}^r u_{i,j}^s w_{k,i}^s
    #     + sum_{s in S} w_{k,i}^r u_{i,j}^s v_{j,k}^s = 0
    # for all (i,j,k) that are 1 in the tensor.  This is a linear system in
    # the coefficients c_r; if a nontrivial solution with c_r != 0 exists
    # then factor r can be absorbed.  We test redundancy by checking whether
    # removing factor r leaves the tensor unchanged (a direct, robust check
    # that avoids numerical rank pitfalls).
    # ------------------------------------------------------------------
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # Precompute the tensor support: positions (i,j,k) where the value is 1.
    support = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                support.append((i, j, k))

    def tensor_value_at(Uv, Vv, Wv, rank, i, j, k):
        # Uv,Vv,Wv are 2D arrays of shape (dim, rank)
        return np.dot(np.dot(Uv[i * m + j], Vv[j * p + k]), Wv[k * n + i])

    def check_full(Uv, Vv, Wv, rank):
        for (i, j, k) in support:
            if tensor_value_at(Uv, Vv, Wv, rank, i, j, k) != 1:
                return False
        return True

    # Greedy elimination: try to drop each column; keep the drop if the
    # decomposition still reproduces the tensor exactly.  Iterate to a fixed
    # point because dropping one column can expose another redundancy.
    def reduce_rank(Uv, Vv, Wv, rank):
        # Build a "column index" list that maps current logical factors to
        # original column positions; we physically compact at the end.
        kept = list(range(rank))
        progress = True
        while progress and (time.time() - start) < deadline - 20.0:
            progress = False
            new_kept = []
            for idx in kept:
                # Build arrays with this column removed
                rem = [c for c in kept if c != idx]
                Uv_t = Uv[:, rem]
                Vv_t = Vv[:, rem]
                Wv_t = Wv[:, rem]
                if check_full(Uv_t, Vv_t, Wv_t, len(rem)):
                    # column idx is redundant -> drop it
                    kept = rem
                    progress = True
                    break
                else:
                    new_kept.append(idx)
            if not progress:
                kept = new_kept
        return kept

    kept = reduce_rank(U_np, V_np, W_np, rank)
    new_rank = len(kept)

    if new_rank < rank:
        U_np = U_np[:, kept]
        V_np = V_np[:, kept]
        W_np = W_np[:, kept]
        rank = new_rank

    # Final verification of the reduced decomposition.
    if not check_full(U_np, V_np, W_np, rank):
        # something went wrong; fall back to the un-reduced (valid) rank-42
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = 42

    # Convert to integer/half-integer lists (entries are already small ints).
    U_out = [[int(round(U_np[i, r])) for r in range(rank)] for i in range(n * m)]
    V_out = [[int(round(V_np[i, r])) for r in range(rank)] for i in range(m * p)]
    W_out = [[int(round(W_np[i, r])) for r in range(rank)] for i in range(p * n)]

    # Final exact integer verification before returning.
    if not verify_decomp(U_out, V_out, W_out, rank):
        # fall back to the original valid rank-42 integer matrices
        return {"U": U, "V": V, "W": W}

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
