import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Schoolbook 3x3 decomposition (rank 27) as base kernel
    def get_3x3_schoolbook():
        n3 = 3
        rank3 = 27
        U3 = np.zeros((9, rank3))
        V3 = np.zeros((9, rank3))
        W3 = np.zeros((9, rank3))
        for i in range(n3):
            for j in range(n3):
                for k in range(n3):
                    col = i * 9 + j * 3 + k
                    U3[i * 3 + j, col] = 1.0
                    V3[j * 3 + k, col] = 1.0
                    W3[k * 3 + i, col] = 1.0
        return U3, V3, W3

    U3, V3, W3 = get_3x3_schoolbook()
    r3 = 27

    # 2x2 Strassen with 3x3 blocks: total rank = 7 * 27 = 189
    total_rank = 7 * r3
    U6 = np.zeros((36, total_rank))
    V6 = np.zeros((36, total_rank))
    W6 = np.zeros((36, total_rank))

    def add_strassen_term(start_col, a_blocks, b_blocks, c_blocks):
        """Add one Strassen product term to the global factor matrices."""
        for r in range(r3):
            u_mat = U3[:, r].reshape(3, 3)
            v_mat = V3[:, r].reshape(3, 3)
            w_mat = W3[:, r].reshape(3, 3)

            # U part: A-side
            for bi, coeff in a_blocks:
                rb, cb = bi // 2, bi % 2
                for i in range(3):
                    for j in range(3):
                        val = coeff * u_mat[i, j]
                        if val != 0:
                            gi, gj = rb * 3 + i, cb * 3 + j
                            U6[gi * 6 + gj, start_col + r] += val

            # V part: B-side
            for bi, coeff in b_blocks:
                rb, cb = bi // 2, bi % 2
                for j in range(3):
                    for k in range(3):
                        val = coeff * v_mat[j, k]
                        if val != 0:
                            gj, gk = rb * 3 + j, cb * 3 + k
                            V6[gj * 6 + gk, start_col + r] += val

            # W part: C-side (index k*n + i)
            for bi, coeff in c_blocks:
                rb, cb = bi // 2, bi % 2
                for k in range(3):
                    for i in range(3):
                        val = coeff * w_mat[k, i]
                        if val != 0:
                            gk, gi = cb * 3 + k, rb * 3 + i
                            W6[gk * 6 + gi, start_col + r] += val

    # Strassen formulas for 2x2 blocks: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # P1 = (A11 + A22) * (B11 + B22) -> C11, C22
    add_strassen_term(0,
                      a_blocks=[(0, 1), (3, 1)],
                      b_blocks=[(0, 1), (3, 1)],
                      c_blocks=[(0, 1), (3, 1)])

    # P2 = (A21 + A22) * B11 -> C21 (+1), C22 (-1)
    add_strassen_term(r3,
                      a_blocks=[(2, 1), (3, 1)],
                      b_blocks=[(0, 1)],
                      c_blocks=[(2, 1), (3, -1)])

    # P3 = A11 * (B12 - B22) -> C12 (+1), C22 (+1)
    add_strassen_term(2 * r3,
                      a_blocks=[(0, 1)],
                      b_blocks=[(1, 1), (3, -1)],
                      c_blocks=[(1, 1), (3, 1)])

    # P4 = A22 * (B21 - B11) -> C11 (+1), C21 (+1)
    add_strassen_term(3 * r3,
                      a_blocks=[(3, 1)],
                      b_blocks=[(2, 1), (0, -1)],
                      c_blocks=[(0, 1), (2, 1)])

    # P5 = (A11 + A12) * B22 -> C12 (+1), C11 (-1)
    add_strassen_term(4 * r3,
                      a_blocks=[(0, 1), (1, 1)],
                      b_blocks=[(3, 1)],
                      c_blocks=[(1, 1), (0, -1)])

    # P6 = (A21 - A11) * (B11 + B12) -> C22 (+1)
    add_strassen_term(5 * r3,
                      a_blocks=[(2, 1), (0, -1)],
                      b_blocks=[(0, 1), (1, 1)],
                      c_blocks=[(3, 1)])

    # P7 = (A12 - A22) * (B21 + B22) -> C11 (+1)
    add_strassen_term(6 * r3,
                      a_blocks=[(1, 1), (3, -1)],
                      b_blocks=[(2, 1), (3, 1)],
                      c_blocks=[(0, 1)])

    # Fast verification: T[i,j,k] = sum_r U6[i*6+j, r] * V6[j*6+k, r] * W6[k*6+i, r]
    # Use einsum but only once for full verification
    def verify_decomp(U_mat, V_mat, W_mat):
        """Verify that the decomposition gives all-ones tensor."""
        r = U_mat.shape[1]
        Ug = U_mat.reshape(6, 6, r)
        Vg = V_mat.reshape(6, 6, r)
        Wg = W_mat.reshape(6, 6, r)
        rec = np.einsum('ijr,jkr,kir->ijk', Ug, Vg, Wg, optimize=True)
        return np.allclose(rec, np.ones((6, 6, 6)), atol=1e-9)

    if time.time() < deadline:
        try:
            if verify_decomp(U6, V6, W6):
                # Try to find a smaller subset of rank terms
                # First check linear algebraic rank bounds
                rank_U = np.linalg.matrix_rank(U6, tol=1e-10)
                rank_V = np.linalg.matrix_rank(V6, tol=1e-10)
                rank_W = np.linalg.matrix_rank(W6, tol=1e-10)
                min_rank_bound = min(rank_U, rank_V, rank_W)
                
                best_keep = list(range(total_rank))
                
                if min_rank_bound < total_rank and time.time() < deadline - 30.0:
                    # Greedy forward selection: start with all, remove if possible
                    # Use a faster check: verify on a random subset of entries
                    rng = np.random.RandomState(42)
                    # Sample a few (i,j,k) positions to check
                    sample_ijk = [(rng.randint(6), rng.randint(6), rng.randint(6)) for _ in range(50)]
                    
                    def fast_verify(U_mat, V_mat, W_mat, sample):
                        r = U_mat.shape[1]
                        Ug = U_mat.reshape(6, 6, r)
                        Vg = V_mat.reshape(6, 6, r)
                        Wg = W_mat.reshape(6, 6, r)
                        for (i, j, k) in sample:
                            val = np.sum(Ug[i, j, :] * Vg[j, k, :] * Wg[k, i, :])
                            if abs(val - 1.0) > 1e-8:
                                return False
                        return True
                    
                    # Try removing each term, check with fast verification
                    keep = list(range(total_rank))
                    for _ in range(3):  # Multiple passes
                        for r_idx in range(len(keep) - 1, -1, -1):
                            if time.time() > deadline - 10.0:
                                break
                            candidate = keep[r_idx]
                            new_keep = [k for k in keep if k != candidate]
                            if len(new_keep) == 0:
                                continue
                            U_t = U6[:, new_keep]
                            V_t = V6[:, new_keep]
                            W_t = W6[:, new_keep]
                            if fast_verify(U_t, V_t, W_t, sample_ijk):
                                keep = new_keep
                    
                    if len(keep) < len(best_keep):
                        best_keep = keep
                
                # Final full verification of best subset
                if len(best_keep) < total_rank:
                    U_f = U6[:, best_keep]
                    V_f = V6[:, best_keep]
                    W_f = W6[:, best_keep]
                    if verify_decomp(U_f, V_f, W_f):
                        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
                
                return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
            else:
                # Try to fix or fall back
                return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
        except Exception:
            pass

    # Fallback: schoolbook rank 216
    rank = 216
    U_f = np.zeros((36, rank))
    V_f = np.zeros((36, rank))
    W_f = np.zeros((36, rank))
    for i in range(6):
        for j in range(6):
            for k in range(6):
                col = i * 36 + j * 6 + k
                U_f[i * 6 + j, col] = 1.0
                V_f[j * 6 + k, col] = 1.0
                W_f[k * 6 + i, col] = 1.0
    return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

# EVOLVE-BLOCK-END
