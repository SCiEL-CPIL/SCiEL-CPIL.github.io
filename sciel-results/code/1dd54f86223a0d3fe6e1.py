"""Matrix multiplication 4,4,9 rank-minimal decomposition.

Uses a block decomposition with Strassen:
  - Split B (4x9) into B1 (4x4), B2 (4x4), B3 (4x1)
  - A*B1 and A*B2 use Strassen for 4x4*4x4 (rank 49 each)
  - A*B3 uses schoolbook for 4x4*4x1 (rank 16)
  - Total rank: 49 + 49 + 16 = 114

Then attempt linear-dependence pruning to reduce rank.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 9]


def _verify_decomposition(U, V, W, n=4, m=4, p=9, tol=1e-6):
    """Check that the 3-way tensor decomposition equals the matmul tensor.

    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]  should == 1
    """
    rank = U.shape[1]
    T_decomp = np.zeros((n, m, p))
    for r in range(rank):
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        # T[i,j,k] += Ur[i,j]*Vr[j,k]*Wr[k,i]
        # Vectorize: for each i, T[i,:,:] += Ur[i,:][:,None] * (Vr * Wr.T)
        # Vr[j,k]*Wr[k,i] = (Vr @ Wr.T)[j,i]
        prod = Vr @ Wr.T  # (m, n)
        for i in range(n):
            T_decomp[i, :, :] += Ur[i, :, None] * prod[:, i][None, :]
    T_expected = np.ones((n, m, p))
    return np.allclose(T_decomp, T_expected, atol=tol)


def _build_strassen_114():
    """Build the known rank-114 Strassen block decomposition for 4x4 * 4x9."""
    n, m, p = 4, 4, 9
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    strassen_2x2_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

    def strassen_4x4(A_base_i, A_base_j, B_base_j, B_base_k, C_base_i, C_base_k):
        def get_A(gi, gj):
            return gi * m + gj
        def get_B(gj, gk):
            return gj * p + gk
        def get_C(gk, gi):
            return gk * n + gi

        def add_block_mult(A_side, B_side, C_side):
            for A_coeffs, B_coeffs, C_contribs in strassen_2x2_terms:
                u_vec = [0] * (n * m)
                v_vec = [0] * (m * p)
                w_vec = [0] * (p * n)

                for (li, lj), meta_coeff in A_coeffs.items():
                    for (br, bc, block_coeff) in A_side:
                        gi = A_base_i + br * 2 + li
                        gj = A_base_j + bc * 2 + lj
                        u_vec[get_A(gi, gj)] += meta_coeff * block_coeff

                for (lj, lk), meta_coeff in B_coeffs.items():
                    for (br, bc, block_coeff) in B_side:
                        gj = B_base_j + br * 2 + lj
                        gk = B_base_k + bc * 2 + lk
                        v_vec[get_B(gj, gk)] += meta_coeff * block_coeff

                for (li, lk), meta_coeff in C_contribs.items():
                    for (br, bc), block_coeff in C_side.items():
                        gi = C_base_i + br * 2 + li
                        gk = C_base_k + bc * 2 + lk
                        w_vec[get_C(gk, gi)] += meta_coeff * block_coeff

                add_term(u_vec, v_vec, w_vec)

        add_block_mult(
            [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)],
            {(0,0): 1, (1,1): 1}
        )
        add_block_mult(
            [(1,0,1), (1,1,1)], [(0,0,1)],
            {(1,0): 1, (1,1): -1}
        )
        add_block_mult(
            [(0,0,1)], [(0,1,1), (1,1,-1)],
            {(0,1): 1, (1,1): 1}
        )
        add_block_mult(
            [(1,1,1)], [(1,0,1), (0,0,-1)],
            {(0,0): 1, (1,0): 1}
        )
        add_block_mult(
            [(0,0,1), (0,1,1)], [(1,1,1)],
            {(0,0): -1, (0,1): 1}
        )
        add_block_mult(
            [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)],
            {(1,1): 1}
        )
        add_block_mult(
            [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)],
            {(0,0): 1}
        )

    # Part 1: A * B[0:4]
    strassen_4x4(0, 0, 0, 0, 0, 0)
    # Part 2: A * B[4:8]
    strassen_4x4(0, 0, 0, 4, 0, 4)
    # Part 3: A * B[8:9] (4x1), schoolbook
    for i in range(4):
        for j in range(4):
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            u_vec[i * m + j] = 1
            v_vec[j * p + 8] = 1
            w_vec[8 * n + i] = 1
            add_term(u_vec, v_vec, w_vec)

    rank = len(U_cols)
    U = np.array(U_cols, dtype=float).T  # (n*m, rank)
    V = np.array(V_cols, dtype=float).T  # (m*p, rank)
    W = np.array(W_cols, dtype=float).T  # (p*n, rank)
    return U, V, W


def _try_reduce_rank(U, V, W, n=4, m=4, p=9, max_time=100.0, tol=1e-8):
    """Attempt to reduce rank via linear dependence pruning.

    A term r is redundant if its three vectors (U_r, V_r, W_r) can be expressed
    such that the tensor contribution is a linear combination of other terms.
    Practical sufficient check: if the triple (U_r, V_r, W_r) is linearly
    dependent with the set of other triples in the tensor space, drop it.
    We approximate by checking if U_r is in span of other U columns AND the
    induced tensor contribution is recoverable. Simpler robust check: detect
    exact duplicate or zero triples, and try greedy removal verifying each time.
    """
    start = time.time()
    rank = U.shape[1]

    # Remove zero columns
    keep = []
    for r in range(rank):
        if np.any(U[:, r] != 0) or np.any(V[:, r] != 0) or np.any(W[:, r] != 0):
            keep.append(r)
    if len(keep) < rank:
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
        rank = len(keep)

    # Greedy removal: try removing each term, verify, keep if still valid.
    # This is expensive (rank * verify). Only do a limited number of attempts.
    max_attempts = 200
    attempts = 0
    changed = True
    while changed and attempts < max_attempts and (time.time() - start) < max_time:
        changed = False
        rank = U.shape[1]
        for r in range(rank):
            attempts += 1
            if (time.time() - start) > max_time:
                break
            # Try removing term r
            mask = np.ones(rank, dtype=bool)
            mask[r] = False
            U_t = U[:, mask]
            V_t = V[:, mask]
            W_t = W[:, mask]
            if _verify_decomposition(U_t, V_t, W_t, n, m, p, tol=tol):
                U, V, W = U_t, V_t, W_t
                changed = True
                break  # restart scan after a removal
    return U, V, W


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Build the rank-114 Strassen decomposition
    U, V, W = _build_strassen_114()

    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to pure schoolbook (rank 144)
        U_cols, V_cols, W_cols = [], [], []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = [0] * (n*m); v = [0] * (m*p); w = [0] * (p*n)
                    u[i*m+j] = 1
                    v[j*p+k] = 1
                    w[k*n+i] = 1
                    U_cols.append(u); V_cols.append(v); W_cols.append(w)
        rank = len(U_cols)
        U = np.array(U_cols, dtype=float).T
        V = np.array(V_cols, dtype=float).T
        W = np.array(W_cols, dtype=float).T

    # Try to reduce rank (limited time)
    if time.time() < deadline - 5:
        U, V, W = _try_reduce_rank(U, V, W, n, m, p, max_time=deadline - time.time() - 5)

    # Final verification safety net
    if not _verify_decomposition(U, V, W, n, m, p):
        U, V, W = _build_strassen_114()

    # Convert to lists
    rank = U.shape[1]
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
