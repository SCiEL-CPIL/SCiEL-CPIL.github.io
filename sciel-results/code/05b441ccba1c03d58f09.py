"""
Strategy: Direct rank-27 construction for 2x4x4 matrix multiplication tensor.

Key insight: The 2x4x4 tensor can be decomposed as:
- 3 Strassen 2x2x2 sub-decompositions (7 terms each = 21 terms)
- 1 schoolbook 2x2x2 sub-decompositions (4 terms each = 4 terms)
- Total: 25 terms? No...

Actually, let me think more carefully.

The 2x4x4 multiplication: C = A * B where A is 2x4, B is 4x4, C is 2x4.

Partition into 4 blocks of size 2x2x2:
- Block (0,0): A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
- Block (0,1): A[:,0:2] * B[0:2,2:4] -> C[:,2:4]
- Block (1,0): A[:,2:4] * B[2:4,0:2] -> C[:,0:2]
- Block (1,1): A[:,2:4] * B[2:4,2:4] -> C[:,2:4]

Each is an independent 2x2x2 multiplication. Using Strassen on all 4 gives rank 28.

To get rank 27, we need to save 1 term. This is possible if we can share
structure between blocks. However, the blocks are truly independent in the
standard partitioning, so direct sharing isn't trivial.

Alternative approach: Use a different partitioning or a global algebraic identity.

Let me try a different strategy: use rank-7 Strassen on 3 blocks and
rank-4 schoolbook on 1 block, giving 3*7 + 4 = 25. But wait, that's
less than 27, which would be even better. But is it valid?

Actually no - each block must be independently decomposed. The tensor for
the full 2x4x4 is the "direct sum" (in a sense) of the 4 block tensors,
and the rank of a direct sum is at most the sum of ranks. So 3*7+4=25
would give rank at most 25, but we need to verify it's exact.

Wait, but the blocks share A rows and C rows. Let me reconsider.

A is 2x4. The 4 blocks use:
- Block (0,0): A rows 0,1; A cols 0,1
- Block (0,1): A rows 0,1; A cols 0,1
- Block (1,0): A rows 0,1; A cols 2,3
- Block (1,1): A rows 0,1; A cols 2,3

So blocks (0,0) and (0,1) share the same A submatrix A[:,0:2].
Blocks (1,0) and (1,1) share the same A submatrix A[:,2:4].

Similarly for C:
- Block (0,0): C cols 0,1
- Block (0,1): C cols 2,3
- Block (1,0): C cols 0,1
- Block (1,1): C cols 2,3

So blocks (0,0) and (1,0) share C[:,0:2], and blocks (0,1) and (1,1) share C[:,2:4].

This sharing might allow us to reduce the total rank below 28.

Let me try: use Strassen (rank 7) for blocks (0,0), (0,1), (1,0) and
schoolbook (rank 4) for block (1,1). Total = 7+7+7+4 = 25.

But we need to verify this actually works. The issue is that the U, V, W
factors for different blocks must not interfere with each other.

For block (0,0): U uses A[:,0:2] entries, V uses B[0:2,0:2] entries, W uses C[:,0:2] entries
For block (0,1): U uses A[:,0:2] entries, V uses B[0:2,2:4] entries, W uses C[:,2:4] entries
For block (1,0): U uses A[:,2:4] entries, V uses B[2:4,0:2] entries, W uses C[:,0:2] entries
For block (1,1): U uses A[:,2:4] entries, V uses B[2:4,2:4] entries, W uses C[:,2:4] entries

Since the V factors use disjoint B entries and the U/W factors use disjoint
A/C entries within each "row group", there's no interference. So the direct
sum construction is valid, and rank 25 should work!

Let me verify: can we do even better? What about 7+7+4+4 = 22? Or 7+4+7+4 = 22?

Actually, the minimum rank for 2x2x2 is 7 (Strassen), and for the full problem
we need at least some combination. Let me just try rank 25 first (3 Strassen + 1 schoolbook)
and verify. If it works, great. If not, try other combinations.

Actually wait - I realize the rank might be lower. Let me just try to construct
rank 25 and verify it exactly.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = 140

    n, m, p = DIMENSIONS

    # Build the target tensor T[i,j,k] for 2x4x4
    # i = row_a * 4 + col_a (A is 2x4)
    # j = row_b * 4 + col_b (B is 4x4)
    # k = col_c * 2 + row_c (C is 2x4, stored as W[k*n+i,r] convention: k = col*n + row)
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for row_a in range(n):
        for col_a in range(m):
            i = row_a * m + col_a
            for row_b in range(m):
                for col_b in range(p):
                    j = row_b * p + col_b
                    for row_c in range(n):
                        for col_c in range(p):
                            k = col_c * n + row_c
                            if col_a == row_b and row_a == row_c and col_b == col_c:
                                T[i, j, k] = 1.0

    # Strassen 2x2x2 factors
    U_s = np.array([
        [1, 0, 1, 0, 1, 0, 1],
        [0, 0, 1, -1, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, -1],
        [1, 1, 0, 1, 0, -1, 0]
    ], dtype=float)

    V_s = np.array([
        [1, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 1]
    ], dtype=float)

    # Derive W_s by solving the linear system for 2x2x2
    T_2 = np.zeros((4, 4, 4), dtype=float)
    for ra in range(2):
        for ca in range(2):
            ii = ra * 2 + ca
            for rb in range(2):
                for cb in range(2):
                    jj = rb * 2 + cb
                    for rc in range(2):
                        for cc in range(2):
                            kk = cc * 2 + rc
                            if ca == rb and ra == rc and cb == cc:
                                T_2[ii, jj, kk] = 1.0

    M_sys = np.zeros((64, 28), dtype=float)
    T_vec = np.zeros(64, dtype=float)
    for ii in range(4):
        for jj in range(4):
            for kk in range(4):
                eq = ii * 16 + jj * 4 + kk
                T_vec[eq] = T_2[ii, jj, kk]
                for s in range(7):
                    M_sys[eq, kk * 7 + s] = U_s[ii, s] * V_s[jj, s]

    W_vec, _, _, _ = np.linalg.lstsq(M_sys, T_vec, rcond=None)
    W_s = W_vec.reshape(4, 7)

    # Verify Strassen decomposition
    err_s = 0.0
    for ii in range(4):
        for jj in range(4):
            for kk in range(4):
                val = np.sum(U_s[ii, :] * V_s[jj, :] * W_s[kk, :])
                err_s = max(err_s, abs(val - T_2[ii, jj, kk]))

    if err_s > 1e-9:
        # Fallback: schoolbook rank 32
        U_fb = np.zeros((8, 32))
        V_fb = np.zeros((16, 32))
        W_fb = np.zeros((8, 32))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Schoolbook 2x2x2 factors (rank 4)
    U_sb = np.zeros((4, 4), dtype=float)
    V_sb = np.zeros((4, 4), dtype=float)
    W_sb = np.zeros((4, 4), dtype=float)
    for ra in range(2):
        for cb in range(2):
            col = ra * 2 + cb
            U_sb[ra * 2 + cb, col] = 1.0
            V_sb[cb * 2 + ra, col] = 1.0  # Wait, this isn't right
            W_sb[cb * 2 + ra, col] = 1.0  # This isn't right either

    # Let me redo schoolbook properly.
    # For 2x2x2: C[ra, cb] = sum_rb A[ra, rb] * B[rb, cb]
    # Term: A[ra, rb] * B[rb, cb] -> C[ra, cb]
    # A flat index: ra * 2 + rb
    # B flat index: rb * 2 + cb
    # C flat index: cb * 2 + ra
    U_sb = np.zeros((4, 4), dtype=float)
    V_sb = np.zeros((4, 4), dtype=float)
    W_sb = np.zeros((4, 4), dtype=float)
    col = 0
    for ra in range(2):
        for rb in range(2):
            for cb in range(2):
                U_sb[ra * 2 + rb, col] = 1.0
                V_sb[rb * 2 + cb, col] = 1.0
                W_sb[cb * 2 + ra, col] = 1.0
                col += 1

    # Verify schoolbook
    err_sb = 0.0
    for ii in range(4):
        for jj in range(4):
            for kk in range(4):
                val = np.sum(U_sb[ii, :] * V_sb[jj, :] * W_sb[kk, :])
                err_sb = max(err_sb, abs(val - T_2[ii, jj, kk]))

    if err_sb > 1e-9:
        U_fb = np.zeros((8, 32))
        V_fb = np.zeros((16, 32))
        W_fb = np.zeros((8, 32))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Now try different rank combinations
    # 4 blocks, each can be Strassen (7) or schoolbook (4)
    # Total rank = sum of chosen ranks
    # Try: 7+7+7+4 = 25, 7+7+4+4 = 22, 7+4+7+4 = 22, 4+7+7+4 = 22, etc.
    # Also try all Strassen: 7+7+7+7 = 28

    # Define block mappings
    # Block (bi, bj): A cols [2*bi:2*bi+2], B rows [2*bi:2*bi+2], B cols [2*bj:2*bj+2], C cols [2*bj:2*bj+2]
    # A rows are always 0,1
    # C rows are always 0,1

    def build_decomposition(rank_choice):
        """
        rank_choice: list of 4 values, each 7 (Strassen) or 4 (schoolbook)
        Returns U, V, W for the full 2x4x4 problem
        """
        total_rank = sum(rank_choice)
        U_full = np.zeros((8, total_rank), dtype=float)
        V_full = np.zeros((16, total_rank), dtype=float)
        W_full = np.zeros((8, total_rank), dtype=float)

        offset = 0
        for bi in range(2):
            for bj in range(2):
                block_idx = bi * 2 + bj
                r = rank_choice[block_idx]
                if r == 7:
                    U_block = U_s
                    V_block = V_s
                    W_block = W_s
                else:
                    U_block = U_sb
                    V_block = V_sb
                    W_block = W_sb

                # Map local indices to global
                # A: local (ra, ca) -> global (ra, 2*bi + ca) -> flat: ra*4 + (2*bi + ca)
                for ra in range(2):
                    for ca in range(2):
                        local_i = ra * 2 + ca
                        global_i = ra * 4 + (2 * bi + ca)
                        U_full[global_i, offset:offset + r] = U_block[local_i, :]

                # B: local (rb, cb) -> global (2*bi + rb, 2*bj + cb) -> flat: (2*bi+rb)*4 + (2*bj+cb)
                for rb in range(2):
                    for cb in range(2):
                        local_j = rb * 2 + cb
                        global_j = (2 * bi + rb) * 4 + (2 * bj + cb)
                        V_full[global_j, offset:offset + r] = V_block[local_j, :]

                # C: local (rc, cc) -> global (rc, 2*bj + cc) -> flat: (2*bj+cc)*2 + rc
                for rc in range(2):
                    for cc in range(2):
                        local_k = cc * 2 + rc
                        global_k = (2 * bj + cc) * 2 + rc
                        W_full[global_k, offset:offset + r] = W_block[local_k, :]

                offset += r

        return U_full, V_full, W_full

    def verify_decomposition(U_full, V_full, W_full):
        max_err = 0.0
        for i in range(8):
            for j in range(16):
                for k in range(8):
                    val = np.dot(U_full[i, :] * V_full[j, :], W_full[k, :])
                    err = abs(val - T[i, j, k])
                    if err > max_err:
                        max_err = err
        return max_err

    # Try different combinations
    best_rank = 32
    best_U, best_V, best_W = None, None, None

    # Try all 2^4 = 16 combinations
    for mask in range(16):
        rank_choice = []
        for b in range(4):
            if (mask >> b) & 1:
                rank_choice.append(7)
            else:
                rank_choice.append(4)

        total_r = sum(rank_choice)
        if total_r >= best_rank:
            continue

        U_try, V_try, W_try = build_decomposition(rank_choice)
        err = verify_decomposition(U_try, V_try, W_try)

        if err < 1e-9:
            best_rank = total_r
            best_U, best_V, best_W = U_try, V_try, W_try

        if time.time() - start_time > deadline:
            break

    # Also try all Strassen (rank 28) as a guaranteed fallback
    if best_rank > 28 or best_U is None:
        rank_choice = [7, 7, 7, 7]
        U_try, V_try, W_try = build_decomposition(rank_choice)
        err = verify_decomposition(U_try, V_try, W_try)
        if err < 1e-9:
            best_rank = 28
            best_U, best_V, best_W = U_try, V_try, W_try

    # Final fallback: schoolbook rank 32
    if best_U is None:
        U_fb = np.zeros((8, 32))
        V_fb = np.zeros((16, 32))
        W_fb = np.zeros((8, 32))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Round to nearest half-integer and re-verify
    U_rounded = np.round(best_U * 2) / 2
    V_rounded = np.round(best_V * 2) / 2
    W_rounded = np.round(best_W * 2) / 2

    err_rounded = verify_decomposition(U_rounded, V_rounded, W_rounded)
    if err_rounded < 1e-9:
        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
