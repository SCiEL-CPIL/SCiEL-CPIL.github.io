"""Combined program: Strassen 4x4x4 + block-Strassen 4x4x2 + searched 4x4x3 for 4x4x7 matrix multiplication."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # ---- Shared Strassen 2x2x2 terms (x_sub, y_sub, c_sub) ----
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
    ]

    # ---- Strassen 4x4x4 outer structure ----
    strassen_4x4_outer = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}),
        ({0: 1}, {1: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}),
        ({0: 1, 1: 1}, {3: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]
    c_blocks_4x4 = [
        {(0,0): 1.0, (1,1): 1.0},
        {(1,0): 1.0, (1,1): -1.0},
        {(0,1): 1.0, (1,1): 1.0},
        {(0,0): 1.0, (1,0): 1.0},
        {(0,0): -1.0, (0,1): 1.0},
        {(1,1): 1.0},
        {(0,0): 1.0},
    ]

    def build_4x4x4(U, V, W, start_rank):
        """Fill 49 ranks for the 4x4x4 Strassen part (columns 0..3 of B, rows 0..3 of C)."""
        cur = start_rank
        for m_idx in range(7):
            x_def, y_def = strassen_4x4_outer[m_idx]
            c_block = c_blocks_4x4[m_idx]
            for t_idx in range(7):
                r = cur
                cur += 1
                x_sub, y_sub, c_sub = strassen_2x2[t_idx]
                for (li, lj), cx in x_sub.items():
                    for b, cb in x_def.items():
                        bi, bj = b // 2, b % 2
                        i = 2 * bi + li
                        j = 2 * bj + lj
                        U[i * m + j, r] += cx * cb
                for (li, lj), cy in y_sub.items():
                    for b, cb in y_def.items():
                        bi, bj = b // 2, b % 2
                        i = 2 * bi + li
                        j = 2 * bj + lj
                        V[i * p + j, r] += cy * cb
                for (li, lj), cc in c_sub.items():
                    if cc == 0:
                        continue
                    for (br, bc), cb_c in c_block.items():
                        if cb_c == 0:
                            continue
                        i_final = 2 * br + li
                        k_final = 2 * bc + lj
                        W[k_final * n + i_final, r] += cc * cb_c
        return cur  # start_rank + 49

    def add_strassen_2x2(U, V, W, cur, A_block_rows, A_block_cols,
                         B_block_rows, B_block_cols, C_block_rows, C_block_cols):
        """Add a Strassen 2x2x2 block decomposition; returns new cur."""
        for t_idx in range(7):
            r = cur
            cur += 1
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]
            for (li, lj), cx in x_sub.items():
                i = A_block_rows[li]
                j = A_block_cols[lj]
                U[i * m + j, r] += cx
            for (li, lj), cy in y_sub.items():
                i = B_block_rows[li]
                j = B_block_cols[lj]
                V[i * p + j, r] += cy
            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                i_final = C_block_rows[li]
                k_final = C_block_cols[lj]
                W[k_final * n + i_final, r] += cc
        return cur

    # ---- Build the proven rank-93 baseline (4x4x4 + 4x4x2 block + 4x4x1 schoolbook) ----
    def build_baseline():
        total_rank = 93
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        cur = build_4x4x4(U, V, W, 0)  # 49

        # 4x4x2 block Strassen (columns 4,5)
        A00_r = [0, 1]; A00_c = [0, 1]
        A01_r = [0, 1]; A01_c = [2, 3]
        A10_r = [2, 3]; A10_c = [0, 1]
        A11_r = [2, 3]; A11_c = [2, 3]
        B_top_r = [0, 1]; B_top_c = [4, 5]
        B_bot_r = [2, 3]; B_bot_c = [4, 5]
        C_top_r = [0, 1]; C_top_c = [4, 5]
        C_bot_r = [2, 3]; C_bot_c = [4, 5]
        cur = add_strassen_2x2(U, V, W, cur, A00_r, A00_c, B_top_r, B_top_c, C_top_r, C_top_c)
        cur = add_strassen_2x2(U, V, W, cur, A01_r, A01_c, B_bot_r, B_bot_c, C_top_r, C_top_c)
        cur = add_strassen_2x2(U, V, W, cur, A10_r, A10_c, B_top_r, B_top_c, C_bot_r, C_bot_c)
        cur = add_strassen_2x2(U, V, W, cur, A11_r, A11_c, B_bot_r, B_bot_c, C_bot_r, C_bot_c)  # +28

        # 4x4x1 schoolbook (column 6)
        for i in range(n):
            for j in range(m):
                r = cur
                cur += 1
                U[i * m + j, r] = 1.0
                V[j * p + 6, r] = 1.0
                W[6 * n + i, r] = 1.0
        assert cur == total_rank, f"baseline rank {cur} != {total_rank}"
        return U, V, W, total_rank

    # ---- Rank-reduction search for the 4x4x3 sub-problem (columns 4,5,6) ----
    # The sub-tensor: for each (i,j,k) with k in {4,5,6}, value 1.
    # A decomposition of rank r is U (16 x r), V (28 x r), W (28 x r) such that
    #   sum_t U[i,j,t] V[j,k,t] W[k,i,t] = 1  for k in {4,5,6}, else 0.
    def build_sub_4x4x3(U, V, W, start_rank, r):
        """Fill the 4x4x3 sub-problem using the searched (or baseline) factors."""
        cur = start_rank
        if r == 48:
            # schoolbook for columns 4,5,6
            for i in range(n):
                for j in range(m):
                    for k in range(4, p):
                        rr = cur
                        cur += 1
                        U[i * m + j, rr] = 1.0
                        V[j * p + k, rr] = 1.0
                        W[k * n + i, rr] = 1.0
            return cur
        else:
            # Use searched factors stored in module-level cache
            for t in range(r):
                rr = cur
                cur += 1
                for i in range(n):
                    for j in range(m):
                        U[i * m + j, rr] = sub_factors['U'][i, j, t]
                for j in range(m):
                    for k in range(4, p):
                        V[j * p + k, rr] = sub_factors['V'][j, k - 4, t]
                for k in range(4, p):
                    for i in range(n):
                        W[k * n + i, rr] = sub_factors['W'][k - 4, i, t]
            return cur

    # Global cache for the best found 4x4x3 sub-factors
    sub_factors = {'U': None, 'V': None, 'W': None, 'rank': 48}

    def verify_sub(U_sub, V_sub, W_sub, r):
        """Check the 4x4x3 sub-decomposition. U_sub (16,r), V_sub (28,r) for k in {4,5,6}, W_sub (28,r).
        Returns max abs error vs the target tensor (1 for k in {4,5,6})."""
        # Target: T[i,j,k] = 1 if k in {4,5,6} else 0, for i in 0..3, j in 0..3, k in 0..6
        # But sub only covers k in {4,5,6}. We check: sum_t U[i,j,t] V[j,k,t] W[k,i,t] = 1 for k in {4,5,6}
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(4, p):
                    s = 0.0
                    for t in range(r):
                        s += U_sub[i * m + j, t] * V_sub[j * p + k, t] * W_sub[k * n + i, t]
                    e = abs(s - 1.0)
                    if e > err:
                        err = e
        return err

    def search_4x4x3(budget):
        """Try to find a 4x4x3 decomposition with rank < 48 within budget seconds."""
        global sub_factors
        best = {'U': None, 'V': None, 'W': None, 'rank': 48, 'err': 1e9}
        t0 = time.time()
        rng = np.random.default_rng(12345)

        # Target tensor for the sub-problem (only k in {4,5,6} matter)
        # We'll work in a reduced coordinate: k' = k - 4 in {0,1,2}
        # U: 16 x r (i*4+j), V: 28 x r (j*7+k'), W: 28 x r (k'*4+i)
        # Reconstruct: U_full[i*4+j], V_full[j*7 + (k'+4)]... but we only fill the 4x4x3 part.
        # For verification we use the full U,V,W via build_sub_4x4x3 + verify_sub.

        # Structured random initialization: start from schoolbook and perturb, or random.
        # Try a few ranks descending from 47.
        for target_rank in range(47, 20, -1):
            if time.time() - t0 > budget:
                break
            # Try a few random initializations for this rank
            for init in range(3):
                if time.time() - t0 > budget:
                    break
                U_sub = rng.standard_normal((n * m, target_rank))
                V_sub = np.zeros((m * p, target_rank))
                W_sub = np.zeros((p * n, target_rank))
                # Initialize V, W to spread over k in {4,5,6}
                for t in range(target_rank):
                    for j in range(m):
                        for k in range(4, p):
                            V_sub[j * p + k, t] = rng.standard_normal()
                    for k in range(4, p):
                        for i in range(n):
                            W_sub[k * n + i, t] = rng.standard_normal()
                # Iterative refinement: gradient-descent-like on log-sum-exp surrogate
                # Objective: sum over (i,j,k) of (sum_t U[i,j,t] V[j,k,t] W[k,i,t] - 1)^2
                # Use simple coordinate/rank-one updates.
                lr = 0.01
                for it in range(200):
                    if time.time() - t0 > budget:
                        break
                    # Compute gradient w.r.t. U, V, W
                    # dL/dU[i,j,t] = 2 * sum_k (res[i,j,k]) * V[j,k,t] * W[k,i,t]
                    # We'll do a simplified update: for each t, treat as rank-one.
                    # For efficiency, do a few random rank-one coordinate updates.
                    # res[i,j,k] = sum_t U[i,j,t] V[j,k,t] W[k,i,t] - 1
                    # Update U[i,j,t] -= lr * 2 * res[i,j,k] * V[j,k,t] * W[k,i,t] (summed over k)
                    # This is O(n*m*p*rank) per iteration which is 4*4*3*rank ~ 48*rank.
                    # For rank ~ 40, that's ~1920 ops per iter, 200 iters ~ 384k ops, fast.
                    # Compute residual
                    res = np.zeros((n, m, p))
                    for i in range(n):
                        for j in range(m):
                            for k in range(4, p):
                                s = 0.0
                                for t in range(target_rank):
                                    s += U_sub[i * m + j, t] * V_sub[j * p + k, t] * W_sub[k * n + i, t]
                                res[i, j, k] = s - 1.0
                    # Update
                    for i in range(n):
                        for j in range(m):
                            for t in range(target_rank):
                                g = 0.0
                                for k in range(4, p):
                                    g += res[i, j, k] * V_sub[j * p + k, t] * W_sub[k * n + i, t]
                                U_sub[i * m + j, t] -= lr * 2 * g
                    for j in range(m):
                        for k in range(4, p):
                            for t in range(target_rank):
                                g = 0.0
                                for i in range(n):
                                    g += res[i, j, k] * U_sub[i * m + j, t] * W_sub[k * n + i, t]
                                V_sub[j * p + k, t] -= lr * 2 * g
                    for k in range(4, p):
                        for i in range(n):
                            for t in range(target_rank):
                                g = 0.0
                                for j in range(m):
                                    g += res[i, j, k] * U_sub[i * m + j, t] * V_sub[j * p + k, t]
                                W_sub[k * n + i, t] -= lr * 2 * g
                    # Check convergence
                    err = np.max(np.abs(res))
                    if err < 1e-8:
                        # Found a solution!
                        if target_rank < best['rank']:
                            best = {'U': U_sub.copy(), 'V': V_sub.copy(), 'W': W_sub.copy(),
                                     'rank': target_rank, 'err': err}
                        break
                    # Adaptive learning rate
                    if err < 0.1:
                        lr = max(0.001, lr * 0.9)
                # After refinement, check if we got a good solution
                if best['rank'] == 48:
                    # Re-evaluate the last init
                    U_sub2 = U_sub.copy()
                    V_sub2 = V_sub.copy()
                    W_sub2 = W_sub.copy()
                    err = verify_sub(U_sub2, V_sub2, W_sub2, target_rank)
                    if err < 1e-6 and target_rank < best['rank']:
                        best = {'U': U_sub2, 'V': V_sub2, 'W': W_sub2,
                                 'rank': target_rank, 'err': err}
            # If we found a rank, stop searching lower
            if best['rank'] < 48:
                break

        if best['rank'] < 48:
            sub_factors['U'] = best['U']
            sub_factors['V'] = best['V']
            sub_factors['W'] = best['W']
            sub_factors['rank'] = best['rank']
        return sub_factors['rank']

    # ---- Main build ----
    # First build the baseline (rank 93) as the guaranteed fallback
    U_base, V_base, W_base, rank_base = build_baseline()

    # Try the search for the 4x4x3 sub-problem
    search_budget = min(30.0, deadline - (time.time() - start_time))
    if search_budget > 5.0:
        search_4x4x3(search_budget)

    # Build the final solution using the searched sub-factors (or baseline if search failed)
    sub_rank = sub_factors['rank']
    if sub_rank < 48:
        # Use 4x4x4 (49) + 4x4x2 block (28) + searched 4x4x3 (sub_rank)
        # But wait: the 4x4x2 block already handles columns 4,5. The searched 4x4x3 handles 4,5,6.
        # This double-counts columns 4,5! We need to be careful.
        # The baseline is: 4x4x4 (cols 0-3) + 4x4x2 block (cols 4,5) + 4x4x1 schoolbook (col 6).
        # The searched version should be: 4x4x4 (cols 0-3) + searched 4x4x3 (cols 4,5,6).
        # So we drop the 4x4x2 block and 4x4x1 schoolbook, and use the searched 4x4x3.
        total_rank = 49 + sub_rank
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        cur = build_4x4x4(U, V, W, 0)  # 49
        cur = build_sub_4x4x3(U, V, W, cur, sub_rank)  # sub_rank
        assert cur == total_rank, f"searched rank {cur} != {total_rank}"
        rank = total_rank
    else:
        # Fall back to the proven baseline (rank 93)
        U, V, W = U_base, V_base, W_base
        rank = rank_base

    # ---- Verification ----
    def verify_full(U, V, W, r):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        u_idx = i_idx * m + j_idx
        v_idx = j_idx * p + k_idx
        w_idx = k_idx * n + i_idx
        u_vals = Ua[u_idx, :]
        v_vals = Va[v_idx, :]
        w_vals = Wa[w_idx, :]
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        return np.max(np.abs(result - 1.0))

    err = verify_full(U, V, W, rank)
    if err < 1e-10:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # If verification fails, fall back to the schoolbook (rank 112)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
