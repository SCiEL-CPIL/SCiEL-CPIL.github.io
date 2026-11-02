import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin before 150s kill

    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12

    # Construct the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Helper to check if a decomposition is valid and roundable
    def check_and_round(U, V, W):
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        T_calc = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(T_calc, T, atol=1e-9):
            return U_r, V_r, W_r
        return None

    # Helper to perform ALS for a given rank with Strassen-based initialization
    def try_rank(R, max_iters=80, n_restarts=8):
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break

            np.random.seed(42 + restart * 1000 + R)

            # Initialize using Strassen structure as a warm start, then perturb
            U0, V0, W0 = _strassen_init(R)
            if U0 is None:
                U = np.random.randn(nm, R) * 0.5
                V = np.random.randn(mp, R) * 0.5
                W = np.random.randn(pn, R) * 0.5
            else:
                U = U0 + np.random.randn(nm, R) * 0.1
                V = V0 + np.random.randn(mp, R) * 0.1
                W = W0 + np.random.randn(pn, R) * 0.1

            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break

                # Update W: solve linear system
                A_mat = np.zeros((24, pn * R))
                b_vec = np.zeros(24)
                idx = 0
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            if T[a, b, c] != 0:
                                for r in range(R):
                                    A_mat[idx, c * R + r] = U[a, r] * V[b, r]
                                b_vec[idx] = T[a, b, c]
                                idx += 1
                try:
                    W_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W = W_flat.reshape(pn, R)
                except:
                    continue

                # Update U
                A_mat = np.zeros((24, nm * R))
                b_vec = np.zeros(24)
                idx = 0
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            if T[a, b, c] != 0:
                                for r in range(R):
                                    A_mat[idx, a * R + r] = V[b, r] * W[c, r]
                                b_vec[idx] = T[a, b, c]
                                idx += 1
                try:
                    U_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U = U_flat.reshape(nm, R)
                except:
                    continue

                # Update V
                A_mat = np.zeros((24, mp * R))
                b_vec = np.zeros(24)
                idx = 0
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            if T[a, b, c] != 0:
                                for r in range(R):
                                    A_mat[idx, b * R + r] = U[a, r] * W[c, r]
                                b_vec[idx] = T[a, b, c]
                                idx += 1
                try:
                    V_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    V = V_flat.reshape(mp, R)
                except:
                    continue

                # Check convergence
                T_calc = np.einsum('ar,br,cr->abc', U, V, W)
                err = np.linalg.norm(T_calc - T)
                if err < 1e-8:
                    result = check_and_round(U, V, W)
                    if result is not None:
                        return result

        return None

    def _strassen_init(R):
        """Initialize with Strassen structure, padded/truncated to rank R."""
        if R < 7:
            return None, None, None
        strassen_steps = [
            {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
            {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
            {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
            {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
            {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
            {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
            {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
        ]
        U = np.zeros((nm, R))
        V = np.zeros((mp, R))
        W = np.zeros((pn, R))
        for g in range(3):
            for s in range(7):
                r = 7 * g + s
                if r >= R:
                    break
                step = strassen_steps[s]
                k0 = 2 * g
                k1 = 2 * g + 1
                b_map = [6*0+k0, 6*0+k1, 6*1+k0, 6*1+k1]
                c_map = [2*k0+0, 2*k1+0, 2*k0+1, 2*k1+1]
                for l in range(4):
                    U[l, r] = step['u'][l]
                    V[b_map[l], r] = step['v'][l]
                    W[c_map[l], r] = step['w'][l]
        return U, V, W

    # Try slice merging: find linear combination of 3 Strassen blocks that gives rank-1
    def _try_slice_merge():
        """Try to find alpha, beta, gamma such that the combined tensor has rank 1,
        then construct a rank-20 decomposition by absorbing that term."""
        if time.time() > deadline - 15:
            return None

        # Build the 3 block Strassen tensors
        strassen_steps = [
            {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
            {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
            {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
            {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
            {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
            {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
            {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
        ]

        # Build block tensors for each of the 3 blocks
        block_tensors = []
        for g in range(3):
            block_T = np.zeros((nm, mp, pn))
            for s in range(7):
                step = strassen_steps[s]
                k0 = 2 * g
                k1 = 2 * g + 1
                b_map = [6*0+k0, 6*0+k1, 6*1+k0, 6*1+k1]
                c_map = [2*k0+0, 2*k1+0, 2*k0+1, 2*k1+1]
                for l in range(4):
                    block_T[l, b_map[l], c_map[l]] = step['u'][l] * step['v'][l] * step['w'][l]
            block_tensors.append(block_T)

        # We want: alpha*B0 + beta*B1 + gamma*B2 = u ⊗ v ⊗ w (rank 1)
        # Try grid search over small rational values
        candidates = []
        for a_val in [-1, -0.5, 0, 0.5, 1, 1.5, 2]:
            for b_val in [-1, -0.5, 0, 0.5, 1, 1.5, 2]:
                for g_val in [-1, -0.5, 0, 0.5, 1, 1.5, 2]:
                    if a_val == 0 and b_val == 0 and g_val == 0:
                        continue
                    combined = a_val * block_tensors[0] + b_val * block_tensors[1] + g_val * block_tensors[2]
                    # Check if this is rank 1: all 2x2 minors of matricizations should be zero
                    # Matricize as (nm*mp, pn)
                    mat = combined.reshape(nm * mp, pn)
                    # Check rank via SVD
                    try:
                        s_vals = np.linalg.svd(mat, compute_uv=False)
                    except:
                        continue
                    # Rank 1 means only one non-zero singular value
                    if len(s_vals) > 1 and s_vals[1] < 1e-10 and s_vals[0] > 1e-10:
                        candidates.append((a_val, b_val, g_val, combined))

        if not candidates:
            return None

        # Use the first candidate found
        a_val, b_val, g_val, combined = candidates[0]

        # Extract the rank-1 vectors from combined tensor
        # Find the first non-zero element
        nz = np.unravel_index(np.argmax(np.abs(combined)), combined.shape)
        u_vec = np.zeros(nm)
        v_vec = np.zeros(mp)
        w_vec = np.zeros(pn)
        scale = combined[nz]

        # u_vec from fixing b,c to nz[1],nz[2]
        u_vec[:] = combined[:, nz[1], nz[2]] / scale
        # v_vec from fixing a,c to nz[0],nz[2]
        v_vec[:] = combined[nz[0], :, nz[2]] / scale
        # w_vec from fixing a,b to nz[0],nz[1]
        w_vec[:] = combined[nz[0], nz[1], :] / scale

        # Verify: u ⊗ v ⊗ w should equal combined
        recon = np.einsum('a,b,c->abc', u_vec, v_vec, w_vec)
        if not np.allclose(recon, combined, atol=1e-9):
            return None

        # Now build rank-20 decomposition:
        # For each block g, we have 7 Strassen terms.
        # The combined rank-1 term = a*B0 + b*B1 + g*B2
        # We can express one term from each block as part of this rank-1.
        # Strategy: keep 6 terms from block 0, 6 from block 1, 6 from block 2 = 18 terms
        # Plus 1 rank-1 term = 19... but we need exactly 20.
        # Actually: total 21 terms. If we can show that one specific term from each block
        # sums to a rank-1 tensor, we replace 3 terms with 1, getting 21-3+1=19.
        # But that's too aggressive. Let's try: absorb the rank-1 combination.
        #
        # Alternative: The 21-term decomposition is block diagonal.
        # If alpha*B0 + beta*B1 + gamma*B2 = rank-1, then:
        # B0 + B1 + B2 = (1/alpha)*(rank-1) + (1-alpha/alpha)*B0 + ... 
        # This is getting complex. Let's just try ALS for rank 20 with this as init.

        # Build a rank-20 initialization: take all 21 Strassen terms, zero out the
        # contribution that forms the rank-1, and add the rank-1 as a new term.
        # Simpler: just use the 21-term init but set one column to zero and run ALS.

        # Build full 21-term matrices
        U21 = np.zeros((nm, 21))
        V21 = np.zeros((mp, 21))
        W21 = np.zeros((pn, 21))
        for g in range(3):
            for s in range(7):
                r = 7 * g + s
                step = strassen_steps[s]
                k0 = 2 * g
                k1 = 2 * g + 1
                b_map = [6*0+k0, 6*0+k1, 6*1+k0, 6*1+k1]
                c_map = [2*k0+0, 2*k1+0, 2*k0+1, 2*k1+1]
                for l in range(4):
                    U21[l, r] = step['u'][l]
                    V21[b_map[l], r] = step['v'][l]
                    W21[c_map[l], r] = step['w'][l]

        # Zero out the last column (term 20) to create rank-20 init
        U20 = U21[:, :20].copy()
        V20 = V21[:, :20].copy()
        W20 = W21[:, :20].copy()

        # Run ALS from this init
        U = U20 + np.random.randn(nm, 20) * 0.05
        V = V20 + np.random.randn(mp, 20) * 0.05
        W = W20 + np.random.randn(pn, 20) * 0.05

        for it in range(100):
            if time.time() > deadline - 10:
                break

            # Update W
            A_mat = np.zeros((24, pn * 20))
            b_vec = np.zeros(24)
            idx = 0
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        if T[a, b, c] != 0:
                            for r in range(20):
                                A_mat[idx, c * 20 + r] = U[a, r] * V[b, r]
                            b_vec[idx] = T[a, b, c]
                            idx += 1
            try:
                W_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W = W_flat.reshape(pn, 20)
            except:
                continue

            # Update U
            A_mat = np.zeros((24, nm * 20))
            b_vec = np.zeros(24)
            idx = 0
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        if T[a, b, c] != 0:
                            for r in range(20):
                                A_mat[idx, a * 20 + r] = V[b, r] * W[c, r]
                            b_vec[idx] = T[a, b, c]
                            idx += 1
            try:
                U_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                U = U_flat.reshape(nm, 20)
            except:
                continue

            # Update V
            A_mat = np.zeros((24, mp * 20))
            b_vec = np.zeros(24)
            idx = 0
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        if T[a, b, c] != 0:
                            for r in range(20):
                                A_mat[idx, b * 20 + r] = U[a, r] * W[c, r]
                            b_vec[idx] = T[a, b, c]
                            idx += 1
            try:
                V_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                V = V_flat.reshape(mp, 20)
            except:
                continue

            # Check convergence
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_calc - T)
            if err < 1e-8:
                result = check_and_round(U, V, W)
                if result is not None:
                    return result

        return None

    # First try slice merging approach
    merge_result = _try_slice_merge()
    if merge_result is not None:
        U, V, W = merge_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find lower ranks, starting from 20 down to 15
    for R in range(20, 14, -1):
        if time.time() > deadline - 10:
            break
        result = try_rank(R, max_iters=60, n_restarts=6)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to Rank 21 (Strassen based)
    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]

    U = np.zeros((4, 21))
    V = np.zeros((12, 21))
    W = np.zeros((12, 21))

    for g in range(3):
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            k0 = 2 * g
            k1 = 2 * g + 1
            b_map = [6*0+k0, 6*0+k1, 6*1+k0, 6*1+k1]
            c_map = [2*k0+0, 2*k1+0, 2*k0+1, 2*k1+1]
            for l in range(4):
                U[l, r] = step['u'][l]
                V[b_map[l], r] = step['v'][l]
                W[c_map[l], r] = step['w'][l]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
