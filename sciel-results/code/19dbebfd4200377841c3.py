"""Matrix multiplication 4,4,9 tensor decomposition - structured Strassen refinement."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    deadline = time.time() + 145
    n, m, p = DIMENSIONS  # 4, 4, 9
    n_dim, m_dim, p_dim = n * m, m * p, p * n  # 16, 36, 36

    # Build target tensor
    target = np.zeros(n_dim * m_dim * p_dim, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                target[a_idx * (m_dim * p_dim) + b_idx * p_dim + c_idx] = 1.0

    def verify(U, V, W, rank):
        T = np.zeros((n_dim, m_dim, p_dim))
        for r in range(rank):
            T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.allclose(T.flatten(), target, atol=1e-9)

    def construct_strassen():
        """Known rank-114 Strassen-based decomposition."""
        U_cols, V_cols, W_cols = [], [], []

        def add_term(u, v, w):
            U_cols.append(u); V_cols.append(v); W_cols.append(w)

        strassen_2x2 = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
        ]

        def add_strassen_block(col_off):
            def add_inner(A_blocks, B_blocks, C_blocks):
                for uc, vc, wc in strassen_2x2:
                    u = [0] * 16; v = [0] * 36; w = [0] * 36
                    for loc, c in uc.items():
                        li, lj = loc // 2, loc % 2
                        for br, bc, bcc in A_blocks:
                            u[(br * 2 + li) * m + (bc * 2 + lj)] += c * bcc
                    for loc, c in vc.items():
                        lj, lk = loc // 2, loc % 2
                        for br, bc, bcc in B_blocks:
                            v[(br * 2 + lj) * p + col_off + bc * 2 + lk] += c * bcc
                    for loc, c in wc.items():
                        li, lk = loc // 2, loc % 2
                        for br, bc, bcc in C_blocks:
                            w[col_off + bc * 2 + lk] * 0  # placeholder
                            w[(col_off + bc * 2 + lk) * n + br * 2 + li] += c * bcc
                    add_term(u, v, w)
            add_inner([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)])
            add_inner([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)])
            add_inner([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)])
            add_inner([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)])
            add_inner([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)])
            add_inner([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)])
            add_inner([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)])

        add_strassen_block(0)
        add_strassen_block(4)
        for i in range(4):
            for j in range(4):
                u = [0] * 16; v = [0] * 36; w = [0] * 36
                u[i * m + j] = 1; v[j * p + 8] = 1; w[8 * n + i] = 1
                add_term(u, v, w)

        rank = len(U_cols)
        U = np.array([[U_cols[c][r] for c in range(rank)] for r in range(16)], dtype=float)
        V = np.array([[V_cols[c][r] for c in range(rank)] for r in range(36)], dtype=float)
        W = np.array([[W_cols[c][r] for c in range(rank)] for r in range(36)], dtype=float)
        return U, V, W

    # Fallback
    U_f, V_f, W_f = construct_strassen()
    if not verify(U_f, V_f, W_f, 114):
        # Should not happen, but return anyway
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

    best_rank = 114
    best_U, best_V, best_W = U_f, V_f, W_f

    # Structured search: try to merge/cancel terms
    # Approach: fix supports from Strassen, optimize values, try to nullify some columns
    def try_merge(rank_target, seed=0):
        """Try to find rank < 114 by merging Strassen terms."""
        rng = random.Random(seed)
        U0, V0, W0 = construct_strassen()
        R0 = 114

        for attempt in range(200):
            if time.time() > deadline - 10:
                return None

            # Pick rank_target columns to keep, try to express the rest
            # Randomly select which columns to "absorb"
            keep = list(range(rank_target))
            absorb = list(range(rank_target, R0))

            # Build system: for each absorbed column r, we need to find
            # coefficients alpha_r such that sum_{r in absorb} alpha_r * (U0[:,r] ⊗ V0[:,r] ⊗ W0[:,r])
            # can be represented as a linear combination of the kept columns
            # This is: sum_{r in absorb} alpha_r * U0[:,r] * V0[:,r] * W0[:,r] = sum_{s in keep} beta_s * ...
            # This is nonlinear. Instead, try: fix U and V supports, solve for W.

            # Simpler: try random perturbations of the Strassen solution
            U_try = U0.copy()
            V_try = V0.copy()
            W_try = W0.copy()

            # Add small random noise to some entries
            for _ in range(5):
                pos = rng.randint(0, 2)
                mat = [U_try, V_try, W_try][pos]
                r, c = rng.randint(0, mat.shape[0]-1), rng.randint(0, mat.shape[1]-1)
                mat[r, c] += rng.choice([-1, 1]) * 0.5

            # Re-solve W given U, V
            D = np.zeros((n_dim * m_dim, rank_target))
            for a in range(n_dim):
                for b in range(m_dim):
                    D[a * m_dim + b] = U_try[a] * V_try[b]

            T_reshaped = target.reshape(n_dim * m_dim, p_dim)
            try:
                X, res, rk, sv = np.linalg.lstsq(D, T_reshaped.T, rcond=None)
                W_sol = X.T
            except:
                continue

            T_check = np.einsum('ar,br,cr->abc', U_try, V_try, W_sol)
            if np.allclose(T_check.flatten(), target, atol=1e-8):
                # Check for half-integer values
                allv = np.concatenate([U_try.flat, V_try.flat, W_sol.flat])
                scaled = allv * 2
                if np.allclose(scaled, np.round(scaled), atol=1e-4):
                    U_e = np.round(U_try * 2) / 2
                    V_e = np.round(V_try * 2) / 2
                    W_e = np.round(W_sol * 2) / 2
                    T_f = np.einsum('ar,br,cr->abc', U_e, V_e, W_e)
                    if np.allclose(T_f.flatten(), target, atol=1e-9):
                        return U_e, V_e, W_e, rank_target
        return None

    # Try lower ranks
    for rk in [112, 110, 108, 106, 104]:
        if time.time() > deadline - 15:
            break
        res = try_merge(rk, seed=rk)
        if res is not None:
            U, V, W, r = res
            if r < best_rank:
                best_rank, best_U, best_V, best_W = r, U, V, W

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
