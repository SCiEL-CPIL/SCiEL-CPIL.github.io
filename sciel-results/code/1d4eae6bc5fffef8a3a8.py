import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 3, 3, 3
    nm, mp, pn = 9, 9, 9

    # Target tensor indices: T[i,j,k] = 1 for all i,j,k in {0,1,2}
    ones_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                ones_indices.append((a, b, c))

    ones_set = set(ones_indices)
    all_indices = [(a, b, c) for a in range(nm) for b in range(mp) for c in range(pn)]
    zero_indices = [idx for idx in all_indices if idx not in ones_set]

    def verify(U, V, W, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for a, b, c in ones_indices:
            s = np.sum(W[c, :] * U[a, :] * V[b, :])
            if abs(s - 1.0) > tol:
                return False
        for a, b, c in zero_indices:
            s = np.sum(W[c, :] * U[a, :] * V[b, :])
            if abs(s) > tol:
                return False
        return True

    def snap_and_verify(U, V, W, scale=2):
        """Try snapping to half-integers (scale=2) or integers (scale=1)."""
        for s in (scale, 1):
            U_r = np.round(U * s) / s
            V_r = np.round(V * s) / s
            W_r = np.round(W * s) / s
            if verify(U_r, V_r, W_r, tol=1e-9):
                return U_r, V_r, W_r
        return None

    # --- Explicit rank-26 construction: Strassen for 2x2 core + border ---
    R26 = 26
    U26 = np.zeros((nm, R26))
    V26 = np.zeros((mp, R26))
    W26 = np.zeros((pn, R26))

    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i

    # Strassen terms for the 2x2 core (indices 0,1)
    # r=0: (A00+A11)(B00+B11) -> C00, C11
    U26[idx_u(0,0), 0] = 1; U26[idx_u(1,1), 0] = 1
    V26[idx_v(0,0), 0] = 1; V26[idx_v(1,1), 0] = 1
    W26[idx_w(0,0), 0] = 1; W26[idx_w(1,1), 0] = 1

    # r=1: (A10+A11)B00 -> C10, -C11
    U26[idx_u(1,0), 1] = 1; U26[idx_u(1,1), 1] = 1
    V26[idx_v(0,0), 1] = 1
    W26[idx_w(0,1), 1] = 1; W26[idx_w(1,1), 1] = -1

    # r=2: A00(B01-B11) -> C01, C11
    U26[idx_u(0,0), 2] = 1
    V26[idx_v(0,1), 2] = 1; V26[idx_v(1,1), 2] = -1
    W26[idx_w(1,0), 2] = 1; W26[idx_w(1,1), 2] = 1

    # r=3: A11(B10-B00) -> C00, C10
    U26[idx_u(1,1), 3] = 1
    V26[idx_v(1,0), 3] = 1; V26[idx_v(0,0), 3] = -1
    W26[idx_w(0,0), 3] = 1; W26[idx_w(0,1), 3] = 1

    # r=4: (A00+A01)B11 -> -C00, C01
    U26[idx_u(0,0), 4] = 1; U26[idx_u(0,1), 4] = 1
    V26[idx_v(1,1), 4] = 1
    W26[idx_w(0,0), 4] = -1; W26[idx_w(1,0), 4] = 1

    # r=5: (A10-A00)(B00+B01) -> C11
    U26[idx_u(1,0), 5] = 1; U26[idx_u(0,0), 5] = -1
    V26[idx_v(0,0), 5] = 1; V26[idx_v(0,1), 5] = 1
    W26[idx_w(1,1), 5] = 1

    # r=6: (A01-A11)(B10+B11) -> C00
    U26[idx_u(0,1), 6] = 1; U26[idx_u(1,1), 6] = -1
    V26[idx_v(1,0), 6] = 1; V26[idx_v(1,1), 6] = 1
    W26[idx_w(0,0), 6] = 1

    # Border terms: all (i,j,k) where max(i,j,k) == 2
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U26[idx_u(i,j), col] = 1
                V26[idx_v(j,k), col] = 1
                W26[idx_w(k,i), col] = 1
                col += 1

    assert col == 26, f"Expected 26 columns, got {col}"

    if not verify(U26, V26, W26, tol=1e-9):
        # Fallback to schoolbook if rank-26 construction is broken
        rank_sb = 27
        U27 = np.zeros((nm, rank_sb))
        V27 = np.zeros((mp, rank_sb))
        W27 = np.zeros((pn, rank_sb))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U27[idx_u(i,j), c] = 1
                    V27[idx_v(j,k), c] = 1
                    W27[idx_w(k,i), c] = 1
                    c += 1
        return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}

    # Fallback: schoolbook rank 27 (only used if rank-26 verification fails)
    rank_sb = 27
    U27 = np.zeros((nm, rank_sb))
    V27 = np.zeros((mp, rank_sb))
    W27 = np.zeros((pn, rank_sb))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U27[idx_u(i,j), c] = 1
                V27[idx_v(j,k), c] = 1
                W27[idx_w(k,i), c] = 1
                c += 1

    best_U, best_V, best_W = U26, V26, W26
    best_rank = 26

    # --- SVD / null-space search for lower rank (25, 24, 23) ---
    # Strategy: build a linear system from the flattening of the tensor.
    # For a rank-R decomposition, the flattening matrices (U*V^T) must have
    # the target tensor's flattening in their column space. We use SVD of
    # the rank-26 solution's flattening to identify the kernel structure,
    # then solve a linear system for the lower-rank case.

    from scipy.optimize import least_squares

    def build_residual_func(R, subset_zeros=None):
        """Build residual function for rank R."""
        if subset_zeros is None:
            subset_zeros = zero_indices

        def residual(x):
            U_f = x[:nm*R].reshape(nm, R)
            V_f = x[nm*R:2*nm*R].reshape(mp, R)
            W_f = x[2*nm*R:3*nm*R].reshape(pn, R)

            res = np.zeros(len(ones_indices) + len(subset_zeros))
            idx = 0
            for a, b, c in ones_indices:
                s = np.sum(W_f[c, :] * U_f[a, :] * V_f[b, :])
                res[idx] = s - 1.0
                idx += 1
            for a, b, c in subset_zeros:
                s = np.sum(W_f[c, :] * U_f[a, :] * V_f[b, :])
                res[idx] = s
                idx += 1
            return res

        return residual

    # SVD-based initialization: use the rank-26 solution's structure
    # to build a better starting point for lower-rank searches.
    # The flattening of the rank-26 tensor has rank at most 26.
    # For rank R < 26, we project onto the top-R singular vectors.

    for R in [25, 24, 23]:
        if time.time() > deadline - 30:
            break

        # Build flattening matrices from the rank-26 solution
        # Flattening 1: (a,b) -> c, entries T[a,b,c]
        # For rank-R decomposition: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Fix c=0: T[a,b,0] = sum_r U[a,r]*V[b,r]*W[0,r]
        # This is a matrix M_c = (U * diag(W[c,:])) * V^T

        # Use SVD of the rank-26 flattening to get a good initialization
        # Build M for c=0 from the rank-26 solution
        U26_f = U26.copy()
        V26_f = V26.copy()
        W26_f = W26.copy()

        # SVD-based initialization for rank R
        for attempt in range(15):
            if time.time() > deadline - 10:
                break

            np.random.seed(attempt * 104729 + R * 7919)

            # Initialize with random perturbation of the rank-26 core
            # Use the first R columns plus noise to break symmetry
            U_init = U26[:, :R].copy()
            V_init = V26[:, :R].copy()
            W_init = W26[:, :R].copy()

            # Add structured perturbation
            scale = 0.05 + 0.02 * attempt
            U_init += np.random.randn(nm, R) * scale
            V_init += np.random.randn(mp, R) * scale
            W_init += np.random.randn(pn, R) * scale

            # Use a random subset of zero constraints
            shuffled_zeros = zero_indices.copy()
            np.random.shuffle(shuffled_zeros)
            opt_zeros = shuffled_zeros[:min(200, len(zero_indices))]

            residual_func = build_residual_func(R, opt_zeros)
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

            try:
                sol = least_squares(
                    residual_func, x0, method='trf',
                    max_nfev=3000, ftol=1e-13, xtol=1e-13, gtol=1e-13
                )

                if sol.cost < 1e-10:
                    U_sol = sol.x[:nm*R].reshape(nm, R)
                    V_sol = sol.x[nm*R:2*nm*R].reshape(mp, R)
                    W_sol = sol.x[2*nm*R:3*nm*R].reshape(pn, R)

                    # Polish with all constraints
                    residual_full = build_residual_func(R, zero_indices)
                    try:
                        sol2 = least_squares(
                            residual_full, sol.x, method='trf',
                            max_nfev=5000, ftol=1e-15, xtol=1e-15, gtol=1e-15
                        )
                        if sol2.cost < sol.cost:
                            U_sol = sol2.x[:nm*R].reshape(nm, R)
                            V_sol = sol2.x[nm*R:2*nm*R].reshape(mp, R)
                            W_sol = sol2.x[2*nm*R:3*nm*R].reshape(pn, R)
                    except Exception:
                        pass

                    # Try snapping to half-integers
                    snapped = snap_and_verify(U_sol, V_sol, W_sol, scale=2)
                    if snapped is not None:
                        U_r, V_r, W_r = snapped
                        if verify(U_r, V_r, W_r, tol=1e-9):
                            best_U, best_V, best_W = U_r, V_r, W_r
                            best_rank = R
                            break
            except Exception:
                pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
