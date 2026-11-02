"""Matrix multiplication 2x4 x 4x4 tensor decomposition - rank 27 search via linear-algebraic rank compression."""

import numpy as np
import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    start_time = time.time()
    deadline = start_time + 115

    def build_strassen_28():
        rank = 28
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        col = 0
        for b in range(2):
            for a_blk in range(2):
                for s in range(7):
                    c = col
                    j0 = 2 * a_blk
                    j1 = 2 * a_blk + 1

                    U_contrib = {
                        0: [(0, j0, 1), (1, j1, 1)],
                        1: [(1, j0, 1), (1, j1, 1)],
                        2: [(0, j0, 1)],
                        3: [(1, j1, 1)],
                        4: [(0, j0, 1), (0, j1, 1)],
                        5: [(1, j0, 1), (0, j0, -1)],
                        6: [(0, j1, 1), (1, j1, -1)],
                    }
                    for (i, j, coeff) in U_contrib[s]:
                        U[i * 4 + j, c] += coeff

                    V_contrib = {
                        0: [(2*a_blk, 2*b, 1), (2*a_blk+1, 2*b+1, 1)],
                        1: [(2*a_blk, 2*b, 1)],
                        2: [(2*a_blk, 2*b+1, 1), (2*a_blk+1, 2*b+1, -1)],
                        3: [(2*a_blk+1, 2*b, 1), (2*a_blk, 2*b, -1)],
                        4: [(2*a_blk+1, 2*b+1, 1)],
                        5: [(2*a_blk, 2*b, 1), (2*a_blk, 2*b+1, 1)],
                        6: [(2*a_blk+1, 2*b, 1), (2*a_blk+1, 2*b+1, 1)],
                    }
                    for (j, k, coeff) in V_contrib[s]:
                        V[j * 4 + k, c] += coeff

                    W_contrib = {
                        0: {(0, 0): 1, (1, 1): 1},
                        1: {(1, 0): 1, (1, 1): -1},
                        2: {(0, 1): 1, (1, 1): 1},
                        3: {(0, 0): 1, (1, 0): 1},
                        4: {(0, 0): -1, (0, 1): 1},
                        5: {(1, 1): 1},
                        6: {(0, 0): 1},
                    }
                    for (i_loc, k_loc), coeff in W_contrib[s].items():
                        W[(2 * b + k_loc) * 2 + i_loc, c] += coeff

                    col += 1

        return U, V, W

    def validate_decomposition(U, V, W, tol=1e-8):
        """Check if U,V,W form an exact decomposition of the matmul tensor."""
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        rank = U_np.shape[1]

        # Check all entries: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = np.dot(U_np[a], V_np[b] * W_np[c])
                    if abs(s - 1.0) > tol:
                        return False

        # Also check that off-diagonal terms are zero
        # This is implicitly checked by the above if we verify all (i,j,k) triples
        # But let's also verify that for incompatible indices, the sum is 0
        for a in range(n * m):
            i1 = a // m
            j1 = a % m
            for b in range(m * p):
                j2 = b // p
                k1 = b % p
                for c in range(p * n):
                    i2 = c // n
                    k2 = c % n
                    if i1 != i2 or j1 != j2 or k1 != k2:
                        s = np.dot(U_np[a], V_np[b] * W_np[c])
                        if abs(s) > tol:
                            return False
        return True

    def is_half_integer(arr, tol=1e-8):
        """Check if all entries are integers or half-integers."""
        doubled = 2 * np.array(arr, dtype=np.float64)
        return np.allclose(doubled, np.round(doubled), atol=tol)

    def snap_to_half_int(arr):
        """Snap array values to nearest integer or half-integer."""
        return np.round(2 * np.array(arr, dtype=np.float64)) / 2.0

    def validate_half_int(U, V, W, tol=1e-8):
        """Validate that U, V, W contain only integers/half-integers and form an exact decomposition."""
        if not is_half_integer(U, tol) or not is_half_integer(V, tol) or not is_half_integer(W, tol):
            return False
        return validate_decomposition(U, V, W, tol)

    # Build baseline rank-28
    U28, V28, W28 = build_strassen_28()

    best_result = {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
    best_rank = 28
    found = False

    # Strategy 1: Try dropping each column and solving for W exactly
    def try_drop_column(drop_col):
        """Try to find rank-27 by dropping column drop_col and solving for W."""
        keep_cols = [c for c in range(28) if c != drop_col]
        U = U28[:, keep_cols].copy()  # 8 x 27
        V = V28[:, keep_cols].copy()  # 16 x 27
        W = np.zeros((p * n, 27))  # 8 x 27

        for c_idx in range(p * n):
            i2 = c_idx // n
            k2 = c_idx % n
            # Build system: for each (a,b), U[a] * V[b] dot W[c] = T[a,b,c]
            M = np.zeros((n * m * m * p, 27))
            rhs = np.zeros(n * m * m * p)
            idx = 0
            for a in range(n * m):
                i1 = a // m
                j1 = a % m
                for b in range(m * p):
                    j2 = b // p
                    k1 = b % p
                    M[idx, :] = U[a, :] * V[b, :]
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        rhs[idx] = 1.0
                    else:
                        rhs[idx] = 0.0
                    idx += 1

            try:
                sol, residuals, rank_M, sv = np.linalg.lstsq(M, rhs, rcond=None)
                W[c_idx, :] = sol
            except:
                return None

        # Snap to half-integers
        U_snap = snap_to_half_int(U)
        V_snap = snap_to_half_int(V)
        W_snap = snap_to_half_int(W)

        if validate_half_int(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap
        return None

    # Strategy 2: ALS refinement with exact W solve
    def als_refine(U_init, V_init, W_init, max_iters=40, time_limit=None):
        """ALS refinement for rank-27."""
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        target_rank = 27

        for step in range(max_iters):
            if time_limit is not None and time.time() > time_limit:
                break

            # Update U rows
            for a in range(n * m):
                i1 = a // m
                j1 = a % m
                M = np.zeros((m * p * p * n, target_rank))
                rhs = np.zeros(m * p * p * n)
                idx = 0
                for b in range(m * p):
                    j2 = b // p
                    k1 = b % p
                    for c in range(p * n):
                        i2 = c // n
                        k2 = c % n
                        M[idx, :] = V[b, :] * W[c, :]
                        if i1 == i2 and j1 == j2 and k1 == k2:
                            rhs[idx] = 1.0
                        else:
                            rhs[idx] = 0.0
                        idx += 1
                try:
                    U[a, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                except:
                    pass

            # Update V rows
            for b in range(m * p):
                j2 = b // p
                k1 = b % p
                M = np.zeros((n * m * p * n, target_rank))
                rhs = np.zeros(n * m * p * n)
                idx = 0
                for a in range(n * m):
                    i1 = a // m
                    j1 = a % m
                    for c in range(p * n):
                        i2 = c // n
                        k2 = c % n
                        M[idx, :] = U[a, :] * W[c, :]
                        if i1 == i2 and j1 == j2 and k1 == k2:
                            rhs[idx] = 1.0
                        else:
                            rhs[idx] = 0.0
                        idx += 1
                try:
                    V[b, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                except:
                    pass

            # Update W rows (exact per-output solve)
            for c in range(p * n):
                i2 = c // n
                k2 = c % n
                M = np.zeros((n * m * m * p, target_rank))
                rhs = np.zeros(n * m * m * p)
                idx = 0
                for a in range(n * m):
                    i1 = a // m
                    j1 = a % m
                    for b in range(m * p):
                        j2 = b // p
                        k1 = b % p
                        M[idx, :] = U[a, :] * V[b, :]
                        if i1 == i2 and j1 == j2 and k1 == k2:
                            rhs[idx] = 1.0
                        else:
                            rhs[idx] = 0.0
                        idx += 1
                try:
                    W[c, :], _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
                except:
                    pass

            # Check validity with snapping
            U_snap = snap_to_half_int(U)
            V_snap = snap_to_half_int(V)
            W_snap = snap_to_half_int(W)
            if validate_half_int(U_snap, V_snap, W_snap):
                return U_snap, V_snap, W_snap

        return None

    # Try each dropped column
    for drop_col in range(28):
        if time.time() > deadline - 25:
            break

        result = try_drop_column(drop_col)
        if result is not None:
            U_sol, V_sol, W_sol = result
            best_rank = 27
            best_result = {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
            found = True
            break

    # If not found, try ALS with perturbed initializations
    if not found and time.time() < deadline - 15:
        np.random.seed(12345)
        for attempt in range(30):
            if time.time() > deadline - 15:
                break

            drop_col = attempt % 28
            keep_cols = [c for c in range(28) if c != drop_col]

            U0 = U28[:, keep_cols].copy()
            V0 = V28[:, keep_cols].copy()
            W0 = W28[:, keep_cols].copy()

            # Perturb to escape the Strassen manifold
            scale = 0.03 * (1 + attempt * 0.02)
            U0 += np.random.randn(n * m, 27) * scale
            V0 += np.random.randn(m * p, 27) * scale
            W0 += np.random.randn(p * n, 27) * scale

            result = als_refine(U0, V0, W0, max_iters=25, time_limit=deadline - 15)
            if result is not None:
                U_sol, V_sol, W_sol = result
                best_rank = 27
                best_result = {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
                found = True
                break

    # If still not found, try without perturbation
    if not found and time.time() < deadline - 5:
        for drop_col in range(28):
            if time.time() > deadline - 5:
                break

            keep_cols = [c for c in range(28) if c != drop_col]
            U0 = U28[:, keep_cols].copy()
            V0 = V28[:, keep_cols].copy()
            W0 = W28[:, keep_cols].copy()

            result = als_refine(U0, V0, W0, max_iters=20, time_limit=deadline - 5)
            if result is not None:
                U_sol, V_sol, W_sol = result
                best_rank = 27
                best_result = {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
                found = True
                break

    return best_result

# EVOLVE-BLOCK-END
