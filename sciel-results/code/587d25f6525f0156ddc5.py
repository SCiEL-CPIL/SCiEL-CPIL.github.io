import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 11]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Build the multiplication tensor
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor_entries.append((u_idx, v_idx, w_idx))

    num_entries = len(tensor_entries)  # 44

    # Pre-compute entries grouped by u, v, w for fast ALS
    entries_by_u = {}
    entries_by_v = {}
    entries_by_w = {}
    for (u_idx, v_idx, w_idx) in tensor_entries:
        entries_by_u.setdefault(u_idx, []).append((v_idx, w_idx))
        entries_by_v.setdefault(v_idx, []).append((u_idx, w_idx))
        entries_by_w.setdefault(w_idx, []).append((u_idx, v_idx))

    # Set of non-zero entries for quick lookup
    nz_set = set(tensor_entries)

    # Known rank-39 decomposition (Strassen blocks + schoolbook)
    def make_rank39():
        rank = 39
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        r = 0
        for block in range(5):
            k0 = block * 2
            k1 = block * 2 + 1
            v_e = k0; v_f = k1; v_g = 11 + k0; v_h = 11 + k1
            w_c00 = 2 * k0; w_c10 = 2 * k0 + 1; w_c01 = 2 * k1; w_c11 = 2 * k1 + 1

            U[0][r] = 1; U[3][r] = 1
            V[v_e][r] = 1; V[v_h][r] = 1
            W[w_c00][r] = 1; W[w_c11][r] = 1
            r += 1
            U[2][r] = 1; U[3][r] = 1
            V[v_e][r] = 1
            W[w_c10][r] = 1; W[w_c11][r] = -1
            r += 1
            U[0][r] = 1
            V[v_f][r] = 1; V[v_h][r] = -1
            W[w_c01][r] = 1; W[w_c11][r] = 1
            r += 1
            U[3][r] = 1
            V[v_g][r] = 1; V[v_e][r] = -1
            W[w_c00][r] = 1; W[w_c10][r] = 1
            r += 1
            U[0][r] = 1; U[1][r] = 1
            V[v_h][r] = 1
            W[w_c00][r] = -1; W[w_c01][r] = 1
            r += 1
            U[2][r] = 1; U[0][r] = -1
            V[v_e][r] = 1; V[v_f][r] = 1
            W[w_c11][r] = 1
            r += 1
            U[1][r] = 1; U[3][r] = -1
            V[v_g][r] = 1; V[h_h if False else v_h][r] = 1
            W[w_c00][r] = 1
            r += 1

        k2 = 10
        v_e2 = k2; v_g2 = 11 + k2
        w_c02 = 2 * k2; w_c12 = 2 * k2 + 1
        U[0][r] = 1; V[v_e2][r] = 1; W[w_c02][r] = 1; r += 1
        U[1][r] = 1; V[v_g2][r] = 1; W[w_c02][r] = 1; r += 1
        U[2][r] = 1; V[v_e2][r] = 1; W[w_c12][r] = 1; r += 1
        U[3][r] = 1; V[v_g2][r] = 1; W[w_c12][r] = 1; r += 1

        return {"U": U, "V": V, "W": W}

    def make_rank39_np():
        """Return rank-39 as numpy arrays for fast computation."""
        r39 = make_rank39()
        U = np.array(r39['U'], dtype=float)
        V = np.array(r39['V'], dtype=float)
        W = np.array(r39['W'], dtype=float)
        return U, V, W

    # Full verification: check non-zero entries = 1 AND all other entries = 0
    def verify_full(U, V, W, tol=1e-9):
        if isinstance(U, list):
            U = np.array(U, dtype=float)
            V = np.array(V, dtype=float)
            W = np.array(W, dtype=float)

        rank = U.shape[1]

        # Check non-zero entries: should equal 1
        for (u_idx, v_idx, w_idx) in tensor_entries:
            total = np.dot(U[u_idx] * V[v_idx], W[w_idx])
            if abs(total - 1) > tol:
                return False

        # Check all zero entries: should equal 0
        for u_idx in range(n * m):
            for v_idx in range(m * p):
                for w_idx in range(p * n):
                    if (u_idx, v_idx, w_idx) not in nz_set:
                        total = np.dot(U[u_idx] * V[v_idx], W[w_idx])
                        if abs(total) > tol:
                            return False
        return True

    # Fast verification for non-zero entries only (used during search)
    def verify_nz(U, V, W, tol=1e-9):
        if isinstance(U, list):
            U = np.array(U, dtype=float)
            V = np.array(V, dtype=float)
            W = np.array(W, dtype=float)

        for (u_idx, v_idx, w_idx) in tensor_entries:
            total = np.dot(U[u_idx] * V[v_idx], W[w_idx])
            if abs(total - 1) > tol:
                return False
        return True

    # Compute full error including zero-constraint penalty
    def compute_error(U, V, W, zero_weight=1.0):
        rank = U.shape[1]
        err = 0.0
        # Non-zero entries
        for (u_idx, v_idx, w_idx) in tensor_entries:
            val = np.dot(U[u_idx] * V[v_idx], W[w_idx])
            err += (val - 1.0) ** 2
        # Zero entries
        for u_idx in range(n * m):
            for v_idx in range(m * p):
                for w_idx in range(p * n):
                    if (u_idx, v_idx, w_idx) not in nz_set:
                        val = np.dot(U[u_idx] * V[v_idx], W[w_idx])
                        err += zero_weight * val * val
        return err

    # Local search on half-integer/integer values to polish a near-exact solution
    def local_search_polish(U, V, W, max_iters=500, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)

        # Round to nearest half-integer as starting point
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2

        # Candidate values: small integers and half-integers
        candidates = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]

        best_err = compute_error(U, V, W)

        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break

            improved = False

            # Try changing one element at a time in U, V, W
            for mat_name in ['U', 'V', 'W']:
                if mat_name == 'U':
                    mat = U
                    dims = (n * m, U.shape[1])
                elif mat_name == 'V':
                    mat = V
                    dims = (m * p, V.shape[1])
                else:
                    mat = W
                    dims = (p * n, W.shape[1])

                for idx in range(dims[0]):
                    for col in range(dims[1]):
                        old_val = mat[idx, col]
                        for new_val in candidates:
                            if abs(new_val - old_val) < 1e-12:
                                continue
                            mat[idx, col] = new_val
                            new_err = compute_error(U, V, W)
                            if new_err < best_err - 1e-12:
                                best_err = new_err
                                improved = True
                                break
                            else:
                                mat[idx, col] = old_val
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

            if not improved:
                break

            # Check if we've reached exactness
            if verify_full(U, V, W, tol):
                return U.tolist(), V.tolist(), W.tolist()

        # Final check
        if verify_full(U, V, W, tol):
            return U.tolist(), V.tolist(), W.tolist()
        return None

    # Try to find a lower rank decomposition using ALS + local search polish
    def try_als_rank(target_rank, max_iters=150, seed=42, init_type='random'):
        if time.time() > deadline - 15:
            return None

        np.random.seed(seed)

        if init_type == 'random':
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
        elif init_type == 'strassen_perturb':
            # Start from rank-39 and truncate/perturb
            U_full, V_full, W_full = make_rank39_np()
            # Keep first target_rank columns and add small perturbation
            U = U_full[:, :target_rank].copy() + np.random.randn(n * m, target_rank) * 0.1
            V = V_full[:, :target_rank].copy() + np.random.randn(m * p, target_rank) * 0.1
            W = W_full[:, :target_rank].copy() + np.random.randn(p * n, target_rank) * 0.1
        elif init_type == 'strassen_merge':
            # Try to merge pairs of columns from rank-39 to reduce rank
            U_full, V_full, W_full = make_rank39_np()
            # Keep first target_rank columns
            U = U_full[:, :target_rank].copy()
            V = V_full[:, :target_rank].copy()
            W = W_full[:, :target_rank].copy()
            # Add moderate perturbation to escape local minima
            U += np.random.randn(n * m, target_rank) * 0.3
            V += np.random.randn(m * p, target_rank) * 0.3
            W += np.random.randn(p * n, target_rank) * 0.3
        else:
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5

        best_err = float('inf')
        best_U, best_V, best_W = None, None, None

        for it in range(max_iters):
            if time.time() > deadline - 15:
                break

            # Update U
            for u_idx in range(n * m):
                entries_u = entries_by_u.get(u_idx, [])
                if not entries_u:
                    continue
                A_mat = np.array([V[v_idx] * W[w_idx] for (v_idx, w_idx) in entries_u])
                b_vec = np.ones(len(entries_u))
                try:
                    x, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    U[u_idx] = x
                except:
                    pass

            # Update V
            for v_idx in range(m * p):
                entries_v = entries_by_v.get(v_idx, [])
                if not entries_v:
                    continue
                A_mat = np.array([U[u_idx] * W[w_idx] for (u_idx, w_idx) in entries_v])
                b_vec = np.ones(len(entries_v))
                try:
                    x, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    V[v_idx] = x
                except:
                    pass

            # Update W
            for w_idx in range(p * n):
                entries_w = entries_by_w.get(w_idx, [])
                if not entries_w:
                    continue
                A_mat = np.array([U[u_idx] * V[v_idx] for (u_idx, v_idx) in entries_w])
                b_vec = np.ones(len(entries_w))
                try:
                    x, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W[w_idx] = x
                except:
                    pass

            # Compute full error (non-zero + zero entries)
            err = compute_error(U, V, W, zero_weight=1.0)

            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()

            if err < 1e-14:
                break

        if best_U is None:
            return None

        # Try local search polish
        polished = local_search_polish(best_U, best_V, best_W, max_iters=300)
        if polished is not None:
            U_list, V_list, W_list = polished
            if verify_full(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}

        # Also try direct rounding without local search
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2

        U_list = U_rounded.tolist()
        V_list = V_rounded.tolist()
        W_list = W_rounded.tolist()

        if verify_full(U_list, V_list, W_list):
            return {"U": U_list, "V": V_list, "W": W_list}

        return None

    # Try ranks from 38 down to 35
    for target_rank in [38, 37, 36, 35]:
        if time.time() > deadline - 20:
            break

        # Try multiple seeds with different initialization strategies
        for seed in range(8):
            if time.time() > deadline - 20:
                break

            if seed < 3:
                init_type = 'random'
            elif seed < 6:
                init_type = 'strassen_perturb'
            else:
                init_type = 'strassen_merge'

            result = try_als_rank(target_rank, max_iters=100, seed=seed, init_type=init_type)
            if result is not None:
                return result

    # Fall back to known rank-39
    return make_rank39()
