import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14

    start_time = time.time()
    deadline = start_time + 135.0

    # Schoolbook rank
    rank_sb = n * m * p  # 42

    # Build schoolbook factor matrices
    U_sb = np.zeros((nm, rank_sb), dtype=np.int64)
    V_sb = np.zeros((mp, rank_sb), dtype=np.int64)
    W_sb = np.zeros((pn, rank_sb), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * (m * p) + j * p + k
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1

    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    best_rank = rank_sb

    def verify(U, V, W):
        """Check exact decomposition using integer arithmetic where possible."""
        try:
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                        if abs(val - 1.0) > 1e-8:
                            return False
            return True
        except:
            return False

    def snap_half(arr, tol=1e-6):
        """Snap to nearest half-integer if close."""
        s = np.round(arr * 2) / 2
        if np.allclose(arr, s, atol=tol):
            return s
        return None

    def snap_int(arr, tol=1e-6):
        """Snap to nearest integer if close."""
        s = np.round(arr)
        if np.allclose(arr, s, atol=tol):
            return s
        return None

    def solve_W(U, V, r):
        """Solve for W given U and V. Returns W or None."""
        W = np.zeros((pn, r))
        for i in range(n):
            for k in range(p):
                A = np.zeros((m, r))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                b = np.ones(m)
                try:
                    w, res, rk, sv = np.linalg.lstsq(A, b, rcond=None)
                except:
                    return None
                if np.max(np.abs(A @ w - b)) > 1e-7:
                    return None
                W[k * n + i, :] = w
        return W

    def try_rank(r, num_trials, entry_pool, structured=False):
        """Try to find a decomposition of rank r."""
        for t in range(num_trials):
            if time.time() > deadline:
                return None, None, None

            if structured and r < rank_sb:
                # Structured: exploit n=2 symmetry
                U = np.random.choice(entry_pool, size=(nm, r))
                V = np.random.choice(entry_pool, size=(mp, r))
                # Impose some structure
                strat = np.random.randint(4)
                if strat == 0 and m >= 2:
                    U[m, :] = U[0, :].copy()
                elif strat == 1 and m >= 2:
                    U[m, :] = -U[0, :].copy()
                elif strat == 2:
                    # Make V rows symmetric
                    if mp >= 2:
                        V[mp // 2, :] = V[0, :].copy()
                elif strat == 3 and r >= 2:
                    # Zero out some columns to create structure
                    idx = np.random.randint(r // 2, r)
                    U[:, idx] = 0
                    V[:, idx] = 0
            else:
                U = np.random.choice(entry_pool, size=(nm, r))
                V = np.random.choice(entry_pool, size=(mp, r))

            W = solve_W(U, V, r)
            if W is None:
                continue

            W_s = snap_half(W)
            if W_s is None:
                W_s = snap_int(W)
            if W_s is None:
                continue

            if verify(U, V, W_s):
                return U, V, W_s

        return None, None, None

    def column_elimination(U, V, W, r):
        """Try to eliminate columns that are linearly dependent or contribute zero."""
        # Check which columns are needed
        keep = []
        for col in range(r):
            # Check if removing this column still gives valid decomposition
            if col == 0:
                continue  # keep at least one
            U_t = np.delete(U, col, axis=1)
            V_t = np.delete(V, col, axis=1)
            W_t = np.delete(W, col, axis=1)
            if verify(U_t, V_t, W_t):
                # Try to remove this column
                return U_t, V_t, W_t, r - 1
        return U, V, W, r

    def random_permutation_search(r, num_trials, entry_pool):
        """Search with random permutations and signs."""
        for t in range(num_trials):
            if time.time() > deadline:
                return None, None, None

            U = np.random.choice(entry_pool, size=(nm, r))
            V = np.random.choice(entry_pool, size=(mp, r))

            # Apply random sign flips to columns
            signs = np.random.choice([-1, 1], size=r)
            U = U * signs
            V = V * signs

            # Apply random column permutation
            perm = np.random.permutation(r)
            U = U[:, perm]
            V = V[:, perm]

            W = solve_W(U, V, r)
            if W is None:
                continue

            W_s = snap_half(W)
            if W_s is None:
                W_s = snap_int(W)
            if W_s is None:
                continue

            if verify(U, V, W_s):
                return U, V, W_s

        return None, None, None

    # Search strategy: try ranks from just below schoolbook downward
    # Use multiple search strategies and entry sets

    entry_sets = [
        [-1, 0, 1],
        [0, 1],
        [-1, 1],
        [-1, 0, 1, 2],
        [1, -1, 0],
    ]

    # First pass: structured search
    for r in range(rank_sb - 1, max(28, rank_sb - 12), -1):
        if time.time() > deadline:
            break
        for pool in entry_sets[:3]:
            U_f, V_f, W_f = try_rank(r, 25, pool, structured=True)
            if U_f is not None:
                best_U, best_V, best_W = U_f, V_f, W_f
                best_rank = r
                # Try column elimination
                for elim_pass in range(10):
                    if time.time() > deadline:
                        break
                    U_e, V_e, W_e, new_r = column_elimination(best_U, best_V, best_W, best_rank)
                    if new_r < best_rank:
                        best_U, best_V, best_W = U_e, V_e, W_e
                        best_rank = new_r
                    else:
                        break
                break
        if best_rank < rank_sb:
            break

    # Second pass: random search with permutations
    if best_rank == rank_sb:
        for r in range(rank_sb - 1, max(25, rank_sb - 18), -1):
            if time.time() > deadline:
                break
            for pool in entry_sets:
                U_f, V_f, W_f = random_permutation_search(r, 20, pool)
                if U_f is not None:
                    best_U, best_V, best_W = U_f, V_f, W_f
                    best_rank = r
                    # Column elimination
                    for elim_pass in range(10):
                        if time.time() > deadline:
                            break
                        U_e, V_e, W_e, new_r = column_elimination(best_U, best_V, best_W, best_rank)
                        if new_r < best_rank:
                            best_U, best_V, best_W = U_e, V_e, W_e
                            best_rank = new_r
                        else:
                            break
                    break
            if best_rank < rank_sb:
                break

    # Third pass: more aggressive random search if still at schoolbook
    if best_rank == rank_sb and time.time() < deadline - 20:
        for r in range(rank_sb - 1, max(20, rank_sb - 25), -1):
            if time.time() > deadline - 5:
                break
            for pool in entry_sets[:4]:
                U_f, V_f, W_f = try_rank(r, 15, pool, structured=False)
                if U_f is not None:
                    best_U, best_V, best_W = U_f, V_f, W_f
                    best_rank = r
                    for elim_pass in range(10):
                        if time.time() > deadline - 2:
                            break
                        U_e, V_e, W_e, new_r = column_elimination(best_U, best_V, best_W, best_rank)
                        if new_r < best_rank:
                            best_U, best_V, best_W = U_e, V_e, W_e
                            best_rank = new_r
                        else:
                            break
                    break
            if best_rank < rank_sb:
                break

    # Final: ensure we return a valid decomposition
    if not verify(best_U, best_V, best_W):
        # Fall back to schoolbook
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = rank_sb

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
