import time
import numpy as np

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
    U_sb = np.zeros((nm, rank_sb), dtype=np.float64)
    V_sb = np.zeros((mp, rank_sb), dtype=np.float64)
    W_sb = np.zeros((pn, rank_sb), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * (m * p) + j * p + k
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1

    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    best_rank = rank_sb

    def verify(U, V, W, tol=1e-10):
        """Check exact decomposition."""
        try:
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                        if abs(val - 1.0) > tol:
                            return False
            return True
        except:
            return False

    def snap_values(arr, tol=1e-6):
        """Snap to nearest half-integer or integer if close."""
        # Try integer first
        s_int = np.round(arr)
        if np.allclose(arr, s_int, atol=tol):
            return s_int
        # Try half-integer
        s_half = np.round(arr * 2) / 2
        if np.allclose(arr, s_half, atol=tol):
            return s_half
        return None

    def try_column_merge(U, V, W, r):
        """Try to merge two columns by finding linear combination that preserves decomposition."""
        # For each pair of columns, try to see if one can be eliminated
        for c1 in range(r):
            if time.time() > deadline - 5:
                return None
            for c2 in range(c1 + 1, r):
                # Try removing c2: check if we can adjust other columns
                # Simple approach: just check if c2 is redundant
                U_t = np.delete(U, c2, axis=1)
                V_t = np.delete(V, c2, axis=1)
                W_t = np.delete(W, c2, axis=1)
                if verify(U_t, V_t, W_t, tol=1e-8):
                    return U_t, V_t, W_t, r - 1
        return None

    def systematic_reduction(U, V, W, r):
        """Systematically try to reduce rank by removing columns."""
        current_U, current_V, current_W, current_r = U, V, W, r
        progress = True
        while progress:
            progress = False
            if time.time() > deadline - 10:
                break
            result = try_column_merge(current_U, current_V, current_W, current_r)
            if result is not None:
                current_U, current_V, current_W, current_r = result
                progress = True
        return current_U, current_V, current_W, current_r

    def solve_W_from_UV(U, V, r):
        """Solve for W given U and V using least squares."""
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
                if np.max(np.abs(A @ w - b)) > 1e-6:
                    return None
                W[k * n + i, :] = w
        return W

    def random_search_rank(r, num_trials, entry_pool):
        """Random search for rank r decomposition."""
        for t in range(num_trials):
            if time.time() > deadline - 10:
                return None, None, None

            U = np.random.choice(entry_pool, size=(nm, r))
            V = np.random.choice(entry_pool, size=(mp, r))

            W = solve_W_from_UV(U, V, r)
            if W is None:
                continue

            W_s = snap_values(W)
            if W_s is None:
                continue

            if verify(U, V, W_s, tol=1e-8):
                return U, V, W_s

        return None, None, None

    def perturbation_search(U_init, V_init, W_init, r_init, num_trials):
        """Start from a known decomposition and perturb to find lower rank."""
        current_U, current_V, current_W = U_init.copy(), V_init.copy(), W_init.copy()
        current_r = r_init
        
        for t in range(num_trials):
            if time.time() > deadline - 10:
                break
            
            # Try to find a column to eliminate by perturbing
            col_to_try = np.random.randint(current_r)
            
            # Zero out this column in U and V, see if we can solve for W
            U_test = current_U.copy()
            V_test = current_V.copy()
            U_test[:, col_to_try] = 0
            V_test[:, col_to_try] = 0
            
            # Now try to solve for W with the modified U, V
            # But we need to keep the other columns' W values
            W_test = current_W.copy()
            
            # Actually, let's try a different approach: 
            # Try to find if the decomposition works with one column removed
            U_rm = np.delete(current_U, col_to_try, axis=1)
            V_rm = np.delete(current_V, col_to_try, axis=1)
            W_rm = np.delete(current_W, col_to_try, axis=1)
            
            if verify(U_rm, V_rm, W_rm, tol=1e-8):
                current_U, current_V, current_W = U_rm, V_rm, W_rm
                current_r -= 1
                continue
            
            # Try small perturbations
            if np.random.random() < 0.3 and current_r > 5:
                # Randomly zero out a column and try to re-solve
                U_new = current_U.copy()
                V_new = current_V.copy()
                U_new[:, col_to_try] = 0
                V_new[:, col_to_try] = 0
                # Keep the column but with zero contribution
                # This doesn't directly help, so skip
                pass
        
        return current_U, current_V, current_W, current_r

    # Entry pools for random search
    entry_sets = [
        [-1, 0, 1],
        [0, 1],
        [-1, 1],
        [-1, 0, 1, 2],
        [-2, -1, 0, 1, 2],
    ]

    # Phase 1: Start from schoolbook and try systematic column elimination
    U_cur, V_cur, W_cur = U_sb.copy(), V_sb.copy(), W_sb.copy()
    r_cur = rank_sb
    
    # Try to eliminate columns from schoolbook
    result = systematic_reduction(U_cur, V_cur, W_cur, r_cur)
    if result is not None:
        U_cur, V_cur, W_cur, r_cur = result
        if r_cur < best_rank:
            best_U, best_V, best_W = U_cur, V_cur, W_cur
            best_rank = r_cur

    # Phase 2: Random search for rank reductions
    for target_r in range(min(best_rank - 1, rank_sb - 1), max(30, rank_sb - 15), -1):
        if time.time() > deadline - 20:
            break
        for pool in entry_sets:
            U_f, V_f, W_f = random_search_rank(target_r, 30, pool)
            if U_f is not None:
                # Try to reduce further
                U_r, V_r, W_r, r_r = systematic_reduction(U_f, V_f, W_f, target_r)
                if r_r < best_rank:
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = r_r
                break
        if best_rank < rank_sb - 5:
            break

    # Phase 3: Perturbation from best found
    if best_rank < rank_sb and time.time() < deadline - 15:
        U_p, V_p, W_p, r_p = perturbation_search(best_U, best_V, best_W, best_rank, 50)
        if r_p < best_rank:
            best_U, best_V, best_W = U_p, V_p, W_p
            best_rank = r_p

    # Phase 4: More aggressive random search
    if time.time() < deadline - 10:
        for target_r in range(min(best_rank - 1, rank_sb - 1), max(25, rank_sb - 20), -1):
            if time.time() > deadline - 5:
                break
            for pool in entry_sets[:3]:
                U_f, V_f, W_f = random_search_rank(target_r, 20, pool)
                if U_f is not None:
                    U_r, V_r, W_r, r_r = systematic_reduction(U_f, V_f, W_f, target_r)
                    if r_r < best_rank:
                        best_U, best_V, best_W = U_r, V_r, W_r
                        best_rank = r_r
                    break
            if best_rank <= 35:
                break

    # Final verification
    if not verify(best_U, best_V, best_W, tol=1e-8):
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = rank_sb

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
