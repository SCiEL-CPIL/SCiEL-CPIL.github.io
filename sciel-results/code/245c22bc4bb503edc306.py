"""Find a rank-minimal exact tensor decomposition for 2x4 * 4x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0

    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    def verify(U, V, W, tol=1e-9):
        r = U.shape[1]
        Um = U.reshape(n, m, r)
        Vm = V.reshape(m, p, r)
        Wm = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', Um, Vm, Wm)
        return np.allclose(T, np.ones((n, m, p)), atol=tol, rtol=0)

    def error_of(U, V, W):
        r = U.shape[1]
        Um = U.reshape(n, m, r)
        Vm = V.reshape(m, p, r)
        Wm = W.reshape(p, n, r)
        T = np.einsum('ijc,jkc,kic->ijk', Um, Vm, Wm)
        return np.max(np.abs(T - 1.0))

    def solve_U(U, V, W):
        """Solve for U given V, W."""
        r = U.shape[1]
        Vm = V.reshape(m, p, r)
        Wm = W.reshape(p, n, r)
        for i in range(n):
            for j in range(m):
                A = Vm[j] * Wm[:, i]  # (p, r)
                b = np.ones(p)
                try:
                    x, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                    U[i * m + j] = x
                except np.linalg.LinAlgError:
                    pass
        return U

    def solve_V(U, V, W):
        """Solve for V given U, W."""
        r = V.shape[1]
        Um = U.reshape(n, m, r)
        Wm = W.reshape(p, n, r)
        for j in range(m):
            for k in range(p):
                A = Um[:, j] * Wm[k]  # (n, r)
                b = np.ones(n)
                try:
                    x, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                    V[j * p + k] = x
                except np.linalg.LinAlgError:
                    pass
        return V

    def solve_W(U, V, W):
        """Solve for W given U, V."""
        r = W.shape[1]
        Um = U.reshape(n, m, r)
        Vm = V.reshape(m, p, r)
        for k in range(p):
            for i in range(n):
                A = Um[i] * Vm[:, k]  # (m, r)
                b = np.ones(m)
                try:
                    x, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                    W[k * n + i] = x
                except np.linalg.LinAlgError:
                    pass
        return W

    def am_single(target_rank, seed, max_iters=500, tol=1e-12):
        """Single AM run with given seed."""
        rng = np.random.RandomState(seed)
        scale = 1.0 / np.sqrt(target_rank)
        U = rng.randn(n * m, target_rank) * scale
        V = rng.randn(m * p, target_rank) * scale
        W = rng.randn(p * n, target_rank) * scale
        
        err = error_of(U, V, W)
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            U = solve_U(U, V, W)
            V = solve_V(U, V, W)
            W = solve_W(U, V, W)
            err = error_of(U, V, W)
            if err < tol:
                break
            if it % 100 == 99 and err > 10.0:
                break
        return U, V, W, err

    def try_rank(target_rank, n_restarts=100, max_iters=500):
        """Try to find decomposition at target_rank."""
        best_err = float('inf')
        best_result = None
        for seed in range(n_restarts):
            if time.time() > deadline - 5:
                break
            U, V, W, err = am_single(target_rank, seed, max_iters)
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            if err < 1e-9:
                break
        return best_result, best_err

    # Try ranks from low to high
    target_ranks = [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 44, 48, 52]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 15:
            break
        if target_rank >= best_rank:
            continue
        
        result, err = try_rank(target_rank, n_restarts=150, max_iters=400)
        
        if result is not None:
            U_try, V_try, W_try = result
            if verify(U_try, V_try, W_try, tol=1e-9):
                best_rank = target_rank
                best_U, best_V, best_W = U_try, V_try, W_try
                # Try to go lower
                continue
            elif err < 1e-6:
                # Close but not exact - do a polish
                for _ in range(50):
                    if time.time() > deadline - 5:
                        break
                    U_try = solve_U(U_try, V_try, W_try)
                    V_try = solve_V(U_try, V_try, W_try)
                    W_try = solve_W(U_try, V_try, W_try)
                    if error_of(U_try, V_try, W_try) < 1e-10:
                        break
                if verify(U_try, V_try, W_try, tol=1e-9):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_try, V_try, W_try
                    continue
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
