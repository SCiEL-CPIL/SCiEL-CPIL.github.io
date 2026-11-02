"""Program to find a rank-minimal exact tensor decomposition for 2x2 by 2x5 matrix multiplication."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_start_time = time.time()
DEADLINE = 143.0


def _verify_full(U, V, W, n=2, m=2, p=5, tol=1e-8):
    """Verify the full 2x2x5 tensor decomposition. Returns True if exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    target = 1.0 if j == jp else 0.0
                    val = 0.0
                    for c in range(rank):
                        val += U[i * m + j][c] * V[jp * p + k][c] * W[k * n + i][c]
                    if abs(val - target) > tol:
                        return False
    return True


def _get_rank_18_solution(n=2, m=2, p=5):
    """Fallback: 2 Strassen blocks (rank 7 each) for cols 0-3, schoolbook (rank 4) for col 4."""
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(t_offset, k_start):
        U[0][t_offset + 0] = 1
        U[3][t_offset + 0] = 1
        U[2][t_offset + 1] = 1
        U[3][t_offset + 1] = 1
        U[0][t_offset + 2] = 1
        U[3][t_offset + 3] = 1
        U[0][t_offset + 4] = 1
        U[1][t_offset + 4] = 1
        U[2][t_offset + 5] = 1
        U[0][t_offset + 5] = -1
        U[1][t_offset + 6] = 1
        U[3][t_offset + 6] = -1

        k0 = k_start
        k1 = k_start + 1
        V[0 * p + k0][t_offset + 0] = 1
        V[1 * p + k1][t_offset + 0] = 1
        V[0 * p + k0][t_offset + 1] = 1
        V[0 * p + k1][t_offset + 2] = 1
        V[1 * p + k1][t_offset + 2] = -1
        V[1 * p + k0][t_offset + 3] = 1
        V[0 * p + k0][t_offset + 3] = -1
        V[1 * p + k1][t_offset + 4] = 1
        V[0 * p + k0][t_offset + 5] = 1
        V[0 * p + k1][t_offset + 5] = 1
        V[1 * p + k0][t_offset + 6] = 1
        V[1 * p + k1][t_offset + 6] = 1

        W[k0 * n + 0][t_offset + 0] = 1
        W[k1 * n + 1][t_offset + 0] = 1
        W[k0 * n + 1][t_offset + 1] = 1
        W[k1 * n + 1][t_offset + 1] = -1
        W[k1 * n + 0][t_offset + 2] = 1
        W[k1 * n + 1][t_offset + 2] = 1
        W[k0 * n + 0][t_offset + 3] = 1
        W[k0 * n + 1][t_offset + 3] = 1
        W[k0 * n + 0][t_offset + 4] = -1
        W[k1 * n + 0][t_offset + 4] = 1
        W[k1 * n + 1][t_offset + 5] = 1
        W[k0 * n + 0][t_offset + 6] = 1

    add_strassen_block(0, 0)
    add_strassen_block(7, 2)

    t = 14
    U[0][t] = 1; V[0 * p + 4][t] = 1; W[4 * n + 0][t] = 1; t += 1
    U[1][t] = 1; V[1 * p + 4][t] = 1; W[4 * n + 0][t] = 1; t += 1
    U[2][t] = 1; V[0 * p + 4][t] = 1; W[4 * n + 1][t] = 1; t += 1
    U[3][t] = 1; V[1 * p + 4][t] = 1; W[4 * n + 1][t] = 1

    return {"U": U, "V": V, "W": W}


def _compute_error_223(U, V, W, rank):
    """Compute SSE error for 2x2x3 sub-decomposition."""
    n_sub, m_sub, p_sub = 2, 2, 3
    err = 0.0
    for i in range(n_sub):
        for j in range(m_sub):
            for jp in range(m_sub):
                for k in range(p_sub):
                    target = 1.0 if j == jp else 0.0
                    val = np.dot(U[i * m_sub + j, :], V[jp * p_sub + k, :] * W[k * n_sub + i, :])
                    err += (val - target) ** 2
    return err


def _check_exact_223(U, V, W, rank, tol=1e-6):
    """Check if 2x2x3 decomposition is exact within tolerance."""
    n_sub, m_sub, p_sub = 2, 2, 3
    for i in range(n_sub):
        for j in range(m_sub):
            for jp in range(m_sub):
                for k in range(p_sub):
                    target = 1.0 if j == jp else 0.0
                    val = np.dot(U[i * m_sub + j, :], V[jp * p_sub + k, :] * W[k * n_sub + i, :])
                    if abs(val - target) > tol:
                        return False
    return True


def _snap_to_half_int(U, V, W):
    """Snap values to nearest half-integer and return snapped arrays."""
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    return U_r, V_r, W_r


def _als_223(rank, n_iter=200, seed=0, time_budget=None):
    """
    ALS (Alternating Least Squares) for 2x2x3 tensor decomposition.
    Fixes two matrices, solves linear system for the third.
    """
    n_sub, m_sub, p_sub = 2, 2, 3
    np.random.seed(seed)
    
    U = np.random.randn(n_sub * m_sub, rank) * 0.5
    V = np.random.randn(m_sub * p_sub, rank) * 0.5
    W = np.random.randn(p_sub * n_sub, rank) * 0.5
    
    for it in range(n_iter):
        if time_budget is not None and time.time() > time_budget:
            break
        
        # Fix U, V, solve for W: for each (k, i)
        for k in range(p_sub):
            for i in range(n_sub):
                A_mat = np.zeros((m_sub * m_sub, rank))
                b_vec = np.zeros(m_sub * m_sub)
                for j in range(m_sub):
                    for jp in range(m_sub):
                        A_mat[j * m_sub + jp, :] = U[i * m_sub + j, :] * V[jp * p_sub + k, :]
                        b_vec[j * m_sub + jp] = 1.0 if j == jp else 0.0
                w, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W[k * n_sub + i, :] = w
        
        # Fix U, W, solve for V: for each (j, k)
        for j in range(m_sub):
            for k in range(p_sub):
                A_mat = np.zeros((n_sub * m_sub, rank))
                b_vec = np.zeros(n_sub * m_sub)
                for i in range(n_sub):
                    for jp in range(m_sub):
                        A_mat[i * m_sub + jp, :] = U[i * m_sub + jp, :] * W[k * n_sub + i, :]
                        b_vec[i * m_sub + jp] = 1.0 if j == jp else 0.0
                v, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                V[j * p_sub + k, :] = v
        
        # Fix V, W, solve for U: for each (i, j)
        for i in range(n_sub):
            for j in range(m_sub):
                A_mat = np.zeros((m_sub * p_sub, rank))
                b_vec = np.zeros(m_sub * p_sub)
                for jp in range(m_sub):
                    for k in range(p_sub):
                        A_mat[jp * p_sub + k, :] = V[jp * p_sub + k, :] * W[k * n_sub + i, :]
                        b_vec[jp * p_sub + k] = 1.0 if j == jp else 0.0
                u, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                U[i * m_sub + j, :] = u
        
        # Check convergence
        err = _compute_error_223(U, V, W, rank)
        if err < 1e-12:
            break
    
    err = _compute_error_223(U, V, W, rank)
    return U, V, W, err


def _polish_223(U, V, W, rank, tol=1e-12, max_iter=20, time_budget=None):
    """High-precision polish using L-BFGS-B."""
    n_sub, m_sub, p_sub = 2, 2, 3
    size = n_sub * m_sub * rank + m_sub * p_sub * rank + p_sub * n_sub * rank
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

    def _err(params):
        U_p = params[:n_sub * m_sub * rank].reshape((n_sub * m_sub, rank))
        V_p = params[n_sub * m_sub * rank: n_sub * m_sub * rank + m_sub * p_sub * rank].reshape((m_sub * p_sub, rank))
        W_p = params[n_sub * m_sub * rank + m_sub * p_sub * rank:].reshape((p_sub * n_sub, rank))
        return _compute_error_223(U_p, V_p, W_p, rank)

    for iteration in range(max_iter):
        if time_budget is not None and time.time() > time_budget:
            break
        res = minimize(_err, x0, method='L-BFGS-B',
                       options={'maxiter': 100, 'ftol': 1e-16, 'gtol': 1e-16, 'disp': False})
        x0 = res.x
        if res.fun < tol:
            break

    U_p = x0[:n_sub * m_sub * rank].reshape((n_sub * m_sub, rank))
    V_p = x0[n_sub * m_sub * rank: n_sub * m_sub * rank + m_sub * p_sub * rank].reshape((m_sub * p_sub, rank))
    W_p = x0[n_sub * m_sub * rank + m_sub * p_sub * rank:].reshape((p_sub * n_sub, rank))
    return U_p, V_p, W_p


def _try_find_223_decomposition(target_rank, time_budget):
    """
    Attempt to find an exact rank-target_rank decomposition for 2x2x3.
    Uses ALS with multiple restarts, then polishing.
    Returns (U, V, W) if found, else None.
    """
    n_sub, m_sub, p_sub = 2, 2, 3
    best_solution = None
    best_error = float('inf')
    
    n_restarts = 30
    for restart in range(n_restarts):
        if time.time() > time_budget:
            break
        
        # Vary the number of ALS iterations and seed
        n_iter = 100 + (restart % 3) * 50
        U, V, W, err = _als_223(target_rank, n_iter=n_iter, seed=restart * 7 + 1, time_budget=time_budget)
        
        if err < best_error:
            best_error = err
            best_solution = (U.copy(), V.copy(), W.copy())
        
        # If close, try polishing
        if err < 1e-4:
            U_p, V_p, W_p = _polish_223(U, V, W, target_rank, tol=1e-14, max_iter=10, time_budget=time_budget)
            err_p = _compute_error_223(U_p, V_p, W_p, target_rank)
            
            if err_p < best_error:
                best_error = err_p
                best_solution = (U_p.copy(), V_p.copy(), W_p.copy())
            
            if err_p < 1e-8:
                U_final, V_final, W_final = best_solution
                if _check_exact_223(U_final, V_final, W_final, target_rank, tol=1e-6):
                    U_r, V_r, W_r = _snap_to_half_int(U_final, V_final, W_final)
                    if _check_exact_223(U_r, V_r, W_r, target_rank, tol=1e-6):
                        return (U_r, V_r, W_r)
                    return (U_final, V_final, W_final)
    
    # Final polish on best solution
    if best_solution is not None and best_error < 1e-2:
        U_t, V_t, W_t = best_solution
        U_p, V_p, W_p = _polish_223(U_t, V_t, W_t, target_rank, tol=1e-16, max_iter=15, time_budget=time_budget)
        err_p = _compute_error_223(U_p, V_p, W_p, target_rank)
        
        if err_p < 1e-8:
            if _check_exact_223(U_p, V_p, W_p, target_rank, tol=1e-6):
                U_r, V_r, W_r = _snap_to_half_int(U_p, V_p, W_p)
                if _check_exact_223(U_r, V_r, W_r, target_rank, tol=1e-6):
                    return (U_r, V_r, W_r)
                return (U_p, V_p, W_p)
    
    return None


def construct():
    n, m, p = DIMENSIONS

    # Try to find rank 10 for 2x2x3 first (total rank 17)
    sub_decomp = None
    sub_rank = None

    if time.time() < DEADLINE - 30:
        sub_decomp = _try_find_223_decomposition(10, time_budget=DEADLINE - 30)
        if sub_decomp is not None:
            sub_rank = 10

    # Try to find rank 9 for 2x2x3 (total rank 16) if rank 10 failed
    if sub_decomp is None and time.time() < DEADLINE - 15:
        sub_decomp = _try_find_223_decomposition(9, time_budget=DEADLINE - 15)
        if sub_decomp is not None:
            sub_rank = 9

    if sub_decomp is not None:
        U_sub, V_sub, W_sub = sub_decomp
        rank_sub = U_sub.shape[1]
        total_rank = 7 + rank_sub

        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]

        # Strassen block for cols 0, 1
        U[0][0] = 1; U[3][0] = 1
        U[2][1] = 1; U[3][1] = 1
        U[0][2] = 1
        U[3][3] = 1
        U[0][4] = 1; U[1][4] = 1
        U[2][5] = 1; U[0][5] = -1
        U[1][6] = 1; U[3][6] = -1

        V[0 * p + 0][0] = 1
        V[1 * p + 1][0] = 1
        V[0 * p + 0][1] = 1
        V[0 * p + 1][2] = 1
        V[1 * p + 1][2] = -1
        V[1 * p + 0][3] = 1
        V[0 * p + 0][3] = -1
        V[1 * p + 1][4] = 1
        V[0 * p + 0][5] = 1
        V[0 * p + 1][5] = 1
        V[1 * p + 0][6] = 1
        V[1 * p + 1][6] = 1

        W[0 * n + 0][0] = 1
        W[1 * n + 1][0] = 1
        W[0 * n + 1][1] = 1
        W[1 * n + 1][1] = -1
        W[1 * n + 0][2] = 1
        W[1 * n + 1][2] = 1
        W[0 * n + 0][3] = 1
        W[0 * n + 1][3] = 1
        W[0 * n + 0][4] = -1
        W[1 * n + 0][4] = 1
        W[1 * n + 1][5] = 1
        W[0 * n + 0][6] = 1

        # Copy sub-decomposition (cols 2,3,4 of B)
        for i in range(2):
            for j in range(2):
                for c in range(rank_sub):
                    U[i * m + j][7 + c] = U_sub[i * 2 + j, c]

        for j in range(2):
            for k_sub in range(3):
                k_global = 2 + k_sub
                for c in range(rank_sub):
                    V[j * p + k_global][7 + c] = V_sub[j * 3 + k_sub, c]

        for k_sub in range(3):
            k_global = 2 + k_sub
            for i in range(2):
                for c in range(rank_sub):
                    W[k_global * n + i][7 + c] = W_sub[k_sub * 2 + i, c]

        # Verify the full solution
        if _verify_full(U, V, W, n, m, p, tol=1e-6):
            return {"U": U, "V": V, "W": W}
        else:
            return _get_rank_18_solution(n, m, p)
    else:
        return _get_rank_18_solution(n, m, p)

# EVOLVE-BLOCK-END
