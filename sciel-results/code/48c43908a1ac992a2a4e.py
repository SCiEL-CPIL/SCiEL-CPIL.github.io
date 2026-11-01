"""Program to find a rank-minimal exact tensor decomposition for 2x2 by 2x5 matrix multiplication."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_start_time = time.time()
DEADLINE = 145.0

def construct():
    n, m, p = DIMENSIONS

    def get_rank_18_solution():
        rank = 18
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        def add_strassen_block(t_offset, k_start):
            U[0][t_offset+0] = 1
            U[3][t_offset+0] = 1
            U[2][t_offset+1] = 1
            U[3][t_offset+1] = 1
            U[0][t_offset+2] = 1
            U[3][t_offset+3] = 1
            U[0][t_offset+4] = 1
            U[1][t_offset+4] = 1
            U[2][t_offset+5] = 1
            U[0][t_offset+5] = -1
            U[1][t_offset+6] = 1
            U[3][t_offset+6] = -1

            k0 = k_start
            k1 = k_start + 1
            V[0*p + k0][t_offset+0] = 1
            V[1*p + k1][t_offset+0] = 1
            V[0*p + k0][t_offset+1] = 1
            V[0*p + k1][t_offset+2] = 1
            V[1*p + k1][t_offset+2] = -1
            V[1*p + k0][t_offset+3] = 1
            V[0*p + k0][t_offset+3] = -1
            V[1*p + k1][t_offset+4] = 1
            V[0*p + k0][t_offset+5] = 1
            V[0*p + k1][t_offset+5] = 1
            V[1*p + k0][t_offset+6] = 1
            V[1*p + k1][t_offset+6] = 1

            W[k0*n + 0][t_offset+0] = 1
            W[k1*n + 1][t_offset+0] = 1
            W[k0*n + 1][t_offset+1] = 1
            W[k1*n + 1][t_offset+1] = -1
            W[k1*n + 0][t_offset+2] = 1
            W[k1*n + 1][t_offset+2] = 1
            W[k0*n + 0][t_offset+3] = 1
            W[k0*n + 1][t_offset+3] = 1
            W[k0*n + 0][t_offset+4] = -1
            W[k1*n + 0][t_offset+4] = 1
            W[k1*n + 1][t_offset+5] = 1
            W[k0*n + 0][t_offset+6] = 1

        add_strassen_block(0, 0)
        add_strassen_block(7, 2)

        t = 14
        U[0][t] = 1
        V[0*p + 4][t] = 1
        W[4*n + 0][t] = 1
        t += 1
        U[1][t] = 1
        V[1*p + 4][t] = 1
        W[4*n + 0][t] = 1
        t += 1
        U[2][t] = 1
        V[0*p + 4][t] = 1
        W[4*n + 1][t] = 1
        t += 1
        U[3][t] = 1
        V[1*p + 4][t] = 1
        W[4*n + 1][t] = 1

        return {"U": U, "V": V, "W": W}

    n_sub = 2
    m_sub = 2
    p_sub = 3
    total_params = n_sub*m_sub + m_sub*p_sub + p_sub*n_sub  # 4+6+6=16

    def error_223(params, rank):
        U = params[:n_sub*m_sub*rank].reshape((n_sub*m_sub, rank))
        V = params[n_sub*m_sub*rank : n_sub*m_sub*rank + m_sub*p_sub*rank].reshape((m_sub*p_sub, rank))
        W = params[n_sub*m_sub*rank + m_sub*p_sub*rank :].reshape((p_sub*n_sub, rank))

        err = 0.0
        for i in range(n_sub):
            for j in range(m_sub):
                for jp in range(m_sub):
                    for k in range(p_sub):
                        target = 1.0 if j == jp else 0.0
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m_sub + j, c] * V[jp*p_sub + k, c] * W[k*n_sub + i, c]
                        err += (val - target) ** 2
        return err

    def check_exact_223(U, V, W, rank, tol=1e-6):
        for i in range(n_sub):
            for j in range(m_sub):
                for jp in range(m_sub):
                    for k in range(p_sub):
                        target = 1.0 if j == jp else 0.0
                        val = 0.0
                        for c in range(rank):
                            val += U[i*m_sub + j, c] * V[jp*p_sub + k, c] * W[k*n_sub + i, c]
                        if abs(val - target) > tol:
                            return False
        return True

    def snap_to_half_integers(arr, tol=1e-4):
        snapped = np.round(arr * 2) / 2
        if np.allclose(arr, snapped, atol=tol):
            return snapped
        return None

    def snap_to_integers(arr, tol=1e-4):
        snapped = np.round(arr)
        if np.allclose(arr, snapped, atol=tol):
            return snapped
        return None

    def try_find_223_decomposition(target_rank):
        if time.time() > DEADLINE - 15:
            return None

        max_restarts = 40
        for restart in range(max_restarts):
            if time.time() > DEADLINE - 15:
                break

            np.random.seed(restart * 97 + target_rank * 31 + 7)

            if restart % 4 == 0:
                x0 = np.random.uniform(-1, 1, size=total_params * target_rank)
            elif restart % 4 == 1:
                x0 = np.random.uniform(-2, 2, size=total_params * target_rank)
            elif restart % 4 == 2:
                # Sparse: half zeros, half ±1
                x0 = np.zeros(total_params * target_rank)
                n_nonzero = total_params * target_rank // 2
                idx = np.random.choice(len(x0), size=n_nonzero, replace=False)
                x0[idx] = np.random.choice([-1, 1], size=n_nonzero)
            else:
                # Sparse: quarter nonzero, values in {-1, 0, 1}
                x0 = np.zeros(total_params * target_rank)
                n_nonzero = total_params * target_rank // 4
                idx = np.random.choice(len(x0), size=n_nonzero, replace=False)
                x0[idx] = np.random.choice([-1, 1], size=n_nonzero)

            res = minimize(error_223, x0, args=(target_rank,), method='Nelder-Mead',
                           options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-12, 'disp': False})

            if res.fun < 1e-8:
                U = res.x[:n_sub*m_sub*target_rank].reshape((n_sub*m_sub, target_rank))
                V = res.x[n_sub*m_sub*target_rank : n_sub*m_sub*target_rank + m_sub*p_sub*target_rank].reshape((m_sub*p_sub, target_rank))
                W = res.x[n_sub*m_sub*target_rank + m_sub*p_sub*target_rank :].reshape((p_sub*n_sub, target_rank))

                # Try snapping to integers
                U_i = snap_to_integers(U)
                V_i = snap_to_integers(V)
                W_i = snap_to_integers(W)
                if U_i is not None and V_i is not None and W_i is not None:
                    if check_exact_223(U_i, V_i, W_i, target_rank, tol=1e-8):
                        return (U_i, V_i, W_i)

                # Try snapping to half-integers
                U_r = snap_to_half_integers(U)
                V_r = snap_to_half_integers(V)
                W_r = snap_to_half_integers(W)
                if U_r is not None and V_r is not None and W_r is not None:
                    if check_exact_223(U_r, V_r, W_r, target_rank, tol=1e-8):
                        return (U_r, W_r, V_r) if False else (U_r, V_r, W_r)

                # If raw float is exact, try harder snapping with looser tol
                if check_exact_223(U, V, W, target_rank, tol=1e-5):
                    U_i2 = np.round(U)
                    V_i2 = np.round(V)
                    W_i2 = np.round(W)
                    if check_exact_223(U_i2, V_i2, W_i2, target_rank, tol=1e-6):
                        return (U_i2, V_i2, W_i2)

                    U_r2 = np.round(U * 2) / 2
                    V_r2 = np.round(V * 2) / 2
                    W_r2 = np.round(W * 2) / 2
                    if check_exact_223(U_r2, V_r2, W_r2, target_rank, tol=1e-6):
                        return (U_r2, V_r2, W_r2)

        return None

    # Try to find rank 9 for 2x2x3 (total rank 16)
    sub_decomp = None
    if time.time() < DEADLINE - 30:
        sub_decomp = try_find_223_decomposition(9)

    # Try to find rank 10 for 2x2x3 (total rank 17)
    if sub_decomp is None and time.time() < DEADLINE - 30:
        sub_decomp = try_find_223_decomposition(10)

    if sub_decomp is not None:
        U_sub, V_sub, W_sub = sub_decomp
        rank_sub = U_sub.shape[1]
        total_rank = 7 + rank_sub

        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]

        def add_strassen_block(t_offset, k_start):
            U[0][t_offset+0] = 1
            U[3][t_offset+0] = 1
            U[2][t_offset+1] = 1
            U[3][t_offset+1] = 1
            U[0][t_offset+2] = 1
            U[3][t_offset+3] = 1
            U[0][t_offset+4] = 1
            U[1][t_offset+4] = 1
            U[2][t_offset+5] = 1
            U[0][t_offset+5] = -1
            U[1][t_offset+6] = 1
            U[3][t_offset+6] = -1

            k0 = k_start
            k1 = k_start + 1
            V[0*p + k0][t_offset+0] = 1
            V[1*p + k1][t_offset+0] = 1
            V[0*p + k0][t_offset+1] = 1
            V[0*p + k1][t_offset+2] = 1
            V[1*p + k1][t_offset+2] = -1
            V[1*p + k0][t_offset+3] = 1
            V[0*p + k0][t_offset+3] = -1
            V[1*p + k1][t_offset+4] = 1
            V[0*p + k0][t_offset+5] = 1
            V[0*p + k1][t_offset+5] = 1
            V[1*p + k0][t_offset+6] = 1
            V[1*p + k1][t_offset+6] = 1

            W[k0*n + 0][t_offset+0] = 1
            W[k1*n + 1][t_offset+0] = 1
            W[k0*n + 1][t_offset+1] = 1
            W[k1*n + 1][t_offset+1] = -1
            W[k1*n + 0][t_offset+2] = 1
            W[k1*n + 1][t_offset+2] = 1
            W[k0*n + 0][t_offset+3] = 1
            W[k0*n + 1][t_offset+3] = 1
            W[k0*n + 0][t_offset+4] = -1
            W[k1*n + 0][t_offset+4] = 1
            W[k1*n + 1][t_offset+5] = 1
            W[k0*n + 0][t_offset+6] = 1

        add_strassen_block(0, 0)

        for i in range(n_sub):
            for j in range(m_sub):
                for c in range(rank_sub):
                    U[i*m_sub + j][7 + c] = U_sub[i*m_sub + j, c]

        for j in range(m_sub):
            for k_sub in range(p_sub):
                k_global = 2 + k_sub
                for c in range(rank_sub):
                    V[j*p + k_global][7 + c] = V_sub[j*p_sub + k_sub, c]

        for k_sub in range(p_sub):
            k_global = 2 + k_sub
            for i in range(n_sub):
                for c in range(rank_sub):
                    W[k_global*n + i][7 + c] = W_sub[k_sub*n_sub + i, c]

        return {"U": U, "V": V, "W": W}
    else:
        return get_rank_18_solution()

# EVOLVE-BLOCK-END
