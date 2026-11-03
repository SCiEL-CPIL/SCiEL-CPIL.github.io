"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
import scipy.optimize as opt

N, M, P = 2, 4, 4
TIME_LIMIT = 130  # safety margin before the 150s kill
START_TIME = time.time()


def _time_up():
    return (time.time() - START_TIME) > TIME_LIMIT


def _target_tensor():
    """The matrix-multiplication tensor T[i,j,k] = 1 for all i,j,k."""
    return np.ones((N, M, P), dtype=float)


def _recon(U, V, W):
    """Reconstruct T from factor matrices (already numpy, shape (n*m, r) etc.)."""
    r = U.shape[1]
    # U: (N*M, r), V: (M*P, r), W: (P*N, r)
    # T[i,j,k] = sum_r U[i*M+j, r] * V[j*P+k, r] * W[k*N+i, r]
    T = np.zeros((N, M, P), dtype=float)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T[i, j, k] = np.einsum('r,r,r->', U[i * M + j], V[j * P + k], W[k * N + i])
    return T


def _residuals(x, rank):
    """Residual vector for the least-squares objective."""
    su = N * M * rank
    sv = M * P * rank
    U = x[:su].reshape(N * M, rank)
    V = x[su:su + sv].reshape(M * P, rank)
    W = x[su + sv:].reshape(P * N, rank)
    T = _recon(U, V, W)
    return T.flatten() - 1.0


def _residual_norm(x, rank):
    return np.linalg.norm(_residuals(x, rank))


def _strassen_factors():
    """Return (U, V, W) numpy arrays for the rank-28 Strassen decomposition."""
    rank = 28
    U = np.zeros((N * M, rank), dtype=float)
    V = np.zeros((M * P, rank), dtype=float)
    W = np.zeros((P * N, rank), dtype=float)

    # Strassen components for 2x2*2x2 (P*Q=R)
    # (U pattern on P, V pattern on Q, W pattern on R)
    strassen_components = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
    ]

    # 4 terms: (a_col_offset, b_row_offset, b_col_offset, c_col_offset)
    terms = [
        (0, 0, 0, 0),  # A0 * B00 -> C0
        (2, 2, 0, 0),  # A1 * B10 -> C0
        (0, 0, 2, 2),  # A0 * B01 -> C1
        (2, 2, 2, 2),  # A1 * B11 -> C1
    ]

    col = 0
    for (a_co, b_ro, b_co, c_co) in terms:
        for (u_pat, v_pat, w_pat) in strassen_components:
            for (pi, pj), val in u_pat.items():
                U[pi * M + (a_co + pj), col] += val
            for (qi, qj), val in v_pat.items():
                V[(b_ro + qi) * P + (b_co + qj), col] += val
            for (ri, rj), val in w_pat.items():
                W[(c_co + rj) * N + ri, col] += val
            col += 1

    return U, V, W


def _verify(U, V, W):
    """Check exactness of the decomposition."""
    T = _recon(U, V, W)
    return np.max(np.abs(T - 1.0)) < 1e-9


def _polish(x, rank, max_iter=200, tol=1e-12):
    """High-precision Newton/LM polish on the current iterate."""
    for _ in range(max_iter):
        if _time_up():
            break
        r = _residuals(x, rank)
        if np.linalg.norm(r) < tol:
            break
        # Jacobian via finite differences (small enough problem to be OK)
        su = N * M * rank
        sv = M * P * rank
        total = su + sv + P * N * rank
        # Use scipy's numerical Jacobian through least_squares for robustness
        try:
            res = opt.least_squares(_residuals, x, args=(rank,), method='lm',
                                    max_nfev=200, xtol=1e-14, ftol=1e-14, gtol=1e-14)
            x = res.x
            if np.linalg.norm(res.fun) < tol:
                break
        except Exception:
            break
    return x


def _search_rank(target_rank, U0, V0, W0, max_restarts=12):
    """Try to find a valid decomposition of target_rank, seeded from Strassen."""
    su = N * M * target_rank
    sv = M * P * target_rank
    total = su + sv + P * N * target_rank

    best_x = None
    best_err = float('inf')

    for restart in range(max_restarts):
        if _time_up():
            break

        # Initialize: take first target_rank columns of Strassen (if available) + perturbation
        if target_rank <= 28:
            U_init = U0[:, :target_rank].copy()
            V_init = V0[:, :target_rank].copy()
            W_init = W0[:, :target_rank].copy()
        else:
            U_init = np.hstack([U0, np.random.randn(N * M, target_rank - 28) * 0.1])
            V_init = np.hstack([V0, np.random.randn(M * P, target_rank - 28) * 0.1])
            W_init = np.hstack([W0, np.random.randn(P * N, target_rank - 28) * 0.1])

        if restart > 0:
            scale = 0.02 + 0.01 * restart
            U_init = U_init + np.random.randn(N * M, target_rank) * scale
            V_init = V_init + np.random.randn(M * P, target_rank) * scale
            W_init = W_init + np.random.randn(P * N, target_rank) * scale

        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

        try:
            res = opt.least_squares(_residuals, x0, args=(target_rank,), method='lm',
                                    max_nfev=3000)
            err = np.linalg.norm(res.fun)
            if err < best_err:
                best_err = err
                best_x = res.x.copy()
            if err < 1e-9:
                # Polish for exactness
                x_pol = _polish(res.x, target_rank)
                U = x_pol[:su].reshape(N * M, target_rank)
                V = x_pol[su:su + sv].reshape(M * P, target_rank)
                W = x_pol[su + sv:].reshape(P * N, target_rank)
                if _verify(U, V, W):
                    return U, V, W
        except Exception:
            pass

    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for 2x4 by 4x4 matrix multiplication."""
    # Guaranteed rank-28 Strassen solution as fallback
    U_s, V_s, W_s = _strassen_factors()

    # Try progressively lower ranks
    for target_rank in [24, 22, 20]:
        if _time_up():
            break
        result = _search_rank(target_rank, U_s, V_s, W_s, max_restarts=15)
        if result is not None:
            U, V, W = result
            # Final polish to ensure exactness
            su = N * M * target_rank
            sv = M * P * target_rank
            x = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            x_pol = _polish(x, target_rank, max_iter=50, tol=1e-13)
            U = x_pol[:su].reshape(N * M, target_rank)
            V = x_pol[su:su + sv].reshape(M * P, target_rank)
            W = x_pol[su + sv:].reshape(P * N, target_rank)
            if _verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: return the Strassen rank-28 solution
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
# EVOLVE-BLOCK-END
