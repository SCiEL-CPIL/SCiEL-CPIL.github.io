import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _build_strassen_28(n, m, p):
    """Build the known rank-28 Strassen-based decomposition."""
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen terms for X(2x2) * Y(2x2) = P(2x2)
    # Each term: (U_factors, V_factors, W_factors) where each is list of (local_idx, coeff)
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    def map_u_a(side, local_idx):
        if side == 'L':
            return [0, 1, 4, 5][local_idx]
        else:
            return [2, 3, 6, 7][local_idx]
            
    def map_v_b(block, local_idx):
        r = local_idx // 2
        c = local_idx % 2
        if block == 'TL':
            return r * 4 + c
        elif block == 'TR':
            return r * 4 + (2 + c)
        elif block == 'BL':
            return (2 + r) * 4 + c
        elif block == 'BR':
            return (2 + r) * 4 + (2 + c)
            
    def map_w_c(side, local_idx):
        r = local_idx // 2
        c = local_idx % 2
        if side == 'L':
            return c * 2 + r
        else:
            return (2 + c) * 2 + r
            
    col = 0
    for side_a, block_b, side_c in [('L','TL','L'), ('R','BL','L'), ('L','TR','R'), ('R','BR','R')]:
        for u_l, v_l, w_l in strassen_terms:
            for idx, coeff in u_l:
                U[map_u_a(side_a, idx), col] += coeff
            for idx, coeff in v_l:
                V[map_v_b(block_b, idx), col] += coeff
            for idx, coeff in w_l:
                W[map_w_c(side_c, idx), col] += coeff
            col += 1
        
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify the factorization against random test matrices."""
    np.random.seed(42)
    for trial in range(3):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        Cf = np.zeros((n, p))
        for ii in range(n):
            for kk in range(p):
                acc = 0.0
                for cc in range(U.shape[1]):
                    for jj in range(m):
                        acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
                Cf[ii, kk] = acc
        if not np.allclose(C, Cf, atol=1e-8):
            return False
    return True

def _try_rank_reduction(U28, V28, W28, n, m, p, target_rank, deadline):
    """Try to find a lower-rank decomposition by optimizing."""
    nU, nV, nW = n * m, m * p, p * n
    r = target_rank
    
    # Precompute the constraint indices
    constraint_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                constraint_indices.append((i * m + j, j * p + k, k * n + i))
    n_constraints = len(constraint_indices)
    
    def make_residuals():
        def residuals(x):
            Uf = x[:nU * r].reshape(nU, r)
            Vf = x[nU * r:nU * r + nV * r].reshape(nV, r)
            Wf = x[nU * r + nV * r:].reshape(nW, r)
            res = np.empty(n_constraints)
            for idx, (ui, vi, wi) in enumerate(constraint_indices):
                res[idx] = np.dot(Uf[ui, :] * Vf[vi, :], Wf[wi, :]) - 1.0
            return res
        return residuals
    
    residuals = make_residuals()
    
    # Strategy: for each column to "remove", start from rank-28 with that column zeroed
    # and optimize all r columns
    np.random.seed(123)
    best_cost = np.inf
    best_x = None
    
    for remove_idx in range(28):
        if time.time() > deadline - 20:
            break
        
        # Initialize: keep all 28 columns except remove_idx, but we need r=27 columns
        # Use the remaining 27 columns as initial guess
        keep = [c for c in range(28) if c != remove_idx]
        U_init = U28[:, keep].copy()
        V_init = V28[:, keep].copy()
        W_init = W28[:, keep].copy()
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        # Try a few initializations: the direct one, and perturbed versions
        n_attempts = 3
        for attempt in range(n_attempts):
            if time.time() > deadline - 20:
                break
            if attempt == 0:
                x_try = x0.copy()
            else:
                x_try = x0 + np.random.randn(len(x0)) * 0.05 * attempt
            try:
                result = least_squares(residuals, x_try, method='lm', max_nfev=2000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
                if result.cost < best_cost:
                    best_cost = result.cost
                    best_x = result.x.copy()
                if np.max(np.abs(result.fun)) < 1e-10:
                    # Found a solution, verify it
                    U_sol = best_x[:nU * r].reshape(nU, r)
                    V_sol = best_x[nU * r:nU * r + nV * r].reshape(nV, r)
                    W_sol = best_x[nU * r + nV * r:].reshape(nW, r)
                    if _verify(U_sol, V_sol, W_sol, n, m, p):
                        return U_sol, V_sol, W_sol
                    break
            except Exception:
                pass
    
    # If we have a near-solution, do a polish pass
    if best_x is not None and best_cost < 1e-6:
        try:
            result = least_squares(residuals, best_x, method='lm', max_nfev=5000, xtol=1e-15, ftol=1e-15, gtol=1e-15)
            if np.max(np.abs(result.fun)) < 1e-8:
                U_sol = result.x[:nU * r].reshape(nU, r)
                V_sol = result.x[nU * r:nU * r + nV * r].reshape(nV, r)
                W_sol = result.x[nU * r + nV * r:].reshape(nW, r)
                if _verify(U_sol, V_sol, W_sol, n, m, p):
                    return U_sol, V_sol, W_sol
        except Exception:
            pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Build the rank-28 Strassen solution
    U28, V28, W28 = _build_strassen_28(n, m, p)
    
    # Verify rank-28 solution
    if not _verify(U28, V28, W28, n, m, p):
        # Should not happen, but fallback
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
    
    # Try to reduce to rank 27
    result = _try_rank_reduction(U28, V28, W28, n, m, p, 27, deadline)
    if result is not None:
        U_sol, V_sol, W_sol = result
        return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
    
    # Return rank 28
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
