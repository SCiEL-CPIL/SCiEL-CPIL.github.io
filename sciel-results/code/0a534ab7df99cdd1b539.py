"""
Rank-minimal exact tensor decomposition for 2x2 * 2x5 matrix multiplication.
Combined: structural rank-18 baseline + targeted algebraic search for rank 17/16.
"""

import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 5]


def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1
    return T


def _get_rank18_solution():
    """Structural rank-18: two Strassen blocks + schoolbook for col 4."""
    n, m, p = DIMENSIONS
    rank = 18
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)

    def add_strassen_block(col0, col1, base):
        # M1: (a00+a11)(b00+b11)
        r = base
        U[0, r] = 1; U[3, r] = 1
        V[0*p+col0, r] = 1; V[1*p+col1, r] = 1
        W[col0*n+0, r] = 1; W[col1*n+1, r] = 1
        # M2: (a10+a11)(b00)
        r = base + 1
        U[2, r] = 1; U[3, r] = 1
        V[0*p+col0, r] = 1
        W[col0*n+1, r] = 1; W[col1*n+1, r] = -1
        # M3: (a00)(b11-b01)
        r = base + 2
        U[0, r] = 1
        V[0*p+col1, r] = 1; V[1*p+col1, r] = -1
        W[col1*n+0, r] = 1; W[col1*n+1, r] = 1
        # M4: (a11)(b10-b00)
        r = base + 3
        U[3, r] = 1
        V[1*p+col0, r] = 1; V[0*p+col0, r] = -1
        W[col0*n+0, r] = 1; W[col0*n+1, r] = 1
        # M5: (a00+a01)(b11)
        r = base + 4
        U[0, r] = 1; U[1, r] = 1
        V[1*p+col1, r] = 1
        W[col0*n+0, r] = -1; W[col1*n+0, r] = 1
        # M6: (a10-a00)(b00+b01)
        r = base + 5
        U[2, r] = 1; U[0, r] = -1
        V[0*p+col0, r] = 1; V[0*p+col1, r] = 1
        W[col1*n+1, r] = 1
        # M7: (a01-a11)(b10+b11)
        r = base + 6
        U[1, r] = 1; U[3, r] = -1
        V[1*p+col0, r] = 1; V[1*p+col1, r] = 1
        W[col0*n+0, r] = 1

    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)

    # Schoolbook for col 4
    r = 14
    U[0, r] = 1; V[0*p+4, r] = 1; W[4*n+0, r] = 1
    r = 15
    U[1, r] = 1; V[1*p+4, r] = 1; W[4*n+0, r] = 1
    r = 16
    U[2, r] = 1; V[0*p+4, r] = 1; W[4*n+1, r] = 1
    r = 17
    U[3, r] = 1; V[1*p+4, r] = 1; W[4*n+1, r] = 1

    return U, V, W


def _check_decomposition_float(U, V, W, T):
    R = U.shape[1]
    result = np.zeros(T.shape, dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    return np.allclose(result, T, atol=1e-9)


def _compute_tensor(U, V, W):
    R = U.shape[1]
    n, m, p = DIMENSIONS
    result = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    return result


def _to_fraction(val):
    if abs(val) < 1e-12:
        return Fraction(0)
    nearest_half = round(val * 2) / 2.0
    if abs(val - nearest_half) < 1e-9:
        return Fraction(int(nearest_half * 2), 2)
    nearest_int = round(val)
    if abs(val - nearest_int) < 1e-9:
        return Fraction(int(nearest_int))
    for denom in [3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 100]:
        num = round(val * denom)
        if abs(val - num / denom) < 1e-9:
            return Fraction(int(num), denom)
    return Fraction(val).limit_denominator(200)


def _check_decomposition_exact(U, V, W, T):
    n, m, p = DIMENSIONS
    R = U.shape[1]
    U_frac = np.zeros_like(U, dtype=object)
    V_frac = np.zeros_like(V, dtype=object)
    W_frac = np.zeros_like(W, dtype=object)
    for i in range(U.shape[0]):
        for j in range(R):
            U_frac[i, j] = _to_fraction(U[i, j])
    for i in range(V.shape[0]):
        for j in range(R):
            V_frac[i, j] = _to_fraction(V[i, j])
    for i in range(W.shape[0]):
        for j in range(R):
            W_frac[i, j] = _to_fraction(W[i, j])
    for u in range(n*m):
        for v in range(m*p):
            for w in range(p*n):
                total = Fraction(0)
                for r in range(R):
                    total += U_frac[u, r] * V_frac[v, r] * W_frac[w, r]
                if total != Fraction(int(T[u, v, w])):
                    return False
    return True


def _als_update_U(T, V, W, R):
    n, m, p = DIMENSIONS
    U_rows = n * m
    VW = np.einsum('vr,wr->vwr', V, W)
    VW_flat = VW.reshape(-1, R)
    U = np.zeros((U_rows, R), dtype=np.float64)
    for u in range(U_rows):
        b = T[u].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(VW_flat, b, rcond=None)
        U[u] = sol
    return U


def _als_update_V(T, U, W, R):
    n, m, p = DIMENSIONS
    V_rows = m * p
    UW = np.einsum('ur,wr->uwr', U, W)
    UW_flat = UW.reshape(-1, R)
    V = np.zeros((V_rows, R), dtype=np.float64)
    for v in range(V_rows):
        b = T[:, v, :].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(UW_flat, b, rcond=None)
        V[v] = sol
    return V


def _als_update_W(T, U, V, R):
    n, m, p = DIMENSIONS
    W_rows = p * n
    UV = np.einsum('ur,vr->uvr', U, V)
    UV_flat = UV.reshape(-1, R)
    W = np.zeros((W_rows, R), dtype=np.float64)
    for w in range(W_rows):
        b = T[:, :, w].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(UV_flat, b, rcond=None)
        W[w] = sol
    return W


def _compute_error(U, V, W, T):
    result = _compute_tensor(U, V, W)
    return np.linalg.norm(result - T)


def _snap_and_check(U, V, W, T, scales=None):
    """Try snapping to various rational grids and verify exactly."""
    if scales is None:
        scales = [1, 2, 4, 8]
    for scale in scales:
        U_s = np.round(U * scale) / scale
        V_s = np.round(V * scale) / scale
        W_s = np.round(W * scale) / scale
        if _check_decomposition_float(U_s, V_s, W_s, T.astype(np.float64)):
            if _check_decomposition_exact(U_s, V_s, W_s, T):
                return U_s, V_s, W_s
    # Also try half-integer snap
    U_s = np.round(U * 2) / 2.0
    V_s = np.round(V * 2) / 2.0
    W_s = np.round(W * 2) / 2.0
    if _check_decomposition_float(U_s, V_s, W_s, T.astype(np.float64)):
        if _check_decomposition_exact(U_s, V_s, W_s, T):
            return U_s, V_s, W_s
    return None


def _try_rank17_structured(deadline):
    """
    Targeted search for rank 17 by trying to merge/replace pairs of rank-18 terms.
    
    Strategy: For each pair of terms in the rank-18 solution, try to express
    their sum as a single rank-1 term with rational coefficients. Also try
    replacing the 4 schoolbook terms for col 4 with 3 terms using cross-block
    sharing.
    """
    n, m, p = DIMENSIONS
    U18, V18, W18 = _get_rank18_solution()
    T = _build_target()
    T_float = T.astype(np.float64)
    
    # Strategy 1: Try to replace 2 schoolbook terms with 1 cross-term
    # The 4 schoolbook terms for col 4 are at indices 14,15,16,17
    # Can we do col 4 with 3 terms by sharing with Strassen blocks?
    
    # Strategy 2: Try to merge two Strassen terms into one
    # For each pair (i,j) from 0..13, check if U[:,i]*V[:,i]*W[:,i] + U[:,j]*V[:,j]*W[:,j]
    # can be expressed as a single rank-1 term
    
    # Strategy 3: ALS with structured initialization from rank-18
    U_rows, V_rows, W_rows = 4, 10, 10
    
    # Try dropping each pair of terms and running ALS
    best_result = None
    
    for drop1 in range(18):
        if time.time() > deadline:
            break
        for drop2 in range(drop1 + 1, 18):
            if time.time() > deadline:
                break
            
            mask = [i for i in range(18) if i not in (drop1, drop2)]
            U_init = U18[:, mask].copy()
            V_init = V18[:, mask].copy()
            W_init = W18[:, mask].copy()
            
            # Add small perturbation
            np.random.seed(drop1 * 100 + drop2)
            U_init += np.random.randn(U_rows, 16) * 0.05
            V_init += np.random.randn(V_rows, 16) * 0.05
            W_init += np.random.randn(W_rows, 16) * 0.05
            
            U_cur, V_cur, W_cur = U_init.copy(), V_init.copy(), W_init.copy()
            prev_err = np.inf
            
            for it in range(60):
                if time.time() > deadline:
                    break
                U_cur = _als_update_U(T_float, V_cur, W_cur, 16)
                V_cur = _als_update_V(T_float, U_cur, W_cur, 16)
                W_cur = _als_update_W(T_float, U_cur, V_cur, 16)
                err = _compute_error(U_cur, V_cur, W_cur, T_float)
                if abs(prev_err - err) < 1e-14:
                    break
                prev_err = err
            
            if err < 1e-8:
                result = _snap_and_check(U_cur, V_cur, W_cur, T)
                if result is not None:
                    return result
            if time.time() > deadline:
                break
    
    return None


def _try_rank17_als(deadline, n_seeds=30):
    """ALS-based rank 17 search with diverse initializations."""
    n, m, p = DIMENSIONS
    U_rows, V_rows, W_rows = 4, 10, 10
    T = _build_target()
    T_float = T.astype(np.float64)
    U18, V18, W18 = _get_rank18_solution()
    
    seeds = list(range(n_seeds))
    
    for seed in seeds:
        if time.time() > deadline:
            break
        np.random.seed(seed)
        
        # Multiple init strategies
        inits = []
        
        # Pure random
        U = np.random.randn(U_rows, 17) * 0.5
        V = np.random.randn(V_rows, 17) * 0.5
        W = np.random.randn(W_rows, 17) * 0.5
        inits.append((U.copy(), V.copy(), W.copy()))
        
        # Sparse random
        U = np.random.randn(U_rows, 17) * (np.random.rand(U_rows, 17) < 0.4)
        V = np.random.randn(V_rows, 17) * (np.random.rand(V_rows, 17) * 0.4)
        W = np.random.randn(W_rows, 17) * (np.random.rand(W_rows, 17) < 0.4)
        inits.append((U.copy(), V.copy(), W.copy()))
        
        # From rank-18: drop one term + perturb
        drop = seed % 18
        mask = [i for i in range(18) if i != drop]
        U_s = U18[:, mask].copy() + np.random.randn(U_rows, 17) * 0.02
        V_s = V18[:, mask].copy() + np.random.randn(V_rows, 17) * 0.02
        W_s = W18[:, mask].copy() + np.random.randn(W_rows, 17) * 0.02
        inits.append((U_s, V_s, W_s))
        
        for U_init, V_init, W_init in inits:
            if time.time() > deadline:
                break
            
            U_cur, V_cur, W_cur = U_init.copy(), V_init.copy(), W_init.copy()
            prev_err = np.inf
            
            for it in range(80):
                if time.time() > deadline:
                    break
                U_cur = _als_update_U(T_float, V_cur, W_cur, 17)
                V_cur = _als_update_V(T_float, U_cur, W_cur, 17)
                W_cur = _als_update_W(T_float, U_cur, V_cur, 17)
                err = _compute_error(U_cur, V_cur, W_cur, T_float)
                if abs(prev_err - err) < 1e-14:
                    break
                prev_err = err
            
            if err < 1e-6:
                result = _snap_and_check(U_cur, V_cur, W_cur, T)
                if result is not None:
                    return result
    
    return None


def _try_rank16_als(deadline, n_seeds=15):
    """ALS-based rank 16 search."""
    n, m, p = DIMENSIONS
    U_rows, V_rows, W_rows = 4, 10, 10
    T = _build_target()
    T_float = T.astype(np.float64)
    U18, V18, W18 = _get_rank18_solution()
    
    for seed in range(n_seeds):
        if time.time() > deadline:
            break
        np.random.seed(1000 + seed)
        
        # Drop 2 terms from rank-18
        drop1, drop2 = seed % 18, (seed + 7) % 18
        if drop1 == drop2:
            drop2 = (drop2 + 1) % 18
        mask = [i for i in range(18) if i not in (drop1, drop2)]
        
        U_init = U18[:, mask].copy() + np.random.randn(U_rows, 16) * 0.1
        V_init = V18[:, mask].copy() + np.random.randn(V_rows, 16) * 0.1
        W_init = W18[:, mask].copy() + np.random.randn(W_rows, 16) * 0.1
        
        U_cur, V_cur, W_cur = U_init, V_init, W_init
        prev_err = np.inf
        
        for it in range(60):
            if time.time() > deadline:
                break
            U_cur = _als_update_U(T_float, V_cur, W_cur, 16)
            V_cur = _als_update_V(T_float, U_cur, W_cur, 16)
            W_cur = _als_update_W(T_float, U_cur, V_cur, 16)
            err = _compute_error(U_cur, V_cur, W_cur, T_float)
            if abs(prev_err - err) < 1e-14:
                break
            prev_err = err
        
        if err < 1e-6:
            result = _snap_and_check(U_cur, V_cur, W_cur, T)
            if result is not None:
                return result
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Baseline: Rank 18 (instant, guaranteed)
    U18, V18, W18 = _get_rank18_solution()
    best_result = {"U": U18.tolist(), "V": V18.tolist(), "W": W18.tolist()}
    best_rank = 18
    
    # Try rank 17 with structured approach first (fast)
    if time.time() < deadline - 60:
        result = _try_rank17_structured(deadline - 60)
        if result is not None:
            U_l, V_l, W_l = result
            best_result = {"U": U_l.tolist(), "V": V_l.tolist(), "W": W_l.tolist()}
            best_rank = 17
    
    # Try rank 17 with ALS if not found
    if best_rank == 18 and time.time() < deadline - 30:
        result = _try_rank17_als(deadline - 30, n_seeds=25)
        if result is not None:
            U_l, V_l, W_l = result
            best_result = {"U": U_l.tolist(), "V": V_l.tolist(), "W": W_l.tolist()}
            best_rank = 17
    
    # Try rank 16 if time permits
    if time.time() < deadline - 10:
        result = _try_rank16_als(deadline - 5, n_seeds=12)
        if result is not None:
            U_l, V_l, W_l = result
            best_result = {"U": U_l.tolist(), "V": V_l.tolist(), "W": W_l.tolist()}
            best_rank = 16
    
    return best_result
