"""
Rank-minimal exact tensor decomposition for 2x2 * 2x5 matrix multiplication.

Combined approach:
1. Structured ALS with multiple initialization strategies
2. Direct small-integer/half-integer search for low ranks
3. Schoolbook fallback (rank 20)
"""

import numpy as np
import time
from itertools import product, combinations_with_replacement
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


def _check_decomposition(U, V, W, T):
    """Check if sum_r U_u,r * V_v,r * W_w,r == T_u,v,w for all u,v,w."""
    R = U.shape[1]
    result = np.zeros_like(T, dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    return np.allclose(result, T, atol=1e-9)


def _check_decomposition_exact(U, V, W, T):
    """Exact check using Fraction to avoid floating point issues."""
    n, m, p = DIMENSIONS
    U_frac = [[Fraction(int(x*2), 2) if abs(x*2 - round(x*2)) < 1e-9 else Fraction(x).limit_denominator(100) for x in row] for row in U]
    V_frac = [[Fraction(int(x*2), 2) if abs(x*2 - round(x*2)) < 1e-9 else Fraction(x).limit_denominator(100) for x in row] for row in V]
    W_frac = [[Fraction(int(x*2), 2) if abs(x*2 - round(x*2)) < 1e-9 else Fraction(x).limit_denominator(100) for x in row] for row in W]
    
    R = U.shape[1]
    for u in range(n*m):
        for v in range(m*p):
            for w in range(p*n):
                total = Fraction(0)
                for r in range(R):
                    total += U_frac[u][r] * V_frac[v][r] * W_frac[w][r]
                if total != Fraction(int(T[u, v, w])):
                    return False
    return True


def _schoolbook_decomposition():
    """Return the standard schoolbook decomposition with rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U[u, col] = 1
                V[v, col] = 1
                W[w, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()


def _als_update_U(T, V, W, R):
    """Solve for U given V, W using least squares."""
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
    """Solve for V given U, W using least squares."""
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
    """Solve for W given U, V using least squares."""
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


def _compute_error(U, V, W, T, R):
    """Compute the Frobenius norm of the decomposition error."""
    result = np.zeros_like(T, dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    return np.linalg.norm(result - T.astype(np.float64))


def _try_als_rank(R, T, deadline, n_seeds=5, max_iters=80):
    """Try to find a rank-R decomposition using ALS with multiple seeds."""
    n, m, p = DIMENSIONS
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    seeds = [42, 123, 7, 999, 2024, 555, 777, 1111, 2222, 3333]
    
    for seed in seeds[:n_seeds]:
        if time.time() > deadline:
            break
        np.random.seed(seed)
        
        # Multiple initialization strategies
        inits = []
        # Random small
        inits.append((np.random.randn(U_rows, R) * 0.3,
                      np.random.randn(V_rows, R) * 0.3,
                      np.random.randn(W_rows, R) * 0.3))
        # Random medium
        inits.append((np.random.randn(U_rows, R) * 1.0,
                      np.random.randn(V_rows, R) * 1.0,
                      np.random.randn(W_rows, R) * 1.0))
        # Structured: initialize based on schoolbook pattern
        U_init = np.zeros((U_rows, R))
        V_init = np.zeros((V_rows, R))
        W_init = np.zeros((W_rows, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col >= R:
                        break
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U_init[u, col] = 1
                    V_init[v, col] = 1
                    W_init[w, col] = 1
                    col += 1
        if col < R:
            U_init[:, col:R] = np.random.randn(U_rows, R - col) * 0.1
            V_init[:, col:R] = np.random.randn(V_rows, R - col) * 0.1
            W_init[:, col:R] = np.random.randn(W_rows, R - col) * 0.1
        inits.append((U_init, V_init, W_init))
        
        for U0, V0, W0 in inits:
            if time.time() > deadline:
                break
            
            U, V, W = U0.copy(), V0.copy(), W0.copy()
            prev_err = np.inf
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                U = _als_update_U(T, V, W, R)
                V = _als_update_V(T, U, W, R)
                W = _als_update_W(T, U, V, R)
                
                err = _compute_error(U, V, W, T, R)
                
                if abs(prev_err - err) < 1e-14:
                    break
                prev_err = err
            
            # Check for exact solution
            if _check_decomposition(U, V, W, T):
                # Try to rationalize to half-integers
                for max_denom in [2, 4, 8, 16]:
                    U_s = np.round(U * max_denom) / max_denom
                    V_s = np.round(V * max_denom) / max_denom
                    W_s = np.round(W * max_denom) / max_denom
                    if _check_decomposition(U_s, V_s, W_s, T):
                        return U_s.tolist(), V_s.tolist(), W_s.tolist()
                # Check if values are already close to rationals
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                if _check_decomposition(U_rounded, V_rounded, W_rounded, T):
                    return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
                return U.tolist(), V.tolist(), W.tolist()
    
    return None


def _try_direct_small_rank(R, T, deadline):
    """
    Direct search for very small ranks using integer/half-integer values.
    Only feasible for R <= 12 or so due to combinatorial explosion.
    """
    n, m, p = DIMENSIONS
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    if R > 12:
        return None
    
    # For very small R, try a limited search space
    # Values from {-1, -0.5, 0, 0.5, 1}
    val_set = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    # This is still too large for direct enumeration.
    # Instead, use a greedy/iterative approach:
    # Build the decomposition one column at a time.
    
    best_U = None
    best_V = None
    best_W = None
    best_err = np.inf
    
    np.random.seed(42)
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        U = np.zeros((U_rows, R))
        V = np.zeros((V_rows, R))
        W = np.zeros((W_rows, R))
        
        for r in range(R):
            if time.time() > deadline:
                break
            
            # Greedily choose column r to minimize residual
            best_col_err = np.inf
            best_u_col = None
            best_v_col = None
            best_w_col = None
            
            # Sample a few candidate columns
            for sample in range(50):
                u_col = np.random.choice(val_set, size=U_rows)
                v_col = np.random.choice(val_set, size=V_rows)
                w_col = np.random.choice(val_set, size=W_rows)
                
                # Compute residual after adding this column
                # Current partial sum: sum_{r'<r} U[:,r']*V[:,r']*W[:,r']
                partial = np.zeros_like(T, dtype=np.float64)
                for r2 in range(r):
                    partial += np.einsum('u,v,w->uvw', U[:, r2], V[:, r2], W[:, r2])
                
                new_sum = partial + np.einsum('u,v,w->uvw', u_col, v_col, w_col)
                err = np.linalg.norm(new_sum - T.astype(np.float64))
                
                if err < best_col_err:
                    best_col_err = err
                    best_u_col = u_col.copy()
                    best_v_col = v_col.copy()
                    best_w_col = w_col.copy()
            
            if best_u_col is not None:
                U[:, r] = best_u_col
                V[:, r] = best_v_col
                W[:, r] = best_w_col
        
        # Final check
        if _check_decomposition(U, V, W, T):
            return U.tolist(), V.tolist(), W.tolist()
        
        err = _compute_error(U, V, W, T, R)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    return None


def construct():
    deadline = time.time() + 145  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    T = _build_target()
    
    # Try ranks from 19 down to 10
    # Start with the most promising (closest to schoolbook)
    for R in range(19, 9, -1):
        if time.time() > deadline:
            break
        
        # For higher ranks, use ALS
        if R >= 14:
            n_seeds = 3 if R > 16 else 5
            max_iters = 60 if R > 16 else 80
            result = _try_als_rank(R, T, deadline, n_seeds=n_seeds, max_iters=max_iters)
            if result is not None:
                U_l, V_l, W_l = result
                return {"U": U_l, "V": W_l, "W": W_l}
        
        # For lower ranks, also try direct search
        else:
            result = _try_direct_small_rank(R, T, deadline)
            if result is not None:
                U_l, V_l, W_l = result
                return {"U": U_l, "V": V_l, "W": W_l}
            
            # Also try ALS for lower ranks
            n_seeds = 5
            max_iters = 100
            result = _try_als_rank(R, T, deadline, n_seeds=n_seeds, max_iters=max_iters)
            if result is not None:
                U_l, V_l, W_l = result
                return {"U": U_l, "V": V_l, "W": W_l}
    
    # Fall back to schoolbook (rank 20)
    U_l, V_l, W_l = _schoolbook_decomposition()
    return {"U": U_l, "V": V_l, "W": W_l}
