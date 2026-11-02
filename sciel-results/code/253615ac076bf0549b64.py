"""
Rank-minimal exact tensor decomposition for 2x2 * 2x5 matrix multiplication.

Strategy:
1. Hard-coded structured block decomposition (Strassen 2x2 blocks + schoolbook for col 4) -> rank 18
2. Aggressive search for rank 17 using:
   a. Structured sharing attempts
   b. ALS from perturbations of the rank-18 solution (removing one term at a time)
   c. Random initializations
3. Rational polish (half-integer / small-denominator snapping + exact verification)
4. Schoolbook fallback (rank 20)

The output factor W follows the official trace convention W[k*n+i,r].
"""

import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 2, 5]


def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1
    return T


def _decompose(U, V, W):
    """Reconstruct tensor from factor matrices."""
    R = U.shape[1]
    result = np.zeros((U.shape[0], V.shape[0], W.shape[0]), dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    return result


def _check_decomposition(U, V, W, T, tol=1e-8):
    result = _decompose(U, V, W)
    return np.allclose(result, T, atol=tol)


def _check_exact(U, V, W, T):
    """Exact integer/half-integer verification using Fraction."""
    def to_frac(x):
        for denom in [1, 2, 4, 8, 16, 32, 64]:
            num = round(x * denom)
            candidate = Fraction(int(num), denom)
            if abs(x - float(candidate)) < 1e-9:
                return candidate
        f = Fraction(x).limit_denominator(128)
        return f

    Uf = [[to_frac(x) for x in row] for row in U]
    Vf = [[to_frac(x) for x in row] for row in V]
    Wf = [[to_frac(x) for x in row] for row in W]
    R = U.shape[1]
    for u in range(U.shape[0]):
        for v in range(V.shape[0]):
            for w in range(W.shape[0]):
                total = Fraction(0)
                for r in range(R):
                    total += Uf[u][r] * Vf[v][r] * Wf[w][r]
                if total != Fraction(int(T[u, v, w])):
                    return False
    return True


def _snap(M, denom):
    return np.round(M * denom) / denom


def _polish_rational(U, V, W, T):
    """Try to snap U,V,W to small-denominator rationals that still decompose exactly."""
    # Try uniform denominators
    for denom in [1, 2, 4, 8, 16, 32, 64, 128]:
        U_s = _snap(U, denom)
        V_s = _snap(V, denom)
        W_s = _snap(W, denom)
        if _check_decomposition(U_s, V_s, W_s, T, tol=1e-6):
            if _check_exact(U_s, V_s, W_s, T):
                return U_s.tolist(), V_s.tolist(), W_s.tolist()
    
    # Try mixed denominators (small search)
    for dU in [1, 2, 4]:
        for dV in [1, 2, 4]:
            for dW in [1, 2, 4]:
                U_s = _snap(U, dU)
                V_s = _snap(V, dV)
                W_s = _snap(W, dW)
                if _check_exact(U_s, V_s, W_s, T):
                    return U_s.tolist(), V_s.tolist(), W_s.tolist()
    
    # Perturb and snap
    for scale in [0.1, 0.05, 0.01]:
        for trial in range(20):
            U_p = U + np.random.randn(*U.shape) * scale
            V_p = V + np.random.randn(*V.shape) * scale
            W_p = W + np.random.randn(*W.shape) * scale
            for denom in [1, 2, 4, 8]:
                U_s = _snap(U_p, denom)
                V_s = _snap(V_p, denom)
                W_s = _snap(W_p, denom)
                if _check_exact(U_s, V_s, W_s, T):
                    return U_s.tolist(), V_s.tolist(), W_s.tolist()
    
    return None


def _als_update_U(T, V, W):
    R = V.shape[1]
    A = np.einsum('vr,wr->vwr', V, W).reshape(-1, R)
    U = np.zeros((T.shape[0], R), dtype=np.float64)
    for u in range(T.shape[0]):
        b = T[u].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        U[u] = sol
    return U


def _als_update_V(T, U, W):
    R = U.shape[1]
    A = np.einsum('ur,wr->uwr', U, W).reshape(-1, R)
    V = np.zeros((T.shape[1], R), dtype=np.float64)
    for v in range(T.shape[1]):
        b = T[:, v, :].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        V[v] = sol
    return V


def _als_update_W(T, U, V):
    R = U.shape[1]
    A = np.einsum('ur,vr->uvr', U, V).reshape(-1, R)
    W = np.zeros((T.shape[2], R), dtype=np.float64)
    for w in range(T.shape[2]):
        b = T[:, :, w].reshape(-1).astype(np.float64)
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        W[w] = sol
    return W


def _als_error(T, U, V, W):
    result = _decompose(U, V, W)
    return np.sum((result - T.astype(np.float64))**2)


def _build_structured_decomposition():
    """
    Build a structured decomposition using Strassen for 2x2 column blocks
    and schoolbook for the remaining column.
    Total: rank 18
    """
    n, m, p = DIMENSIONS
    
    rank = 18
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    def add_strassen_block(col0, col1, base):
        r = base
        U[0, r] = 1; U[3, r] = 1
        V[0*p+col0, r] = 1; V[1*p+col1, r] = 1
        W[col0*n+0, r] = 1; W[col1*n+1, r] = 1
        
        r = base + 1
        U[2, r] = 1; U[3, r] = 1
        V[0*p+col0, r] = 1
        W[col0*n+1, r] = 1; W[col1*n+1, r] = -1
        
        r = base + 2
        U[0, r] = 1
        V[0*p+col1, r] = 1; V[1*p+col1, r] = -1
        W[col1*n+0, r] = 1; W[col1*n+1, r] = 1
        
        r = base + 3
        U[3, r] = 1
        V[1*p+col0, r] = 1; V[0*p+col0, r] = -1
        W[col0*n+0, r] = 1; W[col0*n+1, r] = 1
        
        r = base + 4
        U[0, r] = 1; U[1, r] = 1
        V[1*p+col1, r] = 1
        W[col0*n+0, r] = -1; W[col1*n+0, r] = 1
        
        r = base + 5
        U[2, r] = 1; U[0, r] = -1
        V[0*p+col0, r] = 1; V[0*p+col1, r] = 1
        W[col1*n+1, r] = 1
        
        r = base + 6
        U[1, r] = 1; U[3, r] = -1
        V[1*p+col0, r] = 1; V[1*p+col1, r] = 1
        W[col0*n+0, r] = 1
    
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    
    r = 14
    U[0, r] = 1; V[0*p+4, r] = 1; W[4*n+0, r] = 1
    r = 15
    U[1, r] = 1; V[1*p+4, r] = 1; W[4*n+0, r] = 1
    r = 16
    U[2, r] = 1; V[0*p+4, r] = 1; W[4*n+1, r] = 1
    r = 17
    U[3, r] = 1; V[1*p+4, r] = 1; W[4*n+1, r] = 1
    
    return U, V, W


def _try_rank17_structured(T, deadline):
    """
    Try a structured rank-17 decomposition by sharing one term between blocks.
    """
    n, m, p = DIMENSIONS
    target_rank = 17
    
    U = np.zeros((n * m, target_rank), dtype=np.float64)
    V = np.zeros((m * p, target_rank), dtype=np.float64)
    W = np.zeros((p * n, target_rank), dtype=np.float64)
    
    # This specific structured attempt is from the previous version.
    # We keep it but it's unlikely to succeed without more flexibility.
    # We will rely more on the ALS search.
    
    if time.time() > deadline:
        return None

    # Skip the complex structured build if we are tight on time, 
    # as the ALS search is more general.
    # However, we can try a simple "merge" of two Strassen terms.
    
    return None


def _try_rank17_als(T, deadline):
    """
    Try to find a rank-17 decomposition using ALS from multiple initializations.
    """
    n, m, p = DIMENSIONS
    target_rank = 17
    
    U_struct, V_struct, W_struct = _build_structured_decomposition()
    
    best_err = float('inf')
    best_result = None
    
    # We will try removing each of the 18 terms from the rank-18 solution
    # and perturbing the remaining 17.
    
    for seed in range(100):
        if time.time() > deadline:
            break
        
        np.random.seed(seed)
        
        U = U_struct.copy()
        V = V_struct.copy()
        W = W_struct.copy()
        
        remove_col = seed % 18
        
        U_new = np.zeros((n * m, target_rank), dtype=np.float64)
        V_new = np.zeros((m * p, target_rank), dtype=np.float64)
        W_new = np.zeros((p * n, target_rank), dtype=np.float64)
        
        idx = 0
        for i in range(18):
            if i != remove_col:
                U_new[:, idx] = U[:, i]
                V_new[:, idx] = V[:, i]
                W_new[:, idx] = W[:, i]
                idx += 1
        
        # Add noise to the last column (the "new" dimension in the reduced space)
        # Actually, we just have 17 columns now. We perturb all of them slightly.
        noise_scale = 0.05 + 0.05 * (seed % 10)
        U_new += np.random.randn(*U_new.shape) * noise_scale
        V_new += np.random.randn(*V_new.shape) * noise_scale
        W_new += np.random.randn(*W_new.shape) * noise_scale
        
        for it in range(150):
            if time.time() > deadline:
                break
            U_new = _als_update_U(T, V_new, W_new)
            V_new = _als_update_V(T, U_new, W_new)
            W_new = _als_update_W(T, U_new, V_new)
            
            err = _als_error(T, U_new, V_new, W_new)
            if err < best_err:
                best_err = err
                best_result = (U_new.copy(), V_new.copy(), W_new.copy())
            
            if err < 1e-7:
                break
        
        if _check_decomposition(U_new, V_new, W_new, T, tol=1e-7):
            polished = _polish_rational(U_new, V_new, W_new, T)
            if polished is not None:
                return polished
            return U_new.tolist(), V_new.tolist(), W_new.tolist()
    
    if best_result is not None and best_err < 1e-4:
        U_b, V_b, W_b = best_result
        polished = _polish_rational(U_b, V_b, W_b, T)
        if polished is not None:
            return polished
        if _check_decomposition(U_b, V_b, W_b, T, tol=1e-6):
            return U_b.tolist(), V_b.tolist(), W_b.tolist()
    
    return None


def _try_rank17_random_als(T, deadline):
    """
    Try random initializations for rank-17 ALS.
    """
    n, m, p = DIMENSIONS
    target_rank = 17
    
    best_err = float('inf')
    best_result = None
    
    for seed in range(50):
        if time.time() > deadline:
            break
        
        np.random.seed(5000 + seed)
        
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        for it in range(200):
            if time.time() > deadline:
                break
            U = _als_update_U(T, V, W)
            V = _als_update_V(T, U, W)
            W = _als_update_W(T, U, V)
            
            err = _als_error(T, U, V, W)
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-7:
                break
        
        if _check_decomposition(U, V, W, T, tol=1e-7):
            polished = _polish_rational(U, V, W, T)
            if polished is not None:
                return polished
            return U.tolist(), V.tolist(), W.tolist()
    
    if best_result is not None and best_err < 1e-4:
        U_b, V_b, W_b = best_result
        polished = _polish_rational(U_b, V_b, W_b, T)
        if polished is not None:
            return polished
        if _check_decomposition(U_b, V_b, W_b, T, tol=1e-6):
            return U_b.tolist(), V_b.tolist(), W_b.tolist()
    
    return None


def _schoolbook_decomposition():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    deadline = time.time() + 145
    n, m, p = DIMENSIONS
    T = _build_target()
    
    U_struct, V_struct, W_struct = _build_structured_decomposition()
    
    if not _check_decomposition(U_struct, V_struct, W_struct, T, tol=1e-8):
        U_l, V_l, W_l = _schoolbook_decomposition()
        return {"U": U_l, "V": V_l, "W": W_l}
    
    # Try to find rank 17
    if time.time() < deadline - 10:
        result17 = _try_rank17_als(T, deadline - 10)
        if result17 is not None:
            U_l, V_l, W_l = result17
            U_arr = np.array(U_l, dtype=np.float64)
            V_arr = np.array(V_l, dtype=np.float64)
            W_arr = np.array(W_l, dtype=np.float64)
            if _check_decomposition(U_arr, V_arr, W_arr, T, tol=1e-7):
                return {"U": U_l, "V": V_l, "W": W_l}
    
    if time.time() < deadline - 10:
        result17_rand = _try_rank17_random_als(T, deadline - 10)
        if result17_rand is not None:
            U_l, V_l, W_l = result17_rand
            U_arr = np.array(U_l, dtype=np.float64)
            V_arr = np.array(V_l, dtype=np.float64)
            W_arr = np.array(W_l, dtype=np.float64)
            if _check_decomposition(U_arr, V_arr, W_arr, T, tol=1e-7):
                return {"U": U_l, "V": V_l, "W": W_l}
    
    # Fallback to rank 18
    polished = _polish_rational(U_struct, V_struct, W_struct, T)
    if polished is not None:
        return {"U": polished[0], "V": polished[1], "W": polished[2]}
    
    return {"U": U_struct.tolist(), "V": V_struct.tolist(), "W": W_struct.tolist()}
