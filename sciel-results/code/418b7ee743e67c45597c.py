"""
Rank-minimal exact tensor decomposition for 2x2 * 2x5 matrix multiplication.

Combined approach:
1. Structural Rank 18 solution (Strassen + Schoolbook) as baseline.
2. Search for lower ranks (17, 16) using ALS with structured initializations.
3. Randomized algebraic elimination over rationals with sparsity constraints.
4. Verify exactness using Fractions.
"""

import numpy as np
import time
import itertools
from fractions import Fraction
from collections import defaultdict

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


def _build_target_float():
    return _build_target().astype(np.float64)


def _check_decomposition_float(U, V, W, T):
    """Check if sum_r U_u,r * V_v,r * W_w,r == T_u,v,w for all u,v,w."""
    R = U.shape[1]
    result = np.zeros((T.shape[0], T.shape[1], T.shape[2]), dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    return np.allclose(result, T, atol=1e-9)


def _compute_tensor(U, V, W):
    """Compute the tensor from factor matrices."""
    R = U.shape[1]
    n, m, p = DIMENSIONS
    result = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for r in range(R):
        result += np.einsum('u,v,w->uvw', U[:, r], V[:, r], W[:, r])
    return result


def _snap_to_half_integers(M):
    """Snap matrix values to nearest half-integer."""
    return np.round(M * 2) / 2.0


def _snap_to_integers(M):
    """Snap matrix values to nearest integer."""
    return np.round(M)

def _to_fraction(val):
    """Convert a float to Fraction, preferring small denominators."""
    if abs(val) < 1e-12:
        return Fraction(0)
    # Try half-integer first
    nearest_half = np.round(val * 2) / 2.0
    if abs(val - nearest_half) < 1e-9:
        return Fraction(int(nearest_half * 2), 2)
    # Try integer
    nearest_int = np.round(val)
    if abs(val - nearest_int) < 1e-9:
        return Fraction(int(nearest_int))
    # Try other small denominators
    for denom in [3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 100]:
        num = np.round(val * denom)
        if abs(val - num / denom) < 1e-9:
            frac = Fraction(int(num), denom)
            return frac
    # Last resort
    return Fraction(val).limit_denominator(200)


def _check_decomposition_exact(U, V, W, T):
    """Exact check using Fraction to avoid floating point issues."""
    n, m, p = DIMENSIONS
    U_frac = np.zeros_like(U, dtype=object)
    V_frac = np.zeros_like(V, dtype=object)
    W_frac = np.zeros_like(W, dtype=object)
    
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            U_frac[i, j] = _to_fraction(U[i, j])
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            V_frac[i, j] = _to_fraction(V[i, j])
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_frac[i, j] = _to_fraction(W[i, j])
    
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


def _get_rank18_solution():
    """
    Returns the structural Rank 18 solution:
    Two Strassen blocks (cols 0-1, 2-3) and Schoolbook for col 4.
    """
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


def _try_direct_rank16(deadline):
    """
    Direct algebraic construction for rank-16.
    Uses structured sparse patterns and solves linear systems.
    """
    n, m, p = DIMENSIONS
    R = 16
    U_rows = n * m  # 4
    V_rows = m * p  # 10
    W_rows = p * n  # 10
    
    T = _build_target()
    T_float = T.astype(np.float64)
    
    # Build the rank-18 solution as a starting point
    U18, V18, W18 = _get_rank18_solution()
    
    # Compute partial tensor from first 7 terms (Strassen for cols 0,1)
    T_partial = np.zeros((4, 10, 10), dtype=np.float64)
    for r in range(7):
        T_partial += np.einsum('u,v,w->uvw', U18[:, r], V18[:, r], W18[:, r])
    
    # Residual that terms 7-15 must satisfy
    residual = T_float - T_partial
    
    # Generate U patterns for terms 7-15
    U_patterns = []
    
    # Pattern 1: Identity-like
    U1 = np.zeros((4, 9))
    U1[0, 0] = 1; U1[0, 1] = 1; U1[0, 2] = 1
    U1[1, 3] = 1; U1[1, 4] = 1; U1[1, 5] = 1
    U1[2, 6] = 1; U1[2, 7] = 1; U1[2, 8] = 1
    U1[3, 0] = 1; U1[3, 3] = 1; U1[3, 6] = 1
    U_patterns.append(U1)
    
    # Pattern 2: Strassen-like combinations
    U2 = np.zeros((4, 9))
    U2[0, 0] = 1; U2[3, 0] = 1
    U2[2, 1] = 1; U2[3, 1] = 1
    U2[0, 2] = 1
    U2[3, 3] = 1
    U2[0, 4] = 1; U2[1, 4] = 1
    U2[2, 5] = 1; U2[0, 5] = -1
    U2[1, 6] = 1; U2[3, 6] = -1
    U2[0, 7] = 1
    U2[1, 8] = 1
    U_patterns.append(U2)
    
    # Pattern 3: All single rows
    U3 = np.zeros((4, 9))
    for r in range(9):
        row = r % 4
        U3[row, r] = 1
    U_patterns.append(U3)
    
    # Pattern 4: Mixed
    U4 = np.zeros((4, 9))
    U4[0, 0] = 1; U4[1, 0] = 1
    U4[2, 1] = 1; U4[3, 1] = 1
    U4[0, 2] = 1; U4[3, 2] = 1
    U4[1, 3] = 1; U4[2, 3] = -1
    U4[0, 4] = 1
    U4[1, 5] = 1
    U4[2, 6] = 1
    U4[3, 7] = 1
    U4[0, 8] = 1; U4[2, 8] = 1
    U_patterns.append(U4)
    
    # Pattern 5: Another Strassen variant
    U5 = np.zeros((4, 9))
    U5[0, 0] = 1; U5[1, 0] = 1  # a00+a01
    U5[2, 1] = 1; U5[3, 1] = 1  # a10+a11
    U5[0, 2] = 1                 # a00
    U5[3, 3] = 1                 # a11
    U5[1, 4] = 1; U5[3, 4] = -1 # a01-a11
    U5[2, 5] = 1; U5[0, 5] = -1 # a10-a00
    U5[0, 6] = 1; U5[3, 6] = 1  # a00+a11
    U5[1, 7] = 1                 # a01
    U5[2, 8] = 1                 # a10
    U_patterns.append(U5)
    
    # Pattern 6: Sparse with half-integers
    U6 = np.zeros((4, 9))
    U6[0, 0] = 1; U6[3, 0] = 1
    U6[1, 1] = 1; U6[2, 1] = 1
    U6[0, 2] = 0.5; U6[1, 2] = 0.5
    U6[2, 3] = 0.5; U6[3, 3] = 0.5
    U6[0, 4] = 1; U6[2, 4] = -1
    U6[1, 5] = 1; U6[3, 5] = -1
    U6[0, 6] = 1
    U6[1, 7] = 1
    U6[2, 8] = 1
    U_patterns.append(U6)
    
    for U_pat in U_patterns:
        if time.time() > deadline:
            break
        
        # Build full U
        U_full = np.zeros((4, 16))
        U_full[:, 0:7] = U18[:, 0:7]
        U_full[:, 7:16] = U_pat
        
        # Generate V patterns
        V_patterns = []
        
        # V pattern 1: Each term handles specific columns
        V1 = np.zeros((10, 9))
        V1[2, 0] = 1; V1[7, 0] = 1
        V1[2, 1] = 1; V1[7, 1] = 1
        V1[3, 2] = 1; V1[8, 2] = 1
        V1[3, 3] = 1; V1[8, 3] = 1
        V1[4, 4] = 1; V1[9, 4] = 1
        V1[4, 5] = 1; V1[9, 5] = 1
        V1[2, 6] = 1; V1[3, 6] = 1
        V1[3, 7] = 1; V1[4, 7] = 1
        V1[4, 8] = 1; V1[2, 8] = 1
        V_patterns.append(V1)
        
        # V pattern 2: Strassen-like
        V2 = np.zeros((10, 9))
        V2[2, 0] = 1; V2[7, 0] = 1
        V2[2, 1] = 1
        V2[7, 1] = -1
        V2[7, 2] = 1
        V2[3, 3] = 1; V2[8, 3] = 1
        V2[3, 4] = 1
        V2[8, 4] = -1
        V2[8, 5] = 1
        V2[4, 6] = 1; V2[9, 6] = 1
        V2[4, 7] = 1
        V2[9, 7] = -1
        V2[9, 8] = 1
        V_patterns.append(V2)
        
        # V pattern 3: Simple column assignment
        V3 = np.zeros((10, 9))
        V3[2, 0] = 1; V3[3, 0] = 1; V3[4, 0] = 1
        V3[7, 1] = 1; V3[8, 1] = 1; V3[9, 1] = 1
        V3[2, 2] = 1; V3[3, 2] = 1; V3[4, 2] = 1
        V3[7, 3] = 1; V3[8, 3] = 1; V3[9, 3] = 1
        V3[2, 4] = 1; V3[3, 4] = 1; V3[4, 4] = 1
        V3[7, 5] = 1; V3[8, 5] = 1; V3[9, 5] = 1
        V3[2, 6] = 1; V3[3, 6] = 1; V3[4, 6] = 1
        V3[7, 7] = 1; V3[8, 7] = 1; V3[9, 7] = 1
        V3[2, 8] = 1; V3[3, 8] = 1; V3[4, 8] = 1
        V_patterns.append(V3)
        
        # V pattern 4: Sparse
        V4 = np.zeros((10, 9))
        V4[2, 0] = 1
        V4[7, 1] = 1
        V4[3, 2] = 1
        V4[8, 3] = 1
        V4[4, 4] = 1
        V4[9, 5] = 1
        V4[2, 6] = 1; V4[7, 6] = 1
        V4[3, 7] = 1; V4[8, 7] = 1
        V4[4, 8] = 1; V4[9, 8] = 1
        V_patterns.append(V4)
        
        for V_pat in V_patterns:
            if time.time() > deadline:
                break
            
            # Solve for W[:, 7:16]
            A = np.zeros((400, 90))
            b = np.zeros(400)
            
            eq_idx = 0
            for u in range(4):
                for v in range(10):
                    for w in range(10):
                        for r in range(9):
                            col = w * 9 + r
                            A[eq_idx, col] = U_full[u, 7+r] * V_pat[v, r]
                        b[eq_idx] = residual[u, v, w]
                        eq_idx += 1
            
            try:
                W_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            W_sol = W_sol.reshape(10, 9)
            
            W_full = np.zeros((10, 16))
            W_full[:, 0:7] = W18[:, 0:7]
            W_full[:, 7:16] = W_sol
            
            V_full = np.zeros((10, 16))
            V_full[:, 0:7] = V18[:, 0:7]
            V_full[:, 7:16] = V_pat
            
            tensor = np.zeros((4, 10, 10))
            for r in range(16):
                tensor += np.einsum('u,v,w->uvw', U_full[:, r], V_full[:, r], W_full[:, r])
            
            if np.allclose(tensor, T_float, atol=1e-9):
                for scale in [1, 2, 4]:
                    U_s = np.round(U_full * scale) / scale
                    V_s = np.round(V_full * scale) / scale
                    W_s = np.round(W_full * scale) / scale
                    
                    if _check_decomposition_float(U_s, V_s, W_s, T_float):
                        valid = True
                        for u in range(4):
                            for v in range(10):
                                for w in range(10):
                                    total = Fraction(0)
                                    for r in range(16):
                                        total += _to_fraction(U_s[u, r]) * _to_fraction(V_s[v, r]) * _to_fraction(W_s[w, r])
                                    if total != Fraction(int(T[u, v, w])):
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                break
                        
                        if valid:
                            return U_s.tolist(), V_s.tolist(), W_s.tolist()
    
    return None


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

def _als_polish(U, V, W, T, R, max_iters=100, tol=1e-15):
    """High-precision ALS polish."""
    prev_err = np.inf
    for it in range(max_iters):
        U = _als_update_U(T, V, W, R)
        V = _als_update_V(T, U, W, R)
        W = _als_update_W(T, U, V, R)
        err = _compute_error(U, V, W, T)
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return U, V, W, err

def _try_als_rank(R, T, deadline, n_seeds=50, max_iters=50):
    """
    Try to find a rank-R decomposition using ALS with multiple initialization strategies.
    """
    n, m, p = DIMENSIONS
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    T_float = T.astype(np.float64)
    
    # Use Rank 18 solution as a structured initialization
    U18, V18, W18 = _get_rank18_solution()
    
    best_err = np.inf
    best_solution = None
    
    seeds = [42, 123, 7, 999, 2024, 555, 777, 1111, 2222, 3333, 
             1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 
             16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
             100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1111, 2222, 3333, 4444, 5555]
    
    for seed_idx in range(min(n_seeds, len(seeds))):
        if time.time() > deadline:
            break
        seed = seeds[seed_idx]
        np.random.seed(seed)
        
        # Multiple initialization strategies
        inits = []
        
        # Strategy 1: Pure random
        U = np.random.randn(U_rows, R) * 0.5
        V = np.random.randn(V_rows, R) * 0.5
        W = np.random.randn(W_rows, R) * 0.5
        inits.append((U.copy(), V.copy(), W.copy()))
        
        # Strategy 2: Random with different scales
        U = np.random.randn(U_rows, R)
        V = np.random.randn(V_rows, R)
        W = np.random.randn(W_rows, R)
        inits.append((U.copy(), V.copy(), W.copy()))
        
        # Strategy 3: Sparse random
        U = np.random.randn(U_rows, R) * (np.random.rand(U_rows, R) < 0.3)
        V = np.random.randn(V_rows, R) * (np.random.rand(V_rows, R) < 0.3)
        W = np.random.randn(W_rows, R) * (np.random.rand(W_rows, R) < 0.3)
        inits.append((U.copy(), V.copy(), W.copy()))
        
        # Strategy 4: For R=17, try structured init from rank-18
        if R == 17:
            # Take first 17 columns of rank-18 and add small perturbation
            U_s = U18[:, :17].copy() + np.random.randn(U_rows, 17) * 0.01
            V_s = V18[:, :17].copy() + np.random.randn(V_rows, 17) * 0.01
            W_s = W18[:, :17].copy() + np.random.randn(W_rows, 17) * 0.01
            inits.append((U_s, V_s, W_s))
            
            # Try dropping different columns from rank-18
            for drop_col in range(18):
                mask = [i for i in range(18) if i != drop_col]
                U_s = U18[:, mask].copy() + np.random.randn(U_rows, 17) * 0.01
                V_s = V18[:, mask].copy() + np.random.randn(V_rows, 17) * 0.01
                W_s = W18[:, mask].copy() + np.random.randn(W_rows, 17) * 0.01
                inits.append((U_s, V_s, W_s))
        
        for U_init, V_init, W_init in inits:
            if time.time() > deadline:
                break
            
            U, V, W = U_init.copy(), V_init.copy(), W_init.copy()
            prev_err = np.inf
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                U = _als_update_U(T_float, V, W, R)
                V = _als_update_V(T_float, U, W, R)
                W = _als_update_W(T_float, U, V, R)
                
                err = _compute_error(U, V, W, T_float)
                
                if abs(prev_err - err) < 1e-14:
                    break
                prev_err = err
            
            if err < best_err:
                best_err = err
                best_solution = (U.copy(), V.copy(), W.copy())
            
            # Check for exact solution
            if _check_decomposition_float(U, V, W, T_float):
                # Polish with high precision
                U, V, W, err = _als_polish(U, V, W, T_float, R, max_iters=50, tol=1e-16)
                
                # Try to rationalize
                for snap_fn in [_snap_to_half_integers, _snap_to_integers]:
                    U_s = snap_fn(U)
                    V_s = snap_fn(V)
                    W_s = snap_fn(W)
                    
                    if _check_decomposition_exact(U_s, V_s, W_s, T):
                        return U_s.tolist(), V_s.tolist(), W_s.tolist()
                
                # Check if original is exact
                if _check_decomposition_exact(U, V, W, T):
                    return U.tolist(), V.tolist(), W.tolist()
    
    # If we found a very low error solution, try one more polish
    if best_solution is not None and best_err < 1e-6:
        U, V, W = best_solution
        U, V, W, err = _als_polish(U, V, W, T_float, R, max_iters=200, tol=1e-16)
        if _check_decomposition_float(U, V, W, T_float):
            for snap_fn in [_snap_to_half_integers, _snap_to_integers]:
                U_s = snap_fn(U)
                V_s = snap_fn(V)
                W_s = snap_fn(W)
                if _check_decomposition_exact(U_s, V_s, W_s, T):
                    return U_s.tolist(), V_s.tolist(), W_s.tolist()
            if _check_decomposition_exact(U, V, W, T):
                return U.tolist(), V.tolist(), W.tolist()
    
    return None


def _try_algebraic_elimination_rank17(deadline, n_seeds=50):
    """
    Randomized algebraic elimination strategy for rank-17.
    Uses sparse patterns with small rational values.
    """
    """
    Randomized algebraic elimination strategy for rank-17.
    
    Treats the rank-17 decomposition as a sparse linear system.
    Uses Gaussian elimination over rationals to solve for factor entries
    while enforcing sparsity constraints that break block-diagonal symmetry.
    
    Key insight: For a rank-R decomposition of T with entries in {0,1},
    we have sum_r U[u,r]*V[v,r]*W[w,r] = T[u,v,w].
    
    We can fix some entries (sparsity pattern) and solve for others.
    The strategy:
    1. Choose a random sparse pattern for U, V, W with exactly R non-zero per row/col
       (or similar structured sparsity)
    2. Fix most entries to small integers/half-integers
    3. Solve the remaining linear system over rationals
    4. Verify the solution
    """
    n, m, p = DIMENSIONS
    R = 17
    U_rows = n * m  # 4
    V_rows = m * p  # 10
    W_rows = p * n  # 10
    
    T = _build_target()
    T_float = T.astype(np.float64)
    
    U18, V18, W18 = _get_rank18_solution()
    
    best_result = None
    
    # Strategy 1: Try sparse random patterns
    for seed in range(n_seeds):
        if time.time() > deadline:
            break
        
        np.random.seed(seed)
        
        U_vals = np.random.choice([-1, -0.5, 0, 0, 0, 0.5, 1], size=(U_rows, R))
        V_vals = np.random.choice([-1, -0.5, 0, 0, 0, 0.5, 1], size=(V_rows, R))
        
        for r in range(R):
            if np.all(U_vals[:, r] == 0):
                U_vals[np.random.randint(U_rows), r] = np.random.choice([-1, 1])
            if np.all(V_vals[:, r] == 0):
                V_vals[np.random.randint(V_rows), r] = np.random.choice([-1, 1])
        
        W_vals = np.random.choice([-1, -0.5, 0, 0, 0, 0.5, 1], size=(W_rows, R))
        for r in range(R):
            if np.all(W_vals[:, r] == 0):
                W_vals[np.random.randint(W_rows), r] = np.random.choice([-1, 1])
        
        if _check_decomposition_float(U_vals, V_vals, W_vals, T_float):
            valid = True
            for u in range(U_rows):
                for v in range(V_rows):
                    for w in range(W_rows):
                        total = Fraction(0)
                        for r in range(R):
                            total += _to_fraction(U_vals[u, r]) * _to_fraction(V_vals[v, r]) * _to_fraction(W_vals[w, r])
                        if total != Fraction(int(T[u, v, w])):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_result = (U_vals.tolist(), V_vals.tolist(), W_vals.tolist())
                break
    
    # Strategy 2: Structured approach - fix first 14 terms (2 Strassen blocks)
    # and solve for remaining 3 terms for col 4
    if best_result is None and time.time() < deadline:
        # First 14 terms are Strassen for cols 0-3
        T_partial = np.zeros((4, 10, 10), dtype=np.float64)
        for r in range(14):
            T_partial += np.einsum('u,v,w->uvw', U18[:, r], V18[:, r], W18[:, r])
        
        residual = T_float - T_partial
        
        # For the remaining 3 terms, we need to handle col 4
        # U[:, 14:17] is 4x3, V[:, 14:17] is 10x3, W[:, 14:17] is 10x3
        
        # Try several U patterns
        U_pats = [
            np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0]], dtype=np.float64),
            np.array([[1,0,0],[0,1,0],[0,0,1],[1,0,0]], dtype=np.float64),
            np.array([[1,0,0],[0,1,0],[0,0,1],[0,1,0]], dtype=np.float64),
            np.array([[1,1,0],[0,1,0],[0,0,1],[0,0,0]], dtype=np.float64),
        ]
        
        V_pats = [
            np.array([[0,0,0],[0,0,0],[0,0,0],[0,0,0],[1,0,0],[0,0,0],[0,0,0],[0,1,0],[0,0,0],[0,0,1]], dtype=np.float64),
            np.array([[0,0,0],[0,0,0],[0,0,0],[0,0,0],[1,1,0],[0,0,0],[0,0,0],[0,1,1],[0,0,0],[0,0,1]], dtype=np.float64),
            np.array([[0,0,0],[0,0,0],[0,0,0],[0,0,0],[1,0,0],[0,0,0],[0,0,0],[0,1,0],[0,0,0],[0,0,1]], dtype=np.float64),
        ]
        
        for U_pat in U_pats:
            if time.time() > deadline:
                break
            
            U_full = np.zeros((4, 17))
            U_full[:, 0:14] = U18[:, 0:14]
            U_full[:, 14:17] = U_pat
            
            for V_pat in V_pats:
                if time.time() > deadline:
                    break
                
                # Solve for W
                A = np.zeros((400, 30))
                b = np.zeros(400)
                
                eq_idx = 0
                for u in range(4):
                    for v in range(10):
                        for w in range(10):
                            for r in range(3):
                                col = w * 3 + r
                                A[eq_idx, col] = U_full[u, 14+r] * V_pat[v, r]
                            b[eq_idx] = residual[u, v, w]
                            eq_idx += 1
                
                try:
                    W_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                except:
                    continue
                
                W_sol = W_sol.reshape(10, 3)
                
                W_full = np.zeros((10, 17))
                W_full[:, 0:14] = W18[:, 0:14]
                W_full[:, 14:17] = W_sol
                
                V_full = np.zeros((10, 17))
                V_full[:, 0:14] = V18[:, 0:14]
                V_full[:, 14:17] = V_pat
                
                tensor = np.zeros((4, 10, 10))
                for r in range(17):
                    tensor += np.einsum('u,v,w->uvw', U_full[:, r], V_full[:, r], W_full[:, r])
                
                if np.allclose(tensor, T_float, atol=1e-9):
                    for scale in [1, 2, 4]:
                        U_s = np.round(U_full * scale) / scale
                        V_s = np.round(V_full * scale) / scale
                        W_s = np.round(W_full * scale) / scale
                        
                        if _check_decomposition_float(U_s, V_s, W_s, T_float):
                            valid = True
                            for u in range(4):
                                for v in range(10):
                                    for w in range(10):
                                        total = Fraction(0)
                                        for r in range(17):
                                            total += _to_fraction(U_s[u, r]) * _to_fraction(V_s[v, r]) * _to_fraction(W_s[w, r])
                                        if total != Fraction(int(T[u, v, w])):
                                            valid = False
                                            break
                                    if not valid:
                                        break
                                if not valid:
                                    break
                            
                            if valid:
                                best_result = (U_s.tolist(), V_s.tolist(), W_s.tolist())
                                break
                if best_result is not None:
                    break
            if best_result is not None:
                break
    
    return best_result


def _try_algebraic_elimination_rank16(deadline, n_seeds=20):
    """
    Try to find a rank-16 decomposition using algebraic elimination.
    Uses structured patterns with Strassen blocks.
    """
    n, m, p = DIMENSIONS
    R = 16
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    T = _build_target()
    T_float = T.astype(np.float64)
    
    U18, V18, W18 = _get_rank18_solution()
    
    best_result = None
    
    for seed in range(n_seeds):
        if time.time() > deadline:
            break
        
        np.random.seed(5000 + seed)
        
        # Start from rank-18, drop 2 terms, and try to recover
        drop1, drop2 = np.random.choice(18, 2, replace=False)
        mask = [i for i in range(18) if i not in [drop1, drop2]]
        
        U_init = U18[:, mask].copy() + np.random.randn(U_rows, R) * 0.1
        V_init = V18[:, mask].copy() + np.random.randn(V_rows, R) * 0.1
        W_init = W18[:, mask].copy() + np.random.randn(W_rows, R) * 0.1
        
        U_cur, V_cur, W_cur = U_init, V_init, W_init
        prev_err = np.inf
        
        for it in range(80):
            if time.time() > deadline:
                break
            
            U_cur = _als_update_U(T_float, V_cur, W_cur, R)
            V_cur = _als_update_V(T_float, U_cur, W_cur, R)
            W_cur = _als_update_W(T_float, U_cur, V_cur, R)
            
            err = _compute_error(U_cur, V_cur, W_cur, T_float)
            if abs(prev_err - err) < 1e-14:
                break
            prev_err = err
        
        if err < 1e-8:
            for scale in [1, 2, 4]:
                U_s = np.round(U_cur * scale) / scale
                V_s = np.round(V_cur * scale) / scale
                W_s = np.round(W_cur * scale) / scale
                
                if _check_decomposition_float(U_s, V_s, W_s, T_float):
                    valid = True
                    for u in range(U_rows):
                        for v in range(V_rows):
                            for w in range(W_rows):
                                total = Fraction(0)
                                for r in range(R):
                                    total += _to_fraction(U_s[u, r]) * _to_fraction(V_s[v, r]) * _to_fraction(W_s[w, r])
                                if total != Fraction(int(T[u, v, w])):
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        best_result = (U_s.tolist(), V_s.tolist(), W_s.tolist())
                        break
        if best_result is not None:
            break
    
    return best_result


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    T = _build_target()
    T_float = T.astype(np.float64)
    
    # Baseline: Rank 18
    U18, V18, W18 = _get_rank18_solution()
    best_result = {"U": U18.tolist(), "V": V18.tolist(), "W": W18.tolist()}
    best_rank = 18
    
    # Try direct rank-16 construction first (fast, structured) - give it more time
    direct_deadline_16 = min(deadline - 20, start_time + 80)
    result = _try_direct_rank16(direct_deadline_16)
    if result is not None:
        U_l, V_l, W_l = result
        best_result = {"U": U_l, "V": V_l, "W": W_l}
        best_rank = 16
    
    # Try algebraic elimination for rank 17
    if best_rank > 17 and time.time() < deadline - 15:
        search_deadline_17 = min(deadline - 15, start_time + 120)
        result = _try_algebraic_elimination_rank17(search_deadline_17, n_seeds=50)
        if result is not None:
            U_l, V_l, W_l = result
            best_result = {"U": U_l, "V": V_l, "W": W_l}
            best_rank = 17
    
    # If time permits, try Rank 16 with algebraic elimination
    if best_rank > 16 and time.time() < deadline - 5:
        result = _try_algebraic_elimination_rank16(deadline - 3, n_seeds=10)
        if result is not None:
            U_l, V_l, W_l = result
            best_result = {"U": U_l, "V": V_l, "W": W_l}
            best_rank = 16
    
    return best_result
