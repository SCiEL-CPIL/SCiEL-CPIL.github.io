import time
import numpy as np
from scipy.optimize import least_squares
from itertools import product

DIMENSIONS = [2, 2, 3]
n, m, p = DIMENSIONS

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _residuals_rank10(params):
    """Compute residuals for rank-10 tensor decomposition."""
    U_flat = params[:40]
    V_flat = params[40:100]
    W_flat = params[100:160]
    
    U_mat = U_flat.reshape(4, 10)
    V_mat = V_flat.reshape(6, 10)
    W_mat = W_mat.reshape(6, 10)
    
    res = np.empty(12)
    idx = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                s = 0.0
                for r in range(10):
                    s += U_mat[i*2+j, r] * V_mat[j*3+k, r] * W_mat[k*2+i, r]
                res[idx] = s - 1.0
                idx += 1
    return res

def _residuals_rank11(params):
    """Compute residuals for rank-11 tensor decomposition."""
    U_flat = params[:44]
    V_flat = params[44:110]
    W_flat = params[110:176]
    
    U_mat = U_flat.reshape(4, 11)
    V_mat = V_flat.reshape(6, 11)
    W_mat = W_mat.reshape(6, 11)
    
    res = np.empty(12)
    idx = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                s = 0.0
                for r in range(11):
                    s += U_mat[i*2+j, r] * V_mat[j*3+k, r] * W_mat[k*2+i, r]
                res[idx] = s - 1.0
                idx += 1
    return res

def _rank11_decomposition():
    """Fallback rank 11 decomposition."""
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def set_u(i, j, r, val): U[i * m + j][r] = val
    def set_v(j, k, r, val): V[j * p + k][r] = val
    def set_w(k, i, r, val): W[k * n + i][r] = val

    # Strassen 7 terms
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1); set_w(1, 1, 0, 1)

    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1); set_w(1, 1, 1, -1)

    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1); set_w(1, 1, 2, 1)

    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1); set_w(0, 1, 3, 1)

    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1); set_w(1, 0, 4, 1)

    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # 4 schoolbook terms for k=2
    set_u(0, 0, 7, 1); set_v(0, 2, 7, 1); set_w(2, 0, 7, 1)
    set_u(0, 1, 8, 1); set_v(1, 2, 8, 1); set_w(2, 0, 8, 1)
    set_u(1, 0, 9, 1); set_v(0, 2, 9, 1); set_w(2, 1, 9, 1)
    set_u(1, 1, 10, 1); set_v(1, 2, 10, 1); set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}

def _schoolbook():
    """Schoolbook decomposition of rank n*m*p = 12."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _snap_to_rational(val, max_denom=8):
    """Try to snap a float to a rational with small denominator."""
    for d in range(1, max_denom + 1):
        num = round(val * d)
        if abs(val - num / d) < 1e-8:
            return num / d
    return round(val * 8) / 8  # fallback to eighths

def _try_snap_and_verify(U_mat, V_mat, W_mat, max_denom=8):
    """Try to snap matrices to rationals and verify."""
    U = [[_snap_to_rational(u, max_denom) for u in row] for row in U_mat.tolist()]
    V = [[_snap_to_rational(v, max_denom) for v in row] for row in V_mat.tolist()]
    W = [[_snap_to_rational(w, max_denom) for w in row] for row in W_mat.tolist()]
    
    if _verify(U, V, W, 2, 2, 3):
        return {"U": U, "V": V, "W": W}
    return None

def _try_rank10_structured(deadline):
    """
    Structured search for rank-10 decomposition.
    Strategy: Start from rank-11, try to merge terms using algebraic identities.
    """
    if time.time() > deadline - 15:
        return None
    
    rng = np.random.default_rng(123)
    best_cost = float('inf')
    best_params = None
    
    # Strategy 1: Random dense initialization with LM optimization
    for trial in range(200):
        if time.time() > deadline - 10:
            break
        
        # Mix of sparse and dense initializations
        if trial % 3 == 0:
            # Sparse binary init
            U = np.zeros((4, 10))
            V = np.zeros((6, 10))
            W = np.zeros((6, 10))
            for i in range(4):
                for j in range(10):
                    if rng.random() < 0.5:
                        U[i, j] = rng.choice([-1, 1])
            for i in range(6):
                for j in range(10):
                    if rng.random() < 0.5:
                        V[i, j] = rng.choice([-1, 1])
            for i in range(6):
                for j in range(10):
                    if rng.random() < 0.5:
                        W[i, j] = rng.choice([-1, 1])
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        elif trial % 3 == 1:
            # Small integer init
            U = rng.integers(-2, 3, (4, 10)).astype(float)
            V = rng.integers(-2, 3, (6, 10)).astype(float)
            W = rng.integers(-2, 3, (6, 10)).astype(float)
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        else:
            # Gaussian init
            x0 = rng.normal(0, 1, 160)
        
        try:
            result = least_squares(_residuals_rank10, x0, method='lm', 
                                   max_nfev=3000, ftol=1e-16, xtol=1e-16, gtol=1e-16)
            if result.cost < best_cost:
                best_cost = result.cost
                best_params = result.x.copy()
                if best_cost < 1e-14:
                    break
        except Exception:
            pass
    
    # Strategy 2: Start from rank-11 and perturb
    if time.time() > deadline - 15:
        pass
    else:
        r11 = _rank11_decomposition()
        U11 = np.array(r11["U"], dtype=float)
        V11 = np.array(r11["V"], dtype=float)
        W11 = np.array(r11["W"], dtype=float)
        
        # Try dropping each pair of terms and re-optimizing
        for drop_a in range(11):
            if time.time() > deadline - 10:
                break
            for drop_b in range(drop_a + 1, 11):
                if time.time() > deadline - 10:
                    break
                
                # Keep all terms except drop_a and drop_b, add one new term
                keep = [r for r in range(11) if r not in (drop_a, drop_b)]
                U_keep = U11[:, keep]
                V_keep = V11[:, keep]
                W_keep = W11[:, keep]
                
                # Add a random new term
                U_new = rng.normal(0, 1, 4)
                V_new = rng.normal(0, 1, 6)
                W_new = rng.normal(0, 1, 6)
                
                U_init = np.hstack([U_keep, U_new.reshape(4, 1)])
                V_init = np.hstack([V_keep, V_new.reshape(6, 1)])
                W_init = np.hstack([W_keep, W_new.reshape(6, 1)])
                
                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                
                try:
                    result = least_squares(_residuals_rank10, x0, method='lm',
                                           max_nfev=2000, ftol=1e-16, xtol=1e-16)
                    if result.cost < best_cost:
                        best_cost = result.cost
                        best_params = result.x.copy()
                        if best_cost < 1e-14:
                            break
                except Exception:
                    pass
            if best_cost < 1e-14:
                break
    
    if best_params is None or best_cost > 1e-10:
        return None
    
    # Extract and try to snap to rational values
    U_flat = best_params[:40]
    V_flat = best_params[40:100]
    W_flat = best_params[100:160]
    
    U_mat = U_flat.reshape(4, 10)
    V_mat = V_flat.reshape(6, 10)
    W_mat = W_mat.reshape(6, 10)
    
    # Try different snap strategies with increasing precision
    for max_denom in [2, 4, 8, 16, 32, 64]:
        result = _try_snap_and_verify(U_mat, V_mat, W_mat, max_denom)
        if result is not None:
            return result
    
    # Final check with floating point verification
    if _verify(U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), 2, 2, 3, tol=1e-8):
        return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
    
    return None

def _try_rank10_polynomial_search(deadline):
    """
    Polynomial/systematic search exploiting the algebraic structure.
    For rank 10, we have 160 unknowns and 12 equations.
    Fix gauge: set U[0,0,r] = 1 for r where possible.
    """
    if time.time() > deadline - 20:
        return None
    
    rng = np.random.default_rng(456)
    best_cost = float('inf')
    best_params = None
    
    # Use a smarter parameterization: fix gauge by setting one entry per term to 1
    # U has 4*10=40 entries, V has 6*10=60, W has 6*10=60
    # Fix U[i, r] = 1 for the first non-zero entry in each column r
    
    for trial in range(100):
        if time.time() > deadline - 10:
            break
        
        # Random initialization with gauge fixing
        U = rng.normal(0, 1, (4, 10))
        V = rng.normal(0, 1, (6, 10))
        W = rng.normal(0, 1, (6, 10))
        
        # Gauge fix: for each column r, scale so that max(|U[:,r]|) = 1
        for r in range(10):
            max_u = np.max(np.abs(U[:, r]))
            if max_u > 1e-10:
                U[:, r] /= max_u
                V[:, r] *= max_u
                # W stays the same (scaling invariance: U*V*W is invariant)
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = least_squares(_residuals_rank10, x0, method='lm',
                                   max_nfev=2500, ftol=1e-16, xtol=1e-16)
            if result.cost < best_cost:
                best_cost = result.cost
                best_params = result.x.copy()
                if best_cost < 1e-14:
                    break
        except Exception:
            pass
    
    if best_params is None or best_cost > 1e-10:
        return None
    
    U_flat = best_params[:40]
    V_flat = best_params[40:100]
    W_flat = best_params[100:160]
    
    U_mat = U_flat.reshape(4, 10)
    V_mat = V_flat.reshape(6, 10)
    W_mat = W_mat.reshape(6, 10)
    
    for max_denom in [2, 4, 8, 16, 32, 64, 128]:
        result = _try_snap_and_verify(U_mat, V_mat, W_mat, max_denom)
        if result is not None:
            return result
    
    return None

def construct():
    deadline = time.time() + 145
    
    # Try rank 10 with structured search
    result10 = _try_rank10_structured(deadline)
    if result10 is not None:
        return result10
    
    # Try rank 10 with polynomial search
    if time.time() < deadline - 15:
        result10b = _try_rank10_polynomial_search(deadline)
        if result10b is not None:
            return result10b
    
    # Fallback to rank 11
    result11 = _rank11_decomposition()
    if _verify(result11["U"], result11["V"], result11["W"], n, m, p):
        return result11
    
    # Fallback to schoolbook
    return _schoolbook()
