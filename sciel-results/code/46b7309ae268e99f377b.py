import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def _build_tensor():
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p), dtype=float)
    return T

def _verify_exact(U, V, W):
    n, m, p = DIMENSIONS
    R = len(U[0])
    for i in range(n):
        for j in range(m):
            for l in range(p):
                s = Fraction(0, 1)
                for c in range(R):
                    u_val = U[i*m+j][c]
                    v_val = V[j*p+l][c]
                    w_val = W[l*n+i][c]
                    # Convert to Fraction exactly
                    if isinstance(u_val, float):
                        u_frac = Fraction(u_val).limit_denominator(100000)
                    else:
                        u_frac = Fraction(u_val)
                    if isinstance(v_val, float):
                        v_frac = Fraction(v_val).limit_denominator(100000)
                    else:
                        v_frac = Fraction(v_val)
                    if isinstance(w_val, float):
                        w_frac = Fraction(w_val).limit_denominator(100000)
                    else:
                        w_frac = Fraction(w_val)
                    s += u_frac * v_frac * w_frac
                if s != 1:
                    return False
    return True

def _to_frac(x, max_den=10000):
    """Snap a float to the nearest simple fraction."""
    # Try integer
    r_int = round(x)
    if abs(x - r_int) < 1e-10:
        return int(r_int)
    # Try half-integer
    r_half = round(x * 2) / 2.0
    if abs(x - r_half) < 1e-10:
        if r_half == int(r_half):
            return int(r_half)
        return r_half
    # Try quarters
    r_q = round(x * 4) / 4.0
    if abs(x - r_q) < 1e-10:
        return r_q
    # Try eighths
    r_e = round(x * 8) / 8.0
    if abs(x - r_e) < 1e-10:
        return r_e
    # Try small denominators
    for den in [3, 5, 6, 7, 10, 12, 14, 15, 16, 20, 24]:
        f = round(x * den) / den
        if abs(x - f) < 1e-10:
            return f
    # Fallback: use Fraction with limit_denominator
    frac = Fraction(x).limit_denominator(max_den)
    return float(frac)

def _verify_float(U, V, W, tol=1e-10):
    n, m, p = DIMENSIONS
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for l in range(p):
                s = 0.0
                for c in range(R):
                    s += U[i*m+j, c] * V[j*p+l, c] * W[l*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _compute_error(U, V, W):
    n, m, p = DIMENSIONS
    R = U.shape[1]
    err = 0.0
    for i in range(n):
        for j in range(m):
            for l in range(p):
                s = 0.0
                for c in range(R):
                    s += U[i*m+j, c] * V[j*p+l, c] * W[l*n+i, c]
                err += (s - 1.0)**2
    return err

def _polish(U, V, W, max_iters=100, tol=1e-12):
    """Final high-precision polish pass."""
    n, m, p = DIMENSIONS
    R = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    prev_err = float('inf')
    for it in range(max_iters):
        # Update U
        for i in range(n):
            for j in range(m):
                idx = i*m + j
                M = np.zeros((p, R))
                b = np.ones(p)
                for l in range(p):
                    for c in range(R):
                        M[l, c] = V[j*p+l, c] * W[l*n+i, c]
                try:
                    sol = np.linalg.lstsq(M, b, rcond=None)[0]
                    U[idx, :] = sol
                except:
                    pass
        
        # Update V
        for j in range(m):
            for l in range(p):
                idx = j*p + l
                M = np.zeros((n, R))
                b = np.ones(n)
                for i in range(n):
                    for c in range(R):
                        M[i, c] = U[i*m+j, c] * W[l*n+i, c]
                try:
                    sol = np.linalg.lstsq(M, b, rcond=None)[0]
                    V[idx, :] = sol
                except:
                    pass
        
        # Update W
        for l in range(p):
            for i in range(n):
                idx = l*n + i
                M = np.zeros((m, R))
                b = np.ones(m)
                for j in range(m):
                    for c in range(R):
                        M[j, c] = U[i*m+j, c] * V[j*p+l, c]
                try:
                    sol = np.linalg.lstsq(M, b, rcond=None)[0]
                    W[idx, :] = sol
                except:
                    pass
        
        err = _compute_error(U, V, W)
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W

def _try_coordinate_descent(R, max_outer=30, max_inner=40, seed=0, init_type='random'):
    """Coordinate descent to find a rank-R decomposition."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for outer in range(max_outer):
        # Different initialization strategies
        if init_type == 'random':
            U = rng.randn(n*m, R)
            V = rng.randn(m*p, R)
            W = rng.randn(p*n, R)
        elif init_type == 'structured':
            # Initialize with some structure
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            for c in range(R):
                # Random pattern
                for i in range(n):
                    for j in range(m):
                        U[i*m+j, c] = rng.choice([-1, 0, 1])
                for j in range(m):
                    for l in range(p):
                        V[j*p+l, c] = rng.choice([-1, 0, 1])
                for l in range(p):
                    for i in range(n):
                        W[l*n+i, c] = rng.choice([-1, 0, 1])
        elif init_type == 'sparse':
            # Sparse initialization
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            for c in range(R):
                # Each column has a few non-zeros
                u_idx = rng.choice(n*m, size=max(1, R//3), replace=False)
                v_idx = rng.choice(m*p, size=max(1, R//3), replace=False)
                w_idx = rng.choice(p*n, size=max(1, R//3), replace=False)
                U[u_idx, c] = rng.choice([-1, 1])
                V[v_idx, c] = rng.choice([-1, 1])
                W[w_idx, c] = rng.choice([-1, 1])
        else:
            U = rng.randn(n*m, R)
            V = rng.randn(m*p, R)
            W = rng.randn(p*n, R)
        
        # Coordinate descent iterations
        prev_err = float('inf')
        for inner in range(max_inner):
            # Update U: for each (i,j), solve for U[i*m+j, :]
            for i in range(n):
                for j in range(m):
                    idx = i*m + j
                    # Build system: sum_c U[idx,c] * (V[j*p+l,c]*W[l*n+i,c]) = 1 for all l
                    M = np.zeros((p, R))
                    b = np.ones(p)
                    for l in range(p):
                        for c in range(R):
                            M[l, c] = V[j*p+l, c] * W[l*n+i, c]
                    try:
                        sol = np.linalg.lstsq(M, b, rcond=None)[0]
                        U[idx, :] = sol
                    except:
                        pass
            
            # Update V: for each (j,l), solve for V[j*p+l, :]
            for j in range(m):
                for l in range(p):
                    idx = j*p + l
                    # Build system: sum_c V[idx,c] * (U[i*m+j,c]*W[l*n+i,c]) = 1 for all i
                    M = np.zeros((n, R))
                    b = np.ones(n)
                    for i in range(n):
                        for c in range(R):
                            M[i, c] = U[i*m+j, c] * W[l*n+i, c]
                    try:
                        sol = np.linalg.lstsq(M, b, rcond=None)[0]
                        V[idx, :] = sol
                    except:
                        pass
            
            # Update W: for each (l,i), solve for W[l*n+i, :]
            for l in range(p):
                for i in range(n):
                    idx = l*n + i
                    # Build system: sum_c W[idx,c] * (U[i*m+j,c]*V[j*p+l,c]) = 1 for all j
                    M = np.zeros((m, R))
                    b = np.ones(m)
                    for j in range(m):
                        for c in range(R):
                            M[j, c] = U[i*m+j, c] * V[j*p+l, c]
                    try:
                        sol = np.linalg.lstsq(M, b, rcond=None)[0]
                        W[idx, :] = sol
                    except:
                        pass
            
            # Check convergence
            err = _compute_error(U, V, W)
            if abs(prev_err - err) < 1e-12:
                break
            prev_err = err
        
        # Evaluate error
        err = _compute_error(U, V, W)
        
        if err < best_err:
            best_err = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if best_err < 1e-10:
            break
    
    if best_U is None:
        return None
    
    if best_err < 1e-8:
        # Polish with higher precision
        U_p, V_p, W_p = _polish(best_U, best_V, best_W, max_iters=50, tol=1e-14)
        err_p = _compute_error(U_p, V_p, W_p)
        
        if err_p < best_err:
            best_U, best_V, best_W = U_p, V_p, W_p
            best_err = err_p
        
        # Snap to rationals with higher precision
        for max_den in [1000, 10000, 100000]:
            U_r = [[_to_frac(x, max_den) for x in row] for row in best_U]
            V_r = [[_to_frac(x, max_den) for x in row] for row in best_V]
            W_r = [[_to_frac(x, max_den) for x in row] for row in best_W]
            
            # Verify exactly
            if _verify_exact(U_r, V_r, W_r):
                return (U_r, V_r, W_r)
            
            # If exact verification fails, try with the float values converted to Fraction
            U_f = [[Fraction(x).limit_denominator(max_den) for x in row] for row in best_U]
            V_f = [[Fraction(x).limit_denominator(max_den) for x in row] for row in best_V]
            W_f = [[Fraction(x).limit_denominator(max_den) for x in row] for row in best_W]
            
            # Convert back to float for verification
            U_check = [[float(x) for x in row] for row in U_f]
            V_check = [[float(x) for x in row] for row in V_f]
            W_check = [[float(x) for x in row] for row in W_f]
            
            if _verify_float(np.array(U_check), np.array(V_check), np.array(W_check), tol=1e-10):
                return (U_r, V_r, W_r)
    
    return None

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 115  # Leave 5s margin before 120s limit
    
    init_types = ['random', 'structured', 'sparse']
    
    # Try ranks from 10 up to 17, skip 18 (schoolbook)
    for R in range(10, 18):
        if time.time() > deadline - 5:
            break
        
        # Multiple random seeds and initialization types
        for init_idx, init_type in enumerate(init_types):
            for seed in range(30):
                if time.time() > deadline - 2:
                    break
                
                # Adjust iterations based on time remaining
                time_left = deadline - time.time()
                if time_left < 5:
                    max_outer = 5
                    max_inner = 10
                elif time_left < 15:
                    max_outer = 10
                    max_inner = 20
                else:
                    max_outer = 15
                    max_inner = 30
                
                res = _try_coordinate_descent(R, max_outer=max_outer, max_inner=max_inner, 
                                              seed=seed, init_type=init_type)
                if res is not None:
                    U, V, W = res
                    return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
