import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    rank = len(U[0])
    U_arr = np.array([[float(x) for x in row] for row in U], dtype=float)
    V_arr = np.array([[float(x) for x in row] for row in V], dtype=float)
    W_arr = np.array([[float(x) for x in row] for row in W], dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = np.dot(U_arr[idx_u, :], V_arr[idx_v, :] * W_arr[idx_w, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_rank28(n, m, p):
    """Build the known rank-28 decomposition using Strassen on sub-blocks."""
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: j=2, all k (10 terms)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: j=0,1, k=4 (4 terms)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: j=0,1, k=0,1 using Strassen (7 terms)
    # M1
    U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1; V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1; W[1 * n + 1][r] = 1
    r += 1
    # M2
    U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    W[0 * n + 1][r] = 1; W[1 * n + 1][r] = -1
    r += 1
    # M3
    U[0 * m + 0][r] = 1
    V[0 * p + 1][r] = 1; V[1 * p + 1][r] = -1
    W[1 * n + 0][r] = 1; W[1 * n + 1][r] = 1
    r += 1
    # M4
    U[1 * m + 1][r] = 1
    V[1 * p + 0][r] = 1; V[0 * p + 0][r] = -1
    W[0 * n + 0][r] = 1; W[0 * n + 1][r] = 1
    r += 1
    # M5
    U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = -1; W[1 * n + 0][r] = 1
    r += 1
    # M6
    U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1
    V[0 * p + 0][r] = 1; V[0 * p + 1][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    # M7
    U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1
    V[1 * p + 0][r] = 1; V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: j=0,1, k=2,3 using Strassen (7 terms)
    # M1
    U[0 * m + 0][r] = 1; U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1; V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1; W[3 * n + 1][r] = 1
    r += 1
    # M2
    U[1 * m + 0][r] = 1; U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    W[2 * n + 1][r] = 1; W[3 * n + 1][r] = -1
    r += 1
    # M3
    U[0 * m + 0][r] = 1
    V[0 * p + 3][r] = 1; V[1 * p + 3][r] = -1
    W[3 * n + 0][r] = 1; W[3 * n + 1][r] = 1
    r += 1
    # M4
    U[1 * m + 1][r] = 1
    V[1 * p + 2][r] = 1; V[0 * p + 2][r] = -1
    W[2 * n + 0][r] = 1; W[2 * n + 1][r] = 1
    r += 1
    # M5
    U[0 * m + 0][r] = 1; U[0 * m + 1][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = -1; W[3 * n + 0][r] = 1
    r += 1
    # M6
    U[1 * m + 0][r] = 1; U[0 * m + 0][r] = -1
    V[0 * p + 2][r] = 1; V[0 * p + 3][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    # M7
    U[0 * m + 1][r] = 1; U[1 * m + 1][r] = -1
    V[1 * p + 2][r] = 1; V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    return U, V, W

def _try_lower_rank(n, m, p, target_rank, time_limit=20.0):
    """
    Try to find a rank-target_rank decomposition using numerical optimization.
    Uses a structured initialization based on the rank-28 solution.
    """
    from scipy.optimize import minimize
    
    nm, mp, pn = n * m, m * p, p * n
    n_params = target_rank * (nm + mp + pn)
    
    start_time = time.time()
    
    def objective(params):
        U = params[:target_rank*nm].reshape(target_rank, nm).T
        V = params[target_rank*nm:target_rank*nm+target_rank*mp].reshape(target_rank, mp).T
        W = params[target_rank*nm+target_rank*mp:].reshape(target_rank, pn).T
        
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    residual += (s - 1.0)**2
        return residual
    
    best_obj = np.inf
    best_params = None
    
    # Try multiple initializations
    n_trials = min(15, int(time_limit / 1.5))
    for trial in range(n_trials):
        if time.time() - start_time > time_limit - 5:
            break
        
        np.random.seed(trial * 31 + 7)
        
        # Initialize with scaled random values
        scale = 0.3 + 0.1 * (trial % 3)
        x0 = np.random.randn(n_params) * scale
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x.copy()
                
                if result.fun < 1e-8:
                    break
                    
        except Exception:
            continue
    
    if best_params is None or best_obj > 1e-6:
        return None
    
    # Extract and round to half-integers
    U = best_params[:target_rank*nm].reshape(target_rank, nm).T
    V = best_params[target_rank*nm:target_rank*nm+target_rank*mp].reshape(target_rank, mp).T
    W = best_params[target_rank*nm+target_rank*mp:].reshape(target_rank, pn).T
    
    # Round to nearest half-integer
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    # Verify rounded version
    if _verify_decomposition(
        [[int(x*2) for x in row] for row in U_r],
        [[int(x*2) for x in row] for row in V_r],
        [[int(x*2) for x in row] for row in W_r],
        n, m, p, tol=1e-6
    ):
        # Convert to proper fractions (half-integers)
        U_frac = [[Fraction(int(x*2), 2) for x in row] for row in U_r]
        V_frac = [[Fraction(int(x*2), 2) for x in row] for row in V_r]
        W_frac = [[Fraction(int(x*2), 2) for x in row] for row in W_r]
        return U_frac, V_frac, W_frac
    
    # Try without rounding - check if exact with rational reconstruction
    # Use sympy for rational approximation if available
    try:
        import sympy
        U_sym = sympy.Matrix(U)
        V_sym = sympy.Matrix(V)
        W_sym = sympy.Matrix(W)
        
        # Check if entries are close to rationals with small denominators
        all_close = True
        for mat in [U_sym, V_sym, W_sym]:
            for val in mat:
                if abs(val) < 1e-10:
                    continue
                frac = sympy.nsimplify(float(val), rational=True, tolerance=1e-8)
                if frac.denom > 4:
                    all_close = False
                    break
            if not all_close:
                break
        
        if all_close:
            U_frac = [[sympy.nsimplify(float(x), rational=True, tolerance=1e-8) for x in row] for row in U.tolist()]
            V_frac = [[sympy.nsimplify(float(x), rational=True, tolerance=1e-8) for x in row] for row in V.tolist()]
            W_frac = [[sympy.nsimplify(float(x), rational=True, tolerance=1e-8) for x in row] for row in W.tolist()]
            
            # Convert sympy Rationals to Fractions
            U_frac = [[Fraction(x.p, x.q) if isinstance(x, sympy.Rational) else Fraction(int(x), 1) for x in row] for row in U_frac]
            V_frac = [[Fraction(x.p, x.q) if isinstance(x, sympy.Rational) else Fraction(int(x), 1) for x in row] for row in V_frac]
            W_frac = [[Fraction(x.p, x.q) if isinstance(x, sympy.Rational) else Fraction(int(x), 1) for x in row] for row in W_frac]
            
            if _verify_decomposition(U_frac, V_frac, W_frac, n, m, p):
                return U_frac, V_frac, W_frac
    except Exception:
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 125  # Leave 25s margin for safety
    
    # Build the known rank-28 solution
    U28, V28, W28 = _build_rank28(n, m, p)
    
    # Verify it's correct
    if not _verify_decomposition(U28, V28, W28, n, m, p):
        raise ValueError("Rank-28 decomposition verification failed")
    
    best_U, best_V, best_W = U28, V28, W28
    best_rank = 28
    
    # Try to find lower rank decompositions
    # Try ranks 27, 26, 25, ... down to a reasonable lower bound
    for test_rank in range(27, 15, -1):
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
        
        time_per_rank = min(25.0, (deadline - 30 - elapsed) / max(1, 28 - test_rank))
        if time_per_rank < 3.0:
            break
        
        result = _try_lower_rank(n, m, p, test_rank, time_limit=time_per_rank)
        
        if result is not None:
            U_new, V_new, W_new = result
            if _verify_decomposition(U_new, V_new, W_new, n, m, p):
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = test_rank
                # Don't break - try to go even lower
                continue
    
    return {"U": best_U, "V": best_V, "W": best_W}
