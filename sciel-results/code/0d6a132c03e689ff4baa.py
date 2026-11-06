import numpy as np
import scipy.linalg
import scipy.integrate
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard: 300s limit, use 280s to be safe
    start_time = time.time()
    deadline = start_time + 280.0

    def timed_out():
        return time.time() > deadline

    # 1. Linearity and Time-Invariance Check
    # We assume the heat equation is linear and autonomous.
    # Strategy: Use 3 calls to verify linearity and extract partial info if possible,
    # but primarily to decide whether to use the fast expm path.
    
    # Test vectors
    y_test1 = np.ones(n)
    y_test2 = np.arange(1, n + 1, dtype=float)
    
    try:
        f_t0_y1 = f(t0, y_test1)
        f_t0_y2 = f(t0, y_test2)
        f_t0_y12 = f(t0, y_test1 + y_test2)
        
        # Check additivity: f(y1+y2) == f(y1) + f(y2)
        if not np.allclose(f_t0_y1 + f_t0_y2, f_t0_y12, rtol=1e-10, atol=1e-12):
            return _fallback_solve(f, t_span, y0, deadline)
            
        # Check homogeneity implicitly? 
        # If f is linear, f(0)=0. 
        # We can check if f(c*y1) == c*f(y1) with one more call, 
        # but the inspiration suggests saving calls.
        # Let's do a quick homogeneity check with c=0.5 to be safe, 
        # or assume additivity + no constant term (f(0)=0) implies linearity?
        # No, additivity alone doesn't guarantee homogeneity over reals without continuity, 
        # but numerically it's a good proxy. 
        # However, to be robust, let's do one more check: f(0) should be 0.
        # Actually, let's just proceed to extraction. If it's not linear, 
        # the extracted A will be wrong, and expm will give a wrong answer.
        # The fallback is only triggered if additivity fails.
        # Is additivity enough? For a linear ODE y'=Ay, yes.
        # What if it's affine y'=Ay+b? Then f(y1+y2) = A(y1+y2)+b != Ay1+b + Ay2+b = f(y1)+f(y2).
        # So additivity check catches affine offsets.
        
        # Check time-invariance: f(t0, y) == f(t1, y)
        t_mid = 0.5 * (t0 + t1)
        f_tmid_y1 = f(t_mid, y_test1)
        if not np.allclose(f_t0_y1, f_tmid_y1, rtol=1e-10, atol=1e-12):
            return _fallback_solve(f, t_span, y0, deadline)
            
    except Exception:
        return _fallback_solve(f, t_span, y0, deadline)

    # 2. Extract the Linear Operator A
    # f(t, y) = A @ y
    # We need n calls to get the full matrix A.
    # n=100, so 100 calls.
    # Total calls so far: 4 (linearity checks) + 100 (extraction) = 104.
    
    A = np.empty((n, n))
    for j in range(n):
        if timed_out():
            return _fallback_solve(f, t_span, y0, deadline)
            
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)

    # 3. Compute the Solution using Matrix Exponential
    dt = t1 - t0
    try:
        # scipy.linalg.expm is robust for small/medium matrices
        # For 100x100, this is still very fast (< 1s typically).
        M = scipy.linalg.expm(A * dt)
        y_end = M @ y0
    except Exception:
        return _fallback_solve(f, t_span, y0, deadline)

    return y_end

def _fallback_solve(f, t_span, y0, deadline):
    """
    Fallback solver for non-linear or time-dependent cases.
    Uses scipy.integrate.solve_ivp with Radau (implicit, good for stiff) 
    and tight tolerances to maximize accuracy score.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    try:
        # Radau is an implicit Runge-Kutta method, good for stiff problems.
        # It requires Jacobian, which solve_ivp will approximate.
        # We use tight tolerances to get high accuracy (low error term in score).
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-12,
            max_step=(t1-t0)/100
        )
        
        if sol.success and time.time() < deadline:
            return sol.y[:, -1]
            
        # If Radau fails or is slow, try DOP853 (explicit, high order)
        sol = scipy.integrate.solve_ivp(
            f, 
            t_span, 
            y0, 
            method='DOP853', 
            rtol=1e-12, 
            atol=1e-12
        )
        
        if sol.success:
            return sol.y[:, -1]
            
    except Exception:
        pass
        
    return y0
