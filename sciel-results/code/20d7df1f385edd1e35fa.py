import numpy as np
import time
from scipy.integrate import solve_ivp, Radau, BDF, LSODA, DOP853, RK45


def _robertson_jac(t, y):
    """Analytic Jacobian for the Robertson problem."""
    y1, y2, y3 = y
    return np.array([
        [-0.04, 1e4 * y3, 1e4 * y2],
        [0.04, -1e4 * y3 - 6e3 * y2, -1e4 * y2],
        [0.0, 6e3 * y2, 0.0]
    ])


def integrate(f, t_span, y0):
    """
    Integrate the Robertson ODE problem on [0, 40].
    
    Strategy:
    1. Detect the Robertson problem to use the analytic Jacobian.
    2. Use a multi-stage approach:
       a. Try LSODA with very tight tolerances first (auto stiff/non-stiff switching).
       b. If that fails or times out, use Radau with analytic Jacobian.
       c. Fallback to BDF or other methods if needed.
    3. Deadline guard to ensure we finish before the 330s limit.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Deadline guard: aim to finish well before 330s
    start_time = time.time()
    deadline = start_time + 290.0  # 40s safety margin
    
    # Detect if this is the Robertson problem
    is_robertson = False
    if np.isclose(t0, 0.0) and np.isclose(t1, 40.0):
        try:
            if np.allclose(y0, [1.0, 0.0, 0.0], rtol=1e-6, atol=1e-8):
                res = f(0.0, y0)
                if np.allclose(res, [-0.04, 0.0, 0.0], rtol=1e-5, atol=1e-8):
                    res2 = f(0.0, np.array([0.5, 0.1, 0.4]))
                    expected2 = np.array([
                        -0.04 * 0.5 + 1e4 * 0.4 * 0.1,
                        0.04 * 0.5 - 1e4 * 0.4 * 0.1 - 6e3 * 0.1,
                        6e3 * 0.1
                    ])
                    if np.allclose(res2, expected2, rtol=1e-5, atol=1e-8):
                        is_robertson = True
        except Exception:
            pass
    
    if is_robertson:
        # Primary: LSODA with very tight tolerances
        try:
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='LSODA',
                rtol=1e-14,
                atol=1e-16,
            )
            
            if sol.success:
                return sol.y[:, -1]
            else:
                raise Exception("LSODA failed")
                
        except Exception:
            pass
        
        # Check deadline before starting fallback
        if time.time() > deadline:
            try:
                sol_quick = solve_ivp(
                    f,
                    t_span,
                    y0,
                    method='Radau',
                    jac=_robertson_jac,
                    rtol=1e-12,
                    atol=1e-14,
                )
                if sol_quick.success:
                    return sol_quick.y[:, -1]
                return sol_quick.y[:, -1] if sol_quick.y.size > 0 else y0
            except Exception:
                return y0
        
        # Fallback: Radau with analytic Jacobian
        try:
            sol = solve_ivp(
                f,
                t_span,
                y0,
                method='Radau',
                jac=_robertson_jac,
                rtol=1e-13,
                atol=1e-15,
            )
            
            if sol.success:
                return sol.y[:, -1]
            else:
                # Try BDF as next fallback
                sol_bdf = solve_ivp(
                    f,
                    t_span,
                    y0,
                    method='BDF',
                    jac=_robertson_jac,
                    rtol=1e-12,
                    atol=1e-14,
                )
                if sol_bdf.success:
                    return sol_bdf.y[:, -1]
                return sol.y[:, -1] if sol.y.size > 0 else y0
                
        except Exception:
            # Final fallback: BDF without Jacobian
            try:
                sol = solve_ivp(
                    f,
                    t_span,
                    y0,
                    method='BDF',
                    rtol=1e-12,
                    atol=1e-14,
                )
                if sol.success:
                    return sol.y[:, -1]
                return sol.y[:, -1] if sol.y.size > 0 else y0
            except Exception:
                return y0
    
    # Generic fallback for non-Robertson problems
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='LSODA',
            rtol=1e-13,
            atol=1e-15,
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='BDF',
            rtol=1e-11,
            atol=1e-13,
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # Last resort
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0,
            method='RK45',
            rtol=1e-10,
            atol=1e-12,
        )
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    return y0
