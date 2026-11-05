"""ODE solver: Robertson problem using high-accuracy stiff methods."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] starting at y0.
    Uses high-order stiff-capable methods with tight tolerances to achieve
    high accuracy while minimizing function evaluations.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    
    from scipy.integrate import solve_ivp
    
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    # Strategy: Use Radau (implicit, stiff) with very tight tolerances.
    # Radau is a 5th-order implicit Runge-Kutta method that is L-stable,
    # making it excellent for stiff problems like Robertson.
    
    # Try Radau first with tight tolerances
    try:
        sol = solve_ivp(
            f,
            t_span,
            y,
            method='Radau',
            rtol=1e-12,
            atol=1e-14,
        )
        
        if sol.success:
            return sol.y[:, -1]
    except Exception:
        pass
    
    # If Radau fails or times out, try BDF
    if (time.time() - start_time) < deadline:
        try:
            sol = solve_ivp(
                f,
                t_span,
                y,
                method='BDF',
                rtol=1e-12,
                atol=1e-14,
            )
            
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
    
    # Fallback: Use LSODA which automatically switches between Adams and BDF
    if (time.time() - start_time) < deadline:
        try:
            sol = solve_ivp(
                f,
                t_span,
                y,
                method='LSODA',
                rtol=1e-12,
                atol=1e-14,
            )
            
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
    
    # Final fallback: custom adaptive RK45 with very tight tolerances
    # This is a simple but robust adaptive RK45 implementation
    h = (t1 - t0) / 100  # Initial step size
    t = t0
    y_curr = y.copy()
    
    def rk45_step(t, y, h):
        """Perform one RK45 step and return the 4th and 5th order solutions."""
        k1 = f(t, y)
        k2 = f(t + h/3, y + h/3 * k1)
        k3 = f(t + 2*h/3, y + h/3 * (2*k2 - k1))
        k4 = f(t + h, y + h * (-k1 + 4*k2 - k3))
        k5 = f(t + h, y + h * (k1 - 5*k2 + 4*k3))
        k6 = f(t + h, y + h * (-k1 + 5*k2 - 4*k3 + k4))
        
        y4 = y + h * (1/9*k1 + 12/13*k2 - 17/36*k3 + 1/9*k4 + 4/45*k5)
        y5 = y + h * (19/60*k1 + 3/4*k2 - 32/45*k3 + 3/10*k4 + 1/12*k5)
        
        return y4, y5, np.max(np.abs(y5 - y4))
    
    while t < t1 and (time.time() - start_time) < deadline:
        if t + h > t1:
            h = t1 - t
        
        y4, y5, err = rk45_step(t, y_curr, h)
        
        # Error control
        rtol = 1e-10
        atol = 1e-12
        scale = atol + rtol * np.max(np.abs(y_curr))
        
        if err <= scale:
            y_curr = y5  # Use higher order solution
            t += h
            # Adjust step size
            if err > 0:
                h *= min(5.0, max(0.1, 0.9 * (scale / err) ** 0.2))
            else:
                h *= 5.0
        else:
            # Reject step, reduce h
            if err > 0:
                h *= max(0.1, 0.9 * (scale / err) ** 0.25)
            else:
                h *= 0.1
        
        if h < 1e-14:
            break
    
    return y_curr
# EVOLVE-BLOCK-END
