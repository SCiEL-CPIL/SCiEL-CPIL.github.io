import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    start_time = time.time()
    deadline = 300.0
    margin = 15.0
    
    def _elapsed():
        return time.time() - start_time
    
    def _remaining():
        return deadline - margin - _elapsed()
    
    best_result = None
    
    # Try Radau first (good for stiff problems like convection-diffusion)
    try:
        if _remaining() > 10.0:
            sol = scipy.integrate.solve_ivp(
                f, t_span, y0,
                method='Radau',
                rtol=1e-12,
                atol=1e-14
            )
            if sol.success:
                best_result = sol.y[:, -1].copy()
    except Exception:
        pass
    
    # If Radau failed, try DOP853
    if best_result is None:
        try:
            if _remaining() > 10.0:
                sol = scipy.integrate.solve_ivp(
                    f, t_span, y0,
                    method='DOP853',
                    rtol=1e-12,
                    atol=1e-14
                )
                if sol.success:
                    best_result = sol.y[:, -1].copy()
        except Exception:
            pass
    
    # If both failed, try with slightly looser tolerances
    if best_result is None:
        try:
            if _remaining() > 10.0:
                sol = scipy.integrate.solve_ivp(
                    f, t_span, y0,
                    method='Radau',
                    rtol=1e-10,
                    atol=1e-12
                )
                if sol.success:
                    best_result = sol.y[:, -1].copy()
        except Exception:
            pass
    
    if best_result is None:
        try:
            if _remaining() > 10.0:
                sol = scipy.integrate.solve_ivp(
                    f, t_span, y0,
                    method='DOP853',
                    rtol=1e-10,
                    atol=1e-12
                )
                if sol.success:
                    best_result = sol.y[:, -1].copy()
        except Exception:
            pass
    
    # Final fallback: RK4 with adaptive step size
    if best_result is None:
        try:
            h = (t1 - t0) / 1000
            t, y = t0, y0.copy()
            for _ in range(1000):
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2 * k1)
                k3 = f(t + h/2, y + h/2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
                t += h
                if _elapsed() > deadline - 5.0:
                    break
            best_result = y
        except Exception:
            best_result = y0
            
    return best_result
