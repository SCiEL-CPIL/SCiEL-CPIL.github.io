import numpy as np
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Use Radau (implicit, L-stable) for potentially stiff convection-diffusion
    # This is a different approach from the explicit DOP853
    try:
        sol = solve_ivp(
            f, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-12
        )
        
        if not sol.success:
            # Fallback to BDF
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='BDF', 
                rtol=1e-10, 
                atol=1e-12
            )
            
        if not sol.success:
            # Last resort: DOP853 (previous approach)
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='DOP853', 
                rtol=1e-10, 
                atol=1e-12
            )
            
        if not sol.success:
            # Fixed step RK4
            h = (t1 - t0) / 2000
            t, y = t0, y0
            for _ in range(2000):
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2*k1)
                k3 = f(t + h/2, y + h/2*k2)
                k4 = f(t + h, y + h*k3)
                y = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
                t += h
            return y
            
        return sol.y[:, -1]
    except Exception:
        # If implicit solvers crash (e.g., singular Jacobian), fall back to explicit
        try:
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='DOP853', 
                rtol=1e-10, 
                atol=1e-12
            )
            return sol.y[:, -1]
        except Exception:
            h = (t1 - t0) / 2000
            t, y = t0, y0
            for _ in range(2000):
                k1 = f(t, y)
                k2 = f(t + h/2, y + h/2*k1)
                k3 = f(t + h/2, y + h/2*k2)
                k4 = f(t + h, y + h*k3)
                y = y + h/6*(k1 + 2*k2 + 2*k3 + k4)
                t += h
            return y
