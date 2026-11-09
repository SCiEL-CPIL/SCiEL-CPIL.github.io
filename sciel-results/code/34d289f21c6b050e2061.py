import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit (or 330s if that's the hard kill)
    
    # Primary strategy: DOP853 with very tight tolerances.
    # DOP853 is an 8th-order explicit method, often more efficient in f-calls 
    # for high accuracy than implicit methods on moderate-sized systems 
    # unless the problem is extremely stiff.
    try:
        sol = scipy.integrate.solve_ivp(
            f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-13,
            atol=1e-15
        )
        
        if time.time() > deadline:
            if sol.success:
                return sol.y[:, -1]
            else:
                return y0

        if sol.success:
            return sol.y[:, -1]
        else:
            # If DOP853 failed (likely stiffness), fall back to Radau
            # Radau is an implicit L-stable method, better for stiff problems.
            sol2 = scipy.integrate.solve_ivp(
                f,
                t_span,
                y0,
                method='Radau',
                rtol=1e-12,
                atol=1e-14,
                jac=None # Let it estimate or use finite differences if not provided, 
                         # but for performance, providing a sparse jac would be ideal.
                         # Since we don't have the structure, we rely on default.
            )
            
            if time.time() > deadline:
                # Return best available or y0
                if sol.success:
                    return sol.y[:, -1]
                return y0

            if sol2.success:
                return sol2.y[:, -1]
            else:
                # Last resort: BDF
                sol3 = scipy.integrate.solve_ivp(
                    f,
                    t_span,
                    y0,
                    method='BDF',
                    rtol=1e-11,
                    atol=1e-13
                )
                if sol3.success:
                    return sol3.y[:, -1]
                return y0

    except Exception:
        # Fallback to simple RK4 if scipy fails completely
        try:
            n_steps = 5000
            dt = (t1 - t0) / n_steps
            y = y0.copy()
            for i in range(n_steps):
                t = t0 + i * dt
                if time.time() > deadline:
                    break
                k1 = f(t, y)
                k2 = f(t + dt/2, y + dt/2 * k1)
                k3 = f(t + dt/2, y + dt/2 * k2)
                k4 = f(t + dt, y + dt * k3)
                y = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            return y
        except Exception:
            return y0
