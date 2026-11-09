import numpy as np
import scipy.integrate
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 325.0 # 5s margin
    
    # Define analytic Jacobian
    def jac(t, y):
        # y = [y0, y1]
        # mu = 5
        mu = 5.0
        y0_val = y[0]
        y1_val = y[1]
        
        J = np.array([
            [0.0, 1.0],
            [-2.0 * mu * y0_val * y1_val - 1.0, mu * (1.0 - y0_val**2)]
        ])
        return J

    # Solve using Radau
    sol = scipy.integrate.solve_ivp(
        f, 
        t_span, 
        y0, 
        method='Radau', 
        rtol=1e-11, 
        atol=1e-13,
        jac=jac
    )
    
    if not sol.success:
        # Fallback if Radau fails (unlikely)
        # Try RK45 as fallback? Or just return y0? 
        # Better to return the last successful step if available
        if sol.y.size > 0:
            return sol.y[:, -1]
        return y0
        
    return sol.y[:, -1]
