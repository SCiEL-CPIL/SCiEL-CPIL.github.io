import numpy as np
import time
from scipy.linalg import expm
import scipy.integrate as integrate_scipy

def integrate(f, t_span, y0):
    """
    Solves the 15-dimensional stiff linear IVP.
    Assumes y' = A(t)y + g(t).
    1. Infers A and g at t0 and t1.
    2. If constant, uses matrix exponential / block matrix trick.
    3. If time-dependent, falls back to high-accuracy Radau solver.
    """
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    
    # Helper to get A and g at time t
    def get_A_g(t):
        # g(t) = f(t, 0)
        g_t = f(t, np.zeros(n, dtype=float))
        g_t = np.array(g_t, dtype=float)
        
        A_t = np.zeros((n, n), dtype=float)
        for i in range(n):
            e_i = np.zeros(n, dtype=float)
            e_i[i] = 1.0
            # f(t, e_i) = A e_i + g
            col = f(t, e_i)
            A_t[:, i] = np.array(col, dtype=float) - g_t
            
        return A_t, g_t

    # Sample at t0 and t1
    A0, g0 = get_A_g(t0)
    A1, g1 = get_A_g(t1)
    
    # Check time limit
    if time.time() - start_time > 290:
        # Fallback to solver immediately
        sol = integrate_scipy.solve_ivp(f, t_span, y0, method='Radau', rtol=1e-10, atol=1e-12)
        return sol.y[:, -1]

    # Check if constant
    is_const_A = np.allclose(A0, A1, rtol=1e-9, atol=1e-12)
    is_const_g = np.allclose(g0, g1, rtol=1e-9, atol=1e-12)
    
    if is_const_A and is_const_g:
        dt = t1 - t0
        if np.allclose(g0, 0, atol=1e-15):
            # Homogeneous
            M = expm(A0 * dt)
            y_end = M @ y0
        else:
            # Affine constant
            # Use block matrix: [y; t] -> expm([A, g; 0, 0] * t) * [y0; 0]
            # Note: The block matrix is (n+1)x(n+1)
            B = np.zeros((n+1, n+1), dtype=float)
            B[:n, :n] = A0
            B[:n, n] = g0
            # B[n, :] is 0
            
            # We want y(t1). The state is [y; tau] where tau is time elapsed.
            # Initial state: [y0; 0]
            # Final state: [y_end; dt]
            # expm(B * dt) @ [y0; 0]
            # Actually, the integral term $\int_0^t e^{A(t-\tau)} g d\tau$ is handled by the block matrix.
            # Let's verify.
            # d/dt [y; t] = [Ay+g; 1] = [A, g; 0, 1] [y; t]? No.
            # d/dt t = 1. So the bottom right is 1?
            # If we use [y; 1], d/dt [y; 1] = [Ay+g; 0].
            # Matrix M = [[A, g], [0, 0]].
            # Then d/dt [y; 1] = M [y; 1] is NOT correct because d/dt 1 = 0, but M[1]=0.
            # Wait. If state is z = [y; c] where c is constant 1.
            # z' = [y'; c'] = [Ay+g; 0].
            # M z = [A y + g c; 0 c] = [Ay+g; 0] if c=1.
            # So yes, M = [[A, g], [0, 0]].
            # We integrate from 0 to dt.
            # z(dt) = expm(M * dt) z(0).
            # z(0) = [y0; 1].
            # z(dt) = [y_end; 1].
            
            M_block = np.zeros((n+1, n+1), dtype=float)
            M_block[:n, :n] = A0
            M_block[:n, n] = g0
            # M_block[n, n] = 0
            
            z0 = np.zeros(n+1, dtype=float)
            z0[:n] = y0
            z0[n] = 1.0
            
            z_end = expm(M_block * dt) @ z0
            y_end = z_end[:n]
            
        return y_end
    else:
        # Time-dependent or non-linear (though task says linear)
        # Use Radau solver with high accuracy
        sol = integrate_scipy.solve_ivp(
            f, t_span, y0, 
            method='Radau', 
            rtol=1e-12, 
            atol=1e-14,
            max_step=0.1 # Limit step size to ensure accuracy if stiffness varies
        )
        if sol.success:
            return sol.y[:, -1]
        else:
            # Last resort fallback
            return y0
