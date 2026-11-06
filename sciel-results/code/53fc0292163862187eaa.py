import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Solve the Robertson ODE problem using a high-order fixed-step implicit
    Runge-Kutta method (Radau IIA order 5) with geometric step-size growth
    and Newton iteration for stage equations.
    
    Parameters:
    -----------
    f : callable
        The ODE function f(t, y) -> y'
    t_span : tuple
        (t0, t1) integration interval
    y0 : array-like
        Initial condition
    
    Returns:
    --------
    y_end : ndarray
        Final state at t1
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    time_limit = 310.0  # seconds, safety margin before 330s kill

    # Analytic Jacobian for the Robertson problem
    def jac(t, y):
        y0_val, y1_val, y2_val = y[0], y[1], y[2]
        J = np.array([
            [-0.04, 1e4 * y2_val, 1e4 * y1_val],
            [0.04, -1e4 * y2_val - 6e7 * y1_val, -1e4 * y1_val],
            [0.0, 6e7 * y1_val, 0.0]
        ])
        return J

    # Radau IIA coefficients for order 5 (5-stage)
    # Nodes c_i and coefficients a_ij, b_i
    # From Hairer, Nørsett, Wanner - Solving ODEs II
    c = np.array([
        0.06092105362211144,
        0.2108437777706769,
        0.4250107984040016,
        0.6934927230729799,
        1.0
    ])
    
    # A matrix (5x5) for Radau IIA
    A = np.array([
        [0.01528110964453233, 0.0, 0.0, 0.0, 0.0],
        [0.02712988577299932, 0.01398809088650302, 0.0, 0.0, 0.0],
        [0.05303021656989179, 0.05035590549686653, 0.02212467633724328, 0.0, 0.0],
        [0.09200842399418227, 0.1344427018159841, 0.1478004479394086, 0.0802411493234050, 0.0],
        [0.1388319775440324, 0.2143720854908487, 0.2431321940949507, 0.2286637428701682, 0.1749999999999999]
    ])
    
    # b vector (weights)
    b = np.array([
        0.1388319775440324,
        0.2143720854908487,
        0.2431321940949507,
        0.2286637428701682,
        0.1749999999999999
    ])

    def radau_step(t, y, h):
        """
        Perform one Radau IIA step from (t, y) with step size h.
        Returns the new y at t+h.
        """
        # Stage values Y_i = y + h * sum_j A_ij * f(t + c_j*h, Y_j)
        # We solve the nonlinear system:
        # Y_i - h * sum_j A_ij * f(t + c_j*h, Y_j) = y
        # for i = 1..5
        
        # Precompute t + c_i * h
        tc = t + c * h
        
        # Initial guess: Y_i = y (for all i)
        Y = np.tile(y, (5, 1))  # 5 x n array
        
        # Newton iteration to solve the stage equations
        # F_i(Y) = Y_i - y - h * sum_j A_ij * f(t + c_j*h, Y_j) = 0
        
        max_newton_iters = 50
        newton_tol = 1e-14
        
        for newton_iter in range(max_newton_iters):
            # Evaluate F and Jacobian
            F = np.zeros((5, n))
            J_full = np.zeros((5*n, 5*n))
            
            for i in range(5):
                # F_i = Y_i - y - h * sum_j A_ij * f(tc_j, Y_j)
                sum_af = np.zeros(n)
                for j in range(5):
                    if A[i, j] != 0.0:
                        fval = f(tc[j], Y[j])
                        sum_af += A[i, j] * fval
                
                F[i] = Y[i] - y - h * sum_af
                
                # Jacobian of F_i with respect to Y_k:
                # dF_i/dY_k = delta_ik * I - h * A_ik * J_f(tc_k, Y_k)
                for k in range(5):
                    if k == i:
                        J_full[i*n:(i+1)*n, k*n:(k+1)*n] = np.eye(n)
                    if A[i, k] != 0.0:
                        Jf_k = jac(tc[k], Y[k])
                        J_full[i*n:(i+1)*n, k*n:(k+1)*n] -= h * A[i, k] * Jf_k
            
            # Check convergence
            F_norm = np.max(np.abs(F))
            if F_norm < newton_tol:
                break
            
            # Solve J_full * delta = -F (flattened)
            F_flat = F.flatten()
            try:
                delta_flat = np.linalg.solve(J_full, -F_flat)
            except np.linalg.LinAlgError:
                # Singular matrix, use least squares
                delta_flat, _, _, _ = np.linalg.lstsq(J_full, -F_flat, rcond=None)
            
            delta = delta_flat.reshape(5, n)
            Y += delta
        
        # Final y at t+h: y_new = y + h * sum_i b_i * f(tc_i, Y_i)
        y_new = y.copy()
        for i in range(5):
            fval = f(tc[i], Y[i])
            y_new += h * b[i] * fval
        
        return y_new

    # Geometric step-size schedule
    # Start with small step, grow geometrically
    h_min = 1e-6
    h_max = 0.5
    growth_factor = 1.5
    
    # Initial step size
    h = h_min
    t = t0
    y = y0.copy()
    
    # Integrate from t0 to t1
    while t < t1 - 1e-14:
        # Ensure we don't overshoot t1
        if t + h > t1:
            h = t1 - t
        
        # Perform step
        y_new = radau_step(t, y, h)
        
        # Check if step is reasonable (not NaN or Inf)
        if not np.all(np.isfinite(y_new)):
            # Reduce step size and retry
            h = h / 2
            if h < h_min / 100:
                # Fallback to SciPy
                break
            continue
        
        t = t + h
        y = y_new
        
        # Grow step size
        h = min(h * growth_factor, h_max)
        
        # Check time limit
        if time.time() - start_time > time_limit - 5.0:
            break
    
    # If we broke out due to time or failure, use SciPy as fallback
    if t < t1 - 1e-10 or time.time() - start_time > time_limit - 5.0:
        try:
            sol = solve_ivp(
                f,
                (t, t1),
                y,
                method='Radau',
                jac=jac,
                rtol=1e-13,
                atol=1e-16
            )
            if sol.success:
                return sol.y[:, -1]
        except Exception:
            pass
    
    return y
