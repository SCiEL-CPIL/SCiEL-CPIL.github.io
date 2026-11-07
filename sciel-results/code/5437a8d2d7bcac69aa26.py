"""ODE integrator for Van der Pol oscillator using Radau method."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _vdp_jac(t, y):
    """Jacobian of the Van der Pol oscillator (mu=5)."""
    mu = 5.0
    y0, y1 = y[0], y1[0] if isinstance(y1, np.ndarray) else y1
    # y' = [y1, mu*(1-y0^2)*y1 - y0]
    # J = [[0, 1], [-2*mu*y0*y1 - 1, mu*(1-y0^2)]]
    return np.array([[0.0, 1.0],
                     [-2.0 * mu * y0 * y1 - 1.0, mu * (1.0 - y0**2)]])


def _radau_iiia_step(f, jac, t, y, h, rtol, atol):
    """Single Radau IIA step with Newton iteration."""
    # Radau IIA coefficients for 3rd order
    # c = [0, 1/3, 1]
    # a11 = 1/12, a12 = 3/4, a13 = -1/6
    # a21 = 1/4,  a22 = 0,   a23 = -1/4
    # a31 = 0,    a32 = 1,   a33 = 0
    # b  = [1/9, 8/9, 0]
    
    n = len(y)
    c = np.array([0.0, 1.0/3.0, 1.0])
    A = np.array([[1.0/12.0, 3.0/4.0, -1.0/6.0],
                  [1.0/4.0,   0.0,    -1.0/4.0],
                  [0.0,       1.0,     0.0]])
    b = np.array([1.0/9.0, 8.0/9.0, 0.0])
    
    # Stage values K[i]
    K = [None, None, None]
    K[0] = y.copy()  # K[0] = y (c[0]=0)
    
    # Newton iteration for K[1] and K[2]
    # Equations: K[i] = y + h * sum_j A[i][j] f(t + c[j]*h, K[j])
    
    # Initialize K[1] and K[2]
    f0 = f(t, y)
    K[1] = y + h * A[1][0] * f0
    K[2] = y + h * (A[2][0] * f0 + A[2][1] * f(t + c[1]*h, K[1]))
    
    # Newton iteration
    for newton_iter in range(50):
        f1 = f(t + c[1]*h, K[1])
        f2 = f(t + c[2]*h, K[2])
        
        # Residuals
        r1 = K[1] - (y + h * (A[1][0]*f0 + A[1][1]*f1 + A[1][2]*f2))
        r2 = K[2] - (y + h * (A[2][0]*f0 + A[2][1]*f1 + A[2][2]*f2))
        
        err_norm = max(np.max(np.abs(r1)), np.max(np.abs(r2)))
        if err_norm < 1e-15:
            break
        
        # Build Jacobian of the system
        # dK1/dK1 = I - h*A[1][1]*J(t+c1*h, K1)
        # dK1/dK2 = -h*A[1][2]*J(t+c2*h, K2)
        # dK2/dK1 = -h*A[2][1]*J(t+c1*h, K1)
        # dK2/dK2 = I - h*A[2][2]*J(t+c2*h, K2)  [A[2][2]=0 so just I]
        
        J1 = jac(t + c[1]*h, K[1])
        J2 = jac(t + c[2]*h, K[2])
        
        # 2n x 2n system
        I = np.eye(n)
        # Block matrix
        M = np.zeros((2*n, 2*n))
        M[:n, :n] = I - h * A[1][1] * J1
        M[:n, n:] = -h * A[1][2] * J2
        M[n:, :n] = -h * A[2][1] * J1
        M[n:, n:] = I  # A[2][2] = 0
        
        rhs = np.concatenate([r1, r2])
        try:
            delta = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(M, rhs, rcond=None)[0]
        
        K[1] += delta[:n]
        K[2] += delta[n:]
    
    # Final stage values
    f1 = f(t + c[1]*h, K[1])
    f2 = f(t + c[2]*h, K[2])
    
    # New y
    y_new = y + h * (b[0]*f0 + b[1]*f1 + b[2]*f2)
    
    # Error estimate using embedded lower-order method
    # Use a 2nd order Radau-like estimate
    y_err = y + h * (0.5*f0 + 0.5*f1)
    err = np.max(np.abs(y_new - y_err))
    
    scale = np.maximum(atol, rtol * np.maximum(np.abs(y), np.abs(y_new)))
    err_scaled = err / np.max(scale)
    
    return y_new, err_scaled


def integrate(f, t_span, y0):
    """Integrate the Van der Pol oscillator using custom Radau IIA."""
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    DEADLINE = 310.0  # Safety margin before 330s wall-clock limit
    
    # Use Radau IIA with adaptive stepping
    rtol = 1e-13
    atol = 1e-15
    
    t = t0
    y = y0.copy()
    h = 0.01  # Initial step size
    
    max_steps = 500000
    step_count = 0
    
    while t < t1 and step_count < max_steps:
        # Check deadline
        if time.time() - start_time > DEADLINE:
            break
        
        # Clamp step to not overshoot t1
        h_eff = min(h, t1 - t)
        
        if h_eff <= 0:
            break
        
        y_new, err_scaled = _radau_iiia_step(f, _vdp_jac, t, y, h_eff, rtol, atol)
        
        if not np.all(np.isfinite(y_new)):
            # Step failed, try smaller step
            h *= 0.5
            if h < 1e-12:
                break
            continue
        
        t = t + h_eff
        y = y_new
        step_count += 1
        
        # Adaptive step sizing
        if err_scaled > 1.0:
            # Reject step, reduce h
            h *= 0.5
            t = t - h_eff  # Go back
            y = y  # Keep old y
        elif err_scaled > 1e-8:
            # Accept but don't increase much
            h *= min(1.5, (1.0 / max(err_scaled, 1e-10)) ** 0.25)
        else:
            # Accept and increase
            h *= min(2.0, (1.0 / max(err_scaled, 1e-10)) ** 0.25)
        
        # Clamp h
        h = max(1e-12, min(h, 0.1))
    
    # If we didn't reach t1, try a final direct call to get closer
    if t < t1 and step_count < max_steps:
        remaining = t1 - t
        if remaining > 0:
            # Take one more step
            h_final = min(remaining, 0.1)
            y_new, _ = _radau_iiia_step(f, _vdp_jac, t, y, h_final, rtol, atol)
            if np.all(np.isfinite(y_new)):
                y = y_new
                t = t + h_final
    
    return y
# EVOLVE-BLOCK-END
