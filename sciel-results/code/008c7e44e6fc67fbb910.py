"""ODE solver for 15-dimensional stiff linear system."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import scipy.linalg
import scipy.integrate

def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns the final state y(t_end).
    
    Strategy:
    1. Detect if the system is autonomous and linear (y' = A y).
    2. If yes, use the matrix exponential for an exact solution (up to numerical precision) with minimal f calls.
    3. If not, fall back to scipy.integrate.solve_ivp with Radau method, providing a Jacobian estimate to speed up convergence.
    4. Track time to ensure we finish within 300s.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Leave 10s safety margin
    
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # --- Step 1: Try to exploit linear structure ---
    # Estimate Jacobian at t0 and t1 to check for autonomy and linearity
    # We assume f(t, y) = A(t) y. If A(t0) == A(t1), we treat it as autonomous.
    
    def estimate_jacobian(t, n):
        J = np.zeros((n, n))
        for i in range(n):
            e_i = np.zeros(n)
            e_i[i] = 1.0
            # f(t, e_i) gives the i-th column of A(t) if the system is homogeneous linear
            # If there is an offset b, f(t, 0) != 0. 
            # Let's check for offset too.
            J[:, i] = f(t, e_i)
        return J

    # Check for offset (affine vs homogeneous)
    # f(t, 0) should be 0 for homogeneous linear.
    y_zero = np.zeros(n)
    f_t0_zero = f(t0, y_zero)
    f_t1_zero = f(t1, y_zero)
    
    # If offset is significant, we handle affine case or fall back.
    # For "linear system", it's usually homogeneous. If not, we can subtract offset if constant.
    # Let's assume homogeneous first.
    
    A_t0 = estimate_jacobian(t0, n)
    A_t1 = estimate_jacobian(t1, n)
    
    # Check if A is constant
    # Tolerance for "constant"
    tol = 1e-10 * (1 + np.max(np.abs(A_t0)))
    is_autonomous_linear = np.allclose(A_t0, A_t1, rtol=1e-8, atol=1e-10)
    
    # Also check if the system is actually linear by verifying superposition or just trusting the "linear" description.
    # The prompt says "linear system", so we trust it.
    
    if is_autonomous_linear and np.linalg.norm(f_t0_zero) < 1e-12:
        # Homogeneous autonomous linear system: y' = A y
        # Solution: y(t) = exp(A * (t - t0)) * y0
        # Here t1 - t0 is the interval length.
        dt = t1 - t0
        try:
            # Use scipy.linalg.expm for high accuracy
            expm_A_dt = scipy.linalg.expm(A_t0 * dt)
            y_end = expm_A_dt @ y0
            return y_end
        except Exception:
            pass # Fall through to general solver

    # --- Step 2: Fallback to general stiff solver ---
    # If not autonomous linear, or if expm failed, use solve_ivp.
    # We can still provide a Jacobian to help Radau/BDF.
    
    def jac(t, y):
        # Return the estimated Jacobian at t0, or re-estimate if t is far from t0
        # For simplicity and speed, we might just return A_t0 if it's close enough,
        # or estimate on the fly if we have time.
        # Given the strict call limit, re-estimating is expensive (n calls).
        # If the system is non-autonomous, the Jacobian changes.
        # Let's just return A_t0 as an approximation, or None to let scipy estimate.
        # Actually, if it's not autonomous, providing a wrong Jacobian might slow it down.
        # But if it's stiff, scipy will struggle without a good Jacobian.
        # Let's try providing A_t0. If the system is slowly varying, this is okay.
        # If it's rapidly varying, we are in trouble, but "fixed problem" suggests it's well-behaved.
        return A_t0

    try:
        from scipy.integrate import solve_ivp
        
        # Use Radau, which is good for stiff problems.
        # Provide Jacobian to reduce Newton iterations.
        sol = solve_ivp(
            f, [t0, t1], y0,
            method='Radau',
            rtol=1e-12, atol=1e-14,
            jac=jac,
            max_step=0.05
        )
        
        if sol.success:
            return sol.y[:, -1]
            
        # If Radau failed, try BDF
        sol = solve_ivp(
            f, [t0, t1], y0,
            method='BDF',
            rtol=1e-12, atol=1e-14,
            jac=jac,
            max_step=0.05
        )
        
        if sol.success:
            return sol.y[:, -1]
            
    except Exception:
        pass
    
    # --- Step 3: Last resort fallback ---
    # Simple RK4 with many steps (slow, but robust)
    # This will likely score poorly due to high call count, but better than crashing.
    steps = 1000
    h = (t1 - t0) / steps
    t, y = t0, y0.copy()
    for _ in range(steps):
        if time.time() > deadline:
            break
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
    return y

# EVOLVE-BLOCK-END
