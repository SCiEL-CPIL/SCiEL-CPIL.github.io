"""High-order ODE integrator for the harmonic oscillator."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] starting at y0.
    Uses a high-order Runge-Kutta method (order 8) with a fixed step size
    chosen to balance accuracy and function call count.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    h_total = t1 - t0
    
    if h_total == 0:
        return y
    
    # Use a high-order RK method. We'll use a fixed step size.
    # For the harmonic oscillator, RK8 with ~10-20 steps should give very high accuracy.
    # Let's try 10 steps first, which means h = 1.0 for t in [0, 10].
    
    # Actually, let's use an adaptive approach: start with a reasonable number of steps
    # and check the result. But since we know the problem is smooth and non-stiff,
    # a fixed high-order method should work well.
    
    # Let's use RK8 (order 8) with 10 steps. That's 10 * 8 = 80 function calls.
    # Actually, RK8 uses 8 stages, so 10 steps * 8 = 80 calls.
    
    # But we can do even better. Let's use a single large step with a very high-order method.
    # For the harmonic oscillator, the solution is y(t) = [cos(t), -sin(t)].
    # But we can't use the analytic solution; we must call f.
    
    # Let's use a high-order Taylor series method. We can compute derivatives of f
    # using the chain rule, since f(t, y) = [y1, -y0] for the harmonic oscillator.
    # But we don't know the specific form of f; we only have the function handle.
    
    # So let's stick with a high-order Runge-Kutta method.
    
    # RK8 coefficients (Butcher tableau). I'll use the standard RK8 method.
    # Actually, let's use a simpler approach: use scipy's solve_ivp with DOP853,
    # which is a high-order adaptive method.
    
    # Wait, the task says we should write integrate(f, t_span, y0). We can use scipy.
    
    from scipy.integrate import solve_ivp
    
    # Use DOP853 with tight tolerances to match the reference.
    sol = solve_ivp(f, t_span, y0, method='DOP853', rtol=1e-13, atol=1e-15)
    
    if not sol.success:
        # Fallback to RK4 if solve_ivp fails
        steps = 10000
        h = h_total / steps
        t, y = t0, np.array(y0, dtype=float)
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h / 2, y + h / 2 * k1)
            k3 = f(t + h / 2, y + h / 2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
    
    return sol.y[:, -1]
# EVOLVE-BLOCK-END
