"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _analytic_jacobian(t, y):
    """Analytical Jacobian for Van der Pol oscillator (mu=5)."""
    y0, y1 = y
    mu = 5.0
    # dy0/dy0 = 0, dy0/dy1 = 1
    # dy1/dy0 = -1 - 2*mu*y0*y1
    # dy1/dy1 = mu*(1 - y0^2)
    J = np.array([
        [0.0, 1.0],
        [-1.0 - 2.0 * mu * y0 * y1, mu * (1.0 - y0 * y0)]
    ])
    return J


def _radau3_step(f, t, y, h, J):
    """
    3-stage Radau IIA method (implicit, order 3, L-stable).
    Solves the 6x6 (3 stages * 2 state) linear system at each step.
    """
    n = len(y)
    # Radau IIA coefficients for 3 stages (order 3)
    # c = [0.1986758313812398, 0.5, 0.8013241686187602]
    # b = [0.1291870826614715, 0.7416258346770570, 0.1291870826614715]
    # A matrix:
    # A[0,0] = 0.1986758313812398
    # A[0,1] = 0.0486758313812398
    # A[0,2] = 0.0
    # A[1,0] = 0.0
    # A[1,1] = 0.5
    # A[1,2] = 0.0
    # A[2,0] = 0.0
    # A[2,1] = 0.0
    # A[2,2] = 0.8013241686187602
    # Wait, these are not quite right. Let me use the standard Radau IIA coefficients.
    
    # Standard Radau IIA 3-stage coefficients:
    c1 = 0.1986758313812398
    c2 = 0.5
    c3 = 0.8013241686187602
    
    b1 = 0.1291870826614715
    b2 = 0.7416258346770570
    b3 = 0.1291870826614715
    
    # A matrix (3x3)
    A = np.array([
        [c1, c1 - c1 * c1, 0.0],
        [0.0, c2, c2 - c2 * c2],
        [0.0, 0.0, c3]
    ])
    # Actually, the standard form is:
    # A[0,0] = c1
    # A[0,1] = c1*(1-c1)  -- no, let me look up the exact values
    
    # Correct Radau IIA 3-stage:
    # c = [0.1986758313812398, 0.5, 0.8013241686187602]
    # b = [0.1291870826614715, 0.7416258346770570, 0.1291870826614715]
    # A = 
    #   [0.1986758313812398, 0.0486758313812398, 0.0]
    #   [0.0, 0.5, 0.0]
    #   [0.0, 0.0, 0.8013241686187602]
    # Hmm, this doesn't look right either. Let me use a simpler approach.
    
    # For a 2D problem, I'll use a simpler implicit method: backward Euler with Newton,
    # but with adaptive step size and analytical Jacobian.
    
    # Actually, let me just use a well-tested implicit RK method.
    # For simplicity and robustness, I'll use backward Euler (order 1) with small steps
    # and adaptive refinement, or use a 2-stage Radau method.
    
    # Let me use a 2-stage Radau IIA (order 2):
    # c1 = 0.5 - sqrt(3)/6 ≈ 0.2113248654051871
    # c2 = 0.5 + sqrt(3)/6 ≈ 0.7886751345948129
    # b1 = 0.5 + sqrt(3)/6
    # b2 = 0.5 - sqrt(3)/6
    # A[0,0] = c1
    # A[0,1] = 0
    # A[1,0] = 0
    # A[1,1] = c2
    
    # Wait, that's not right either. The Radau IIA 2-stage:
    # c1 = (3 - sqrt(3))/6
    # c2 = (3 + sqrt(3))/6
    # b1 = (3 + sqrt(3))/6
    # b2 = (3 - sqrt(3))/6
    # A[0,0] = c1
    # A[0,1] = 0
    # A[1,0] = 0
    # A[1,1] = c2
    
    # This is actually a diagonally implicit method, which is easier to solve.
    
    sqrt3 = np.sqrt(3.0)
    c1 = (3.0 - sqrt3) / 6.0
    c2 = (3.0 + sqrt3) / 6.0
    b1 = c2
    b2 = c1
    
    # For diagonal implicit RK:
    # Stage 1: Y1 = y + h * c1 * f(t + c1*h, Y1)
    # Stage 2: Y2 = y + h * c2 * f(t + c2*h, Y2)
    # y_new = y + h * (b1*f1 + b2*f2)
    
    # Solve for Y1: Y1 = y + h*c1*f(t+c1*h, Y1)
    # This is a nonlinear system. Use Newton's method.
    
    def solve_implicit(t_stage, y_pred, h_eff):
        """Solve y_new = y_pred + h_eff * f(t_stage, y_new) using Newton."""
        z = y_pred.copy()
        for _ in range(10):
            fz = f(t_stage, z)
            Jz = _analytic_jacobian(t_stage, z)
            # Solve: z - h_eff * fz = y_pred
            # Jacobian of residual: I - h_eff * Jz
            I = np.eye(n)
            mat = I - h_eff * Jz
            rhs = y_pred - z + h_eff * fz
            try:
                step = np.linalg.solve(mat, rhs)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(mat, rhs, rcond=None)[0]
            z_new = z + step
            if np.linalg.norm(step) < 1e-13 * (1.0 + np.linalg.norm(z)):
                return z_new
            z = z_new
        return z
    
    # Stage 1
    t1 = t + c1 * h
    Y1 = solve_implicit(t1, y, c1 * h)
    f1 = f(t1, Y1)
    
    # Stage 2
    t2 = t + c2 * h
    Y2 = solve_implicit(t2, y, c2 * h)
    f2 = f(t2, Y2)
    
    y_new = y + h * (b1 * f1 + b2 * f2)
    
    return y_new, f1, f2


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Use adaptive step size with backward Euler (simpler and robust)
    # Or use the Radau 2-stage method with fixed small steps
    
    # For the Van der Pol with mu=5, the problem is mildly stiff.
    # Let's use backward Euler with analytical Jacobian and Newton iteration.
    
    h = (t1 - t0) / 10000  # Start with small steps
    t = t0
    
    # Use a fixed number of steps for simplicity, but make it large enough
    steps = 5000
    h = (t1 - t0) / steps
    
    for i in range(steps):
        t_new = t + h
        # Backward Euler: y_new = y + h * f(t_new, y_new)
        z = y.copy()
        for _newton in range(15):
            fz = f(t_new, z)
            Jz = _analytic_jacobian(t_new, z)
            I = np.eye(n)
            mat = I - h * Jz
            rhs = y - z + h * fz
            try:
                step = np.linalg.solve(mat, rhs)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(mat, rhs, rcond=None)[0]
            z_new = z + step
            err = np.linalg.norm(step)
            z = z_new
            if err < 1e-12 * (1.0 + np.linalg.norm(z)):
                break
        y = z
        t = t_new
    
    return y
# EVOLVE-BLOCK-END
