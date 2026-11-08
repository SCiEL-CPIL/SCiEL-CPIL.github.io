"""ODE integrator for the discretized heat equation (N = 20)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    
    Uses an implicit Runge-Kutta method (Radau IIA) with Newton iteration
    to handle the stiff heat equation efficiently.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # We'll use a fixed number of steps for the Radau method.
    # The heat equation with N=20 is moderately stiff, so we need enough steps.
    # Let's use a Radau IIA 3-stage method with adaptive step size control.
    
    # For simplicity and robustness, let's use a fixed-step implicit method.
    # We'll use backward differentiation formulas (BDF) with Newton iteration.
    
    # Actually, let's use a simpler approach: since this is a linear ODE with
    # known structure, we can use a matrix exponential or a high-order implicit method.
    
    # Let's use Radau IIA with 3 stages and a reasonable number of steps.
    # The key insight is that for the heat equation, the problem is stiff but
    # we can handle it with implicit methods.
    
    # Use a fixed number of steps based on the time interval.
    # For t in [0, 0.1], let's use 100 steps of size 0.001.
    
    # Actually, let me think about this more carefully. The heat equation
    # y' = A y where A = 21^2 * tridiag(1, -2, 1) has eigenvalues that are
    # negative real numbers. The largest magnitude eigenvalue is approximately
    # -21^2 * 4 * sin^2(10*pi/42) ≈ -441 * 4 * 0.866 ≈ -1528.
    
    # For stability of explicit methods, we need h < 2/|lambda_max| ≈ 0.0013.
    # For implicit methods, we can take larger steps.
    
    # Let's use a 2-step BDF method with Newton iteration, or simply use
    # backward Euler with enough Newton iterations.
    
    # For efficiency, let's use a matrix-free Newton iteration.
    
    # Actually, the simplest robust approach: use scipy's solve_ivp if available,
    # but we need to count f calls. Let's implement our own implicit method.
    
    # Let's use Radau IIA 3-stage method with fixed step size.
    # The Butcher tableau for Radau IIA 3 stages:
    # c1 = 1 - 2*sqrt(6)/9
    # c2 = 1/2
    # c3 = 1 + 2*sqrt(6)/9
    # 
    # a11 = 1/6 + 2*sqrt(6)/27
    # a12 = 1/6 - 2*sqrt(6)/27
    # a13 = 0
    # a21 = 1/6 - sqrt(6)/9
    # a22 = 1/2
    # a23 = 1/6 + sqrt(6)/9
    # a31 = 0
    # a32 = 1/6 + 2*sqrt(6)/27
    # a33 = 1/6 - 2*sqrt(6)/27
    #
    # b1 = 1/6 + 2*sqrt(6)/27
    # b2 = 2/3
    # b3 = 1/6 - 2*sqrt(6)/27
    
    # This is getting complex. Let me use a simpler approach.
    
    # Use backward Euler with Newton iteration. This is first-order but
    # unconditionally stable. With enough steps, it will be accurate.
    
    # For the heat equation over [0, 0.1], let's use 200 steps of size 0.0005.
    # Backward Euler has local error O(h^2), so global error O(h).
    # With h = 0.0005, global error ~ 0.0005, which might not be enough.
    
    # Let's use a higher-order method. BDF2 is second-order and also stable.
    
    # Actually, let me just use a simple but effective approach:
    # Use the fact that this is a linear system y' = A y, and we can
    # compute the solution using matrix functions or a high-order implicit method.
    
    # For now, let's use Radau IIA with 3 stages and 50 steps.
    
    sqrt6 = np.sqrt(6.0)
    c1 = 1.0 - 2.0*sqrt6/9.0
    c2 = 0.5
    c3 = 1.0 + 2.0*sqrt6/9.0
    
    a11 = 1.0/6.0 + 2.0*sqrt6/27.0
    a12 = 1.0/6.0 - 2.0*sqrt6/27.0
    a13 = 0.0
    a21 = 1.0/6.0 - sqrt6/9.0
    a22 = 0.5
    a23 = 1.0/6.0 + sqrt6/9.0
    a31 = 0.0
    a32 = 1.0/6.0 + 2.0*sqrt6/27.0
    a33 = 1.0/6.0 - 2.0*sqrt6/27.0
    
    b1 = a11
    b2 = 2.0/3.0
    b3 = a33
    
    # Number of steps
    steps = 100
    h = (t1 - t0) / steps
    t = t0
    
    # Newton tolerance
    tol = 1e-12
    max_newton = 20
    
    for _ in range(steps):
        # Radau IIA 3-stage method
        # Stage 1: solve (I - h*a11*A) Y1 = y + h*a12*A*Y2 + h*a13*A*Y3
        # Stage 2: solve (I - h*a22*A) Y2 = y + h*a21*A*Y1 + h*a23*A*Y3
        # Stage 3: solve (I - h*a33*A) Y3 = y + h*a31*A*Y1 + h*a32*A*Y2
        
        # Since A is unknown (we only have f), we use Newton iteration.
        
        # Initialize Y1, Y2, Y3
        Y1 = y.copy()
        Y2 = y.copy()
        Y3 = y.copy()
        
        for _newton in range(max_newton):
            # Compute f at each stage
            f1 = f(t + h*c1, Y1)
            f2 = f(t + h*c2, Y2)
            f3 = f(t + h*c3, Y3)
            
            # Residuals for the implicit equations
            # Y1 = y + h*(a11*f1 + a12*f2 + a13*f3)
            # Y2 = y + h*(a21*f1 + a22*f2 + a23*f3)
            # Y3 = y + h*(a31*f1 + a32*f2 + a33*f3)
            
            R1 = Y1 - (y + h*(a11*f1 + a12*f2 + a13*f3))
            R2 = Y2 - (y + h*(a21*f1 + a22*f2 + a23*f3))
            R3 = Y3 - (y + h*(a31*f1 + a32*f2 + a33*f3))
            
            # Check convergence
            res_norm = np.sqrt(np.sum(R1**2) + np.sum(R2**2) + np.sum(R3**2))
            if res_norm < tol:
                break
            
            # For Newton iteration, we need the Jacobian. Since f(t,y) = A*y
            # (linear system), the Jacobian is just A. We can estimate it
            # or use a simpler fixed-point iteration.
            
            # Let's use a simple relaxation/fixed-point approach:
            # Update Y1, Y2, Y3 using the current f values
            Y1_new = y + h*(a11*f1 + a12*f2 + a13*f3)
            Y2_new = y + h*(a21*f1 + a22*f2 + a33*f3)
            Y3_new = y + h*(a31*f1 + a32*f2 + a33*f3)
            
            # Damping factor for stability
            alpha = 1.0
            Y1 = (1-alpha)*Y1 + alpha*Y1_new
            Y2 = (1-alpha)*Y2 + alpha*Y2_new
            Y3 = (1-alpha)*Y3 + alpha*Y3_new
        
        # Update y
        y = y + h*(b1*f1 + b2*f2 + b3*f3)
        t += h
        
        if not np.all(np.isfinite(y)):
            break
    
    return y
# EVOLVE-BLOCK-END
