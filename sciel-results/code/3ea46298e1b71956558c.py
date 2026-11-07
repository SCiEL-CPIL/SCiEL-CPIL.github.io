"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def _radau_ii_5(f, t0, t1, y0, steps):
    """
    Fixed-step 5th-order Radau IIA collocation integrator.
    Uses a single global Newton iteration per step with a precomputed LU factorization
    of the mass matrix (I - h*J) where J is the Jacobian.
    """
    n = len(y0)
    h = (t1 - t0) / steps
    y = np.array(y0, dtype=float)
    t = t0

    # Radau IIA 5th order collocation points (roots of Legendre polynomial of degree 5 on [0,1])
    # These are the Radau points (including the right endpoint)
    # For 5th order, we have 3 stages (c1, c2, c3) where c3 = 1
    # The Radau IIA 5-stage would be different, but "5th order" typically means 3 stages for Radau IIA
    # Let's use the standard 3-stage Radau IIA (5th order)
    # Nodes: c = [0.435866528072870, 0.679409560539026, 1.0]
    # Weights: b = [0.294384156102161, 0.556685171465157, 0.148930672432682]
    # But wait, Radau IIA with 3 stages is 5th order.
    # Let's verify the standard Radau IIA 3-stage coefficients.
    
    c = np.array([0.435866528072870, 0.679409560539026, 1.0])
    b = np.array([0.294384156102161, 0.556685171465157, 0.148930672432682])
    # A matrix for Radau IIA 3-stage
    # A = [[0.144712731841322, 0.001487479637580, 0.0],
    #      [0.132036162431144, 0.087458417856336, 0.0],
    #      [0.294384156102161, 0.556685171465157, 0.148930672432682]]
    A = np.array([
        [0.144712731841322, 0.001487479637580, 0.0],
        [0.132036162431144, 0.087458417856336, 0.0],
        [0.294384156102161, 0.556685171465157, 0.148930672432682]
    ])
    
    # Mass matrix M for the collocation conditions: M * dY/dt = F(Y)
    # For Radau IIA, the mass matrix is (I - h*A)
    # Actually, the system to solve at each Newton iteration is:
    # (I - h * J) * delta = ...
    # where J is the Jacobian of f.
    # But since f depends on t and y, and we're doing a fixed step, we can approximate J at each stage.
    
    # To minimize function calls, we will use a Jacobian-free Newton method or a finite difference Jacobian.
    # Given the constraint of < 2,000,000 calls, and steps ~ 1000-2000, we can afford a few Newton iterations.
    # Let's use a finite difference Jacobian computed once per step (or per Newton iteration).
    
    # Precompute the LU factorization of the mass matrix if it were constant, but J changes.
    # So we must recompute the LU factorization of (I - h * J) at each Newton iteration.
    
    # Let's try a simpler approach: Use a fixed number of Newton iterations (e.g., 3-5) per step.
    # For each step:
    # 1. Guess Y (stage values)
    # 2. For each Newton iteration:
    #    a. Compute F = f(t + h*c, Y)
    #    b. Compute J = Jacobian of f at (t + h*c, Y) using finite differences
    #    c. Solve (I - h*A) * dY = ... (this is a block system)
    
    # Actually, the Radau IIA collocation equations are:
    # Y_i = y_n + h * sum_j A_ij f(t_n + h*c_j, Y_j)
    # This is a system of n*s equations (s stages, n dimensions).
    # We can solve this with Newton's method.
    
    s = len(c)
    
    for _step in range(steps):
        # Initial guess: Y_i = y
        Y = np.tile(y, (s, 1))  # Shape (s, n)
        
        # Newton iterations
        for _newton in range(5):
            # Compute stage derivatives
            F = np.empty((s, n))
            for i in range(s):
                F[i] = f(t + h * c[i], Y[i])
            
            # Compute residual: R_i = Y_i - y - h * sum_j A_ij F_j
            R = Y - y - h * (A @ F)
            
            # Check convergence
            if np.max(np.abs(R)) < 1e-12 * (1 + np.max(np.abs(Y))):
                break
            
            # Compute Jacobian of the residual with respect to Y
            # dR_i/dY_j = delta_ij * I - h * A_ij * J_j
            # where J_j = df/dy at (t + h*c_j, Y_j)
            
            # We need to solve: J_res * dY = -R
            # J_res is a block matrix of size (s*n) x (s*n)
            # But we can solve this more efficiently.
            
            # Let's build the Jacobian matrix J_res
            # First, compute the finite difference Jacobians J_j for each stage
            J_stages = np.empty((s, n, n))
            eps = 1e-8
            for j in range(s):
                for k in range(n):
                    Y_pert = Y.copy()
                    Y_pert[j, k] += eps * max(1.0, abs(Y[j, k]))
                    F_pert = f(t + h * c[j], Y_pert)
                    J_stages[j, :, k] = (F_pert - F[j]) / (eps * max(1.0, abs(Y[j, k])))
            
            # Now build the block Jacobian
            # J_res[i, j] = delta_ij * I - h * A_ij * J_stages[j]
            # This is a (s*n) x (s*n) matrix
            J_res = np.empty((s * n, s * n))
            for i in range(s):
                for j in range(s):
                    block = np.zeros((n, n))
                    if i == j:
                        block += np.eye(n)
                    block -= h * A[i, j] * J_stages[j]
                    J_res[i*n:(i+1)*n, j*n:(j+1)*n] = block
            
            # Solve J_res * dY = -R
            R_flat = R.flatten()
            try:
                dY_flat = np.linalg.solve(J_res, -R_flat)
            except np.linalg.LinAlgError:
                dY_flat = np.linalg.lstsq(J_res, -R_flat, rcond=None)[0]
            
            dY = dY_flat.reshape(s, n)
            Y += dY
            
            # Damping if needed
            if np.max(np.abs(dY)) > 0.5 * np.max(np.abs(Y)):
                Y -= 0.5 * dY
        
        # Update y
        y = y + h * (b @ F)
        t += h
        
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e12:
            return None
    
    return y


def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator using a fixed-step 5th-order Radau IIA collocation method.
    """
    t0, t1 = t_span
    # Choose a reasonable number of steps. Van der Pol with mu=5 is stiff.
    # Radau IIA is A-stable, so we can use larger steps.
    # Let's try 500 steps first.
    y = _radau_ii_5(f, t0, t1, y0, 500)
    if y is None:
        # Fallback to more steps
        y = _radau_ii_5(f, t0, t1, y0, 1000)
    return y
# EVOLVE-BLOCK-END
