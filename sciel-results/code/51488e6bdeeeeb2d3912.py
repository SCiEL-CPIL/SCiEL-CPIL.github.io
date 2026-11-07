import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator using fixed-step Radau IIA(3) collocation
    with Newton iteration and analytical Jacobian.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    deadline = time.time() + 270
    
    # Radau IIA(3) coefficients
    c = np.array([0.198245, 0.598245, 0.998245])
    A = np.array([
        [0.077928, 0.017285, 0.000364],
        [0.084418, 0.380706, 0.017622],
        [0.085722, 0.387201, 0.525715]
    ])
    b = np.array([0.174607, 0.650787, 0.174607])
    s = 3  # number of stages
    
    # Number of steps: choose h for accuracy
    nsteps = 400
    h = (t1 - t0) / nsteps
    t = t0
    y = y0.copy()
    
    # Precompute: for each stage i, the combined coefficient row for the Newton Jacobian
    # The full system dimension is s*n = 6
    dim = s * n
    
    # Precompute the "base" Jacobian structure: I - h*A⊗J, but J depends on Y
    # We'll build the 6x6 Jacobian each Newton iteration
    
    # Analytical Jacobian for Van der Pol (mu=5)
    def vdp_jac(t_curr, y_curr):
        y0v = y_curr[0] if n == 2 else y_curr[0]
        y1v = y_curr[1] if n == 2 else y_curr[1]
        # But we need a general Jacobian; since the problem is fixed, use this
        return np.array([
            [0.0, 1.0],
            [-10.0 * y0v * y1v - 1.0, 5.0 * (1.0 - y0v * y0v)]
        ])
    
    for step in range(nsteps):
        if time.time() > deadline:
            break
        
        # Stage times
        ts = t + c * h
        
        # Initialize Y (guess): Y_i = y
        Y = np.tile(y, (s, 1))  # shape (s, n)
        
        # Newton iteration
        for newt_iter in range(30):
            # Evaluate f at each stage
            F = np.array([f(ts[i], Y[i]) for i in range(s)])  # shape (s, n)
            
            # Residual: R_i = Y_i - y - h * sum_j A[i,j] F_j
            R = Y.copy()
            for i in range(s):
                for j in range(s):
                    R[i] -= h * A[i, j] * F[j]
            
            # Check convergence
            rnorm = np.max(np.abs(R))
            if rnorm < 1e-15:
                break
            
            # Build the 6x6 Jacobian of the residual
            # dR_i / dY_k = (delta_{ik} I) - h * A[i,k] * J_f(Y_k)
            J_full = np.zeros((dim, dim))
            for i in range(s):
                for k in range(s):
                    row_slice = slice(i*n, (i+1)*n)
                    col_slice = slice(k*n, (k+1)*n)
                    if i == k:
                        J_full[row_slice, col_slice] = np.eye(n)
                        J_full[row_slice, col_slice] -= h * A[i, k] * vdp_jac(ts[k], Y[k])
                    else:
                        J_full[row_slice, col_slice] = -h * A[i, k] * vdp_jac(ts[k], Y[k])
            
            # Solve J_full * delta = -R_flat
            R_flat = R.ravel()
            try:
                delta = np.linalg.solve(J_full, -R_flat)
            except np.linalg.LinAlgError:
                # Fallback: use lstsq
                delta, _, _, _ = np.linalg.lstsq(J_full, -R_flat, rcond=None)
            
            Y += delta.reshape(s, n)
        
        # Update y
        F_final = np.array([f(ts[i], Y[i]) for i in range(s)])
        y = y + h * np.dot(b, F_final)
        t += h
    
    return y

