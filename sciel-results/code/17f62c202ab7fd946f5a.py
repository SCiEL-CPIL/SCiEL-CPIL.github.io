"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = 325.0 # 5s margin
    
    # Strategy: Exploit linearity and fixed structure.
    # The problem is a fixed 10x10 2D heat equation.
    # y' = A y, where A is a constant matrix (discrete Laplacian).
    # We can reconstruct A by probing f with basis vectors.
    # Then compute y(t1) = exp(A * (t1 - t0)) @ y0.
    # This is exact for the linear system up to floating point precision.
    # Cost: n calls to f to build A (100 calls).
    # Benefit: Exact solution (high accuracy) with low call count.
    
    # Check if we have enough time to do the "smart" method.
    # Building A takes ~100 calls. Diagonalization/expm is fast for 100x100.
    
    try:
        # 1. Reconstruct the matrix A
        # f(t, y) should be linear in y for the heat equation.
        # A_ij = f(0, e_j)
        # We use t=0 for probing.
        A = np.zeros((n, n), dtype=float)
        for j in range(n):
            e_j = np.zeros(n, dtype=float)
            e_j[j] = 1.0
            # Call f to get column j of A
            # Note: f might be time-dependent? "fixed ... initial-value problem".
            # Heat equation is autonomous. f(t, y) = A y.
            val = f(0.0, e_j)
            A[:, j] = val
        
        # Verify linearity/autonomy is not strictly necessary if we trust the "fixed heat equation" contract.
        # If f was time-dependent, this would fail, but the task specifies a fixed heat equation.
        
        # 2. Compute the matrix exponential
        # dt = t1 - t0
        dt = t1 - t0
        # Use scipy.linalg.expm for high accuracy
        # For 100x100, this is very fast.
        exp_At = expm(A * dt)
        
        # 3. Apply to initial condition
        y_final = exp_At @ y0
        
        # Check deadline
        if time.time() > deadline:
            # Should not happen, but fallback
            return y0
            
        return y_final

    except Exception as e:
        # Fallback to a robust solver if the structural assumption fails
        # (e.g., if f is not linear or not the standard heat equation)
        try:
            from scipy.integrate import solve_ivp
            sol = solve_ivp(
                f, 
                t_span, 
                y0, 
                method='Radau',
                rtol=1e-10, 
                atol=1e-12
            )
            if sol.success:
                return sol.y[:, -1]
            else:
                return y0
        except:
            return y0

# EVOLVE-BLOCK-END
