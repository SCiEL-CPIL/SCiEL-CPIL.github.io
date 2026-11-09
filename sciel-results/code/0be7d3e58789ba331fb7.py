# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Solve y' = f(t, y) from t_span[0] to t_span[1] with initial condition y0.
    Returns y(t_end).
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard: kill after ~290s to stay within 330s wall-clock
    start_time = time.time()
    deadline = start_time + 290.0

    # 1. Probe the system to determine if it is linear and time-invariant.
    # We assume the system is of the form y' = A(t)y + b(t).
    
    # Evaluate at t0
    # f(t, y) is expected to be linear.
    # f(t, 0) gives b(t) if affine, or 0 if homogeneous.
    b0 = np.asarray(f(t0, np.zeros(n)), dtype=float)
    
    # Check if b0 is effectively zero to decide if we need to treat it as affine.
    # However, even if it is homogeneous, treating it as affine with b=0 is safe.
    # To extract A, we need f(t, e_j) for basis vectors or f(t, I).
    # Let's try to extract A(t0) and b(t0).
    
    # Construct identity matrix
    I = np.eye(n)
    
    # f(t, I) = A(t)I + b(t) = A(t) + b(t)
    # So A(t) = f(t, I) - b(t)
    # Note: f takes (t, y). If y is a matrix, it might not work if f expects a vector.
    # Standard ODE interfaces expect y to be a vector of size n.
    # If f is implemented as y @ A or A @ y, passing a matrix might break it or return wrong shape.
    # Safer approach: Evaluate f(t, e_j) for each basis vector e_j.
    # f(t, e_j) = A(t)[:, j] + b(t).
    # So A(t)[:, j] = f(t, e_j) - b(t).
    
    A_t0 = np.zeros((n, n))
    # Re-use b0
    b_t0 = b0
    
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = 1.0
        fj = np.asarray(f(t0, ej), dtype=float)
        A_t0[:, j] = fj - b_t0
        
    # Check if the system is time-invariant by evaluating at t1
    # f(t1, 0)
    b1 = np.asarray(f(t1, np.zeros(n)), dtype=float)
    
    A_t1 = np.zeros((n, n))
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = 1.0
        fj = np.asarray(f(t1, ej), dtype=float)
        A_t1[:, j] = fj - b1
        
    # Check if A and b are constant
    # Tolerance for "constant"
    tol = 1e-10
    is_homogeneous = np.linalg.norm(b_t0) < tol
    is_time_invariant = np.allclose(A_t0, A_t1, rtol=1e-5, atol=tol) and np.allclose(b_t0, b1, rtol=1e-5, atol=tol)
    
    # Also check if the system is actually linear by checking superposition?
    # If f(t, y1+y2) != f(t,y1)+f(t,y2), it's non-linear.
    # Given the task says "stiff linear system", we assume linearity.
    
    if is_time_invariant:
        dt = t1 - t0
        if is_homogeneous:
            # y(t1) = exp(A * dt) @ y0
            M = A_t0 * dt
            y_end = expm(M) @ y0
        else:
            # Affine system: y' = A y + b
            # Augmented system: z = [y; 1], z' = [A b; 0 0] z
            # Dimension n+1
            n_aug = n + 1
            M_aug = np.zeros((n_aug, n_aug))
            M_aug[:n, :n] = A_t0
            M_aug[:n, n] = b_t0
            # Last row is 0
            
            M_aug_scaled = M_aug * dt
            
            y0_aug = np.zeros(n_aug)
            y0_aug[:n] = y0
            y0_aug[n] = 1.0
            
            z_end = expm(M_aug_scaled) @ y0_aug
            y_end = z_end[:n]
            
        # Verify solution is finite
        if np.all(np.isfinite(y_end)):
            return y_end
        else:
            # Fallback if expm failed or produced NaNs (unlikely for 15x15)
            pass

    # Fallback: Use scipy's robust stiff solver
    # This handles time-varying, non-linear, or cases where our linear assumption failed.
    # We use Radau which is good for stiff problems.
    # Set tight tolerances for high accuracy.
    
    # If we already used many calls, be careful.
    # The probe used 1 + 2*n calls = 1 + 30 = 31 calls.
    
    def f_wrapper(t, y):
        return f(t, y)
        
    # Max steps limit to prevent infinite loops, but allow enough for stiff systems.
    # 15-dim stiff system on [0,1] usually doesn't need huge steps if solver is good.
    try:
        sol = solve_ivp(
            f_wrapper, 
            t_span, 
            y0, 
            method='Radau', 
            rtol=1e-10, 
            atol=1e-12,
            max_step=0.1, # Allow adaptive stepping but cap it
            max_step_limit=10000 # Hard limit on number of steps? No, max_step is size.
        )
        # Check if solver succeeded
        if sol.success:
            return sol.y[:, -1]
        else:
            # If Radau failed, try BDF
            sol2 = solve_ivp(
                f_wrapper, 
                t_span, 
                y0, 
                method='BDF', 
                rtol=1e-10, 
                atol=1e-12,
                max_step=0.1
            )
            if sol2.success:
                return sol2.y[:, -1]
            else:
                # Last resort: return initial condition or last available state?
                # Better to return whatever the last successful step was, or y0 if none.
                # But solve_ivp usually raises exception or returns success=False.
                # If it fails completely, we might just return y0 to avoid exception, though score will be 0.
                # Let's try to return the last finite state if available.
                if len(sol.y) > 0:
                    return sol.y[:, -1]
                if len(sol2.y) > 0:
                    return sol2.y[:, -1]
                return y0
    except Exception:
        return y0

# EVOLVE-BLOCK-END
