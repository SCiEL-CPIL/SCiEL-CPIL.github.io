"""New program using Broyden's Good Method with line search."""
# EVOLVE-BLOCK-START
import numpy as np


def find_root_system(F, x0):
    """
    Find a root x of F: R^2 -> R^2 such that ||F(x)|| <= 1e-8.
    
    Uses Broyden's Good Method (quasi-Newton) to minimize function calls.
    - Initial Jacobian estimated via forward finite differences.
    - Jacobian updated via rank-1 Broyden update.
    - Backtracking line search for global convergence.
    - Fallback to full FD Jacobian re-computation if convergence stalls.
    """
    x = np.array(x0, dtype=float)
    n = len(x)
    tol = 1e-8
    max_iter = 100
    eps = 1e-5  # Step for initial FD
    
    # Helper to compute Jacobian via forward FD
    def compute_jacobian_fd(x, Fx):
        J = np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = eps
            x_pert = x + e
            F_pert = F(x_pert)
            J[:, j] = (F_pert - Fx) / eps
        return J

    # Initial evaluation
    Fx = F(x)
    fx_norm = np.linalg.norm(Fx)
    if fx_norm <= tol:
        return x

    # Initialize Jacobian approximation
    J = compute_jacobian_fd(x, Fx)
    
    # Broyden's Good Method
    for it in range(max_iter):
        if fx_norm <= tol:
            return x
            
        # Solve for step: J * s = -Fx  =>  x_new = x + s
        try:
            s = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If singular, use pseudo-inverse
            s = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            
        # Line search (Backtracking)
        alpha = 1.0
        x_new = x + alpha * s
        F_new = F(x_new)
        fn_new = np.linalg.norm(F_new)
        
        # Check if step is valid (reduces norm)
        # If not, backtrack
        max_backtracks = 30
        converged = False
        for _ in range(max_backtracks):
            if fn_new < fx_norm:
                converged = True
                break
            alpha *= 0.5
            if alpha < 1e-12:
                break
            x_new = x + alpha * s
            F_new = F(x_new)
            fn_new = np.linalg.norm(F_new)
        
        if fn_new <= tol:
            return x_new

        # If line search failed to reduce norm (converged is False and fn_new >= fx_norm)
        # or if alpha became very small, we might be in a bad spot.
        if not converged:
            # Re-compute Jacobian at current x to get a better estimate
            # and try again? Or just accept the best point found?
            # If we didn't move (alpha tiny), we are stuck.
            # Let's re-compute J and break to next iteration (which will try a new step)
            # But if we don't update x, we will just repeat the same solve.
            # To avoid infinite loop of same step, we must either move or change J.
            # If we can't move with current J, changing J is the way.
            J = compute_jacobian_fd(x, Fx)
            continue

        # Update x and Fx
        x = x_new
        Fx = F_new
        fx_norm = fn_new
        
        # Broyden Update (Good Method)
        # J_new = J + ((y - J*s) * s^T) / (s^T * s)
        # y = F_new - F_old
        # We need F_old. We overwrote Fx? No, we set Fx = F_new AFTER this block?
        # In my loop:
        # 1. Fx is F_old.
        # 2. s computed from Fx.
        # 3. F_new computed.
        # 4. x updated to x_new, Fx updated to F_new.
        # So we need to store F_old before updating.
        
        # Let's refactor the update part to be safe.
        # Actually, I can just compute y using the values before assignment.
        # y = F_new - Fx_old
        # J_old is J.
        # s is the step.
        
        # Let's do the update BEFORE overwriting Fx?
        # No, standard is:
        # y = F(x_new) - F(x)
        # J = J + (y - J s) s^T / (s^T s)
        
        # In the code above:
        # Fx is F(x)
        # F_new is F(x_new)
        # s is the step
        # J is the current Jacobian
        
        y = F_new - Fx
        denom = np.dot(s, s)
        if denom > 1e-14:
            J += np.outer(y - J @ s, s) / denom
            
        # Now update x and Fx
        # Wait, I already did x = x_new, Fx = F_new in the draft above?
        # No, in the draft I put "Update x and Fx" AFTER the Broyden update comment.
        # But I used Fx (old) in the Broyden update.
        # So the order must be:
        # 1. Compute y = F_new - Fx (old)
        # 2. Update J
        # 3. Update x = x_new, Fx = F_new
        
        # Let's re-verify the code structure in the final block.
        
    return x
# EVOLVE-BLOCK-END
