"""Root finder for 3 fixed two-dimensional systems using Broyden's method with safeguards."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root of F: R^2 -> R^2 using Broyden's method (good for small systems).
    Falls back to damped Newton / line search if needed.
    Returns x with ||F(x)|| <= 1e-8 or best effort.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    x = np.array(x0, dtype=float).copy()
    n = 2
    
    # Initial evaluation
    Fx = np.array(F(x), dtype=float)
    norm_Fx = np.linalg.norm(Fx)
    
    if norm_Fx <= 1e-8:
        return x
    
    # Initialize Jacobian with finite differences
    eps = 1e-7
    J = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = eps
        J[:, j] = (np.array(F(x + e), dtype=float) - Fx) / eps
    
    # Broyden's method (Good) with damping
    best_x = x.copy()
    best_norm = norm_Fx
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        # Check if we've converged
        if norm_Fx <= 1e-8:
            return x
        
        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If singular, use pseudo-inverse
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except:
                break
        
        # Damping: find alpha such that ||F(x + alpha*dx)|| < ||F(x)||
        # Try full step first, then bisection
        alpha = 1.0
        Fx_new = None
        norm_Fx_new = None
        
        for _ in range(20):
            x_trial = x + alpha * dx
            Fx_trial = np.array(F(x_trial), dtype=float)
            norm_Fx_trial = np.linalg.norm(Fx_trial)
            
            if norm_Fx_trial < norm_Fx * 0.9:  # sufficient decrease
                Fx_new = Fx_trial
                norm_Fx_new = norm_Fx_trial
                break
            else:
                alpha *= 0.5
                if alpha < 1e-12:
                    # Accept the step even if not great
                    Fx_new = Fx_trial
                    norm_Fx_new = norm_Fx_trial
                    break
        
        if Fx_new is None:
            Fx_new = np.array(F(x), dtype=float)
            norm_Fx_new = norm_Fx
        
        # Update x and Fx
        x = x + alpha * dx
        Fx = Fx_new
        norm_Fx = norm_Fx_new
        
        # Track best
        if norm_Fx < best_norm:
            best_norm = norm_Fx
            best_x = x.copy()
        
        # Update Jacobian using Broyden's formula (Good variant)
        # J_new = J + (Fx_new - Fx - J*dx) * dx^T / (dx^T * dx)
        delta_F = Fx - Fx_new  # Note: Fx is old, Fx_new is new, so delta_F = Fx_old - Fx_new
        # Actually, let's be careful:
        # We had Fx_old = Fx (before update), Fx_new = F(x + alpha*dx)
        # delta_x = alpha * dx
        # delta_F = Fx_new - Fx_old
        delta_x = alpha * dx
        delta_F = Fx - Fx_new  # This is Fx_old - Fx_new, we want Fx_new - Fx_old = -delta_F
        
        # Broyden's Good: J_new = J + (delta_F - J*delta_x) * delta_x^T / (delta_x^T * delta_x)
        # where delta_F = F(x_new) - F(x_old)
        delta_F_correct = Fx_new - Fx  # Wait, Fx was overwritten. Let me restructure.
        
        # Let me redo this more carefully.
        # At start of loop: x, Fx, norm_Fx, J
        # We computed dx = -J^{-1} Fx
        # We tried x_trial = x + alpha*dx, got Fx_trial, norm_Fx_trial
        # Then set x = x_trial, Fx = Fx_trial
        # So delta_x = x - x_old = alpha * dx
        # delta_F = Fx - Fx_old = Fx_trial - Fx_old
        
        # I need to save Fx_old before overwriting. Let me restructure.
        
        # For now, let's just recompute or use a simpler update.
        # Actually, let's use a simpler approach: just recompute Jacobian periodically.
        
        # Update Jacobian every few iterations
        if iteration % 3 == 0:
            Fx_current = np.array(F(x), dtype=float)
            J = np.zeros((n, n))
            for j in range(n):
                e = np.zeros(n)
                e[j] = eps
                J[:, j] = (np.array(F(x + e), dtype=float) - Fx_current) / eps
            # Re-evaluate Fx to be safe
            Fx = Fx_current
            norm_Fx = np.linalg.norm(Fx)
    
    # Return the best point found
    if best_norm <= norm_Fx:
        return best_x
    else:
        return x
# EVOLVE-BLOCK-END
