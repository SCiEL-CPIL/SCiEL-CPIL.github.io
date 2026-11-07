import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses a hybrid Broyden-Newton method with adaptive Jacobian recomputation.
    """
    x = np.array(x0, dtype=float).copy()
    if x.shape != (2,):
        x = x.reshape(2)
        
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def time_up():
        return time.time() > deadline

    def compute_jacobian(f, x, h=1e-5):
        """Compute Jacobian using central differences."""
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (f(x + e) - f(x - e)) / (2 * h)
        return J

    # Initial evaluation
    Fx = F(x)
    if not np.all(np.isfinite(Fx)):
        # Try a few perturbations
        for dx in [1e-3, -1e-3, 1e-2, -1e-2, 1e-1, -1e-1]:
            x_try = x + np.array([dx, -dx])
            Fx_try = F(x_try)
            if np.all(np.isfinite(Fx_try)):
                x = x_try
                Fx = Fx_try
                break
        else:
            return x

    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x

    # Initial Jacobian
    J = compute_jacobian(F, x)
    
    best_x = x.copy()
    best_norm = norm_Fx
    prev_norm = norm_Fx
    prev_Fx = Fx.copy()
    prev_x = x.copy()

    for iteration in range(100):
        if time_up():
            break

        # Solve for step direction
        try:
            d = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If singular, recompute Jacobian and try again
            J = compute_jacobian(F, x)
            try:
                d = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break

        if np.linalg.norm(d) < 1e-12:
            break

        # Line search with backtracking
        alpha = 1.0
        step_accepted = False
        Fx_new = None
        x_new = None

        for ls_iter in range(20):
            if time_up():
                break
            
            x_new = x + alpha * d
            Fx_new = F(x_new)
            
            if not np.all(np.isfinite(Fx_new)):
                alpha *= 0.5
                continue

            norm_Fx_new = np.linalg.norm(Fx_new)
            
            if norm_Fx_new <= 1e-8:
                return x_new

            # Track best
            if norm_Fx_new < best_norm:
                best_norm = norm_Fx_new
                best_x = x_new.copy()

            # Accept if we reduce the norm (simple sufficient decrease)
            # Using a looser condition to avoid getting stuck
            if norm_Fx_new < norm_Fx * 0.99: 
                step_accepted = True
                break
            else:
                alpha *= 0.5

        if not step_accepted:
            # If no step was accepted, try a Newton step with fresh Jacobian
            # This costs 4 calls but might break the deadlock
            if iteration > 2:
                J = compute_jacobian(F, x)
                try:
                    d = np.linalg.solve(J, -Fx)
                    x_new = x + d
                    Fx_new = F(x_new)
                    if np.all(np.isfinite(Fx_new)) and np.linalg.norm(Fx_new) < best_norm:
                        x = x_new
                        Fx = Fx_new
                        norm_Fx = np.linalg.norm(Fx)
                        if norm_Fx <= 1e-8:
                            return x
                        # Update prev values for Broyden update next time
                        prev_Fx = Fx.copy()
                        prev_x = x.copy()
                        # Note: J is now the true Jacobian, so Broyden update 
                        # from here will be valid if we use the new step.
                        # But we just took a Newton step. We can continue.
                        continue
                except:
                    pass
            break

        # Update state
        # Broyden Update
        delta_F = Fx_new - Fx
        delta_x = x_new - x
        dot_xx = np.dot(delta_x, delta_x)
        
        if dot_xx > 1e-15:
            J = J + np.outer(delta_F - J @ delta_x, delta_x) / dot_xx

        x = x_new
        Fx = Fx_new
        norm_Fx = np.linalg.norm(Fx)

        # Stagnation Check
        if iteration > 2 and norm_Fx > 0.9 * prev_norm:
            # Stagnation detected, recompute Jacobian
            J = compute_jacobian(F, x)
        
        prev_norm = norm_Fx
        prev_Fx = Fx.copy()
        prev_x = x.copy()

    return best_x
