"""Root finder for 3 fixed two-dimensional systems using Broyden and damped Newton."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Use Broyden's method with damped line search as the primary solver.
       Broyden updates the Jacobian approximation, requiring only 1 F(x) call per iteration
       (plus initial setup), which is much more efficient than finite-difference Newton.
    2. If Broyden fails or stagnates, try a few restarts with different initial points.
    3. Perform a final polishing step with Newton's method to ensure high precision.
    4. Time-guarded to stay within budget.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin for 150s limit
    
    x0 = np.array(x0, dtype=float)
    
    def time_ok():
        return time.time() < deadline

    def eval_F(x):
        """Safely evaluate F and return norm and value."""
        try:
            fx = F(x)
            fx = np.asarray(fx, dtype=float)
            if fx.shape != (2,) or not np.all(np.isfinite(fx)):
                return np.inf, None
            return np.linalg.norm(fx), fx
        except:
            return np.inf, None

    def update_best(x, best_state):
        """Update best solution if x is better."""
        n, _ = eval_F(x)
        if n < best_state['norm']:
            best_state['norm'] = n
            best_state['x'] = x.copy()
            if n <= 1e-8:
                return True
        return False

    def broyden_method(x_init, max_iter=50, tol=1e-10):
        """
        Broyden's Good method for root finding.
        Returns (x, success).
        """
        x = x_init.copy()
        n, Fx = eval_F(x)
        if not np.isfinite(n):
            return x, False
        
        if n <= tol:
            return x, True

        # Initialize Jacobian using finite differences
        h = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            n_plus, Fx_plus = eval_F(x + e)
            n_minus, Fx_minus = eval_F(x - e)
            if np.isfinite(n_plus) and np.isfinite(n_minus):
                J[:, j] = (Fx_plus - Fx_minus) / (2 * h)
            else:
                # Fallback if FD fails
                J[:, j] = 1.0 
        
        # Check if J is singular initially
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            J = J + 1e-6 * np.eye(2)
            try:
                dx = np.linalg.solve(J, -Fx)
            except:
                return x, False

        for _ in range(max_iter):
            if not time_ok():
                break
            
            # Damped Newton step with Broyden update
            # Line search: find alpha such that ||F(x + alpha*dx)|| < ||F(x)||
            alpha = 1.0
            x_new = x + alpha * dx
            n_new, Fx_new = eval_F(x_new)
            
            if not np.isfinite(n_new):
                # Backtrack
                alpha = 0.5
                x_new = x + alpha * dx
                n_new, Fx_new = eval_F(x_new)
                
                if not np.isfinite(n_new):
                    # Try smaller step
                    alpha = 0.1
                    x_new = x + alpha * dx
                    n_new, Fx_new = eval_F(x_new)
                    
                    if not np.isfinite(n_new):
                        break

            if n_new < n:
                # Accept step
                # Broyden update: J_new = J + (Fx_new - Fx - J*dx) * (x_new - x)^T / ||x_new - x||^2
                delta_x = x_new - x
                delta_F = Fx_new - Fx
                denom = np.dot(delta_x, delta_x)
                if denom > 1e-12:
                    rank1 = (delta_F - J @ delta_x) * delta_x.T / denom
                    J = J + rank1
                
                x = x_new
                Fx = Fx_new
                n = n_new
                
                if n <= tol:
                    return x, True
                
                # Solve for next step
                try:
                    dx = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    # Reset Jacobian if it becomes singular
                    h = 1e-7
                    J = np.zeros((2, 2))
                    for j in range(2):
                        e = np.zeros(2)
                        e[j] = h
                        n_plus, Fx_plus = eval_F(x + e)
                        n_minus, Fx_minus = eval_F(x - e)
                        if np.isfinite(n_plus) and np.isfinite(n_minus):
                            J[:, j] = (Fx_plus - Fx_minus) / (2 * h)
                        else:
                            J[:, j] = 1.0
                    try:
                        dx = np.linalg.solve(J, -Fx)
                    except:
                        break
            else:
                # Step rejected, shrink step
                alpha = 0.5 * alpha
                if alpha < 1e-8:
                    break
                x_new = x + alpha * dx
                n_new, Fx_new = eval_F(x_new)
                if not np.isfinite(n_new):
                    break
                # Update x for next iteration even if we don't fully accept? 
                # Usually we just break or try a different direction. 
                # For simplicity, if line search fails, we break.
                break
        
        return x, False

    def newton_polish(x_init, max_iter=10):
        """
        Final polishing with Newton's method.
        """
        x = x_init.copy()
        h = 1e-8
        for _ in range(max_iter):
            if not time_ok():
                break
            n, Fx = eval_F(x)
            if n <= 1e-12:
                return x
            
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                n_plus, Fx_plus = eval_F(x + e)
                n_minus, Fx_minus = eval_F(x - e)
                if np.isfinite(n_plus) and np.isfinite(n_minus):
                    J[:, j] = (Fx_plus - Fx_minus) / (2 * h)
                else:
                    break
            
            try:
                dx = np.linalg.solve(J, -Fx)
                x_new = x + dx
                n_new, _ = eval_F(x_new)
                if np.isfinite(n_new) and n_new < n:
                    x = x_new
                else:
                    break
            except:
                break
        return x

    best_state = {'x': x0.copy(), 'norm': np.inf}
    update_best(x0, best_state)

    # Try Broyden from x0
    if time_ok():
        x_sol, success = broyden_method(x0)
        update_best(x_sol, best_state)
        if best_state['norm'] <= 1e-8:
             # Polish
             x_final = newton_polish(x_sol)
             update_best(x_final, best_state)
             return best_state['x']

    # Try other initial points if x0 didn't work
    scale = max(1.0, np.linalg.norm(x0))
    perturbations = [
        np.array([1.0, 0.0]) * scale,
        np.array([0.0, 1.0]) * scale,
        np.array([1.0, 1.0]) * scale / np.sqrt(2),
        np.array([-1.0, 0.0]) * scale,
        np.array([0.0, -1.0]) * scale,
        np.array([1.0, -1.0]) * scale / np.sqrt(2),
        x0 * 0.5,
        x0 * 2.0,
    ]
    
    for p in perturbations:
        if not time_ok():
            break
        x_init = x0 + p
        x_sol, success = broyden_method(x_init)
        update_best(x_sol, best_state)
        if best_state['norm'] <= 1e-8:
             # Polish
             x_final = newton_polish(x_sol)
             update_best(x_final, best_state)
             return best_state['x']
             
    # If we still have time, try a few random restarts
    for _ in range(5):
        if not time_ok():
            break
        x_init = x0 + np.random.uniform(-scale, scale, size=2)
        x_sol, success = broyden_method(x_init)
        update_best(x_sol, best_state)
        if best_state['norm'] <= 1e-8:
             # Polish
             x_final = newton_polish(x_sol)
             update_best(x_final, best_state)
             return best_state['x']

    # Final polish on best found
    if best_state['norm'] < np.inf and best_state['norm'] > 1e-8 and time_ok():
        x_final = newton_polish(best_state['x'])
        update_best(x_final, best_state)

    return best_state['x'] if np.all(np.isfinite(best_state['x'])) else x0

# EVOLVE-BLOCK-END
