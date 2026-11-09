"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Strategy: Stochastic Global Search (Simulated Annealing) + Newton Polish
    
    1. The problem is a 2D root finding with a tight tolerance (1e-8).
    2. Deterministic methods (Newton, Homotopy) can fail if the initial guess x0 
       is far from the root or if the landscape is complex (multiple roots, narrow basins).
    3. We have a budget of ~150s, which allows for thousands of function evaluations.
    4. Strategy:
       a. Use Simulated Annealing (SA) to minimize ||F(x)||^2. This is a global 
          optimization heuristic that can escape local minima of the residual norm.
       b. SA uses a cooling schedule to balance exploration (early) and exploitation (late).
       c. Periodically (or when stuck), run a fast Damped Newton polish from the current 
          best point to see if we can converge to the high-precision root.
       d. If Newton converges to ||F(x)|| < 1e-8, return immediately.
       e. If time runs out, return the best point found.
    """
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    def time_left():
        return deadline - time.time()
        
    def norm(vec):
        return np.linalg.norm(vec)
    
    def is_good(x):
        if not np.all(np.isfinite(x)):
            return False
        try:
            Fx = F(x)
            if not np.all(np.isfinite(Fx)):
                return False
            return norm(Fx) <= 1e-8
        except Exception:
            return False
            
    def finite_diff_jac(F, x, eps=1e-7):
        """
        Compute Jacobian using central differences for better accuracy.
        Requires 4 function calls.
        """
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = eps
            try:
                J[:, j] = (F(x + e) - F(x - e)) / (2 * eps)
            except Exception:
                raise
        return J
        
    def newton_damped(F, x0, max_iter=20, tol=1e-12):
        """
        Fast Damped Newton. Uses central difference Jacobian.
        Returns (x, converged, calls_used).
        """
        x = np.array(x0, dtype=float)
        calls = 0
        best_x = x.copy()
        best_norm = np.inf
        
        for i in range(max_iter):
            if time_left() < 0.1:
                break
                
            try:
                Fx = F(x)
                calls += 1
            except Exception:
                break
                
            if not np.all(np.isfinite(Fx)):
                break
                
            current_norm = norm(Fx)
            if current_norm < best_norm:
                best_norm = current_norm
                best_x = x.copy()
                
            if current_norm <= tol:
                return x, True, calls
                
            try:
                J = finite_diff_jac(F, x)
                calls += 4
            except Exception:
                break
                
            if not np.all(np.isfinite(J)):
                break
                
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
                
            if not np.all(np.isfinite(dx)):
                break
                
            # Damped step: try full step, then halves
            step_found = False
            for lam in [1.0, 0.5, 0.25, 0.125]:
                x_new = x + lam * dx
                try:
                    F_new = F(x_new)
                    calls += 1
                except Exception:
                    continue
                    
                if not np.all(np.isfinite(F_new)):
                    continue
                    
                new_norm = norm(F_new)
                if new_norm < current_norm:
                    x = x_new
                    step_found = True
                    break
                    
            if not step_found:
                # If we can't make progress, break and return best
                break
                
            if norm(dx) < 1e-12:
                break
                
        return best_x, False, calls

    def simulated_annealing(F, x0, max_iter=5000):
        """
        Simulated Annealing to minimize ||F(x)||^2.
        Returns (best_x, best_val, calls_used).
        """
        x = np.array(x0, dtype=float)
        calls = 0
        
        try:
            Fx = F(x)
            calls += 1
            if not np.all(np.isfinite(Fx)):
                Fx = np.array([1e10, 1e10])
            current_val = np.dot(Fx, Fx)
        except Exception:
            current_val = 1e20
            Fx = np.array([1e10, 1e10])
            
        best_x = x.copy()
        best_val = current_val
        
        # Initial temperature
        T = 1.0
        cooling_rate = 0.99
        # Step size scale
        step_scale = max(1.0, norm(x0)) * 0.5
        
        for i in range(max_iter):
            if time_left() < 1.0:
                break
                
            # Propose a new point
            # Gaussian perturbation
            delta = np.random.normal(0, step_scale, size=2)
            x_new = x + delta
            
            try:
                F_new = F(x_new)
                calls += 1
            except Exception:
                continue
                
            if not np.all(np.isfinite(F_new)):
                continue
                
            new_val = np.dot(F_new, F_new)
            
            # Acceptance criterion
            if new_val < current_val:
                accept = True
            else:
                delta_e = new_val - current_val
                # Metropolis criterion
                if T > 1e-12:
                    accept = np.random.random() < np.exp(-delta_e / T)
                else:
                    accept = False
                    
            if accept:
                x = x_new
                current_val = new_val
                if new_val < best_val:
                    best_val = new_val
                    best_x = x.copy()
                    
                    # If we are close, try Newton immediately
                    if new_val < 1e-4: # ||F|| < 0.01
                         x_pol, conv, n_calls = newton_damped(F, x, max_iter=10)
                         calls += n_calls
                         if conv:
                             return x_pol, 0.0, calls
                         # Update best if Newton improved it
                         try:
                             F_pol = F(x_pol)
                             calls += 1
                             v_pol = np.dot(F_pol, F_pol)
                             if v_pol < best_val:
                                 best_val = v_pol
                                 best_x = x_pol.copy()
                         except Exception:
                             pass
            
            # Cool down
            T *= cooling_rate
            # Shrink step size slightly as we cool
            step_scale *= 0.999
            
        return best_x, best_val, calls

    # --- Main Execution ---
    
    x0 = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x0)):
        x0 = np.array([0.0, 0.0])
        
    # 1. Quick Newton try from x0
    x_newton, conv, calls = newton_damped(F, x0, max_iter=10)
    if conv:
        return x_newton
        
    best_x = x_newton
    best_val = np.inf
    try:
        F_best = F(best_x)
        calls += 1
        if np.all(np.isfinite(F_best)):
            best_val = np.dot(F_best, F_best)
    except Exception:
        best_val = 1e20

    # 2. Simulated Annealing with multiple restarts if needed
    # We will run SA for a while. If it finds a good spot, Newton will finish it.
    # If it stagnates, we might restart SA from a different point.
    
    rng = np.random.default_rng(42)
    
    # Set up SA parameters
    # We want to use most of the time.
    # Estimate max_iter based on time? No, let it run until time_left < 1.0
    
    # Initial SA run
    x_sa, val_sa, calls_sa = simulated_annealing(F, x0, max_iter=100000)
    calls += calls_sa
    
    if val_sa < best_val:
        best_val = val_sa
        best_x = x_sa
        
    # Check if SA found a root
    if is_good(best_x):
        return best_x
        
    # If SA didn't find a root, try a few random restarts of SA if time permits
    # This helps if the first SA got stuck in a local minimum of ||F||^2 that isn't the global root.
    
    restarts = 5
    for r in range(restarts):
        if time_left() < 5.0:
            break
            
        # Generate a random restart point
        # Scale based on initial x0 or a fixed large radius
        scale = max(1.0, norm(x0)) * 5.0
        # Random point in a large box
        x_restart = rng.uniform(-scale, scale, size=2)
        
        x_sa, val_sa, calls_sa = simulated_annealing(F, x_restart, max_iter=50000)
        calls += calls_sa
        
        if val_sa < best_val:
            best_val = val_sa
            best_x = x_sa
            
        if is_good(best_x):
            return best_x

    # Final polish on the best candidate found
    if time_left() > 1.0:
        x_final, conv, calls_final = newton_damped(F, best_x, max_iter=50)
        calls += calls_final
        if conv:
            return x_final
        
        # Update best if final polish improved it
        try:
            F_final = F(x_final)
            calls += 1
            if np.all(np.isfinite(F_final)):
                v_final = np.dot(F_final, F_final)
                if v_final < best_val:
                    best_val = v_final
                    best_x = x_final
        except Exception:
            pass

    return best_x
# EVOLVE-BLOCK-END
