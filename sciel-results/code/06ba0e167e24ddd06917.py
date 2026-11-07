# EVOLVE-BLOCK-START
import numpy as np
import time

def _safe_norm(F, x):
    """Safely compute the norm of F(x), returning inf on failure."""
    try:
        Fx = F(x)
        if not np.all(np.isfinite(Fx)):
            return np.inf
        return np.linalg.norm(Fx)
    except Exception:
        return np.inf

def _levenberg_marquardt(F, x0, max_iter=100, tol=1e-12, mu_init=1e-3, mu_max=1e12):
    """
    Levenberg-Marquardt method for solving F(x) = 0.
    Minimizes ||F(x)||^2.
    Uses 1 function evaluation per iteration after initial Jacobian estimation.
    """
    x = np.array(x0, dtype=float)
    n = len(x)
    
    # Initial Jacobian estimate using forward differences
    J = np.zeros((n, n))
    h = 1e-7
    try:
        Fx = F(x)
        if not np.all(np.isfinite(Fx)):
            return x, np.inf, False
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= tol:
            return x, norm_Fx, True
            
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            Fx_plus = F(x + e)
            if not np.all(np.isfinite(Fx_plus)):
                return x, np.inf, False
            J[:, j] = (Fx_plus - Fx) / h
    except Exception:
        return x, np.inf, False

    mu = mu_init
    lambda_factor = 10.0
    success = False
    
    for _ in range(max_iter):
        try:
            norm_Fx = np.linalg.norm(Fx)
            if norm_Fx <= tol:
                return x, norm_Fx, True
            
            # Compute (J^T J + mu * I) dx = -J^T Fx
            JT = J.T
            JTJ = JT @ J
            JTF = JT @ Fx
            
            # Damped normal equations
            # We solve (J^T J + mu * diag(J^T J)) dx = -J^T Fx
            # Using diag(J^T J) is more robust than I, scaling mu appropriately.
            # However, for simplicity and speed in 2D, we can use I or diag.
            # Let's use diag(J^T J) for better scaling.
            diag_JTJ = np.diag(JTJ)
            # Avoid division by zero in diag
            diag_JTJ_safe = np.where(diag_JTJ > 1e-30, diag_JTJ, 1.0)
            
            A = JTJ + mu * np.diag(diag_JTJ_safe)
            b = -JTF
            
            try:
                dx = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(A, b, rcond=None)[0]
                
            x_new = x + dx
            if not np.all(np.isfinite(x_new)):
                # Step too large or numerical issue, increase mu
                mu *= lambda_factor
                if mu > mu_max:
                    return x, norm_Fx, False
                continue
            
            Fx_new = F(x_new)
            if not np.all(np.isfinite(Fx_new)):
                mu *= lambda_factor
                if mu > mu_max:
                    return x, norm_Fx, False
                continue
            
            norm_Fx_new = np.linalg.norm(Fx_new)
            
            if norm_Fx_new < norm_Fx:
                # Accept step
                x = x_new
                Fx = Fx_new
                # Update Jacobian
                # Simple finite difference update for J
                # This costs n extra calls. 
                # Alternative: Use Broyden update on J? 
                # LM typically re-estimates or uses secant updates.
                # For 2D, re-estimating J costs 2 calls. 
                # Total cost per accepted iter: 1 (eval) + 2 (J est) = 3 calls.
                # This is higher than Broyden (1 call/iter after init) but more robust.
                # Let's try a Broyden-like update to save calls.
                # J_new = J + ( (Fx_new - Fx_old) - J*dx ) * dx^T / (dx^T * dx)
                y = Fx_new - Fx
                Jdx = J @ dx
                delta = y - Jdx
                dx_norm_sq = np.dot(dx, dx)
                if dx_norm_sq > 1e-30:
                    J = J + np.outer(delta, dx) / dx_norm_sq
                else:
                    # If step is tiny, re-estimate
                    pass # Skip update if step is 0
                
                # Decrease mu
                mu = max(mu / lambda_factor, 1e-15)
                success = True
            else:
                # Reject step
                mu *= lambda_factor
                if mu > mu_max:
                    return x, norm_Fx, False

        except Exception:
            return x, norm_Fx if 'norm_Fx' in locals() else np.inf, False

    return x, np.linalg.norm(Fx) if np.all(np.isfinite(Fx)) else np.inf, success

def find_root_system(F, x0):
    """
    Finds a root x such that ||F(x)|| <= 1e-8.
    Strategy: Multi-start Levenberg-Marquardt.
    """
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin for 120s limit
    x0 = np.array(x0, dtype=float)
    
    # 1. Check x0
    norm0 = _safe_norm(F, x0)
    if norm0 <= 1e-8:
        return x0
        
    best_x = x0
    best_norm = norm0
    
    # Generate a set of diverse starting points.
    # We want to cover a range of scales and angles.
    # Since the problem is 2D, we can use a polar grid.
    
    # Radii: Log spaced from small to large
    # Angles: 8 directions
    radii = np.logspace(-3, 3, 15) # 1e-3 to 1e3
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    
    # Create candidate list
    candidates = [x0]
    
    # Add points on grid
    # To save time, we don't evaluate all, we just pass them to LM.
    # LM will do the initial evaluation.
    # We limit the number of starts to avoid timeout.
    # Each LM run might take ~10-50 calls.
    # 15 radii * 8 angles = 120 starts. Too many if each takes 50 calls (6000 calls).
    # We need to be selective or limit total time.
    
    # Let's try a subset first, then expand if time permits.
    # Start with x0 and a few nearby points.
    
    # Sort radii so we try nearby first? Or random?
    # Nearby first is safer for local minima, but far might be needed if x0 is bad.
    # Let's interleave.
    
    starts = []
    # Always include x0
    starts.append(x0)
    
    # Add some structured points
    # We'll generate points and shuffle them slightly to avoid bias, 
    # but keep a deterministic order for reproducibility if needed.
    # Actually, deterministic is better for debugging.
    # Let's try increasing radius.
    
    for r in radii:
        for theta in angles:
            x_try = x0 + r * np.array([np.cos(theta), np.sin(theta)])
            starts.append(x_try)
            
    # We have ~120 starts.
    # We will run LM on each until we find a root or run out of time.
    
    # Optimization: Skip starts that are very close to already tried ones?
    # Not necessary for 2D.
    
    # We'll iterate through starts.
    # To maximize success, we should probably try x0 first, then close, then far.
    
    # Let's run LM.
    # To prevent one bad start from eating all time, we limit iterations per start.
    # But LM converges fast. 100 iters is plenty.
    
    # We can also early-terminate if we find a very good solution.
    
    for x_start in starts:
        if time.time() - start_time > DEADLINE:
            break
            
        # Run LM
        x_sol, norm_sol, success = _levenberg_marquardt(F, x_start, max_iter=50, tol=1e-12)
        
        if success and norm_sol <= 1e-8:
            return x_sol
            
        if norm_sol < best_norm:
            best_norm = norm_sol
            best_x = x_sol
            
        # If we found a very small norm, we might be close. 
        # If best_norm is already < 1e-6, we can stop trying new starts 
        # and just return, or try a few more to be sure.
        # Actually, if we have a valid root, we returned.
        # If best_norm is small but > 1e-8, we might still find a better one.
        # But usually LM converges to the root.
        
        # Heuristic: If best_norm < 1e-4, we are likely in the basin of attraction.
        # We can stop expanding radius and focus on polishing?
        # LM already polishes.
        
    return best_x
# EVOLVE-BLOCK-END
