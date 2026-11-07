"""Root finder for 3 fixed 2D systems using hybrid Nelder-Mead + Broyden with restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Finds x in R^2 such that ||F(x)|| <= 1e-8.
    Hybrid approach: Nelder-Mead on ||F||^2 to find basin, then Broyden for polish.
    """
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
    
    tol = 1e-8
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    def time_left():
        return time_limit - (time.time() - start_time)
    
    # Track best solution found
    best_x = x.copy()
    best_norm = np.linalg.norm(F(x))
    
    # Phase 1: Nelder-Mead to find a good basin
    def merit(xv):
        fx = F(xv)
        return np.dot(fx, fx)
    
    # Initial simplex for Nelder-Mead
    delta0 = 0.1
    simplex = np.array([
        x,
        x + [delta0, 0],
        x + [0, delta0]
    ])
    fvals = [merit(simplex[0]), merit(simplex[1]), merit(simplex[2])]
    
    # Nelder-Mead parameters
    alpha = 1.0    # reflection
    gamma = 2.0    # expansion
    rho = 0.5      # contraction
    sigma = 0.5    # shrink
    
    nm_max_iter = 50
    nm_iter = 0
    
    for nm_iter in range(nm_max_iter):
        if time_left() < 10:
            break
        
        # Sort simplex by function value
        idx = np.argsort(fvals)
        simplex = simplex[idx]
        fvals = [fvals[i] for i in idx]
        
        # Check convergence
        if abs(fvals[-1] - fvals[0]) < 1e-12:
            break
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = merit(xr)
        
        if fr < fvals[0]:
            # Try expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = merit(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            if fr < fvals[-1]:
                # Outside contraction
                xc = centroid + rho * (xr - centroid)
                fc = merit(xc)
                if fc <= fr:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, 3):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = merit(simplex[i])
            else:
                # Inside contraction
                xc = centroid - rho * (centroid - simplex[-1])
                fc = merit(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, 3):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = merit(simplex[i])
        
        # Update best
        for i in range(3):
            if fvals[i] < best_norm ** 2:
                best_norm = np.sqrt(fvals[i])
                best_x = simplex[i].copy()
        
        if best_norm <= tol:
            return best_x
    
    # Use the best point from Nelder-Mead as starting point for Broyden
    x_broyden = best_x.copy()
    
    # Phase 2: Broyden's method for high-precision polishing
    # Compute initial Jacobian using forward differences
    def fd_jacobian(xv):
        J = np.zeros((2, 2))
        h = 1e-7
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (F(xv + e) - F(xv)) / h
        return J
    
    Fx = F(x_broyden)
    J = fd_jacobian(x_broyden)
    
    broyden_max_iter = 50
    for b_iter in range(broyden_max_iter):
        if time_left() < 5:
            break
        
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx < best_norm:
            best_norm = norm_Fx
            best_x = x_broyden.copy()
        
        if norm_Fx <= tol:
            return x_broyden
        
        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Regularize
            try:
                JtJ = J.T @ J + 1e-8 * np.eye(2)
                dx = np.linalg.solve(JtJ, -J.T @ Fx)
            except np.linalg.LinAlgError:
                break
        
        # Damped step with line search
        alpha = 1.0
        accepted = False
        for _ in range(15):
            x_try = x_broyden + alpha * dx
            F_try = F(x_try)
            norm_try = np.linalg.norm(F_try)
            
            if norm_try < norm_Fx:
                # Broyden update
                s = x_try - x_broyden
                y = F_try - Fx
                denom = np.dot(y, s)
                if abs(denom) > 1e-14:
                    J = J + np.outer(y - J @ s, s) / denom
                x_broyden = x_try
                Fx = F_try
                accepted = True
                break
            else:
                alpha *= 0.5
        
        if not accepted:
            break
    
    # If Broyden didn't converge, try a few more restarts from random points
    for restart in range(5):
        if time_left() < 10:
            break
        
        # Random restart around best_x or x0
        if np.random.rand() < 0.5:
            x_restart = x0 + np.random.uniform(-5, 5, size=2)
        else:
            radius = 0.5 * (restart + 1)
            dir_vec = np.random.randn(2)
            dir_vec /= np.linalg.norm(dir_vec)
            x_restart = best_x + radius * dir_vec
        
        # Quick Broyden from this point
        Fx_r = F(x_restart)
        norm_r = np.linalg.norm(Fx_r)
        if norm_r >= best_norm:
            continue
        
        J_r = fd_jacobian(x_restart)
        x_b = x_restart.copy()
        Fx_b = Fx_r
        
        for b_iter in range(30):
            if time_left() < 5:
                break
            
            norm_Fx_b = np.linalg.norm(Fx_b)
            if norm_Fx_b < best_norm:
                best_norm = norm_Fx_b
                best_x = x_b.copy()
            
            if norm_Fx_b <= tol:
                return x_b
            
            try:
                dx_b = np.linalg.solve(J_r, -Fx_b)
            except np.linalg.LinAlgError:
                break
            
            alpha_b = 1.0
            accepted_b = False
            for _ in range(10):
                x_try = x_b + alpha_b * dx_b
                F_try = F(x_try)
                norm_try = np.linalg.norm(F_try)
                
                if norm_try < norm_Fx_b:
                    s = x_try - x_b
                    y = F_try - Fx_b
                    denom = np.dot(y, s)
                    if abs(denom) > 1e-14:
                        J_r = J_r + np.outer(y - J_r @ s, s) / denom
                    x_b = x_try
                    Fx_b = F_try
                    accepted_b = True
                    break
                else:
                    alpha_b *= 0.5
            
            if not accepted_b:
                break
    
    return best_x
# EVOLVE-BLOCK-END
