"""Root finder for 3 fixed 2D systems using homotopy/continuation with Broyden tracking."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Finds x such that ||F(x)|| <= 1e-8.
    Uses homotopy continuation F_t(x) = t*F(x) + (1-t)*(x - x0) for t in [0,1],
    tracking the root path with Broyden's method, then polishes with Newton.
    """
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
    tol = 1e-8
    start_time = time.time()
    time_limit = 130.0  # Safety margin for 150s kill

    def elapsed():
        return time.time() - start_time

    def time_left():
        return time_limit - elapsed()

    best_x = x.copy()
    best_norm = np.linalg.norm(F(x))

    if best_norm <= tol:
        return best_x

    # Homotopy function: G_t(x) = t*F(x) + (1-t)*(x - x0)
    # At t=0: G_0(x) = x - x0, root is x0
    # At t=1: G_1(x) = F(x), root is the desired solution
    def homotopy_G(x, t):
        return t * F(x) + (1.0 - t) * (x - x0)

    # Finite difference Jacobian for homotopy (costs 2 F calls for t>0, 0 for t=0)
    def homotopy_jacobian(x, Gx, t, h=1e-5):
        J = np.zeros((2, 2))
        if t >= 1.0:
            # Pure F Jacobian
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Gx_pert = F(x + e)
                J[:, j] = (Gx_pert - Gx) / h
        elif t <= 0.0:
            # G_0(x) = x - x0, Jacobian is identity
            J = np.eye(2)
        else:
            # G_t(x) = t*F(x) + (1-t)*(x - x0)
            # dG/dx = t*dF/dx + (1-t)*I
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Gx_pert = homotopy_G(x + e, t)
                J[:, j] = (Gx_pert - Gx) / h
        return J

    # Broyden solver for homotopy at fixed t
    def broyden_homotopy(x_start, t, max_iters=30):
        nonlocal best_x, best_norm
        x_local = x_start.copy()
        
        # Compute initial G and Jacobian
        if t >= 1.0:
            Gx = F(x_local)
        else:
            Gx = homotopy_G(x_local, t)
        norm_Gx = np.linalg.norm(Gx)
        
        # Check if this is a good solution for F
        if t >= 0.99:
            Fx_check = F(x_local)
            norm_Fx_check = np.linalg.norm(Fx_check)
            if norm_Fx_check < best_norm:
                best_norm = norm_Fx_check
                best_x = x_local.copy()
            if norm_Fx_check <= tol:
                return x_local
        
        if norm_Gx <= tol:
            return x_local
        
        # Initial Jacobian
        h = 1e-5 * max(1.0, np.linalg.norm(x_local))
        J = homotopy_jacobian(x_local, Gx, t, h)
        
        for i in range(max_iters):
            if time_left() < 2.0:
                break
            
            # Solve J dx = -Gx
            try:
                dx = np.linalg.solve(J, -Gx)
            except np.linalg.LinAlgError:
                try:
                    J_reg = J + 1e-8 * np.eye(2)
                    dx = np.linalg.solve(J_reg, -Gx)
                except np.linalg.LinAlgError:
                    break
            
            # Backtracking line search
            alpha = 1.0
            accepted = False
            for _ in range(8):
                if time_left() < 1.0:
                    break
                x_try = x_local + alpha * dx
                if t >= 1.0:
                    G_try = F(x_try)
                else:
                    G_try = homotopy_G(x_try, t)
                norm_try = np.linalg.norm(G_try)
                
                if norm_try < norm_Gx:
                    # Broyden update
                    s = alpha * dx
                    y = G_try - Gx
                    Js = J @ s
                    denom = s @ s
                    if denom > 1e-14:
                        J += np.outer(y - Js, s) / denom
                    x_local = x_try
                    Gx = G_try
                    norm_Gx = norm_try
                    accepted = True
                    
                    # Track best for F
                    if t >= 0.99:
                        Fx_check = F(x_local)
                        norm_Fx_check = np.linalg.norm(Fx_check)
                        if norm_Fx_check < best_norm:
                            best_norm = norm_Fx_check
                            best_x = x_local.copy()
                        if norm_Fx_check <= tol:
                            return x_local
                    
                    if norm_Gx <= tol:
                        return x_local
                    break
                else:
                    alpha *= 0.5
            
            if not accepted:
                break
        
        return x_local

    # Newton polish for the final F system
    def newton_polish(x_start, max_iters=20):
        nonlocal best_x, best_norm
        x_local = x_start.copy()
        Fx = F(x_local)
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx < best_norm:
            best_norm = norm_Fx
            best_x = x_local.copy()
        if norm_Fx <= tol:
            return x_local
        
        h = 1e-6 * max(1.0, np.linalg.norm(x_local))
        for _ in range(max_iters):
            if time_left() < 1.0:
                break
            if norm_Fx <= tol:
                return x_local
            
            # Fresh Jacobian
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_pert = F(x_local + e)
                J[:, j] = (Fx_pert - Fx) / h
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    J_reg = J + 1e-10 * np.eye(2)
                    dx = np.linalg.solve(J_reg, -Fx)
                except np.linalg.LinAlgError:
                    break
            
            x_try = x_local + dx
            F_try = F(x_try)
            norm_try = np.linalg.norm(F_try)
            
            if norm_try < norm_Fx:
                x_local = x_try
                Fx = F_try
                norm_Fx = norm_try
                if norm_Fx < best_norm:
                    best_norm = norm_Fx
                    best_x = x_local.copy()
            else:
                # Try half step
                x_try = x_local + 0.5 * dx
                F_try = F(x_try)
                norm_try = np.linalg.norm(F_try)
                if norm_try < norm_Fx:
                    x_local = x_try
                    Fx = F_try
                    norm_Fx = norm_try
                    if norm_Fx < best_norm:
                        best_norm = norm_Fx
                        best_x = x_local.copy()
                else:
                    break
        
        return x_local

    # Homotopy continuation: track root from t=0 to t=1
    # Start at x0 where G_0(x0) = 0
    x_current = x0.copy()
    
    # Adaptive t steps: start with larger steps, refine as needed
    t_values = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 1.0]
    
    for t in t_values:
        if time_left() < 5.0:
            break
        
        if t == 0.0:
            x_current = x0.copy()
            continue
        
        # Solve G_t(x) = 0 starting from previous solution
        result = broyden_homotopy(x_current, t, max_iters=30)
        x_current = result
        
        # Check if we've converged for F
        if t >= 0.99:
            Fx_check = F(x_current)
            norm_Fx_check = np.linalg.norm(Fx_check)
            if norm_Fx_check < best_norm:
                best_norm = norm_Fx_check
                best_x = x_current.copy()
            if norm_Fx_check <= tol:
                return best_x
    
    # Final polish
    if best_norm < 1e-4 and time_left() > 3.0:
        newton_polish(best_x, max_iters=20)
    
    # If homotopy didn't work well, try direct Broyden from x0
    if best_norm > tol and time_left() > 5.0:
        # Direct Broyden on F
        x_local = x0.copy()
        Fx = F(x_local)
        norm_Fx = np.linalg.norm(Fx)
        
        if norm_Fx < best_norm:
            best_norm = norm_Fx
            best_x = x_local.copy()
        
        if norm_Fx > tol:
            h = 1e-5 * max(1.0, np.linalg.norm(x_local))
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_pert = F(x_local + e)
                J[:, j] = (Fx_pert - Fx) / h
            
            for i in range(30):
                if time_left() < 2.0:
                    break
                
                try:
                    dx = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    try:
                        J_reg = J + 1e-8 * np.eye(2)
                        dx = np.linalg.solve(J_reg, -Fx)
                    except np.linalg.LinAlgError:
                        break
                
                alpha = 1.0
                accepted = False
                for _ in range(8):
                    if time_left() < 1.0:
                        break
                    x_try = x_local + alpha * dx
                    F_try = F(x_try)
                    norm_try = np.linalg.norm(F_try)
                    
                    if norm_try < norm_Fx:
                        s = alpha * dx
                        y = F_try - Fx
                        Js = J @ s
                        denom = s @ s
                        if denom > 1e-14:
                            J += np.outer(y - Js, s) / denom
                        x_local = x_try
                        Fx = F_try
                        norm_Fx = norm_try
                        accepted = True
                        if norm_Fx < best_norm:
                            best_norm = norm_Fx
                            best_x = x_local.copy()
                        if norm_Fx <= tol:
                            return best_x
                        break
                    else:
                        alpha *= 0.5
                
                if not accepted:
                    break
    
    # Final polish if close
    if best_norm < 1e-4 and time_left() > 2.0:
        newton_polish(best_x, max_iters=15)
    
    return best_x
# EVOLVE-BLOCK-END
