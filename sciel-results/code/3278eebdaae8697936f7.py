import numpy as np
import time

def find_root_system(F, x0):
    x = np.array(x0, dtype=float)
    calls = 0
    t_start = time.time()
    
    # Helper to call F safely
    def safe_F(x):
        nonlocal calls
        if calls >= 490:
            return None
        try:
            y = F(x)
            y = np.asarray(y, dtype=float)
            calls += 1
            if y.shape != (2,) or not np.all(np.isfinite(y)):
                return None
            return y
        except Exception:
            return None

    Fx = safe_F(x)
    if Fx is None:
        return x
        
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-10:
        return x

    # Initial FD Jacobian
    J = np.zeros((2, 2))
    h = 1e-6
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_plus = safe_F(x + e)
        if Fx_plus is None:
            return x
        J[:, j] = (Fx_plus - Fx) / h
        
    x_old = x.copy()
    Fx_old = Fx.copy()
    best_x = x.copy()
    best_norm = norm_Fx
    
    max_iter = 450
    for _ in range(max_iter):
        if time.time() - t_start > 110:
            break
        if calls >= 490:
            break
            
        # Solve for step
        try:
            d = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Recompute J
            Fx = safe_F(x)
            if Fx is None: break
            norm_Fx = np.linalg.norm(Fx)
            if norm_Fx < best_norm:
                best_x = x.copy()
                best_norm = norm_Fx
            if norm_Fx <= 1e-10: return x
            
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = 1e-6
                Fx_plus = safe_F(x + e)
                if Fx_plus is None: break
                J[:, j] = (Fx_plus - Fx) / h
            if Fx_plus is None: break # exhausted calls
            x_old = x.copy()
            Fx_old = Fx.copy()
            continue

        # Line search
        alpha = 1.0
        accepted = False
        while alpha > 1e-8:
            x_new = x + alpha * d
            Fx_new = safe_F(x_new)
            if Fx_new is None:
                break
            norm_new = np.linalg.norm(Fx_new)
            
            if norm_new < best_norm:
                best_x = x_new.copy()
                best_norm = norm_new
                
            if norm_new <= 1e-10:
                return x_new
                
            # Accept if sufficient decrease
            if norm_new < norm_Fx * (1 - 1e-4 * alpha):
                # Broyden update
                s = x_new - x
                y = Fx_new - Fx
                denom = np.dot(s, s)
                if denom > 1e-12:
                    J = J + np.outer(y - J @ s, s) / denom
                    
                x = x_new
                Fx = Fx_new
                norm_Fx = norm_new
                x_old = x_old # not strictly needed for Broyden, but for logic
                Fx_old = Fx_old
                accepted = True
                break
            else:
                alpha *= 0.5
                
        if not accepted:
            # Step failed, recompute Jacobian
            # We already have Fx_new from the last failed attempt? No, we might have exited loop with None.
            # Let's just recompute from current x.
            Fx = safe_F(x)
            if Fx is None: break
            norm_Fx = np.linalg.norm(Fx)
            if norm_Fx < best_norm:
                best_x = x.copy()
                best_norm = norm_Fx
            if norm_Fx <= 1e-10: return x
            
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = 1e-6
                Fx_plus = safe_F(x + e)
                if Fx_plus is None: break
                J[:, j] = (Fx_plus - Fx) / h
            if Fx_plus is None: break
            x_old = x.copy()
            Fx_old = Fx.copy()

    return best_x
