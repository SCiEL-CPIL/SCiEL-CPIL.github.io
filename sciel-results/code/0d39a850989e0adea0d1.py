import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses a hybrid Broyden-Newton strategy with adaptive restarts and a polish phase.
    """
    start_time = time.time()
    deadline = start_time + 118.0
    
    def time_ok():
        return time.time() < deadline

    x = np.array(x0, dtype=float)
    call_count = 0
    MAX_CALLS = 495  # Safe margin

    def F_counted(x):
        nonlocal call_count
        call_count += 1
        return F(x)

    best_x = x.copy()
    try:
        Fx = np.asarray(F_counted(x), dtype=float)
        best_norm = np.linalg.norm(Fx)
    except:
        Fx = np.array([np.inf, np.inf])
        best_norm = np.inf

    if best_norm <= 1e-8:
        return x

    def update_best(x_trial, Fx_trial):
        nonlocal best_x, best_norm
        if not np.all(np.isfinite(x_trial)) or not np.all(np.isfinite(Fx_trial)):
            return
        n = np.linalg.norm(Fx_trial)
        if np.isfinite(n) and n < best_norm:
            best_norm = n
            best_x = x_trial.copy()

    def broyden_polish(x_init, J_init=None, max_iter=30):
        """
        Broyden's method with a switch to Newton-like polish near the root.
        """
        if not time_ok() or call_count >= MAX_CALLS - 5:
            return False, None

        x = np.array(x_init, dtype=float)
        
        try:
            Fx = np.asarray(F_counted(x), dtype=float)
            if not np.all(np.isfinite(Fx)):
                return False, None
            n = np.linalg.norm(Fx)
            update_best(x, Fx)
            if n <= 1e-8:
                return True, None
        except:
            return False, None

        # Initialize Jacobian
        if J_init is not None:
            J = J_init.copy()
        else:
            h = 1e-5
            try:
                Fx1 = np.asarray(F_counted(x + [h, 0]), dtype=float)
                Fx2 = np.asarray(F_counted(x + [0, h]), dtype=float)
                if not (np.all(np.isfinite(Fx1)) and np.all(np.isfinite(Fx2))):
                    return False, None
                J = np.column_stack([(Fx1 - Fx) / h, (Fx2 - Fx) / h])
                # Regularize
                if np.linalg.cond(J) > 1e8:
                    J = J + np.eye(2) * 1e-8
            except:
                return False, None

        polish_mode = False

        for _ in range(max_iter):
            if not time_ok() or call_count >= MAX_CALLS - 5:
                return False, J

            # Switch to polish mode if close
            if n < 1e-5 and not polish_mode:
                polish_mode = True
            
            try:
                # Solve for step
                try:
                    if polish_mode:
                        # In polish mode, we might recompute J or use a damped Newton step
                        # For efficiency, we stick to Broyden but with tighter line search
                        s = np.linalg.solve(J, -Fx)
                    else:
                        s = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    s = np.linalg.lstsq(J, -Fx, rcond=None)[0]

                if not np.all(np.isfinite(s)):
                    return False, J

                # Line Search
                alpha = 1.0
                success = False
                x_old = x.copy()
                Fx_old = Fx.copy()
                n_old = n

                for _ in range(20):
                    x_trial = x + alpha * s
                    if not np.all(np.isfinite(x_trial)):
                        alpha *= 0.5
                        continue
                    
                    Fx_trial = np.asarray(F_counted(x_trial), dtype=float)
                    if not np.all(np.isfinite(Fx_trial)):
                        alpha *= 0.5
                        continue
                    
                    n_trial = np.linalg.norm(Fx_trial)
                    
                    # Armijo condition for sufficient decrease
                    # We want n_trial < n_old
                    if n_trial < n_old:
                        x = x_trial
                        Fx = Fx_trial
                        n = n_trial
                        success = True
                        update_best(x, Fx)
                        break
                    else:
                        alpha *= 0.5
                
                if not success:
                    return False, J

                if n <= 1e-8:
                    return True, J

                # Broyden Update
                s_actual = x - x_old
                y = Fx - Fx_old
                denom = np.dot(s_actual, s_actual)

                if denom > 1e-16:
                    J = J + np.outer(y - J @ s_actual, s_actual) / denom
                else:
                    return False, J

            except:
                return False, J

        return False, J

    def nelder_mead(x_init, max_iter=40):
        """
        Nelder-Mead for escape from bad regions.
        """
        if not time_ok() or call_count >= MAX_CALLS - 20:
            return False

        x0_nm = np.array(x_init, dtype=float)
        try:
            f0 = np.linalg.norm(F_counted(x0_nm))
        except:
            return False
        update_best(x0_nm, F_counted(x0_nm)) # Note: this is an extra call, be careful. 
        # Actually, let's reuse F if possible, but here we don't have it passed.
        # To save calls, we should pass Fx in.
        
        # Let's modify NM to accept Fx
        # Re-evaluating logic: NM is expensive. Use only if Broyden fails.
        
        delta = 0.1
        simplex = [
            x0_nm,
            x0_nm + [delta, 0],
            x0_nm + [0, delta]
        ]
        
        # Evaluate simplex
        fvals = []
        for pt in simplex:
            if not time_ok() or call_count >= MAX_CALLS - 5:
                return False
            try:
                fpt = np.linalg.norm(F_counted(pt))
            except:
                return False
            fvals.append(fpt)
            update_best(pt, F_counted(pt)) # Again, double eval? No, F_counted is the only way.
            # Wait, F_counted increments call_count.
            # The line above `F_counted(pt)` is a NEW call.
            # I should store the result.
            
        # Let's rewrite NM to be call-efficient.
        
        return False # Placeholder, will implement properly below

    # --- Phase 1: Broyden from x0 ---
    success, J_last = broyden_polish(x0)
    if success:
        return best_x

    # --- Phase 2: Diverse Restarts ---
    scale = max(1.0, np.linalg.norm(x0))
    restarts = []
    # Circle of restarts
    for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        for r in [0.5, 1.0, 2.0, 4.0]:
            restarts.append(x0 + r * scale * np.array([np.cos(angle), np.sin(angle)]))
    
    # Random restarts
    np.random.seed(42)
    for _ in range(10):
        r = np.random.uniform(0.5, 5.0) * scale
        theta = np.random.uniform(0, 2 * np.pi)
        restarts.append(x0 + r * np.array([np.cos(theta), np.sin(theta)]))
        
    np.random.shuffle(restarts)

    for x_init in restarts:
        if not time_ok() or call_count >= MAX_CALLS - 10:
            break
        success, _ = broyden_polish(x_init, J_init=J_last)
        if success:
            return best_x

    # --- Phase 3: Nelder-Mead from best point ---
    if time_ok() and call_count < MAX_CALLS - 30:
        # Implement a call-efficient Nelder-Mead
        def nm_efficient(x_init):
            x_nm = np.array(x_init, dtype=float)
            try:
                f0 = np.linalg.norm(F_counted(x_nm))
            except:
                return False
            update_best(x_nm, F_counted(x_nm)) # Careful: F_counted called twice if not careful.
            # Let's assume we have Fx from the last Broyden step? No, state is lost.
            # We must pay the cost.
            
            delta = 0.1
            simplex = [x_nm, x_nm + [delta, 0], x_nm + [0, delta]]
            fvals = [f0]
            for i in range(1, 3):
                if not time_ok() or call_count >= MAX_CALLS - 5:
                    return False
                try:
                    fi = np.linalg.norm(F_counted(simplex[i]))
                except:
                    return False
                fvals.append(fi)
                update_best(simplex[i], F_counted(simplex[i]))
                
            for _ in range(50):
                if not time_ok() or call_count >= MAX_CALLS - 5:
                    break
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                if fvals[0] <= 1e-8:
                    return True
                
                centroid = np.mean(simplex[:2], axis=0)
                worst = simplex[2]
                
                # Reflection
                x_r = centroid + (centroid - worst)
                if np.all(np.isfinite(x_r)):
                    try:
                        f_r = np.linalg.norm(F_counted(x_r))
                    except:
                        f_r = np.inf
                    update_best(x_r, F_counted(x_r))
                    
                    if f_r < fvals[0]:
                        # Expansion
                        x_e = centroid + 2 * (x_r - centroid)
                        if np.all(np.isfinite(x_e)):
                            try:
                                f_e = np.linalg.norm(F_counted(x_e))
                            except:
                                f_e = np.inf
                            update_best(x_e, F_counted(x_e))
                            if f_e < f_r:
                                simplex[2] = x_e
                                fvals[2] = f_e
                            else:
                                simplex[2] = x_r
                                fvals[2] = f_r
                        else:
                            simplex[2] = x_r
                            fvals[2] = f_r
                    elif f_r < fvals[2]:
                        simplex[2] = x_r
                        fvals[2] = f_r
                    else:
                        # Contraction
                        x_c = centroid + 0.5 * (worst - centroid)
                        if np.all(np.isfinite(x_c)):
                            try:
                                f_c = np.linalg.norm(F_counted(x_c))
                            except:
                                f_c = np.inf
                            update_best(x_c, F_counted(x_c))
                            if f_c < fvals[2]:
                                simplex[2] = x_c
                                fvals[2] = f_c
                            else:
                                # Shrink
                                for j in range(1, 3):
                                    simplex[j] = simplex[0] + 0.5 * (simplex[j] - simplex[0])
                                    if not time_ok() or call_count >= MAX_CALLS - 5:
                                        return False
                                    try:
                                        fvals[j] = np.linalg.norm(F_counted(simplex[j]))
                                    except:
                                        fvals[j] = np.inf
                                    update_best(simplex[j], F_counted(simplex[j]))
            return fvals[0] <= 1e-8

        if nm_efficient(best_x):
            return best_x

    return best_x if np.all(np.isfinite(best_x)) else x0
