import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8 using a robust hybrid strategy:
    1. Broyden's method with adaptive line search, returning Jacobian for reuse.
    2. Diverse restarts using the last Jacobian estimate to reduce initialization cost.
    3. Nelder-Mead for derivative-free global search.
    4. Forward-difference polish for final convergence.
    """
    start_time = time.time()
    deadline = start_time + 118.0  # 2s safety margin before 120s limit

    def time_ok():
        return time.time() < deadline

    x0 = np.array(x0, dtype=float)
    call_count = 0
    MAX_CALLS = 490  # Stay well under 500

    def F_counted(x):
        nonlocal call_count
        call_count += 1
        return F(x)

    best_x = x0.copy()
    try:
        best_norm = np.linalg.norm(F_counted(x0))
    except:
        best_norm = np.inf

    if best_norm <= 1e-8:
        return x0

    def update_best(x):
        nonlocal best_x, best_norm
        if not np.all(np.isfinite(x)):
            return
        try:
            fx = np.asarray(F_counted(x), dtype=float)
            if fx.shape != (2,) or not np.all(np.isfinite(fx)):
                return
            n = np.linalg.norm(fx)
            if np.isfinite(n) and n < best_norm:
                best_norm = n
                best_x = x.copy()
        except:
            pass

    def broyden(x_init, J_init=None, max_iter=25):
        """
        Broyden's method with backtracking line search.
        Returns (success, J_final) where J_final is the last Jacobian estimate.
        """
        if not time_ok() or call_count >= MAX_CALLS - 10:
            return False, None

        x = np.array(x_init, dtype=float)
        h = 1e-5

        try:
            Fx = np.asarray(F_counted(x), dtype=float)
            if not np.all(np.isfinite(Fx)):
                return False, None
            n = np.linalg.norm(Fx)
            update_best(x)
            if n <= 1e-8:
                return True, None
        except:
            return False, None

        # Initialize Jacobian
        if J_init is not None:
            J = J_init.copy()
        else:
            try:
                Fx1 = np.asarray(F_counted(x + [h, 0]), dtype=float)
                Fx2 = np.asarray(F_counted(x + [0, h]), dtype=float)
                if not (np.all(np.isfinite(Fx1)) and np.all(np.isfinite(Fx2))):
                    return False, None
                J = np.column_stack([(Fx1 - Fx) / h, (Fx2 - Fx) / h])
                # Regularize if ill-conditioned
                try:
                    if np.linalg.cond(J) > 1e10:
                        J = J + np.eye(2) * 1e-6
                except:
                    pass
            except:
                return False, None

        for _ in range(max_iter):
            if not time_ok() or call_count >= MAX_CALLS - 5:
                return False, J

            try:
                try:
                    s = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    s = np.linalg.lstsq(J, -Fx, rcond=None)[0]

                if not np.all(np.isfinite(s)):
                    return False, J

                alpha = 1.0
                success = False
                x_old = x.copy()
                Fx_old = Fx.copy()

                for _ in range(12):
                    x_trial = x + alpha * s
                    if not np.all(np.isfinite(x_trial)):
                        alpha *= 0.5
                        continue
                    Fx_trial = np.asarray(F_counted(x_trial), dtype=float)
                    if not np.all(np.isfinite(Fx_trial)):
                        alpha *= 0.5
                        continue
                    n_trial = np.linalg.norm(Fx_trial)

                    if n_trial < n:
                        x = x_trial
                        Fx = Fx_trial
                        n = n_trial
                        success = True
                        update_best(x)
                        break
                    alpha *= 0.5

                if not success:
                    return False, J

                if n <= 1e-8:
                    return True, J

                # Broyden rank-1 update
                s_actual = x - x_old
                y = Fx - Fx_old
                denom = np.dot(s_actual, s_actual)

                if denom > 1e-14:
                    J = J + np.outer(y - J @ s_actual, s_actual) / denom
                else:
                    return False, J

            except:
                return False, J

        return False, J

    def nelder_mead(x_init, max_iter=50):
        """
        Nelder-Mead minimizer for ||F(x)||^2.
        Returns True if root found.
        """
        if not time_ok() or call_count >= MAX_CALLS - 30:
            return False

        x0_nm = np.array(x_init, dtype=float)
        try:
            f0 = np.linalg.norm(F_counted(x0_nm))**2
        except:
            return False
        update_best(x0_nm)
        if f0 <= 1e-16:
            return True

        delta = 0.1
        simplex = [
            x0_nm,
            x0_nm + [delta, 0],
            x0_nm + [0, delta]
        ]
        fvals = [f0, np.inf, np.inf]

        for i in range(1, 3):
            if not time_ok() or call_count >= MAX_CALLS - 10:
                return False
            try:
                fvals[i] = np.linalg.norm(F_counted(simplex[i]))**2
            except:
                return False
            update_best(simplex[i])
            if fvals[i] <= 1e-16:
                return True

        for _ in range(max_iter):
            if not time_ok() or call_count >= MAX_CALLS - 10:
                return False

            idx = np.argsort(fvals)
            simplex = [simplex[i] for i in idx]
            fvals = [fvals[i] for i in idx]

            if fvals[0] <= 1e-16:
                return True

            centroid = np.mean(simplex[:2], axis=0)
            worst = simplex[2]
            f_worst = fvals[2]

            x_r = centroid + (centroid - worst)
            if not np.all(np.isfinite(x_r)):
                continue
            try:
                f_r = np.linalg.norm(F_counted(x_r))**2
            except:
                continue
            update_best(x_r)
            if f_r <= 1e-16:
                return True

            if f_r < fvals[0]:
                x_e = centroid + 2 * (x_r - centroid)
                if np.all(np.isfinite(x_e)):
                    try:
                        f_e = np.linalg.norm(F_counted(x_e))**2
                    except:
                        f_e = np.inf
                    update_best(x_e)
                    if f_e <= 1e-16:
                        return True
                    if f_e < f_r:
                        simplex[2] = x_e
                        fvals[2] = f_e
                    else:
                        simplex[2] = x_r
                        fvals[2] = f_r
                else:
                    simplex[2] = x_r
                    fvals[2] = f_r
            elif f_r < f_worst:
                simplex[2] = x_r
                fvals[2] = f_r
            else:
                x_c = centroid + 0.5 * (worst - centroid)
                if np.all(np.isfinite(x_c)):
                    try:
                        f_c = np.linalg.norm(F_counted(x_c))**2
                    except:
                        f_c = np.inf
                    update_best(x_c)
                    if f_c <= 1e-16:
                        return True
                    if f_c < f_worst:
                        simplex[2] = x_c
                        fvals[2] = f_c
                    else:
                        for j in range(1, 3):
                            simplex[j] = simplex[0] + 0.5 * (simplex[j] - simplex[0])
                            if not time_ok() or call_count >= MAX_CALLS - 10:
                                return False
                            try:
                                fvals[j] = np.linalg.norm(F_counted(simplex[j]))**2
                            except:
                                fvals[j] = np.inf
                            update_best(simplex[j])
                            if fvals[j] <= 1e-16:
                                return True

        return False

    def polish(x_init, max_iter=50):
        """
        Forward-difference Newton polish for final convergence.
        Uses 2 calls per iteration (F(x) and F(x+h*e_i) for one coordinate at a time).
        Returns True if root found.
        """
        if not time_ok() or call_count >= MAX_CALLS - 10:
            return False

        x = np.array(x_init, dtype=float)
        h = 1e-6

        for _ in range(max_iter):
            if not time_ok() or call_count >= MAX_CALLS - 5:
                return False

            try:
                Fx = np.asarray(F_counted(x), dtype=float)
                if not np.all(np.isfinite(Fx)):
                    return False
                n = np.linalg.norm(Fx)
                update_best(x)
                if n <= 1e-8:
                    return True
            except:
                return False

            # Compute Jacobian using forward differences (2 calls)
            try:
                Fx1 = np.asarray(F_counted(x + [h, 0]), dtype=float)
                Fx2 = np.asarray(F_counted(x + [0, h]), dtype=float)
                if not (np.all(np.isfinite(Fx1)) and np.all(np.isfinite(Fx2))):
                    return False
                J = np.column_stack([(Fx1 - Fx) / h, (Fx2 - Fx) / h])
            except:
                return False

            try:
                s = np.linalg.solve(J, -Fx)
                if not np.all(np.isfinite(s)):
                    return False
                x_new = x + s
                if not np.all(np.isfinite(x_new)):
                    return False
                x = x_new
            except:
                return False

        return False

    # 1. Try Broyden from x0
    success, J_last = broyden(x0)
    if success and best_norm <= 1e-8:
        return best_x

    # 2. Generate diverse restarts
    scale = max(1.0, np.linalg.norm(x0))
    restarts = []
    
    # Origin and axis points
    restarts.append([0.0, 0.0])
    restarts.append([scale, 0.0])
    restarts.append([0.0, scale])
    restarts.append([-scale, 0.0])
    restarts.append([0.0, -scale])
    
    # Small scale perturbations
    for s in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        for dx, dy in [(1,0), (0,1), (1,1), (1,-1), (-1,0), (0,-1), (-1,-1), (-1,1)]:
            restarts.append(x0 + s * np.array([dx, dy]) * scale)
    
    # Circular perturbations
    for i in range(24):
        angle = 2 * np.pi * i / 24
        r = scale * (0.3 + i * 0.08)
        restarts.append(x0 + r * np.array([np.cos(angle), np.sin(angle)]))
    
    # Shuffle to avoid bias
    np.random.seed(42)
    np.random.shuffle(restarts)
    
    # 3. Try Broyden from restarts, reusing J_last if available
    for x_init in restarts:
        if not time_ok() or call_count >= MAX_CALLS - 10:
            break
        if not np.all(np.isfinite(x_init)):
            continue
        # Use J_last from previous Broyden run if we have one
        J_to_use = J_last if J_last is not None else None
        success, J_new = broyden(x_init, J_init=J_to_use)
        if J_new is not None:
            J_last = J_new
        if success and best_norm <= 1e-8:
            return best_x

    # 4. Nelder-Mead from best point and a few restarts
    nm_starts = [best_x]
    for x_init in restarts[:8]:
        if not time_ok() or call_count >= MAX_CALLS - 40:
            break
        if not np.all(np.isfinite(x_init)):
            continue
        nm_starts.append(x_init)
    
    for x_init in nm_starts:
        if not time_ok() or call_count >= MAX_CALLS - 40:
            break
        if nelder_mead(x_init):
            if best_norm <= 1e-8:
                return best_x

    # 5. Polish the best point found
    if time_ok() and call_count < MAX_CALLS - 10:
        if polish(best_x):
            if best_norm <= 1e-8:
                return best_x

    # 6. Final fallback: try a few more polishes from promising points
    if time_ok() and call_count < MAX_CALLS - 10:
        for x_init in restarts[:5]:
            if not time_ok() or call_count >= MAX_CALLS - 5:
                break
            if not np.all(np.isfinite(x_init)):
                continue
            if polish(x_init):
                if best_norm <= 1e-8:
                    return best_x

    return best_x if np.all(np.isfinite(best_x)) else x0
