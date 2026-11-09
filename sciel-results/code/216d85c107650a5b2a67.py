import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8 using Broyden's method with
    adaptive restarts and a tight call budget.
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

        x0 = np.array(x_init, dtype=float)
        try:
            f0 = np.linalg.norm(F_counted(x0))**2
        except:
            return False
        update_best(x0)
        if f0 <= 1e-16:
            return True

        delta = 0.1
        simplex = [
            x0,
            x0 + [delta, 0],
            x0 + [0, delta]
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

        return False

    # Phase 1: Broyden from x0
    success, J_last = broyden(x0)
    if success and best_norm <= 1e-8:
        return best_x

    # Phase 2: Broyden from diverse restarts, reusing Jacobian when possible
    scale = max(1.0, np.linalg.norm(x0))
    restarts = []
    for s in [0.5, 1.0, 2.0, 4.0, 8.0]:
        for angle in [0, np.pi/2, np.pi, 3*np.pi/2, np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]:
            dx = s * scale * np.cos(angle)
            dy = s * scale * np.sin(angle)
            restarts.append(x0 + np.array([dx, dy]))

    # Add some random restarts
    np.random.seed(42)
    for _ in range(8):
        r = np.random.uniform(0.5, 10.0) * scale
        theta = np.random.uniform(0, 2 * np.pi)
        restarts.append(x0 + r * np.array([np.cos(theta), np.sin(theta)]))

    # Shuffle for diversity
    np.random.shuffle(restarts)

    for x_init in restarts:
        if not time_ok() or call_count >= MAX_CALLS - 15:
            break
        # Try Broyden with the last known Jacobian as initial guess
        success, _ = broyden(x_init, J_init=J_last if J_last is not None else None)
        if success and best_norm <= 1e-8:
            return best_x

    # Phase 3: Nelder-Mead from best point and a few restarts
    if time_ok() and call_count < MAX_CALLS - 20:
        if nelder_mead(best_x):
            if best_norm <= 1e-8:
                return best_x

        # Try NM from a couple of diverse points
        for x_init in restarts[:4]:
            if not time_ok() or call_count >= MAX_CALLS - 15:
                break
            if nelder_mead(x_init):
                if best_norm <= 1e-8:
                    return best_x

    # Phase 4: Coarse grid search + Broyden polish
    if time_ok() and call_count < MAX_CALLS - 40:
        radius = max(5.0, 3 * scale)
        steps = 6
        dx = 2 * radius / steps

        best_grid_x = best_x
        best_grid_norm = best_norm

        for i in range(steps + 1):
            if not time_ok() or call_count >= MAX_CALLS - 10:
                break
            for j in range(steps + 1):
                if not time_ok() or call_count >= MAX_CALLS - 10:
                    break
                x_g = x0 + np.array([i * dx - radius, j * dx - radius])
                if not np.all(np.isfinite(x_g)):
                    continue
                try:
                    fg = np.asarray(F_counted(x_g), dtype=float)
                    if not np.all(np.isfinite(fg)):
                        continue
                    ng = np.linalg.norm(fg)
                    if ng < best_grid_norm:
                        best_grid_norm = ng
                        best_grid_x = x_g
                        update_best(x_g)
                        if ng <= 1e-8:
                            return best_grid_x
                except:
                    pass

        # Polish the best grid point
        if best_grid_norm < best_norm * 0.5:
            success, _ = broyden(best_grid_x)
            if success and best_norm <= 1e-8:
                return best_x

    return best_x if np.all(np.isfinite(best_x)) else x0
