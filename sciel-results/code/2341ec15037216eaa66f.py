import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8 using a hybrid strategy:
    1. Broyden's method (fast, Jacobian-based).
    2. Nelder-Mead (robust, derivative-free, minimizes ||F(x)||^2).
    3. Sparse Grid Search + Polish.
    """
    start_time = time.time()
    deadline = start_time + 115.0

    def time_ok():
        return time.time() < deadline

    x0 = np.array(x0, dtype=float)
    call_count = 0
    MAX_CALLS = 480

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

    def broyden_method(x_init, max_iter=20, max_line_search=10):
        """
        Broyden's method with backtracking line search.
        Returns True if a root is found (||F(x)|| <= 1e-8).
        """
        nonlocal call_count
        if not time_ok() or call_count >= MAX_CALLS - 5:
            return False

        x = np.array(x_init, dtype=float)
        h = 1e-6

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

        # Compute initial Jacobian
        try:
            Fx1 = np.asarray(F_counted(x + [h, 0]), dtype=float)
            Fx2 = np.asarray(F_counted(x + [0, h]), dtype=float)
            if not (np.all(np.isfinite(Fx1)) and np.all(np.isfinite(Fx2))):
                return False
            J = np.column_stack([(Fx1 - Fx) / h, (Fx2 - Fx) / h])

            # Regularize if singular
            try:
                cond = np.linalg.cond(J)
                if cond > 1e12:
                    J = J + np.eye(2) * 1e-8
            except:
                pass
        except:
            return False

        for i in range(max_iter):
            if not time_ok() or call_count >= MAX_CALLS - 2:
                return False

            try:
                # Solve J * s = -Fx
                try:
                    s = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    s = np.linalg.lstsq(J, -Fx, rcond=None)[0]

                if not np.all(np.isfinite(s)):
                    return False

                # Adaptive step size: try full step, then halve if norm increases
                alpha = 1.0
                success = False
                x_new = x
                Fx_new = Fx
                n_new = n

                for _ in range(max_line_search):
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
                        x_new = x_trial
                        Fx_new = Fx_trial
                        n_new = n_trial
                        success = True
                        update_best(x_new)
                        break
                    alpha *= 0.5

                if not success:
                    return False

                if n_new <= 1e-8:
                    return True

                # Broyden Update
                s_actual = x_new - x
                y = Fx_new - Fx
                denom = np.dot(s_actual, s_actual)

                if denom > 1e-12:
                    J = J + np.outer(y - J @ s_actual, s_actual) / denom
                else:
                    return False

                x = x_new
                Fx = Fx_new
                n = n_new
            except:
                return False
        return False

    def nelder_mead(x_init, max_iter=50, tol=1e-10):
        """
        Nelder-Mead simplex method minimizing ||F(x)||^2.
        Returns True if a root is found.
        """
        nonlocal call_count
        if not time_ok() or call_count >= MAX_CALLS - 10:
            return False

        # Initialize simplex
        h = 0.1
        x0 = np.array(x_init, dtype=float)
        simplex = [x0, x0 + [h, 0], x0 + [0, h]]
        fvals = []
        
        for x in simplex:
            if not time_ok() or call_count >= MAX_CALLS - 5:
                return False
            try:
                fx = np.asarray(F_counted(x), dtype=float)
                if not np.all(np.isfinite(fx)):
                    return False
                fvals.append(np.dot(fx, fx))
            except:
                return False

        for i in range(max_iter):
            if not time_ok() or call_count >= MAX_CALLS - 5:
                return False

            # Sort simplex by function value
            idx = np.argsort(fvals)
            simplex = [simplex[j] for j in idx]
            fvals = [fvals[j] for j in idx]

            # Check convergence
            if fvals[0] <= 1e-16:  # ||F||^2 <= 1e-16 means ||F|| <= 1e-8
                return True

            # Centroid of best 2 points
            centroid = np.mean(simplex[:2], axis=0)

            # Reflection
            alpha = 1.0
            x_ref = centroid + alpha * (centroid - simplex[2])
            try:
                fx_ref = np.asarray(F_counted(x_ref), dtype=float)
                if np.all(np.isfinite(fx_ref)):
                    f_ref = np.dot(fx_ref, fx_ref)
                else:
                    f_ref = np.inf
            except:
                f_ref = np.inf

            if f_ref < fvals[0]:
                # Expansion
                gamma = 2.0
                x_exp = centroid + gamma * (centroid - simplex[2])
                try:
                    fx_exp = np.asarray(F_counted(x_exp), dtype=float)
                    if np.all(np.isfinite(fx_exp)):
                        f_exp = np.dot(fx_exp, fx_exp)
                    else:
                        f_exp = np.inf
                except:
                    f_exp = np.inf

                if f_exp < f_ref:
                    simplex[2] = x_exp
                    fvals[2] = f_exp
                else:
                    simplex[2] = x_ref
                    fvals[2] = f_ref
            elif f_ref < fvals[1]:
                simplex[2] = x_ref
                fvals[2] = f_ref
            else:
                # Contraction
                beta = 0.5
                x_con = centroid + beta * (simplex[2] - centroid)
                try:
                    fx_con = np.asarray(F_counted(x_con), dtype=float)
                    if np.all(np.isfinite(fx_con)):
                        f_con = np.dot(fx_con, fx_con)
                    else:
                        f_con = np.inf
                except:
                    f_con = np.inf

                if f_con < f_ref:
                    simplex[2] = x_con
                    fvals[2] = f_con
                else:
                    # Shrink
                    for j in range(1, 3):
                        simplex[j] = simplex[0] + 0.5 * (simplex[j] - simplex[0])
                        try:
                            fx_sh = np.asarray(F_counted(simplex[j]), dtype=float)
                            if np.all(np.isfinite(fx_sh)):
                                fvals[j] = np.dot(fx_sh, fx_sh)
                            else:
                                fvals[j] = np.inf
                        except:
                            fvals[j] = np.inf

            # Update best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_norm**2:
                best_norm = np.sqrt(fvals[min_idx])
                best_x = simplex[min_idx].copy()

        return best_norm <= 1e-8

    # 1. Try Broyden from x0
    if broyden_method(x0):
        if best_norm <= 1e-8:
            return best_x

    # 2. Try Nelder-Mead from x0
    if nelder_mead(x0):
        if best_norm <= 1e-8:
            return best_x

    # 3. Generate diverse restarts
    scale = max(1.0, np.linalg.norm(x0))
    restarts = []
    for s in [0.5, 1.0, 2.0, 5.0, 10.0]:
        for dx, dy in [(1,0), (0,1), (1,1), (1,-1), (-1,0), (0,-1), (-1,-1), (-1,1)]:
            restarts.append(x0 + s * np.array([dx, dy]) * scale)

    # Add some random-ish perturbations based on x0
    for i in range(20):
        angle = 2 * np.pi * i / 20
        r = scale * (0.5 + i * 0.1)
        restarts.append(x0 + r * np.array([np.cos(angle), np.sin(angle)]))

    # Shuffle to avoid bias
    np.random.seed(42)
    np.random.shuffle(restarts)

    for x_init in restarts:
        if not time_ok() or call_count >= MAX_CALLS - 10:
            break
        if not np.all(np.isfinite(x_init)):
            continue
        if broyden_method(x_init):
            if best_norm <= 1e-8:
                return best_x

    # 4. Try Nelder-Mead on best few restart points if Broyden failed
    for x_init in restarts[:5]:
        if not time_ok() or call_count >= MAX_CALLS - 10:
            break
        if not np.all(np.isfinite(x_init)):
            continue
        if nelder_mead(x_init):
            if best_norm <= 1e-8:
                return best_x

    # 5. Fallback: Grid search if still not found
    if time_ok() and call_count < MAX_CALLS - 50:
        center = x0
        radius = max(10.0, 5 * np.linalg.norm(x0))
        steps = 8
        dx = 2 * radius / steps

        for i in range(steps + 1):
            for j in range(steps + 1):
                if not time_ok() or call_count >= MAX_CALLS - 5:
                    break
                x_g = center + np.array([i * dx - radius, j * dx - radius])
                if broyden_method(x_g):
                    if best_norm <= 1e-8:
                        return best_x

    return best_x if np.all(np.isfinite(best_x)) else x0
