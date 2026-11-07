import numpy as np
import time

def find_root_system(F, x0):
    tol = 1e-8
    tol_sq = tol * tol
    start_time = time.time()
    time_limit = 145.0

    def time_ok():
        return time.time() - start_time < time_limit

    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)

    call_count = 0

    def F_count(x_pt):
        nonlocal call_count
        call_count += 1
        return F(x_pt)

    # Initial evaluation
    try:
        Fx = F_count(x)
    except:
        return np.zeros(2)

    if not np.all(np.isfinite(Fx)):
        return x

    norm_sq = float(np.dot(Fx, Fx))
    if norm_sq <= tol_sq:
        return x

    best_x = x.copy()
    best_norm_sq = norm_sq

    def compute_jacobian_fd(x_pt, Fx_pt):
        h = 1e-7
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        try:
            Fx1 = F_count(x_pt + e1)
            Fx2 = F_count(x_pt + e2)
            J = np.array([(Fx1 - Fx_pt) / h, (Fx2 - Fx_pt) / h]).T
            if not np.all(np.isfinite(J)):
                return None
            return J
        except:
            return None

    def broyden_update(J, x_old, x_new, F_old, F_new):
        dx = x_new - x_old
        dF = F_new - F_old
        dx_norm_sq = float(np.dot(dx, dx))
        if dx_norm_sq < 1e-20:
            return J
        # Broyden rank-1 update: J_new = J + (dF - J dx) dx^T / (dx^T dx)
        Jdx = J @ dx
        diff = dF - Jdx
        J_new = J + np.outer(diff, dx) / dx_norm_sq
        if not np.all(np.isfinite(J_new)):
            return J
        return J_new

    # Main solver: Broyden with adaptive step control
    # Start with finite-difference Jacobian
    J = compute_jacobian_fd(x, Fx)
    if J is None:
        J = np.eye(2)

    max_iter = 150
    stagnation = 0
    best_norm_local = norm_sq

    for it in range(max_iter):
        if not time_ok() or call_count > 480:
            break
        if norm_sq <= tol_sq:
            return x

        # Solve J s = -Fx
        try:
            s = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Singular or near-singular: use pseudo-inverse
            try:
                s = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except:
                # Fallback: steepest descent step
                s = -Fx / (np.linalg.norm(Fx) + 1e-15)

        s_norm = np.linalg.norm(s)
        if s_norm < 1e-15:
            break

        # Adaptive step control: start with full step, shrink if needed
        step_scale = 1.0
        accepted = False
        for _try in range(10):
            x_try = x + step_scale * s
            try:
                Fx_try = F_count(x_try)
            except:
                step_scale *= 0.5
                continue

            if not np.all(np.isfinite(Fx_try)):
                step_scale *= 0.5
                continue

            norm_sq_try = float(np.dot(Fx_try, Fx_try))

            if norm_sq_try < norm_sq:
                # Accept
                x_old = x.copy()
                F_old = Fx.copy()
                x = x_try
                Fx = Fx_try
                norm_sq = norm_sq_try
                accepted = True
                break
            else:
                step_scale *= 0.5

        if not accepted:
            stagnation += 1
            if stagnation > 8:
                break
            continue

        stagnation = 0

        # Broyden rank-1 update
        J = broyden_update(J, x_old, x, F_old, Fx)

        # Periodically refresh Jacobian to correct drift
        if it % 8 == 0 and it > 0 and call_count < 460:
            J_new = compute_jacobian_fd(x, Fx)
            if J_new is not None:
                J = J_new

        if norm_sq < best_norm_sq:
            best_norm_sq = norm_sq
            best_x = x.copy()

    if norm_sq <= tol_sq:
        return x

    # If main loop didn't converge, try a few restarts with Broyden
    if call_count < 450 and time_ok():
        rng = np.random.default_rng(12345)
        for restart in range(5):
            if call_count > 470 or not time_ok():
                break

            # Random restart around current best
            x_r = best_x + rng.normal(0, 1.0, size=2)
            try:
                Fx_r = F_count(x_r)
            except:
                continue
            if not np.all(np.isfinite(Fx_r)):
                continue

            nr = float(np.dot(Fx_r, Fx_r))
            if nr < best_norm_sq:
                best_norm_sq = nr
                best_x = x_r.copy()
                if nr <= tol_sq:
                    return best_x

            # Run Broyden from x_r
            x_cur = x_r
            Fx_cur = Fx_r
            norm_cur = nr
            J_cur = compute_jacobian_fd(x_cur, Fx_cur)
            if J_cur is None:
                J_cur = np.eye(2)

            for it2 in range(60):
                if call_count > 480 or not time_ok():
                    break
                if norm_cur <= tol_sq:
                    return x_cur

                try:
                    s2 = np.linalg.solve(J_cur, -Fx_cur)
                except np.linalg.LinAlgError:
                    try:
                        s2 = np.linalg.lstsq(J_cur, -Fx_cur, rcond=None)[0]
                    except:
                        s2 = -Fx_cur / (np.linalg.norm(Fx_cur) + 1e-15)

                s2_norm = np.linalg.norm(s2)
                if s2_norm < 1e-15:
                    break

                step_scale2 = 1.0
                acc2 = False
                for _t2 in range(8):
                    x_t2 = x_cur + step_scale2 * s2
                    try:
                        Fx_t2 = F_count(x_t2)
                    except:
                        step_scale2 *= 0.5
                        continue
                    if not np.all(np.isfinite(Fx_t2)):
                        step_scale2 *= 0.5
                        continue
                    nt2 = float(np.dot(Fx_t2, Fx_t2))
                    if nt2 < norm_cur:
                        x_old2 = x_cur.copy()
                        F_old2 = Fx_cur.copy()
                        x_cur = x_t2
                        Fx_cur = Fx_t2
                        norm_cur = nt2
                        acc2 = True
                        break
                    else:
                        step_scale2 *= 0.5

                if not acc2:
                    break

                J_cur = broyden_update(J_cur, x_old2, x_cur, F_old2, Fx_cur)

                if it2 % 8 == 0 and it2 > 0 and call_count < 460:
                    J_new2 = compute_jacobian_fd(x_cur, Fx_cur)
                    if J_new2 is not None:
                        J_cur = J_new2

                if norm_cur < best_norm_sq:
                    best_norm_sq = norm_cur
                    best_x = x_cur.copy()

    return best_x
