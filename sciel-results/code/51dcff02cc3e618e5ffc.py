# EVOLVE-BLOCK-START
import numpy as np


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8 using a trust-region dogleg method
    with Broyden-updated Jacobian approximation, followed by a Broyden-Newton polish.
    """
    x = np.array(x0, dtype=float, copy=True)
    n = 2
    max_total_calls = 480
    call_count = [0]

    def fcounted(z):
        call_count[0] += 1
        out = F(z)
        out = np.asarray(out, dtype=float)
        return out

    # --- Initial evaluation ---
    Fx = fcounted(x)
    if not np.all(np.isfinite(Fx)):
        return x
    fval = np.dot(Fx, Fx)
    if fval <= (1e-8) ** 2:
        return x

    # --- Initial Jacobian via finite differences ---
    def finite_diff_J(z, Fz, h=1e-7):
        J = np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            Fp = fcounted(z + e)
            J[:, j] = (Fp - Fz) / h
        return J

    J = finite_diff_J(x, Fx)
    if not np.all(np.isfinite(J)):
        J = finite_diff_J(x, Fx, h=1e-5)
    if not np.all(np.isfinite(J)):
        J = np.eye(n)

    # --- Trust Region Parameters ---
    delta = 1.0
    delta_min = 1e-14
    delta_max = 1e8
    eta = 0.1
    gamma = 0.25

    # Broyden update helper
    def broyden_update(J, Fx_old, Fx_new, s):
        y = Fx_new - Fx_old
        Js = J @ s
        diff = y - Js
        sTs = s @ s
        if sTs > 1e-30:
            J = J + np.outer(diff, s) / sTs
        return J

    # --- Phase 1: Trust Region Dogleg ---
    for iteration in range(80):
        if call_count[0] >= max_total_calls:
            break

        fval = np.dot(Fx, Fx)
        if fval <= (1e-8) ** 2:
            return x

        # Gradient and Gauss-Newton Hessian
        g = J.T @ Fx
        H = J.T @ J

        # Solve H p = -g for Gauss-Newton step
        try:
            p_gn = np.linalg.solve(H, -g)
            if not np.all(np.isfinite(p_gn)):
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            try:
                p_gn = np.linalg.lstsq(H, -g, rcond=None)[0]
            except Exception:
                p_gn = -g / (np.linalg.norm(g) + 1e-30)

        # Steepest Descent direction (unit norm)
        g_norm = np.linalg.norm(g)
        if g_norm > 1e-30:
            p_sd = -g / g_norm
        else:
            p_sd = np.zeros(n)

        # Correct Dogleg Step
        gn_norm = np.linalg.norm(p_gn)
        if gn_norm <= delta:
            p = p_gn
        else:
            if g_norm > 1e-30:
                # Dot product of p_gn with p_sd (unit vector)
                t = np.dot(p_gn, p_sd)
                if t <= 0:
                    # p_gn points opposite to steepest descent; use steepest descent
                    p = p_sd * delta
                else:
                    # Corner point: t * p_sd
                    corner = t * p_sd
                    corner_norm = np.linalg.norm(corner)
                    if corner_norm >= delta:
                        # Delta is before the corner; use steepest descent
                        p = p_sd * delta
                    else:
                        # Beyond corner; interpolate between corner and p_gn
                        v = p_gn - corner
                        v_norm = np.linalg.norm(v)
                        if v_norm < 1e-30:
                            p = corner
                        else:
                            remaining = delta - corner_norm
                            p = corner + v * (remaining / v_norm)
            else:
                p = p_gn * (delta / gn_norm)

        # Predicted Reduction
        pred = fval + np.dot(g, p) + 0.5 * np.dot(p, H @ p)

        # Try Step
        x_new = x + p
        Fx_new = fcounted(x_new)
        if not np.all(np.isfinite(Fx_new)):
            delta *= 0.5
            if delta < delta_min:
                # Escape: try a random direction
                rand_dir = np.random.randn(n)
                rand_dir /= np.linalg.norm(rand_dir)
                x_new = x + delta_min * rand_dir
                Fx_new = fcounted(x_new)
                if not np.all(np.isfinite(Fx_new)):
                    break
                fval_new = np.dot(Fx_new, Fx_new)
                if fval_new < fval:
                    J = broyden_update(J, Fx, Fx_new, x_new - x)
                    x = x_new
                    Fx = Fx_new
                    fval = fval_new
                    delta = 1e-2
                else:
                    break
            continue

        fval_new = np.dot(Fx_new, Fx_new)
        actual = fval - fval_new
        predicted = fval - pred

        if predicted <= 0:
            rho = -1.0
        else:
            rho = actual / predicted

        if rho > eta:
            # Accept
            s = x_new - x
            J = broyden_update(J, Fx, Fx_new, s)
            x = x_new
            Fx = Fx_new
            fval = fval_new

            if rho > 0.75:
                delta = min(2.0 * delta, delta_max)
            elif rho > 0.5:
                delta = min(1.5 * delta, delta_max)
            else:
                delta = min(1.25 * delta, delta_max)
        else:
            # Reject: shrink delta
            delta *= 0.5
            if delta < delta_min:
                # Escape strategy
                rand_dir = np.random.randn(n)
                rand_dir /= np.linalg.norm(rand_dir)
                x_new = x + 1e-3 * rand_dir
                Fx_new = fcounted(x_new)
                if np.all(np.isfinite(Fx_new)):
                    fval_new = np.dot(Fx_new, Fx_new)
                    if fval_new < fval:
                        J = broyden_update(J, Fx, Fx_new, x_new - x)
                        x = x_new
                        Fx = Fx_new
                        fval = fval_new
                        delta = 1e-2
                    else:
                        break
                else:
                    break

    # --- Phase 2: Newton Polish with Broyden Jacobian ---
    for _ in range(30):
        if call_count[0] >= max_total_calls:
            break

        fval = np.dot(Fx, Fx)
        if fval <= (1e-8) ** 2:
            return x

        # Try Newton step with current J
        try:
            dx = np.linalg.solve(J, -Fx)
            if not np.all(np.isfinite(dx)):
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            # If J is singular or bad, recompute via finite diff
            if call_count[0] + 2 <= max_total_calls:
                J = finite_diff_J(x, Fx)
            else:
                break
            if not np.all(np.isfinite(J)):
                break
            try:
                dx = np.linalg.solve(J, -Fx)
                if not np.all(np.isfinite(dx)):
                    break
            except np.linalg.LinAlgError:
                break

        # Backtracking line search
        accepted = False
        t = 1.0
        for _ls in range(15):
            x_new = x + t * dx
            Fx_new = fcounted(x_new)
            if not np.all(np.isfinite(Fx_new)):
                t *= 0.5
                continue

            fval_new = np.dot(Fx_new, Fx_new)
            if fval_new < fval * (1 - gamma * t * np.dot(dx, J.T @ Fx)):
                # Sufficient decrease (Armijo-like)
                s = t * dx
                J = broyden_update(J, Fx, Fx_new, s)
                x = x_new
                Fx = Fx_new
                fval = fval_new
                accepted = True
                break
            t *= 0.5

        if not accepted:
            break

    return x
# EVOLVE-BLOCK-END
