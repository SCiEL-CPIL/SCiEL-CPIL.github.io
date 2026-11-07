# EVOLVE-BLOCK-START
import numpy as np


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8 using a trust-region dogleg method
    with Broyden-updated Jacobian approximation, plus a Newton polish phase.
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
    fval = 0.5 * np.dot(Fx, Fx)
    if fval <= (1e-8) ** 2:
        return x

    # --- Initial Jacobian via finite differences ---
    def finite_diff_J(z, Fz, h=1e-6):
        J = np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            Fp = fcounted(z + e)
            J[:, j] = (Fp - Fz) / h
        return J

    J = finite_diff_J(x, Fx)
    if not np.all(np.isfinite(J)):
        # Try a different step
        J = finite_diff_J(x, Fx, h=1e-4)
    if not np.all(np.isfinite(J)):
        J = np.eye(n)

    # --- Trust region parameters ---
    delta = 1.0  # initial trust radius
    delta_min = 1e-12
    delta_max = 1e6
    gamma = 0.5
    eta = 0.1  # ratio threshold for accepting a step

    # Broyden update state
    s_last = None
    y_last = None

    for iteration in range(80):
        if call_count[0] >= max_total_calls:
            break

        Fx = fcounted(x)
        if not np.all(np.isfinite(Fx)):
            break
        fval = 0.5 * np.dot(Fx, Fx)
        if fval <= (1e-8) ** 2:
            return x

        # Gradient and Gauss-Newton Hessian approximation
        g = J.T @ Fx
        H = J.T @ J  # Gauss-Newton Hessian approximation

        # Solve H p = -g for the Gauss-Newton step
        try:
            p_gn = np.linalg.solve(H, -g)
            if not np.all(np.isfinite(p_gn)):
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            # Fallback: use pseudoinverse or steepest descent
            try:
                p_gn = np.linalg.lstsq(H, -g, rcond=None)[0]
            except Exception:
                p_gn = -g / (np.linalg.norm(g) + 1e-30)

        # Steepest descent step
        if np.linalg.norm(g) > 1e-30:
            p_sd = -g / np.linalg.norm(g)
        else:
            p_sd = np.zeros(n)

        # Dogleg step: interpolate between p_sd and p_gn
        # If ||p_gn|| <= delta, take p_gn
        # Else, take the point on the dogleg path at distance delta
        gn_norm = np.linalg.norm(p_gn)
        if gn_norm <= delta:
            p = p_gn
        else:
            # Find the point on the line from p_sd to p_gn at distance delta
            # The dogleg path: first go along p_sd to the point closest to p_gn,
            # then go toward p_gn.
            # Project p_gn onto the direction of p_sd
            if np.linalg.norm(p_sd) > 1e-30:
                t = np.dot(p_gn, p_sd) / np.linalg.norm(p_sd)
                # The corner point is t * p_sd (if t > 0)
                corner = t * p_sd if t > 0 else np.zeros(n)
                # Vector from corner to p_gn
                v = p_gn - corner
                v_norm = np.linalg.norm(v)
                if v_norm > 1e-30:
                    # Distance from origin to corner
                    corner_norm = np.linalg.norm(corner)
                    remaining = delta - corner_norm
                    if remaining <= 0:
                        # We're before the corner
                        p = p_sd * (delta / np.linalg.norm(p_sd))
                    else:
                        p = corner + v * (remaining / v_norm)
                else:
                    p = corner
            else:
                p = p_gn * (delta / gn_norm)

        # Predicted reduction
        # f(x+p) ≈ fval + g^T p + 0.5 p^T H p
        pred = fval + np.dot(g, p) + 0.5 * np.dot(p, H @ p)

        # Try the step
        x_new = x + p
        Fx_new = fcounted(x_new)
        if not np.all(np.isfinite(Fx_new)):
            delta *= 0.5
            continue
        fval_new = 0.5 * np.dot(Fx_new, Fx_new)
        actual = fval - fval_new
        predicted = fval - pred
        if predicted <= 0:
            rho = -1.0
        else:
            rho = actual / predicted

        if rho > eta:
            # Accept the step
            # Broyden update
            s = p
            y = (J_new_T_Fx_new) if False else None  # placeholder, compute properly
            # Compute new Jacobian direction: y = J(x_new) * s - J(x) * s ≈ F(x_new) - F(x) - J*s
            # Actually, for Broyden on the Jacobian of F:
            # We want to update J so that J_new * s ≈ F(x_new) - F(x)
            # y = F(x_new) - F(x) - J * s
            y = Fx_new - Fx - J @ s
            # Broyden rank-1 update
            if np.linalg.norm(s) > 1e-15:
                J = J + (y - J @ s) @ s.T / (s.T @ s)
            x = x_new
            Fx = Fx_new
            fval = fval_new
            # Grow trust region
            if rho > 0.75:
                delta = min(2.0 * delta, delta_max)
            else:
                delta = min(1.5 * delta, delta_max)
        else:
            # Reject the step, shrink trust region
            delta *= 0.5
            if delta < delta_min:
                # Try a small perturbation to escape
                x = x + 1e-4 * np.random.randn(n)
                delta = 1e-2

    # --- Polish phase: Newton with finite-difference Jacobian ---
    for _ in range(20):
        if call_count[0] >= max_total_calls:
            break
        Fx = fcounted(x)
        if not np.all(np.isfinite(Fx)):
            break
        if np.linalg.norm(Fx) <= 1e-8:
            return x
        J = finite_diff_J(x, Fx)
        if not np.all(np.isfinite(J)):
            break
        try:
            dx = np.linalg.solve(J, -Fx)
            if not np.all(np.isfinite(dx)):
                break
            x_new = x + dx
            Fx_new = fcounted(x_new)
            if not np.all(np.isfinite(Fx_new)):
                break
            if np.linalg.norm(Fx_new) < np.linalg.norm(Fx):
                x = x_new
            else:
                # Damped step
                for t in [0.5, 0.25, 0.1]:
                    x_trial = x + t * dx
                    Fx_trial = fcounted(x_trial)
                    if np.all(np.isfinite(Fx_trial)) and np.linalg.norm(Fx_trial) < np.linalg.norm(Fx):
                        x = x_trial
                        break
                else:
                    break
        except np.linalg.LinAlgError:
            break

    return x
# EVOLVE-BLOCK-END
