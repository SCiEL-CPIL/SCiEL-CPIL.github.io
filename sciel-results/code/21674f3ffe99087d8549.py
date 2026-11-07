# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Robust root finder for 2D systems using Broyden's method with
    line search, fallback to full Newton, and time guard.
    """
    deadline = time.time() + 110.0  # safety margin before 120s kill
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)

    # Evaluate F at initial point
    try:
        Fx = np.array(F(x), dtype=float)
    except Exception:
        return x
    if not np.all(np.isfinite(Fx)):
        # Try a small perturbation
        x = x + np.array([1e-4, 1e-4])
        try:
            Fx = np.array(F(x), dtype=float)
        except Exception:
            return x
        if not np.all(np.isfinite(Fx)):
            return x

    fx_norm = np.linalg.norm(Fx)
    if fx_norm <= 1e-12:
        return x

    # Initialize Jacobian via finite differences
    h = 1e-7
    J = np.zeros((2, 2))
    for j in range(2):
        ej = np.zeros(2)
        ej[j] = h
        try:
            Fx_pert = np.array(F(x + ej), dtype=float)
        except Exception:
            Fx_pert = Fx
        if np.all(np.isfinite(Fx_pert)):
            J[:, j] = (Fx_pert - Fx) / h
        else:
            J[:, j] = np.array([1.0, 0.0]) if j == 0 else np.array([0.0, 1.0])

    # Broyden's method with line search
    best_x = x
    best_norm = fx_norm

    for iteration in range(80):
        if time.time() > deadline:
            break

        # Try to solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                break
        except Exception:
            break

        if not np.all(np.isfinite(dx)):
            break

        # Line search: try full step, then halve
        step = 1.0
        found_step = False
        for ls_iter in range(20):
            if time.time() > deadline:
                break
            x_new = x + step * dx
            if not np.all(np.isfinite(x_new)):
                step *= 0.5
                continue
            try:
                Fx_new = np.array(F(x_new), dtype=float)
            except Exception:
                step *= 0.5
                continue
            if not np.all(np.isfinite(Fx_new)):
                step *= 0.5
                continue

            fx_new_norm = np.linalg.norm(Fx_new)
            if fx_new_norm < best_norm:
                best_norm = fx_new_norm
                best_x = x_new.copy()

            # Accept if we made sufficient decrease
            if fx_new_norm < 0.5 * fx_norm or fx_new_norm <= 1e-12:
                # Update Broyden
                s = x_new - x
                y = Fx_new - Fx
                if np.linalg.norm(y) > 1e-14:
                    try:
                        J = J + np.outer(y - J @ s, s) / (s @ s)
                    except Exception:
                        pass
                x = x_new
                Fx = Fx_new
                fx_norm = fx_new_norm
                found_step = True
                break
            else:
                step *= 0.5

        if fx_norm <= 1e-12:
            return x

        if not found_step:
            # If line search failed, try a small random perturbation
            if fx_norm < best_norm * 1.1:
                # Accept current as best and try restarting with perturbation
                x = best_x + np.random.normal(0, 1e-3, 2)
                try:
                    Fx = np.array(F(x), dtype=float)
                except Exception:
                    break
                if not np.all(np.isfinite(Fx)):
                    break
                fx_norm = np.linalg.norm(Fx)
                # Reinitialize Jacobian
                for j in range(2):
                    ej = np.zeros(2)
                    ej[j] = h
                    try:
                        Fx_pert = np.array(F(x + ej), dtype=float)
                    except Exception:
                        Fx_pert = Fx
                    if np.all(np.isfinite(Fx_pert)):
                        J[:, j] = (Fx_pert - Fx) / h
                    else:
                        J[:, j] = np.eye(2)[:, j]
            else:
                # Stuck, try full Newton restart
                x = best_x
                try:
                    Fx = np.array(F(x), dtype=float)
                except Exception:
                    break
                fx_norm = np.linalg.norm(Fx)
                # Recompute Jacobian
                for j in range(2):
                    ej = np.zeros(2)
                    ej[j] = h
                    try:
                        Fx_pert = np.array(F(x + ej), dtype=float)
                    except Exception:
                        Fx_pert = Fx
                    if np.all(np.isfinite(Fx_pert)):
                        J[:, j] = (Fx_pert - Fx) / h
                    else:
                        J[:, j] = np.eye(2)[:, j]

    # Return the best point found
    if np.all(np.isfinite(best_x)):
        return best_x
    return x
# EVOLVE-BLOCK-END
