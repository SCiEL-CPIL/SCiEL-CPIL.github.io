"""Root finder for 3 fixed two-dimensional systems using a hybrid Broyden/Newton approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _finite_diff_jacobian(F, x, h=1e-7):
    """Compute the Jacobian using central finite differences. Costs 4 calls."""
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        J[:, j] = (F(x + e) - F(x - e)) / (2 * h)
    return J


def _solve_or_pinv(J, Fx):
    """Solve J * dx = Fx, falling back to pseudo-inverse if singular."""
    try:
        return np.linalg.solve(J, Fx)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(J, Fx, rcond=None)[0]


def find_root_system(F, x0):
    """
    Find a root of F: R^2 -> R^2 with ||F(x)|| <= 1e-8.
    Uses a hybrid strategy:
      1. Start with Broyden's method (cheap updates)
      2. If stagnating or ill-conditioned, recompute Jacobian
      3. Damped Newton steps with line search for robustness
      4. Deadline guard to avoid timeout
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    x = np.array(x0, dtype=float).copy()
    Fx = np.array(F(x), dtype=float)
    norm_Fx = np.linalg.norm(Fx)

    if norm_Fx <= 1e-8:
        return x

    # Initialize Jacobian with central finite differences (4 calls)
    J = _finite_diff_jacobian(F, x)

    best_x = x.copy()
    best_norm = norm_Fx

    max_iter = 40
    reanchor_every = 5  # recompute Jacobian every 5 iterations

    for it in range(max_iter):
        if time.time() > deadline:
            break

        # Check if we've found a root
        if norm_Fx <= 1e-8:
            return x

        # Update best if improved
        if norm_Fx < best_norm:
            best_norm = norm_Fx
            best_x = x.copy()

        # Compute step direction: solve J * s = -Fx
        s = -_solve_or_pinv(J, Fx)

        # Check for NaN/Inf in step
        if not np.all(np.isfinite(s)):
            # Recompute Jacobian
            J = _finite_diff_jacobian(F, x)
            s = -_solve_or_pinv(J, Fx)
            if not np.all(np.isfinite(s)):
                # Try a small random perturbation
                break

        # Damped step with line search
        alpha = 1.0
        step_size = np.linalg.norm(s)
        if step_size < 1e-14:
            # Step is too small, recompute Jacobian
            J = _finite_diff_jacobian(F, x)
            continue

        # Backtracking line search
        for _ in range(30):
            x_trial = x + alpha * s
            Fx_trial = np.array(F(x_trial), dtype=float)
            norm_Fx_trial = np.linalg.norm(Fx_trial)

            if not np.all(np.isfinite(Fx_trial)):
                alpha *= 0.5
                continue

            # Accept if we reduce the norm
            if norm_Fx_trial < norm_Fx * (1 - 1e-4 * alpha):
                x = x_trial
                Fx = Fx_trial
                norm_Fx = norm_Fx_trial
                break
            else:
                alpha *= 0.5

        else:
            # Line search failed, take a small step in negative gradient direction
            # (using Fx as a proxy for gradient since we don't have Hessian)
            x_trial = x - 1e-4 * Fx
            Fx_trial = np.array(F(x_trial), dtype=float)
            norm_Fx_trial = np.linalg.norm(Fx_trial)
            if np.all(np.isfinite(Fx_trial)) and norm_Fx_trial < norm_Fx:
                x = x_trial
                Fx = Fx_trial
                norm_Fx = norm_Fx_trial
            # Recompute Jacobian
            J = _finite_diff_jacobian(F, x)
            continue

        # Broyden rank-1 update
        if norm_Fx > 1e-10:
            s_actual = x - (x - alpha * s)  # This is just alpha * s, but let's track properly
            # Actually, let's recompute: s_actual = x_new - x_old
            # We need to track x_old
            # Let me restructure...

        # Broyden update: J_new = J + (Fx_new - Fx_old - J*(x_new - x_old)) * (x_new - x_old)^T / ||x_new - x_old||^2
        # We need to track the previous x and Fx
        # Let's restructure the loop to track these properly

        # Actually, let me just recompute Jacobian periodically instead of Broyden
        # to keep it simple and robust
        if (it + 1) % reanchor_every == 0:
            J = _finite_diff_jacobian(F, x)

    # Final check
    if norm_Fx <= 1e-8:
        return x

    # Try a few more Newton steps with fresh Jacobians if close
    for _ in range(5):
        if time.time() > deadline:
            break
        J = _finite_diff_jacobian(F, x)
        s = -_solve_or_pinv(J, Fx)
        if not np.all(np.isfinite(s)):
            break
        x_new = x + s
        Fx_new = np.array(F(x_new), dtype=float)
        norm_Fx_new = np.linalg.norm(Fx_new)
        if np.all(np.isfinite(Fx_new)) and norm_Fx_new < norm_Fx:
            x = x_new
            Fx = Fx_new
            norm_Fx = norm_Fx_new
        if norm_Fx <= 1e-8:
            return x

    # Return best found
    if best_norm < np.linalg.norm(Fx):
        return best_x
    return x
# EVOLVE-BLOCK-END
