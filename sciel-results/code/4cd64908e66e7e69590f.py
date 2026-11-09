"""Root finder for 3 fixed two-dimensional systems using Broyden's method with Newton polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 such that ||F(x)|| <= 1e-8.
    Strategy:
    1. Broyden's method with forward-difference initial Jacobian and step limiting.
    2. If Broyden stalls or fails to converge, apply Newton's method with central differences for polishing.
    3. Explicit call counting to minimize mean_calls for scoring.
    """
    x = np.array(x0, dtype=float, copy=True)
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    call_count = 0
    best_x = x.copy()
    best_norm = np.inf

    def _call_F(x_val):
        nonlocal call_count
        if time.time() > deadline or call_count >= 450:
            return None
        call_count += 1
        try:
            result = F(np.asarray(x_val, dtype=float))
            result = np.asarray(result, dtype=float)
            if result.shape != (2,):
                result = result.reshape(2)
            if not np.all(np.isfinite(result)):
                return None
            return result
        except Exception:
            return None

    # Initial evaluation
    Fx = _call_F(x)
    if Fx is None:
        return best_x
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x
    best_x = x.copy()
    best_norm = norm_Fx

    # Compute initial Jacobian via forward differences (2 extra calls)
    h = 1e-6
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_plus = _call_F(x + e)
        if Fx_plus is None:
            return best_x
        J[:, j] = (Fx_plus - Fx) / h

    # Ensure J is invertible
    if abs(np.linalg.det(J)) < 1e-12:
        J += np.eye(2) * 1e-6

    # Phase 1: Broyden's Method
    max_broyden_iter = 30
    for it in range(max_broyden_iter):
        if time.time() > deadline or call_count >= 440:
            break

        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break

        # Step limiting
        norm_dx = np.linalg.norm(dx)
        if norm_dx > 5.0:
            dx = dx / norm_dx * 5.0

        x_new = x + dx
        if not np.all(np.isfinite(x_new)):
            dx = dx * 0.1
            x_new = x + dx
            if not np.all(np.isfinite(x_new)):
                break

        Fx_new = _call_F(x_new)
        if Fx_new is None:
            break

        norm_Fx_new = np.linalg.norm(Fx_new)

        if norm_Fx_new < best_norm:
            best_x = x_new.copy()
            best_norm = norm_Fx_new

        if norm_Fx_new <= 1e-8:
            return x_new

        # Broyden rank-1 update
        s = x_new - x
        y = Fx_new - Fx
        s_norm_sq = np.dot(s, s)
        if s_norm_sq > 1e-16:
            J = J + np.outer(y - J @ s, s) / s_norm_sq

        # Accept step if it makes progress
        if norm_Fx_new < norm_Fx * 0.9:
            x = x_new
            Fx = Fx_new
            norm_Fx = norm_Fx_new
        else:
            # If no significant progress, break to try Newton polish
            break

    # Phase 2: Newton Polish (Central Differences)
    # Use best_x as starting point
    if best_norm > 1e-8 and call_count < 445 and time.time() < deadline:
        x = best_x.copy()
        h = 1e-5
        for _ in range(5):  # Max 5 Newton steps
            if call_count >= 445 or time.time() > deadline:
                break
            
            Fx = _call_F(x)
            if Fx is None:
                break
            norm_Fx = np.linalg.norm(Fx)
            if norm_Fx <= 1e-8:
                return x
            
            # Compute Jacobian via Central Differences (4 extra calls)
            J = np.zeros((2, 2))
            valid_J = True
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fp = _call_F(x + e)
                Fm = _call_F(x - e)
                if Fp is None or Fm is None:
                    valid_J = False
                    break
                J[:, j] = (Fp - Fm) / (2 * h)
            
            if not valid_J:
                break

            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break

            if not np.all(np.isfinite(dx)):
                break

            # Damped Newton step
            alpha = 1.0
            for _ in range(10):
                x_new = x + alpha * dx
                Fx_new = _call_F(x_new)
                if Fx_new is None:
                    alpha *= 0.5
                    continue
                norm_new = np.linalg.norm(Fx_new)
                if norm_new < norm_Fx:
                    x = x_new
                    Fx = Fx_new
                    norm_Fx = norm_new
                    if norm_Fx < best_norm:
                        best_x = x.copy()
                        best_norm = norm_Fx
                    if norm_Fx <= 1e-8:
                        return x
                    break
                alpha *= 0.5

    return best_x
# EVOLVE-BLOCK-END
