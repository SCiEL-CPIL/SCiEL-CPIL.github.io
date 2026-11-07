import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Broyden's method with flexible step acceptance (any reduction).
    2. Adaptive Jacobian restart when progress stalls.
    3. Final Newton polish with central differences for high precision.
    4. Strict call and time limits.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    x = np.array(x0, dtype=float)
    call_count = 0
    max_calls = 490  # Leave buffer for safety

    def safe_F(x_in):
        nonlocal call_count
        if time.time() > deadline:
            raise TimeoutError()
        if call_count >= max_calls:
            raise TimeoutError()
        call_count += 1
        result = F(x_in)
        result = np.asarray(result, dtype=float)
        if result.shape != (2,):
            result = result.flatten()[:2]
        if not np.all(np.isfinite(result)):
            return np.array([1e10, 1e10])
        return result

    def compute_J_forward(x_curr, Fx_curr):
        """Forward finite differences for Jacobian (2 calls)."""
        nonlocal call_count
        if call_count + 2 > max_calls:
            raise TimeoutError()
        h = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = safe_F(x_curr + e)
            J[:, j] = (Fx_e - Fx_curr) / h
        return J

    def compute_J_central(x_curr):
        """Central finite differences for Jacobian (4 calls)."""
        nonlocal call_count
        if call_count + 4 > max_calls:
            raise TimeoutError()
        h = 1e-6
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_plus = safe_F(x_curr + e)
            Fx_minus = safe_F(x_curr - e)
            J[:, j] = (Fx_plus - Fx_minus) / (2 * h)
        return J

    try:
        # 1. Initial Evaluation
        Fx = safe_F(x)
        fx_norm = np.linalg.norm(Fx)

        if fx_norm <= 1e-8:
            return x

        # 2. Initial Jacobian via Forward Finite Differences
        J = compute_J_forward(x, Fx)

        # Ensure J is well-conditioned
        try:
            det = np.linalg.det(J)
            if abs(det) < 1e-12:
                J += np.eye(2) * 1e-6
        except:
            J += np.eye(2) * 1e-6

        best_x = x.copy()
        best_norm = fx_norm
        best_Fx = Fx.copy()

        # Broyden's method with flexible acceptance
        for iteration in range(300):
            if time.time() > deadline:
                break
            if call_count >= max_calls - 10:
                break

            if best_norm <= 1e-8:
                return best_x

            # Solve for step
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except:
                    break

            # Limit step size
            step_norm = np.linalg.norm(dx)
            if step_norm > 2.0:
                dx = dx / step_norm * 2.0
            elif step_norm < 1e-12:
                # Step is tiny, restart Jacobian
                J = compute_J_central(x)
                continue

            x_new = x + dx
            Fx_new = safe_F(x_new)
            fx_new_norm = np.linalg.norm(Fx_new)

            # Check for success
            if fx_new_norm <= 1e-8:
                return x_new

            # Accept any step that reduces residual (flexible acceptance)
            if fx_new_norm < fx_norm:
                s = x_new - x
                y = Fx_new - Fx
                x = x_new
                Fx = Fx_new
                fx_norm = fx_new_norm

                # Broyden rank-1 update
                s_sq = s @ s
                if s_sq > 1e-16:
                    J = J + np.outer(y - J @ s, s) / s_sq

                if fx_norm < best_norm:
                    best_norm = fx_norm
                    best_x = x.copy()
                    best_Fx = Fx.copy()
            else:
                # Step did not improve. Try with damped step.
                for damp in [0.5, 0.25, 0.1]:
                    x_damped = x + damp * dx
                    Fx_damped = safe_F(x_damped)
                    fx_damped_norm = np.linalg.norm(Fx_damped)

                    if fx_damped_norm <= 1e-8:
                        return x_damped

                    if fx_damped_norm < fx_norm:
                        s = x_damped - x
                        y = Fx_damped - Fx
                        x = x_damped
                        Fx = Fx_damped
                        fx_norm = fx_damped_norm

                        s_sq = s @ s
                        if s_sq > 1e-16:
                            J = J + np.outer(y - J @ s, s) / s_sq

                        if fx_norm < best_norm:
                            best_norm = fx_norm
                            best_x = x.copy()
                            best_Fx = Fx.copy()
                        break
                else:
                    # No improvement even with damping. Restart Jacobian.
                    J = compute_J_central(x)

    except TimeoutError:
        pass

    # 3. Final Newton polish with central differences
    # Use the best point found so far
    x_polish = best_x.copy()
    try:
        Fx_polish = safe_F(x_polish)
        norm_polish = np.linalg.norm(Fx_polish)

        if norm_polish <= 1e-8:
            return x_polish

        # Perform a few Newton iterations with central difference Jacobian
        for _ in range(10):
            if time.time() > deadline:
                break
            if call_count >= max_calls - 5:
                break

            norm_polish = np.linalg.norm(Fx_polish)
            if norm_polish <= 1e-8:
                return x_polish

            J_polish = compute_J_central(x_polish)

            try:
                dx = np.linalg.solve(J_polish, -Fx_polish)
            except np.linalg.LinAlgError:
                break

            step_norm = np.linalg.norm(dx)
            if step_norm < 1e-14:
                break

            # Damped Newton step
            for alpha in [1.0, 0.5, 0.25, 0.1]:
                x_trial = x_polish + alpha * dx
                Fx_trial = safe_F(x_trial)
                norm_trial = np.linalg.norm(Fx_trial)

                if norm_trial <= 1e-8:
                    return x_trial

                if norm_trial < norm_polish:
                    x_polish = x_trial
                    Fx_polish = Fx_trial
                    norm_polish = norm_trial
                    break
            else:
                break

        if np.linalg.norm(Fx_polish) < best_norm:
            best_x = x_polish
            best_norm = np.linalg.norm(Fx_polish)

    except TimeoutError:
        pass

    return best_x
