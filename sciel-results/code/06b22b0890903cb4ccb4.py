import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Broyden's quasi-Newton with rank-1 updates (1 F-call per iteration after setup).
    2. Accept any step that reduces residual; backtracking only as fallback.
    3. Restart Jacobian via finite differences when progress stalls.
    4. Final polish: high-precision finite-difference Newton to converge to 1e-8.
    5. If still not converged, do a local minimization of ||F(x)||^2 via
       Gauss-Newton (J^T J) to squeeze out the last few orders of magnitude.
    """
    start_time = time.time()
    deadline = start_time + 118.0

    x = np.array(x0, dtype=float)
    call_count = 0
    max_calls = 495

    def safe_F(x_in):
        nonlocal call_count
        if time.time() > deadline:
            raise TimeoutError()
        if call_count >= max_calls:
            raise TimeoutError()
        call_count += 1
        result = F(x_in)
        result = np.asarray(result, dtype=float).flatten()
        if result.shape[0] < 2:
            raise ValueError("bad shape")
        result = result[:2]
        if not np.all(np.isfinite(result)):
            return np.array([1e10, 1e10])
        return result

    def compute_J(x_cur, Fx_cur):
        """Forward finite-difference Jacobian (2 calls)."""
        h = 1e-7 * max(1.0, np.linalg.norm(x_cur))
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = safe_F(x_cur + e)
            J[:, j] = (Fx_e - Fx_cur) / h
        det = np.linalg.det(J)
        if abs(det) < 1e-12:
            J += np.eye(2) * 1e-8
        return J

    def compute_J_central(x_cur):
        """Central finite-difference Jacobian (4 calls) for polishing."""
        h = 1e-6 * max(1.0, np.linalg.norm(x_cur))
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_p = safe_F(x_cur + e)
            Fx_m = safe_F(x_cur - e)
            J[:, j] = (Fx_p - Fx_m) / (2 * h)
        det = np.linalg.det(J)
        if abs(det) < 1e-12:
            J += np.eye(2) * 1e-8
        return J

    def gauss_newton_step(x_cur, Fx_cur):
        """
        One Gauss-Newton step for minimizing ||F(x)||^2.
        Uses 2 extra F-calls for central-diff Jacobian, then solves (J^T J) dx = -J^T F.
        Returns (x_new, Fx_new, norm_new, calls_used) or None on failure.
        """
        nonlocal call_count
        if call_count + 5 > max_calls:
            return None
        J = compute_J_central(x_cur)
        JTJ = J.T @ J
        JTF = J.T @ Fx_cur
        # Add small ridge for stability
        JTJ += np.eye(2) * 1e-12
        try:
            dx = np.linalg.solve(JTJ, -JTF)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(JTJ, -JTF, rcond=None)[0]
        
        step_norm = np.linalg.norm(dx)
        if step_norm > 5.0:
            dx = dx / step_norm * 5.0
        
        x_new = x_cur + dx
        Fx_new = safe_F(x_new)
        norm_new = np.linalg.norm(Fx_new)
        return (x_new, Fx_new, norm_new, 5)  # 4 for J + 1 for eval

    try:
        # Initial evaluation
        Fx = safe_F(x)
        fx_norm = np.linalg.norm(Fx)
        if fx_norm <= 1e-8:
            return x

        # Initial Jacobian
        J = compute_J(x, Fx)

        best_x = x.copy()
        best_norm = fx_norm

        stall_count = 0
        prev_norm = fx_norm

        for it in range(450):
            if time.time() > deadline:
                break
            if call_count >= max_calls - 10:
                break
            if fx_norm <= 1e-8:
                return x

            # Solve for Newton step
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except:
                    break

            # Limit step magnitude
            step_norm = np.linalg.norm(dx)
            if step_norm > 10.0:
                dx = dx / step_norm * 10.0

            # Try full step first
            x_new = x + dx
            Fx_new = safe_F(x_new)
            fx_new_norm = np.linalg.norm(Fx_new)

            accepted = False
            s = np.zeros(2)
            y = np.zeros(2)

            if fx_new_norm < fx_norm:
                # Accept: reduces residual
                s = x_new - x
                y = Fx_new - Fx
                x = x_new
                Fx = Fx_new
                fx_norm = fx_new_norm
                accepted = True
            else:
                # Backtracking
                for factor in [0.5, 0.25, 0.125, 0.0625, 0.03125]:
                    if call_count >= max_calls - 5:
                        break
                    x_try = x + factor * dx
                    Fx_try = safe_F(x_try)
                    fx_try_norm = np.linalg.norm(Fx_try)
                    if fx_try_norm < fx_norm * 0.9:
                        s = x_try - x
                        y = Fx_try - Fx
                        x = x_try
                        Fx = Fx_try
                        fx_norm = fx_try_norm
                        accepted = True
                        break

            if not accepted:
                # Stall: restart Jacobian
                stall_count += 1
                if stall_count >= 3 or call_count + 3 > max_calls:
                    break
                J = compute_J(x, Fx)
                continue

            # Broyden rank-1 update
            s_sq = s @ s
            if s_sq > 1e-16:
                J = J + np.outer(y - J @ s, s) / s_sq

            # Track progress
            if fx_norm < best_norm:
                best_norm = fx_norm
                best_x = x.copy()
                stall_count = 0
            else:
                stall_count += 1
                if stall_count >= 4:
                    # Restart Jacobian
                    if call_count + 3 > max_calls:
                        break
                    J = compute_J(x, Fx)
                    stall_count = 0

            if fx_norm <= 1e-8:
                return x

        # If we haven't converged, try a polish phase with central differences
        if fx_norm > 1e-8 and call_count < max_calls - 20:
            for polish_iter in range(15):
                if time.time() > deadline:
                    break
                if call_count >= max_calls - 4:
                    break
                if fx_norm <= 1e-8:
                    return x
                try:
                    J_c = compute_J_central(x)
                    dx = np.linalg.solve(J_c, -Fx)
                    x_new = x + dx
                    Fx_new = safe_F(x_new)
                    fx_new_norm = np.linalg.norm(Fx_new)
                    if fx_new_norm < fx_norm:
                        x = x_new
                        Fx = Fx_new
                        fx_norm = fx_new_norm
                        if fx_norm < best_norm:
                            best_norm = fx_norm
                            best_x = x.copy()
                    else:
                        # Try damped
                        for fac in [0.5, 0.25]:
                            x_try = x + fac * dx
                            Fx_try = safe_F(x_try)
                            n_try = np.linalg.norm(Fx_try)
                            if n_try < fx_norm:
                                x = x_try
                                Fx = Fx_try
                                fx_norm = n_try
                                if fx_norm < best_norm:
                                    best_norm = fx_norm
                                    best_x = x.copy()
                                break
                        else:
                            break
                except:
                    break

        # Final phase: Gauss-Newton minimization of ||F||^2 if still above tolerance
        if fx_norm > 1e-8 and call_count < max_calls - 10:
            for gn_iter in range(8):
                if time.time() > deadline:
                    break
                if call_count >= max_calls - 5:
                    break
                if fx_norm <= 1e-8:
                    return x
                result = gauss_newton_step(x, Fx)
                if result is None:
                    break
                x_new, Fx_new, norm_new, calls_used = result
                if norm_new < fx_norm:
                    x = x_new
                    Fx = Fx_new
                    fx_norm = norm_new
                    if fx_norm < best_norm:
                        best_norm = fx_norm
                        best_x = x.copy()
                else:
                    # Try half step
                    if call_count + 2 <= max_calls:
                        x_try = x + 0.5 * (x_new - x)
                        Fx_try = safe_F(x_try)
                        n_try = np.linalg.norm(Fx_try)
                        if n_try < fx_norm:
                            x = x_try
                            Fx = Fx_try
                            fx_norm = n_try
                            if fx_norm < best_norm:
                                best_norm = fx_norm
                                best_x = x.copy()
                        else:
                            break
                    else:
                        break

    except TimeoutError:
        pass
    except:
        pass

    # Return best point found
    return best_x
