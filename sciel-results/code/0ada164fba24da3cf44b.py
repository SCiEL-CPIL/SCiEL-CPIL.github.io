"""Root-finder for 3 fixed 2D systems using a hybrid Broyden + trust-region approach.
Strategy:
  1. Broyden's method (good variant) with damped steps for fast local convergence.
  2. If Broyden stagnates, switch to a derivative-free trust-region dogleg method
     on g(x) = 0.5*||F(x)||^2 for global robustness.
  3. When close to the root, polish with FD-Newton steps for high precision.
This combines Broyden's efficient Jacobian updates with trust-region global convergence.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 of F(x) = 0 with ||F(x)|| <= 1e-8 (Euclidean norm).
    
    Uses Broyden's method with:
    - Forward-difference initial Jacobian
    - Damped Newton steps with adaptive step control
    - Trust-region dogleg fallback when Broyden stagnates
    - Final polish phase with FD-Newton for high accuracy
    
    Budget: <500 F-calls per system. Time-limited to ~135s.
    """
    x = np.array(x0, dtype=float, copy=True)
    t_start = time.time()
    deadline = t_start + 135.0
    call_count = 0
    best_x = x.copy()
    best_norm = np.inf

    def _time_ok():
        return time.time() < deadline

    def _call_F(x_val):
        nonlocal call_count
        if not _time_ok() or call_count >= 495:
            return None
        call_count += 1
        try:
            result = F(np.asarray(x_val, dtype=float))
        except Exception:
            return None
        result = np.asarray(result, dtype=float)
        if result.shape != (2,):
            try:
                result = result.reshape(2)
            except Exception:
                return None
        if not np.all(np.isfinite(result)):
            return None
        return result

    def _fd_jacobian(x_val, Fx_val, h=1e-5):
        """Compute Jacobian via forward differences. Uses 2 calls (Fx already known)."""
        nonlocal call_count
        J = np.zeros((2, 2))
        for j in range(2):
            if not _time_ok() or call_count >= 493:
                break
            e = np.zeros(2)
            e[j] = h
            Fx_plus = _call_F(x_val + e)
            if Fx_plus is not None:
                J[:, j] = (Fx_plus - Fx_val) / h
            else:
                J[:, j] = 1.0
        return J

    def _g(x_val):
        """Objective: 0.5 * ||F(x)||^2"""
        Fx = _call_F(x_val)
        if Fx is None:
            return None
        return 0.5 * np.dot(Fx, Fx)

    def _trust_region_dogleg(x0_tr, max_iter=100):
        """
        Trust-region dogleg method on g(x) = 0.5*||F(x)||^2.
        Derivative-free: uses finite differences for gradient and Hessian approximation.
        Returns (x_best, g_best).
        """
        nonlocal call_count
        x_curr = np.array(x0_tr, dtype=float)
        g_curr = _g(x_curr)
        if g_curr is None:
            return x_curr, np.inf

        # Initial trust radius
        delta = 0.5 * max(1.0, np.linalg.norm(x_curr))
        best_x_tr = x_curr.copy()
        best_g_tr = g_curr

        # Compute gradient via finite differences
        def _grad(x_val):
            nonlocal call_count
            grad = np.zeros(2)
            h = 1e-5 * max(1.0, np.linalg.norm(x_val))
            for j in range(2):
                if not _time_ok() or call_count >= 490:
                    return None
                e = np.zeros(2)
                e[j] = h
                g_plus = _g(x_val + e)
                g_minus = _g(x_val - e)
                if g_plus is not None and g_minus is not None:
                    grad[j] = (g_plus - g_minus) / (2 * h)
                else:
                    return None
            return grad

        # Compute Hessian approximation via finite differences of gradient
        def _hessian(x_val):
            nonlocal call_count
            H = np.zeros((2, 2))
            h = 1e-4 * max(1.0, np.linalg.norm(x_val))
            grad0 = _grad(x_val)
            if grad0 is None:
                return None
            for j in range(2):
                if not _time_ok() or call_count >= 490:
                    return None
                e = np.zeros(2)
                e[j] = h
                grad_plus = _grad(x_val + e)
                if grad_plus is None:
                    return None
                H[:, j] = (grad_plus - grad0) / h
            H = 0.5 * (H + H.T)
            return H

        for it in range(max_iter):
            if not _time_ok() or call_count >= 485:
                break
            if best_g_tr <= 1e-16:
                break

            # Compute gradient and Hessian
            grad = _grad(x_curr)
            if grad is None:
                break
            H = _hessian(x_curr)
            if H is None:
                break

            # Ensure H is positive semi-definite
            eigvals = np.linalg.eigvalsh(H)
            if eigvals[0] < 0:
                H = H + np.eye(2) * (abs(eigvals[0]) + 1e-8)

            # Newton step: solve H @ dx = -grad
            try:
                dx_newton = np.linalg.solve(H, -grad)
            except np.linalg.LinAlgError:
                dx_newton = np.linalg.pinv(H) @ (-grad)

            # Gradient step (steepest descent)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-15:
                break
            dx_gradient = -grad / grad_norm

            # Dogleg step
            newton_norm = np.linalg.norm(dx_newton)
            if newton_norm <= delta:
                # Newton step is within trust region
                dx = dx_newton
            elif np.dot(dx_newton, dx_gradient) >= 0:
                # Newton and gradient are aligned
                # Find t such that ||t * dx_newton + (1-t) * dx_gradient * delta / grad_norm|| = delta
                # Simplified: use gradient step scaled to trust region
                dx = dx_gradient * delta
            else:
                # Find intersection of dogleg path with trust region boundary
                # The dogleg path goes from 0 to dx_gradient * delta / grad_norm, then to dx_newton
                # We need to find where it hits the trust region boundary
                # For simplicity, use a scaled Newton step
                scale = delta / newton_norm
                dx = dx_newton * scale

            # Try the step
            x_new = x_curr + dx
            g_new = _g(x_new)
            if g_new is None:
                delta *= 0.5
                continue

            # Check if step is acceptable
            if g_new < g_curr:
                # Accept step, possibly expand trust region
                x_curr = x_curr + dx
                g_curr = g_new
                if g_curr < best_g_tr:
                    best_x_tr = x_curr.copy()
                    best_g_tr = g_curr
                # Expand trust region if improvement is good
                predicted = -0.5 * np.dot(grad, dx)
                actual = g_curr - g_new
                if predicted > 1e-15 and actual / predicted > 0.75:
                    delta = min(2 * delta, 10.0)
                else:
                    delta = 1.5 * delta
            else:
                # Reject step, shrink trust region
                delta *= 0.5
                if delta < 1e-10:
                    break

            if best_g_tr <= 1e-16:
                break

        return best_x_tr, best_g_tr

    # Initial evaluation
    Fx = _call_F(x)
    if Fx is None:
        return best_x
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x
    best_x = x.copy()
    best_norm = norm_Fx

    # Compute initial Jacobian
    J = _fd_jacobian(x, Fx)
    try:
        det_J = np.linalg.det(J)
        if abs(det_J) < 1e-10:
            J += np.eye(2) * 1e-6
    except np.linalg.LinAlgError:
        J = np.eye(2)

    # Phase 1: Broyden's method main loop
    max_iter = 450
    stalled_count = 0
    prev_norm = norm_Fx

    for it in range(max_iter):
        if not _time_ok() or call_count >= 485:
            break

        # Solve J @ dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.pinv(J) @ (-Fx)
            except Exception:
                break

        norm_dx = np.linalg.norm(dx)
        if norm_dx == 0:
            break

        # Damped step: try full step, back off if needed
        alpha = 1.0
        success = False
        for _ in range(12):
            if not _time_ok() or call_count >= 488:
                break
            x_new = x + alpha * dx
            if not np.all(np.isfinite(x_new)):
                alpha *= 0.5
                continue
            Fx_new = _call_F(x_new)
            if Fx_new is None:
                break
            norm_new = np.linalg.norm(Fx_new)
            if norm_new < norm_Fx * (1 + 1e-4):
                success = True
                break
            alpha *= 0.5
            if alpha < 1e-12:
                break

        if not success:
            # Try smaller fixed step
            x_new = x + 0.1 * dx
            if np.all(np.isfinite(x_new)):
                Fx_new = _call_F(x_new)
                if Fx_new is not None:
                    norm_new = np.linalg.norm(Fx_new)
                    if norm_new < best_norm:
                        best_x = x_new.copy()
                        best_norm = norm_new
                    if norm_new <= 1e-8:
                        return x_new
                    x, Fx, norm_Fx = x_new, Fx_new, norm_new
            break

        # Update Broyden Jacobian (good variant)
        s = x_new - x
        y = Fx_new - Fx
        s_norm_sq = np.dot(s, s)
        if s_norm_sq > 1e-18:
            J = J + np.outer(y - J @ s, s) / s_norm_sq

        x = x_new
        Fx = Fx_new
        norm_Fx = norm_new

        if norm_Fx <= 1e-8:
            return x

        if norm_Fx < best_norm:
            best_x = x.copy()
            best_norm = norm_Fx

        # Detect stalling
        if norm_Fx > prev_norm * 0.999:
            stalled_count += 1
        else:
            stalled_count = 0
        prev_norm = norm_Fx

        if stalled_count >= 15:
            # Switch to trust-region dogleg method
            if _time_ok() and call_count < 480:
                x_tr, g_tr = _trust_region_dogleg(x, max_iter=50)
                if g_tr is not None and np.isfinite(g_tr):
                    norm_tr = np.sqrt(2 * g_tr)
                    if norm_tr < best_norm:
                        best_x = x_tr.copy()
                        best_norm = norm_tr
                    x = x_tr.copy()
                    Fx = _call_F(x)
                    if Fx is not None:
                        norm_Fx = np.linalg.norm(Fx)
                        if norm_Fx <= 1e-8:
                            return x
                        # Recompute Jacobian
                        J = _fd_jacobian(x, Fx)
                        try:
                            if abs(np.linalg.det(J)) < 1e-10:
                                J += np.eye(2) * 1e-6
                        except np.linalg.LinAlgError:
                            J = np.eye(2)
                    stalled_count = 0
                    prev_norm = np.inf

    # Final polish: use FD-Newton for high precision if we're close
    if best_norm < 1e-4 and call_count < 495 and _time_ok():
        x = best_x.copy()
        Fx = _call_F(x)
        if Fx is not None:
            norm_Fx = np.linalg.norm(Fx)
            for _ in range(10):
                if not _time_ok() or call_count >= 495:
                    break
                if norm_Fx <= 1e-8:
                    break
                # Small FD Jacobian
                h = max(1e-7, 1e-5 * np.sqrt(norm_Fx))
                J = _fd_jacobian(x, Fx, h)
                try:
                    if abs(np.linalg.det(J)) < 1e-12:
                        J += np.eye(2) * 1e-8
                    dx = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    break
                norm_dx = np.linalg.norm(dx)
                if norm_dx < 1e-14:
                    break
                # Damped step
                alpha = 1.0
                for _ in range(8):
                    if not _time_ok() or call_count >= 495:
                        break
                    x_new = x + alpha * dx
                    if not np.all(np.isfinite(x_new)):
                        alpha *= 0.5
                        continue
                    Fx_new = _call_F(x_new)
                    if Fx_new is None:
                        break
                    norm_new = np.linalg.norm(Fx_new)
                    if norm_new < norm_Fx:
                        x, Fx, norm_Fx = x_new, Fx_new, norm_new
                        if norm_Fx < best_norm:
                            best_x = x.copy()
                            best_norm = norm_Fx
                        break
                    alpha *= 0.5
                if norm_Fx <= 1e-8:
                    break

    # Final verification
    if call_count < 498 and _time_ok():
        Fx_final = _call_F(best_x)
        if Fx_final is not None:
            nf = np.linalg.norm(Fx_final)
            if nf <= 1e-8:
                return best_x
            if nf < best_norm:
                best_x = best_x.copy()
                best_norm = nf

    return best_x
# EVOLVE-BLOCK-END
