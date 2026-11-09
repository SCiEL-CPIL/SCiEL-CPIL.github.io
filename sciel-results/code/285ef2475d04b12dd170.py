"""Root-finder for 3 fixed 2D systems using Broyden's method with a robust fallback.
Strategy:
  1. Broyden's method (good variant) with damped steps for fast local convergence.
  2. If Broyden stagnates or fails, switch to a robust Nelder-Mead style simplex
     search on g(x) = ||F(x)||^2, which is derivative-free and robust.
  3. Final polish with FD-Newton steps for high precision.
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
    - Robust fallback to Nelder-Mead if Broyden stalls
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

    def _nelder_mead_fallback(x_start, max_iter=100):
        """
        Nelder-Mead simplex method on g(x) = ||F(x)||^2.
        Derivative-free and robust for 2D problems.
        """
        nonlocal call_count
        nonlocal best_x
        nonlocal best_norm

        # Initialize simplex: 3 vertices in 2D
        h0 = 0.1 * max(1.0, np.linalg.norm(x_start))
        simplex = np.array([
            x_start,
            x_start + [h0, 0],
            x_start + [0, h0]
        ])
        
        # Evaluate objective at each vertex
        fvals = np.zeros(3)
        for i in range(3):
            Fx = _call_F(simplex[i])
            if Fx is not None:
                fvals[i] = np.dot(Fx, Fx)
            else:
                fvals[i] = 1e10

        for it in range(max_iter):
            if not _time_ok() or call_count >= 490:
                break

            # Sort vertices by objective value
            idx = np.argsort(fvals)
            simplex = simplex[idx]
            fvals = fvals[idx]

            # Check convergence
            if fvals[0] <= 1e-16:  # ||F||^2 <= 1e-16 => ||F|| <= 1e-8
                return simplex[0].copy()

            # Update best
            if fvals[0] < best_norm ** 2:
                best_x = simplex[0].copy()
                best_norm = np.sqrt(fvals[0])

            # Centroid of all but worst vertex
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            alpha = 1.0
            x_ref = centroid + alpha * (centroid - simplex[-1])
            Fx_ref = _call_F(x_ref)
            f_ref = np.dot(Fx_ref, Fx_ref) if Fx_ref is not None else 1e10

            if f_ref < fvals[0]:
                # Expansion
                gamma = 2.0
                x_exp = centroid + gamma * (centroid - simplex[-1])
                Fx_exp = _call_F(x_exp)
                f_exp = np.dot(Fx_exp, Fx_exp) if Fx_exp is not None else 1e10
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
            elif f_ref < fvals[-2]:
                simplex[-1] = x_ref
                fvals[-1] = f_ref
            else:
                # Contraction
                beta = 0.5
                x_con = centroid + beta * (simplex[-1] - centroid)
                Fx_con = _call_F(x_con)
                f_con = np.dot(Fx_con, Fx_con) if Fx_con is not None else 1e10
                if f_con < fvals[-1]:
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, 3):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        Fx_sh = _call_F(simplex[i])
                        fvals[i] = np.dot(Fx_sh, Fx_sh) if Fx_sh is not None else 1e10

        return simplex[np.argmin(fvals)].copy()

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
    max_iter = 40
    stalled_count = 0
    prev_norm = norm_Fx
    use_fallback = False

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
            # If we couldn't even take a small step, or if we've been stuck, trigger fallback
            use_fallback = True
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

        if stalled_count >= 5:
            # Switch to robust fallback earlier if stalled
            use_fallback = True
            break

    # Phase 2: Robust Fallback if Broyden didn't solve it
    if norm_Fx > 1e-4 and use_fallback and _time_ok() and call_count < 480:
        x_fallback = _nelder_mead_fallback(x, max_iter=50)
        Fx_fb = _call_F(x_fallback)
        if Fx_fb is not None:
            norm_fb = np.linalg.norm(Fx_fb)
            if norm_fb < best_norm:
                best_x = x_fallback.copy()
                best_norm = norm_fb
                x = x_fallback
                Fx = Fx_fb
                norm_Fx = norm_fb

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
