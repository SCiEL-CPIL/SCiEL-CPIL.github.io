"""Root-finder for 3 fixed 2D systems: Broyden's method with dogleg trust-region and Newton polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 of F(x) = 0 with ||F(x)|| <= 1e-8 (Euclidean norm).

    Strategy:
      1. Broyden's method (good-Broyden) with forward-difference initial Jacobian.
      2. Dogleg trust-region fallback for ill-conditioned systems.
      3. Newton polish with high-precision Jacobian when close to root.
      4. Strict call budget and time deadline management.
    """
    x = np.array(x0, dtype=float, copy=True)
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s kill
    call_count = 0
    best_x = x.copy()
    best_norm = np.inf

    def _call_F(x_val):
        nonlocal call_count
        if time.time() > deadline:
            return None
        if call_count >= 495:
            return None
        call_count += 1
        try:
            result = F(np.asarray(x_val, dtype=float))
        except Exception:
            return None
        try:
            result = np.asarray(result, dtype=float)
        except Exception:
            return None
        if result.shape != (2,):
            try:
                result = result.reshape(2)
            except Exception:
                return None
        if not np.all(np.isfinite(result)):
            return None
        return result

    def _fd_jacobian(x, Fx):
        """Forward-difference Jacobian (2 extra calls, reuses Fx)."""
        J = np.zeros((2, 2))
        h = 1e-5
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_plus = _call_F(x + e)
            if Fx_plus is None:
                return None
            J[:, j] = (Fx_plus - Fx) / h
        return J

    def _central_jacobian(x):
        """Central-difference Jacobian (4 calls). More accurate near root."""
        J = np.zeros((2, 2))
        h = 1e-6
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_plus = _call_F(x + e)
            if Fx_plus is None:
                return None
            Fx_minus = _call_F(x - e)
            if Fx_minus is None:
                return None
            J[:, j] = (Fx_plus - Fx_minus) / (2 * h)
        return J

    def _is_finite_2d(v):
        return np.all(np.isfinite(v))

    def _dogleg_step(J, Fx, trust_radius):
        """
        Compute a dogleg step for the trust-region subproblem.
        Returns the step vector.
        """
        # Cauchy point: steepest descent direction scaled to trust boundary
        g = J.T @ Fx  # gradient of 0.5*||F||^2
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-15:
            return np.zeros(2)
        
        # Newton step
        try:
            dx_newton = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            dx_newton = -g / g_norm  # fallback to steepest descent
        
        dx_newton_norm = np.linalg.norm(dx_newton)
        
        # Cauchy point: t* = -g^T (J^{-1} g) / ||J^{-1} g||^2
        try:
            J_inv_g = np.linalg.solve(J, g)
        except np.linalg.LinAlgError:
            J_inv_g = g / g_norm
        
        t_cauchy = -np.dot(g, J_inv_g) / (np.dot(J_inv_g, J_inv_g) + 1e-30)
        t_cauchy = max(0.0, min(t_cauchy, 1.0))
        dx_cauchy = t_cauchy * dx_newton
        
        cauchy_norm = np.linalg.norm(dx_cauchy)
        
        # If Newton step is within trust region, use it
        if dx_newton_norm <= trust_radius:
            return dx_newton
        
        # If Cauchy point is at or beyond trust boundary, scale it
        if cauchy_norm >= trust_radius:
            if cauchy_norm > 1e-15:
                return dx_cauchy / cauchy_norm * trust_radius
            else:
                return -g / g_norm * trust_radius
        
        # Otherwise, find intersection of dogleg path with trust boundary
        # The dogleg path goes from 0 to dx_cauchy to dx_newton
        # We need to find where the segment from dx_cauchy to dx_newton 
        # intersects the trust region boundary
        
        v = dx_newton - dx_cauchy
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-15:
            return dx_cauchy
        
        # Solve ||dx_cauchy + t*v||^2 = trust_radius^2 for t in [0,1]
        a = np.dot(v, v)
        b = 2 * np.dot(dx_cauchy, v)
        c = np.dot(dx_cauchy, dx_cauchy) - trust_radius**2
        
        disc = b*b - 4*a*c
        if disc < 0:
            return dx_cauchy
        sqrt_disc = np.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2*a)
        t2 = (-b + sqrt_disc) / (2*a)
        
        # Choose the root in [0, 1]
        for t in [t1, t2]:
            if 0 <= t <= 1:
                return dx_cauchy + t * v
        
        # Fallback: scale Newton step
        if dx_newton_norm > 1e-15:
            return dx_newton / dx_newton_norm * trust_radius
        return dx_cauchy

    # ---- Initial evaluation ----
    Fx = _call_F(x)
    if Fx is None:
        return best_x
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x
    best_x = x.copy()
    best_norm = norm_Fx

    # ---- Initial Jacobian via forward differences ----
    J = _fd_jacobian(x, Fx)
    if J is None:
        return best_x
    if abs(np.linalg.det(J)) < 1e-12:
        J = J + np.eye(2) * 1e-6

    # ---- Main loop with Broyden, dogleg fallback, and Newton polish ----
    stalled = 0
    max_stall = 2
    newton_mode = False
    newton_iters = 0
    trust_radius = 1.0
    use_dogleg = False

    for it in range(480):
        if time.time() > deadline:
            break
        if call_count >= 490:
            break

        # Switch to Newton mode when very close to root
        if not newton_mode and norm_Fx < 1e-5 and call_count < 460:
            newton_mode = True
            newton_iters = 0
            use_dogleg = False
            # Recompute Jacobian with central differences for higher accuracy
            J = _central_jacobian(x)
            if J is None:
                J = _fd_jacobian(x, Fx)
                if J is None:
                    break
            if abs(np.linalg.det(J)) < 1e-12:
                J = J + np.eye(2) * 1e-6

        # Determine step direction
        if newton_mode:
            # Pure Newton step with damping
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                J = _central_jacobian(x)
                if J is None:
                    J = _fd_jacobian(x, Fx)
                if J is None:
                    break
                if abs(np.linalg.det(J)) < 1e-12:
                    J = J + np.eye(2) * 1e-6
                try:
                    dx = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    dx = -Fx  # fallback
        elif use_dogleg:
            # Dogleg trust-region step
            dx = _dogleg_step(J, Fx, trust_radius)
        else:
            # Broyden step
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                J = _fd_jacobian(x, Fx)
                if J is None:
                    break
                if abs(np.linalg.det(J)) < 1e-12:
                    J = J + np.eye(2) * 1e-6
                try:
                    dx = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    dx = -Fx

        if not _is_finite_2d(dx):
            break

        # Step control
        norm_dx = np.linalg.norm(dx)
        if norm_dx > 10.0:
            dx = dx / norm_dx * 10.0

        # Damping schedule
        if newton_mode:
            dampings = (1.0, 0.5, 0.25, 0.1)
        elif use_dogleg:
            dampings = (1.0,)  # Dogleg already constrains step
        else:
            dampings = (1.0, 0.5, 0.25, 0.1, 0.05)

        # Try steps
        accepted = False
        for damp in dampings:
            x_new = x + damp * dx
            if not _is_finite_2d(x_new):
                continue
            Fx_new = _call_F(x_new)
            if Fx_new is None:
                break
            norm_Fx_new = np.linalg.norm(Fx_new)
            if not _is_finite_2d(Fx_new):
                continue

            # Accept if residual decreases
            if norm_Fx_new < norm_Fx:
                s = x_new - x
                y = Fx_new - Fx
                s_norm_sq = np.dot(s, s)
                
                if not newton_mode and not use_dogleg and s_norm_sq > 1e-16:
                    # Broyden rank-1 update
                    J = J + np.outer(y - J @ s, s) / s_norm_sq
                elif newton_mode:
                    newton_iters += 1
                    if newton_iters % 2 == 0 and call_count < 450:
                        J = _central_jacobian(x_new)
                        if J is not None and abs(np.linalg.det(J)) >= 1e-12:
                            pass
                        else:
                            J = _fd_jacobian(x_new, Fx_new)
                            if J is None:
                                J = J + np.eye(2) * 1e-6
                elif use_dogleg:
                    # Update trust region
                    actual_reduction = norm_Fx - norm_Fx_new
                    predicted_reduction = np.dot(Fx, J @ s)  # linear model reduction
                    if actual_reduction > 0:
                        rho = actual_reduction / (abs(predicted_reduction) + 1e-15)
                        if rho > 0.75:
                            trust_radius = min(2.0 * trust_radius, 5.0)
                        elif rho > 0.25:
                            trust_radius = trust_radius
                        else:
                            trust_radius = 0.5 * trust_radius
                    # Recompute Jacobian periodically
                    if it % 3 == 0 and call_count < 450:
                        J = _central_jacobian(x_new)
                        if J is None:
                            J = _fd_jacobian(x_new, Fx_new)
                        if J is not None and abs(np.linalg.det(J)) < 1e-12:
                            J = J + np.eye(2) * 1e-6

                x = x_new
                Fx = Fx_new
                norm_Fx = norm_Fx_new
                accepted = True

                if norm_Fx_new < best_norm:
                    best_x = x_new.copy()
                    best_norm = norm_Fx_new

                if norm_Fx_new <= 1e-8:
                    return x_new
                break

        if not accepted:
            stalled += 1
            trust_radius = 0.5 * trust_radius  # Shrink trust region
            
            if stalled >= max_stall:
                # Restart strategy
                if newton_mode:
                    # Fall back from Newton
                    newton_mode = False
                    use_dogleg = True
                    trust_radius = 1.0
                elif use_dogleg:
                    # Try Broyden again
                    use_dogleg = False
                    trust_radius = 1.0
                else:
                    # Switch to dogleg
                    use_dogleg = True
                    trust_radius = 1.0
                
                # Recompute Jacobian
                if call_count < 450:
                    J = _central_jacobian(x)
                    if J is None:
                        J = _fd_jacobian(x, Fx)
                    if J is None:
                        break
                    if abs(np.linalg.det(J)) < 1e-12:
                        J = J + np.eye(2) * 1e-6
                else:
                    J = _fd_jacobian(x, Fx)
                    if J is None:
                        break
                    if abs(np.linalg.det(J)) < 1e-12:
                        J = J + np.eye(2) * 1e-6
                stalled = 0
            else:
                # Recompute Jacobian on stall
                if call_count < 450:
                    J = _fd_jacobian(x, Fx)
                    if J is None:
                        break
                    if abs(np.linalg.det(J)) < 1e-12:
                        J = J + np.eye(2) * 1e-6
                stalled = 0
                if call_count > 450:
                    break

    return best_x
# EVOLVE-BLOCK-END
