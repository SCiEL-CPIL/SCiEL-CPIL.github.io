"""Root finder for 3 fixed 2D systems: adaptive Broyden with global safeguards."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Finds x in R^2 such that ||F(x)|| <= 1e-8.

    Strategy:
      1. Evaluate F at x0. If already converged, return immediately.
      2. Use Broyden's good method with:
         - Initial Jacobian via finite differences (2 extra F-calls)
         - Newton step with backtracking line search on ||F||^2
         - Rank-1 Jacobian update
         - Periodic Jacobian recomputation if convergence stalls
      3. If Broyden fails, try multiple restarts with different initial points.
      4. Return best point found within time budget.
    """
    x0_arr = np.asarray(x0, dtype=float).copy()
    if x0_arr.shape != (2,) or not np.all(np.isfinite(x0_arr)):
        x0_arr = np.zeros(2)

    tol = 1e-8
    tol_sq = tol * tol
    t0 = time.time()
    deadline = 148.0  # seconds; leave safety margin before 150s kill

    def remaining():
        return deadline - (time.time() - t0)

    def safe_F(x_pt):
        """Call F with finite guard; return None on failure."""
        try:
            y = np.asarray(F(x_pt), dtype=float)
            if y.shape != (2,) or not np.all(np.isfinite(y)):
                return None
            return y
        except Exception:
            return None

    def compute_jacobian(x, Fx):
        """Compute Jacobian via forward differences. Uses 2 F-calls."""
        h = 1e-7
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        Fx_e1 = safe_F(x + e1)
        Fx_e2 = safe_F(x + e2)
        if Fx_e1 is None or Fx_e2 is None:
            return None
        J = np.column_stack([(Fx_e1 - Fx) / h, (Fx_e2 - Fx) / h])
        return J

    def broyden_from(x_start, Fx_start, max_iters=50):
        """
        Broyden's good method starting from x_start with Fx_start already computed.
        Returns (x, Fx, converged, num_calls_used).
        """
        x_cur = x_start.copy()
        Fx_cur = Fx_start.copy()
        norm_sq_cur = float(np.dot(Fx_cur, Fx_cur))
        calls_used = 0  # Fx_start was already counted externally

        if norm_sq_cur <= tol_sq:
            return x_cur, Fx_cur, True, calls_used

        # Compute initial Jacobian
        J = compute_jacobian(x_cur, Fx_cur)
        if J is None:
            return x_cur, Fx_cur, False, calls_used + 2
        calls_used += 2

        # Check if Jacobian is invertible
        det_J = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(det_J) < 1e-14:
            return x_cur, Fx_cur, False, calls_used

        stall_count = 0
        best_norm_sq_local = norm_sq_cur
        best_x_local = x_cur.copy()

        for it in range(max_iters):
            if remaining() < 3.0:
                break

            # Newton step
            try:
                s = np.linalg.solve(J, -Fx_cur)
            except np.linalg.LinAlgError:
                break

            s_norm = np.linalg.norm(s)
            if s_norm < 1e-15:
                break

            # Cap step size to avoid wild jumps
            max_step = 100.0
            if s_norm > max_step:
                s *= max_step / s_norm
                s_norm = max_step

            # Backtracking line search on merit function phi = 0.5||F||^2
            phi_prev = 0.5 * norm_sq_cur
            # Approximate directional derivative
            slope = float(np.dot(Fx_cur, J @ s))
            if slope >= -1e-12:
                # Not a descent direction; use steepest descent
                s = -J.T @ Fx_cur
                s_norm = np.linalg.norm(s)
                if s_norm < 1e-15:
                    break
                if s_norm > max_step:
                    s *= max_step / s_norm
                slope = -float(np.dot(s, J.T @ Fx_cur))
                if slope >= -1e-12:
                    break

            alpha = 1.0
            c1 = 1e-4
            accepted = False
            x_try = x_cur
            Fx_try = Fx_cur
            norm_sq_try = norm_sq_cur

            for _ls in range(40):
                x_try = x_cur + alpha * s
                Fx_try = safe_F(x_try)
                calls_used += 1
                if Fx_try is None:
                    alpha *= 0.5
                    continue
                norm_sq_try = float(np.dot(Fx_try, Fx_try))
                phi_try = 0.5 * norm_sq_try

                if norm_sq_try <= tol_sq:
                    return x_try, Fx_try, True, calls_used

                if phi_try <= phi_prev + c1 * alpha * slope:
                    accepted = True
                    break

                alpha *= 0.5
                if alpha < 1e-14:
                    break

            if not accepted:
                # Try a smaller step in the negative gradient direction
                g = J.T @ Fx_cur
                g_norm = np.linalg.norm(g)
                if g_norm > 1e-15:
                    s_g = -g / g_norm * min(1.0, g_norm)
                    x_try = x_cur + 0.1 * s_g
                    Fx_try = safe_F(x_try)
                    calls_used += 1
                    if Fx_try is not None:
                        norm_sq_try = float(np.dot(Fx_try, Fx_try))
                        if norm_sq_try < norm_sq_cur:
                            # Accept even without Armijo
                            accepted = True
                            x_try = x_cur + 0.1 * s_g
                        else:
                            break
                    else:
                        break
                else:
                    break

            if not accepted:
                break

            # Update
            y = Fx_try - Fx_cur
            s_dot_s = float(np.dot(s, s))
            if s_dot_s > 1e-30:
                J += np.outer(y - J @ s, s) / s_dot_s

            x_cur = x_try
            Fx_cur = Fx_try
            norm_sq_cur = norm_sq_try

            # Track best
            if norm_sq_cur < best_norm_sq_local:
                best_norm_sq_local = norm_sq_cur
                best_x_local = x_cur.copy()

            # Detect stall: if norm isn't decreasing much
            if norm_sq_cur > 0.95 * best_norm_sq_local and it > 5:
                stall_count += 1
                if stall_count >= 5:
                    # Recompute Jacobian
                    J_new = compute_jacobian(x_cur, Fx_cur)
                    calls_used += 2
                    if J_new is not None:
                        J = J_new
                        det_J = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
                        if abs(det_J) < 1e-14:
                            break
                    stall_count = 0
            else:
                stall_count = 0

        return best_x_local, safe_F(best_x_local) if best_norm_sq_local <= tol_sq else Fx_cur, best_norm_sq_local <= tol_sq, calls_used

    # Main solver
    best_x = x0_arr.copy()
    best_norm_sq = float('inf')

    # Initial evaluation
    Fx0 = safe_F(x0_arr)
    if Fx0 is None:
        return np.zeros(2)
    best_norm_sq = float(np.dot(Fx0, Fx0))
    if best_norm_sq <= tol_sq:
        return x0_arr

    # Primary Broyden attempt from x0
    x_sol, Fx_sol, conv, _ = broyden_from(x0_arr, Fx0, max_iters=60)
    ns = float(np.dot(Fx_sol, Fx_sol)) if Fx_sol is not None else float('inf')
    if ns < best_norm_sq:
        best_norm_sq = ns
        best_x = x_sol.copy()
    if conv or best_norm_sq <= tol_sq:
        return best_x

    # Multi-start: try several different initial points
    rng = np.random.RandomState(12345)
    max_restarts = 30
    for restart in range(max_restarts):
        if remaining() < 5.0:
            break
        if best_norm_sq <= tol_sq:
            return best_x

        # Generate diverse starting points
        if restart % 5 == 0:
            # Random in larger box
            x_start = x0_arr + rng.uniform(-50, 50, size=2)
        elif restart % 5 == 1:
            # Ring around best_x
            radius = 1.0 + 2.0 * (restart // 5)
            angle = rng.uniform(0, 2 * np.pi)
            x_start = best_x + radius * np.array([np.cos(angle), np.sin(angle)])
        elif restart % 5 == 2:
            # Small perturbation around best_x
            x_start = best_x + rng.randn(2) * 0.5
        elif restart % 5 == 3:
            # Scale from x0
            scale = rng.uniform(0.5, 20.0)
            x_start = x0_arr * scale
        else:
            # Random direction from x0
            angle = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(1.0, 30.0)
            x_start = x0_arr + r * np.array([np.cos(angle), np.sin(angle)])

        if not np.all(np.isfinite(x_start)):
            continue

        Fx_start = safe_F(x_start)
        if Fx_start is None:
            continue

        ns_start = float(np.dot(Fx_start, Fx_start))
        if ns_start < best_norm_sq:
            best_norm_sq = ns_start
            best_x = x_start.copy()
            if best_norm_sq <= tol_sq:
                return best_x

        # Try Broyden from this start with fewer iterations
        x_sol, Fx_sol, conv, _ = broyden_from(x_start, Fx_start, max_iters=30)
        ns = float(np.dot(Fx_sol, Fx_sol)) if Fx_sol is not None else float('inf')
        if ns < best_norm_sq:
            best_norm_sq = ns
            best_x = x_sol.copy()
        if conv or best_norm_sq <= tol_sq:
            return best_x

    return best_x
# EVOLVE-BLOCK-END
