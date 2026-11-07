"""Root finder for 3 fixed 2D systems: Nelder-Mead basin finding + Broyden polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Finds x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Quick Newton/Broyden attempt from x0.
    2. If not converged, use Nelder-Mead on ||F(x)||^2 to find a good basin.
    3. Switch to high-precision Broyden/Newton polish when close enough.
    4. Multi-start with diverse initial points if needed.
    """
    x0_arr = np.asarray(x0, dtype=float).copy()
    if x0_arr.shape != (2,) or not np.all(np.isfinite(x0_arr)):
        x0_arr = np.zeros(2)

    tol = 1e-8
    tol_sq = tol * tol
    t0 = time.time()
    deadline = 148.0

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

    def merit(x_pt):
        """Return ||F(x)||^2 or inf on failure."""
        Fx = safe_F(x_pt)
        if Fx is None:
            return float('inf')
        return float(np.dot(Fx, Fx))

    def compute_jacobian(x, Fx):
        """Compute Jacobian via forward differences. Uses 2 F-calls."""
        h = 1e-6
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        Fx_e1 = safe_F(x + e1)
        Fx_e2 = safe_F(x + e2)
        if Fx_e1 is None or Fx_e2 is None:
            return None
        J = np.column_stack([(Fx_e1 - Fx) / h, (Fx_e2 - Fx) / h])
        return J

    def broyden_newton(x_start, Fx_start, max_iters=25, tol_local=1e-8):
        """
        Broyden's method with rank-1 updates and backtracking line search.
        Returns (x, Fx, converged, calls_used).
        """
        x_cur = x_start.copy()
        Fx_cur = Fx_start.copy()
        norm_sq_cur = float(np.dot(Fx_cur, Fx_cur))
        calls_used = 0

        if norm_sq_cur <= tol_local * tol_local:
            return x_cur, Fx_cur, True, calls_used

        # Initial Jacobian
        J = compute_jacobian(x_cur, Fx_cur)
        if J is None:
            return x_cur, Fx_cur, False, 2
        calls_used += 2

        for _ in range(max_iters):
            if remaining() < 1.0:
                break

            # Solve J * s = -Fx
            try:
                s = np.linalg.solve(J, -Fx_cur)
            except np.linalg.LinAlgError:
                # Fallback: use steepest descent direction
                s = -Fx_cur
                s_norm = np.linalg.norm(s)
                if s_norm < 1e-15:
                    break
                s = s / s_norm * 0.1

            s_norm = np.linalg.norm(s)
            if s_norm < 1e-15:
                break

            # Backtracking line search on ||F||^2
            alpha = 1.0
            accepted = False
            x_try = x_cur
            Fx_try = Fx_cur
            norm_sq_try = norm_sq_cur

            for _ls in range(8):
                x_try = x_cur + alpha * s
                Fx_try = safe_F(x_try)
                calls_used += 1
                if Fx_try is None:
                    alpha *= 0.5
                    continue
                norm_sq_try = float(np.dot(Fx_try, Fx_try))
                if norm_sq_try <= tol_local * tol_local:
                    return x_try, Fx_try, True, calls_used
                if norm_sq_try < norm_sq_cur * 0.95:
                    accepted = True
                    break
                alpha *= 0.5

            if not accepted:
                break

            # Broyden rank-1 update
            Fx_diff = Fx_try - Fx_cur
            s_actual = x_try - x_cur
            # Update J: J += (Fx_diff - J @ s_actual) @ s_actual^T / (s_actual^T @ s_actual)
            denom = float(np.dot(s_actual, s_actual))
            if denom > 1e-30:
                J += np.outer(Fx_diff - J @ s_actual, s_actual) / denom

            x_cur = x_try
            Fx_cur = Fx_try
            norm_sq_cur = norm_sq_try

            # Occasionally recompute Jacobian for accuracy
            if _ % 5 == 4 and norm_sq_cur > 1e-12:
                J_new = compute_jacobian(x_cur, Fx_cur)
                if J_new is not None:
                    J = J_new
                    calls_used += 2
                else:
                    break

        return x_cur, Fx_cur, norm_sq_cur <= tol_local * tol_local, calls_used

    def nelder_mead(x_start, max_iters=50, tol_nm=1e-10):
        """
        Nelder-Mead on merit function ||F(x)||^2.
        Returns (x, merit_val, calls_used).
        """
        # Initialize simplex: 3 points in 2D
        step = 0.1
        simplex = [
            x_start.copy(),
            x_start + np.array([step, 0.0]),
            x_start + np.array([0.0, step])
        ]
        fvals = [merit(p) for p in simplex]
        calls_used = 3  # initial evaluations

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        for _ in range(max_iters):
            if remaining() < 2.0:
                break

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Check convergence
            fspread = abs(fvals[-1] - fvals[0])
            if fspread < tol_nm and fvals[0] <= tol * tol:
                break
            if fvals[0] <= tol * tol:
                break

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = merit(xr)
            calls_used += 1
            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = merit(xe)
                calls_used += 1
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = merit(xc)
                calls_used += 1
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = merit(simplex[i])
                        calls_used += 1

        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx], calls_used

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

    # Phase 1: Quick Broyden-Newton from x0
    x_sol, Fx_sol, conv, _ = broyden_newton(x0_arr, Fx0, max_iters=15, tol_local=1e-8)
    ns = float(np.dot(Fx_sol, Fx_sol)) if Fx_sol is not None else float('inf')
    if ns < best_norm_sq:
        best_norm_sq = ns
        best_x = x_sol.copy()
    if conv or best_norm_sq <= tol_sq:
        return best_x

    # Phase 2: Nelder-Mead to find a better basin
    if remaining() > 5.0:
        nm_x, nm_val, nm_calls = nelder_mead(best_x, max_iters=40)
        if nm_val < best_norm_sq:
            best_norm_sq = nm_val
            best_x = nm_x.copy()
            if best_norm_sq <= tol_sq:
                return best_x

        # Phase 3: High-precision polish from NM result
        Fx_nm = safe_F(best_x)
        if Fx_nm is not None:
            x_polish, Fx_polish, conv_polish, _ = broyden_newton(best_x, Fx_nm, max_iters=30, tol_local=1e-10)
            ns_polish = float(np.dot(Fx_polish, Fx_polish)) if Fx_polish is not None else float('inf')
            if ns_polish < best_norm_sq:
                best_norm_sq = ns_polish
                best_x = x_polish.copy()
            if conv_polish or best_norm_sq <= tol_sq:
                return best_x

    # Phase 4: Multi-start with diverse initial points
    rng = np.random.RandomState(12345)
    max_restarts = 12
    for restart in range(max_restarts):
        if remaining() < 3.0:
            break
        if best_norm_sq <= tol_sq:
            return best_x

        # Generate diverse starting points
        if restart % 4 == 0:
            x_start = x0_arr + rng.uniform(-15, 15, size=2)
        elif restart % 4 == 1:
            radius = 1.0 + 0.5 * restart
            angle = rng.uniform(0, 2 * np.pi)
            x_start = best_x + radius * np.array([np.cos(angle), np.sin(angle)])
        elif restart % 4 == 2:
            x_start = x0_arr * rng.uniform(0.3, 8.0)
        else:
            x_start = rng.uniform(-20, 20, size=2)

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

        # Quick Broyden from this start
        x_sol, Fx_sol, conv, _ = broyden_newton(x_start, Fx_start, max_iters=20, tol_local=1e-8)
        ns = float(np.dot(Fx_sol, Fx_sol)) if Fx_sol is not None else float('inf')
        if ns < best_norm_sq:
            best_norm_sq = ns
            best_x = x_sol.copy()
        if conv or best_norm_sq <= tol_sq:
            return best_x

    return best_x
# EVOLVE-BLOCK-END
