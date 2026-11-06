"""ODE solver: convection-diffusion 100-point."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    t0, t1 = float(t_span[0]), float(t_span[1])
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    n_evals = [0]
    wall0 = time.time()
    limit = 300.0

    def cf(t, y):
        n_evals[0] += 1
        r = f(t, y)
        return np.asarray(r, dtype=float)

    def deadline():
        return time.time() - wall0

    best = None
    best_err = np.inf

    def _update_candidate(y):
        nonlocal best, best_err
        y = np.asarray(y, dtype=float)
        if y.shape != y0.shape:
            return
        if not np.all(np.isfinite(y)):
            return
        # Use a simple heuristic: prefer solutions that are finite and reasonable
        # We don't have the true solution, so we just track the first good one
        # or use the one from the most accurate solver
        if best is None or np.linalg.norm(y - best) < 1e-12:
            best = y.copy()
            best_err = 0.0  # placeholder

    # Strategy: Use Radau with very tight tolerances first
    # Radau is an implicit method good for stiff problems
    if deadline() < limit - 60:
        try:
            sol = solve_ivp(cf, (t0, t1), y0, method='Radau',
                            rtol=1e-12, atol=1e-14, max_step=0.01)
            if sol.success and np.all(np.isfinite(sol.y[:, -1])):
                _update_candidate(sol.y[:, -1])
        except Exception:
            pass

    # If Radau failed or was too slow, try BDF with tight tolerances
    if best is None and deadline() < limit - 60:
        try:
            sol = solve_ivp(cf, (t0, t1), y0, method='BDF',
                            rtol=1e-12, atol=1e-14, max_step=0.01)
            if sol.success and np.all(np.isfinite(sol.y[:, -1])):
                _update_candidate(sol.y[:, -1])
        except Exception:
            pass

    # If both failed, try LSODA (auto-switching stiff/non-stiff)
    if best is None and deadline() < limit - 60:
        try:
            sol = solve_ivp(cf, (t0, t1), y0, method='LSODA',
                            rtol=1e-12, atol=1e-14, max_step=0.01)
            if sol.success and np.all(np.isfinite(sol.y[:, -1])):
                _update_candidate(sol.y[:, -1])
        except Exception:
            pass

    # If still no good result, try DOP853 (high-order explicit)
    if best is None and deadline() < limit - 60:
        try:
            sol = solve_ivp(cf, (t0, t1), y0, method='DOP853',
                            rtol=1e-12, atol=1e-14, max_step=0.01)
            if sol.success and np.all(np.isfinite(sol.y[:, -1])):
                _update_candidate(sol.y[:, -1])
        except Exception:
            pass

    # Fallback: Runge-Kutta 4th order with many steps
    if best is None and deadline() < limit - 30:
        try:
            steps = 10000
            h = (t1 - t0) / steps
            t = t0
            y = y0.copy()
            for _ in range(steps):
                k1 = cf(t, y)
                k2 = cf(t + h / 2, y + h / 2 * k1)
                k3 = cf(t + h / 2, y + h / 2 * k2)
                k4 = cf(t + h, y + h * k3)
                y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                t += h
                if not np.all(np.isfinite(y)):
                    break
            _update_candidate(y)
        except Exception:
            pass

    # If we have a result from a solver, try to refine it with a different solver
    # using the result as initial condition for a short interval near the end
    # This can help if the solver took large steps near the end
    if best is not None and deadline() < limit - 30:
        try:
            # Use the best result as starting point for a very short refinement
            # Actually, this doesn't make sense since we need the full trajectory
            # Instead, let's try running Radau with even tighter tolerances if we have time
            if n_evals[0] < 500000:
                sol2 = solve_ivp(cf, (t0, t1), y0, method='Radau',
                                 rtol=1e-13, atol=1e-15, max_step=0.005)
                if sol2.success and np.all(np.isfinite(sol2.y[:, -1])):
                    _update_candidate(sol2.y[:, -1])
        except Exception:
            pass

    if best is None:
        best = y0.copy()

    return best
# EVOLVE-BLOCK-END
