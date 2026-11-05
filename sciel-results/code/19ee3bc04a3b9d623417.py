import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    """Integrate y' = f(t, y) over t_span and return y at t_end."""
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    n = y0_arr.size
    deadline = time.time() + 290.0

    # Strategy: Use DOP853 with optimized step constraints to balance accuracy
    # against the log10(calls) penalty. The score is -log10(rel_err) - 0.5*log10(n_calls).
    # For RK4-like methods, error ~ C*h^p, so relative error ~ C*(T/N)^p.
    # Score ~ p*log10(N) - log10(C) - 0.5*log10(4*N) for RK4 (4 calls/step)
    # = (p - 0.5)*log10(N) - log10(C) - 0.5*log10(4)
    # For p=4 (RK4): score increases with N, so use as many steps as feasible.
    # For DOP853 (p=8, ~12 calls/step): score ~ 7.5*log10(N) - const, also increases with N.
    # But DOP853 is adaptive, so we need to constrain max_step to force resolution.

    best_y = None
    best_score = -np.inf

    # Primary: DOP853 with tight tolerances and constrained max_step
    # For 100-point convection-diffusion on [0, 0.5], we need enough resolution.
    # Start with max_step that gives ~200 steps (conservative for stability)
    try:
        sol = solve_ivp(
            f,
            t_span,
            y0_arr,
            method='DOP853',
            rtol=1e-11,
            atol=1e-13,
            max_step=(t1 - t0) / 200.0,
        )
        if sol.success and np.all(np.isfinite(sol.y[:, -1])):
            n_calls = len(sol.t) * 12  # approximate DOP853 calls per step
            rel_err_est = 1e-11  # rough estimate based on rtol
            score_est = -np.log10(rel_err_est) - 0.5 * np.log10(max(n_calls, 1))
            if score_est > best_score:
                best_score = score_est
                best_y = sol.y[:, -1].copy()
    except Exception:
        pass

    # If we have time, try with more steps for higher accuracy
    if best_y is not None and time.time() < deadline - 60.0:
        try:
            sol2 = solve_ivp(
                f,
                t_span,
                y0_arr,
                method='DOP853',
                rtol=1e-12,
                atol=1e-14,
                max_step=(t1 - t0) / 400.0,
            )
            if sol2.success and np.all(np.isfinite(sol2.y[:, -1])):
                n_calls2 = len(sol2.t) * 12
                rel_err_est2 = 1e-12
                score_est2 = -np.log10(rel_err_est2) - 0.5 * np.log10(max(n_calls2, 1))
                if score_est2 > best_score:
                    best_score = score_est2
                    best_y = sol2.y[:, -1].copy()
        except Exception:
            pass

    if best_y is not None:
        return best_y

    # Fallback: fixed-step RK4 with optimal step count
    # For RK4: score ~ 4*log10(N) - 0.5*log10(4*N) - log10(C)
    # = 3.5*log10(N) - 0.5*log10(4) - log10(C)
    # Maximize N subject to time constraint and 2M call limit
    remaining = deadline - time.time()
    # Estimate: each RK4 step takes ~50 microseconds for n=100
    est_time_per_step = 50e-6
    max_steps_by_time = int(remaining / est_time_per_step)
    max_steps_by_calls = 2000000 // 4
    max_steps = min(max_steps_by_time, max_steps_by_calls)
    max_steps = max(max_steps, 500)
    max_steps = min(max_steps, 500000)  # safety cap

    h = (t1 - t0) / max_steps
    t = t0
    y = y0_arr.copy()

    for i in range(max_steps):
        if time.time() > deadline - 10.0:
            break
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h

    return y
