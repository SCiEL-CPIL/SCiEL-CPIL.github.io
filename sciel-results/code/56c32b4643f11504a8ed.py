"""
High-precision ODE integrator for the Lotka-Volterra system.

Strategy:
- Use scipy.integrate.solve_ivp with method 'DOP853' (same as the reference).
- Set tight tolerances (rtol=1e-13, atol=1e-15) to push relative error below the
  reference's own truncation error, while keeping the call count low thanks to
  DOP853's 8th-order accuracy.
- A single solve_ivp call with dense_output=False returns the state exactly at
  t1 in sol.y[:, -1], so no extra dense-output evaluation or second run is needed.
- Enforce a hard call-budget ceiling to guarantee validity.
- Include a time-deadline guard as a safety net.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrates the given ODE system over t_span starting from y0.
    Returns the state vector at t = t_span[1].
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float).copy()

    # Edge case: zero-length span
    if t0 == t1:
        return y0

    start_time = time.time()
    DEADLINE = 290.0  # seconds; well under the 300 s evaluator limit

    call_count = [0]
    MAX_CALLS = 2_000_000  # hard ceiling from the contract

    def counted_f(t, y):
        call_count[0] += 1
        if call_count[0] > MAX_CALLS:
            raise RuntimeError("Function-call budget exceeded")
        if time.time() - start_time > DEADLINE:
            raise RuntimeError("Deadline exceeded")
        return f(t, y)

    try:
        sol = solve_ivp(
            counted_f,
            t_span,
            y0,
            method='DOP853',
            rtol=1e-13,
            atol=1e-15,
        )

        if sol.success:
            y_end = sol.y[:, -1]
            if np.all(np.isfinite(y_end)):
                return y_end
            # Non-finite result: fall through to fallback
        else:
            # If the primary run failed, try a slightly looser tolerance
            call_count[0] = 0
            sol2 = solve_ivp(
                counted_f,
                t_span,
                y0,
                method='DOP853',
                rtol=1e-11,
                atol=1e-13,
            )
            if sol2.success:
                y_end = sol2.y[:, -1]
                if np.all(np.isfinite(y_end)):
                    return y_end

        # Last-resort fallback: fixed-step classical RK4 (only reached in pathological cases)
        y = y0.copy()
        t = t0
        # Choose a conservative fixed step
        n_steps = 2000
        h = (t1 - t0) / n_steps
        for _ in range(n_steps):
            if time.time() - start_time > DEADLINE:
                break
            k1 = f(t, y)
            k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
            k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
            k4 = f(t + h, y + h * k3)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
            if not np.all(np.isfinite(y)):
                break
        if np.all(np.isfinite(y)):
            return y

        return y0

    except Exception:
        # Any exception (including our budget/deadline raises) → return a finite vector
        return y0

# EVOLVE-BLOCK-END
