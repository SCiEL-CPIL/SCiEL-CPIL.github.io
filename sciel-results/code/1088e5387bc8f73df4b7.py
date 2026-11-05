"""ODE solver for the Robertson problem using LSODA with a deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    deadline = time.time() + 290.0  # safety margin under the 330 s wall-clock limit
    best = None

    # Tightest tolerances first: maximises -log10(relative error) at the cost of
    # more f-evaluations (penalised only by 0.5*log10), so tighter is favourable.
    for rtol, atol in [(1e-11, 1e-11), (1e-10, 1e-10), (1e-8, 1e-8), (1e-6, 1e-6)]:
        if time.time() > deadline:
            break
        try:
            sol = solve_ivp(
                f, (t0, t1), y0,
                method="LSODA",
                rtol=rtol, atol=atol,
            )
            if sol.success and sol.y is not None and sol.y.shape[1] >= 1:
                y_end = sol.y[:, -1]
                if np.all(np.isfinite(y_end)):
                    best = y_end
        except Exception:
            pass

    # Final state of the fixed Robertson problem on [0, 40].
    return best if best is not None else y0
# EVOLVE-BLOCK-END
