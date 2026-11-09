"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.asarray(y0, dtype=float)
    
    # Robertson problem is stiff. We use an implicit method (BDF or Radau) with tight tolerances.
    # The scoring function is: -log10(relative error) - 0.5 * log10(evaluations).
    # We want to minimize error and evaluations.
    
    # First, try a high-precision solve with BDF.
    # solve_ivp counts function calls internally, but we need to know them for the score.
    # Actually, the evaluator counts calls to f. So we should wrap f to count calls?
    # No, the evaluator likely wraps f before passing it to us, or counts calls to the f we receive.
    # The prompt says: "scored by ... against a SciPy reference solution, with at most 2000000 calls (higher is better)."
    # "The evaluator counts calls to f."
    # This implies the evaluator wraps the original f, passes the wrapped f to integrate, and counts calls to that wrapped f.
    # So we don't need to count calls ourselves, just use f efficiently.
    
    # However, to be safe and to optimize the trade-off, we can try to get a very accurate solution with few calls.
    # Radau is often more efficient for stiff problems than BDF in terms of accuracy per function evaluation.
    
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s margin
    
    # Try Radau first with tight tolerances
    # rtol and atol should be small.
    # Let's try rtol=1e-10, atol=1e-12
    sol = solve_ivp(f, t_span, y0_arr, method='Radau', rtol=1e-10, atol=1e-12)
    
    if not sol.success:
        # Try BDF
        sol = solve_ivp(f, t_span, y0_arr, method='BDF', rtol=1e-10, atol=1e-12)
    
    if sol.success:
        return sol.y[:, -1]
    else:
        # Fallback to a very fine explicit method if implicit fails (unlikely for Robertson)
        # Or just return the last computed value
        if sol.y.size > 0:
            return sol.y[:, -1]
        else:
            return y0_arr

# EVOLVE-BLOCK-END
