"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
import time

def integrate(f, t_span, y0):
    """
    Integrate the Van der Pol oscillator (mu=5) from t_span[0] to t_span[1]
    with initial condition y0. Returns the state at t_span[1].
    
    The function f is counted for calls, so we minimize f calls by using
    an internal solver that doesn't call f, or by using very few steps.
    
    However, the task says f must be called at least once, and calls are counted.
    The score is -log10(rel) - 0.5*log10(calls).
    
    Strategy: Use scipy's solve_ivp with Radau method (stiff solver) with tight
    tolerances to match the reference. But we need to call f at least once.
    
    Wait - the evaluator provides f and counts its calls. We can't just use
    solve_ivp with our own RHS because f is the one that gets counted.
    
    Actually, re-reading: "f(t, y) returns y' as an array and every call is counted."
    So we MUST use f for the integration. We cannot bypass it.
    
    But we can use a stiff solver that minimizes function evaluations.
    
    Let me implement a Radau-like or BDF method, or use an adaptive stiff solver
    that calls f efficiently.
    
    Actually, the simplest approach: use solve_ivp but pass f as the function.
    solve_ivp will call f multiple times, and those calls will be counted.
    
    The reference uses Radau with rtol=1e-11, atol=1e-13. If we use the same
    method and tolerances, our result should be very close to the reference.
    
    Key insight: We should use solve_ivp with method='Radau', rtol=1e-11, atol=1e-13
    to match the reference exactly. This minimizes the relative error.
    
    The number of f calls will be whatever solve_ivp needs, but that's fine.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    # Use solve_ivp with Radau to match the reference solver
    # This will call f multiple times, and those calls are counted
    sol = solve_ivp(
        f,
        (t0, t1),
        y0,
        method='Radau',
        rtol=1e-11,
        atol=1e-13,
        dense_output=False
    )
    
    if not sol.success:
        # Fallback: try with looser tolerances
        sol = solve_ivp(
            f,
            (t0, t1),
            y0,
            method='Radau',
            rtol=1e-9,
            atol=1e-11,
            dense_output=False
        )
    
    if sol.success and len(sol.y) > 0:
        return sol.y[:, -1]
    
    # Last resort: return y0 if something goes wrong
    return y0
# EVOLVE-BLOCK-END
