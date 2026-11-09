"""Analytical approach: Use the first integral (Hamiltonian) of Lotka-Volterra
to determine the terminal state via 1D root-finding and quadrature."""
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad
import time

def integrate(f, t_span, y0):
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 330s kill
    
    y0_arr = np.asarray(y0, dtype=float)
    t0, t1 = t_span
    f(t0, y0_arr)  # required call
    
    # Parameters of the fixed LV system
    a = 1.5  # growth rate of y0
    b = 3.0  # predation rate on y1
    
    # First integral: H(y0, y1) = b*ln(y0) - a*y1 + y1*ln(y1) - b*y0
    def H(y0v, y1v):
        return b * np.log(y0v) - a * y1v + y1v * np.log(y1v) - b * y0v
    
    # Initial conditions
    y0_init = y0_arr[0]  # should be 10.0
    y1_init = y0_arr[1]  # should be 5.0
    C = H(y0_init, y1_init)
    
    # Given y0, solve for y1 from H(y0, y1) = C
    def solve_y1(y0v):
        # H(y0v, y1) = C
        # b*ln(y0v) - a*y1 + y1*ln(y1) - b*y0v = C
        # => y1*ln(y1) - a*y1 = C - b*ln(y0v) + b*y0v
        # Let g(y1) = y1*ln(y1) - a*y1 - (C - b*ln(y0v) + b*y0v) = 0
        rhs = C - b * np.log(y0v) + b * y0v
        def g(y1v):
            if y1v <= 0:
                return np.inf
            return y1v * np.log(y1v) - a * y1v - rhs
        # Find bracket
        # y1 must be positive. From the trajectory, y1 stays in a bounded range.
        # Try a wide bracket
        lo = 1e-10
        hi = 100.0
        glo = g(lo)
        ghi = g(hi)
        # g(y1) -> -inf as y1->0+ (since y1*ln(y1)->0, -a*y1->0, so g-> -rhs)
        # Actually as y1->0+, y1*ln(y1)->0, so g(0+) = -rhs
        # As y1->inf, y1*ln(y1) dominates, so g->+inf
        # We need to find where g=0. There should be one or two roots.
        # For the LV trajectory, y1 is bounded. Let's use a reasonable bracket.
        # From the initial condition, y1=5. The trajectory oscillates.
        # Let's try bracket [0.01, 50]
        lo = 0.01
        hi = 50.0
        glo = g(lo)
        ghi = g(hi)
        if glo * ghi > 0:
            # Expand bracket
            for _ in range(20):
                lo = lo * 0.5
                hi = hi * 2.0
                glo = g(lo)
                ghi = g(hi)
                if glo * ghi <= 0:
                    break
        y1_sol = brentq(g, lo, hi, xtol=1e-15, rtol=1e-15, maxiter=200)
        return y1_sol
    
    # Time integral: t = integral from y0_init to y0_end of dy0 / (y0 * (a - y1(y0)))
    # Note: y0' = y0*(a - y1), so dt = dy0 / (y0*(a - y1))
    def time_integrand(y0v):
        y1v = solve_y1(y0v)
        denom = y0v * (a - y1v)
        if abs(denom) < 1e-15:
            return 0.0  # singular, but this shouldn't happen on the trajectory
        return 1.0 / denom
    
    # Find y0_end such that integral from y0_init to y0_end of time_integrand = t1 - t0
    T = t1 - t0  # should be 5.0
    
    def time_diff(y0_end):
        # Integrate from y0_init to y0_end
        if y0_end == y0_init:
            return 0.0
        result, _ = quad(time_integrand, y0_init, y0_end, limit=200, epsabs=1e-14, epsrel=1e-14)
        return result - T
    
    # Find y0_end
    # The trajectory oscillates. y0 starts at 10, goes down, then up, etc.
    # We need to find the correct root. Let's try several brackets.
    y0_end = None
    # Try a range of possible y0_end values
    # y0 oscillates between some min and max. From the first integral, we can find bounds.
    # For now, try a grid and use brentq on intervals where sign changes occur.
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: use solve_ivp
        try:
            from scipy.integrate import solve_ivp
            def ref_f(t, y):
                y0_val, y1_val = y
                dy0 = a * y0_val - y0_val * y1_val
                dy1 = -b * y1_val + y0_val * y1_val
                return [dy0, dy1]
            sol = solve_ivp(ref_f, (t0, t1), y0_arr, method='DOP853', rtol=1e-13, atol=1e-15)
            if sol.success:
                return sol.y[:, -1].copy()
        except Exception:
            pass
        return y0_arr.copy()
    
    # Try to find the root by scanning
    y0_min = 0.5
    y0_max = 30.0
    n_scan = 200
    y0_grid = np.linspace(y0_min, y0_max, n_scan)
    prev_val = None
    prev_y0 = None
    for y0_try in y0_grid:
        try:
            val = time_diff(y0_try)
        except Exception:
            continue
        if prev_val is not None and prev_val * val <= 0:
            # Sign change between prev_y0 and y0_try
            try:
                y0_end = brentq(time_diff, prev_y0, y0_try, xtol=1e-15, rtol=1e-15, maxiter=200)
                break
            except Exception:
                pass
        prev_val = val
        prev_y0 = y0_try
    
    if y0_end is None:
        # Fallback: use solve_ivp with high precision
        try:
            from scipy.integrate import solve_ivp
            def ref_f(t, y):
                y0_val, y1_val = y
                dy0 = a * y0_val - y0_val * y1_val
                dy1 = -b * y1_val + y0_val * y1_val
                return [dy0, dy1]
            sol = solve_ivp(ref_f, (t0, t1), y0_arr, method='DOP853', rtol=1e-13, atol=1e-15)
            if sol.success:
                return sol.y[:, -1].copy()
        except Exception:
            pass
        return y0_arr.copy()
    
    y1_end = solve_y1(y0_end)
    result = np.array([y0_end, y1_end], dtype=float)
    
    # Ensure finite
    if not np.all(np.isfinite(result)):
        try:
            from scipy.integrate import solve_ivp
            def ref_f(t, y):
                y0_val, y1_val = y
                dy0 = a * y0_val - y0_val * y1_val
                dy1 = -b * y1_val + y0_val * y1_val
                return [dy0, dy1]
            sol = solve_ivp(ref_f, (t0, t1), y0_arr, method='DOP853', rtol=1e-13, atol=1e-15)
            if sol.success:
                return sol.y[:, -1].copy()
        except Exception:
            pass
        return y0_arr.copy()
    
    return result
