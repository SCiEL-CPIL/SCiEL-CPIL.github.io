# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    """
    Integrate the Lotka-Volterra system using scipy's DOP853 method.
    This matches the reference method, allowing us to achieve high accuracy
    with a controlled number of function calls.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    
    start_time = time.time()
    deadline = 325.0  # 5s margin before 330s kill

    def elapsed():
        return time.time() - start_time

    # We want to balance accuracy (rel error) and call count.
    # The reference uses DOP853 with rtol=1e-11, atol=1e-13.
    # If we use the same method and similar tolerances, we should get very close results.
    # However, solve_ivp's adaptive stepping might use different steps than the reference.
    # To ensure we are "close" to the reference, we can try to match its behavior or
    # simply use tight tolerances.
    
    # The score is -log10(rel) - 0.5 * log10(calls).
    # To maximize this, we want small rel and small calls.
    # rel < 1e-14 gives no further gain. So we aim for rel ~ 1e-15.
    # Calls: each order of magnitude increase costs 0.5.
    # DOP853 is 8th order.
    
    # Let's try a few configurations with solve_ivp.
    # We need to count calls. solve_ivp doesn't directly expose the count easily in a simple wrapper
    # unless we wrap f.
    
    call_count = [0]
    original_f = f
    
    def counting_f(t, y):
        call_count[0] += 1
        if call_count[0] > 1800000:
            raise RuntimeError("Too many calls")
        return original_f(t, y)

    best_y = None
    best_score = -np.inf

    # Candidate configurations: (rtol, atol, max_step)
    # max_step helps control the number of calls by preventing huge steps, 
    # but too small max_step increases calls.
    # t_span is [0, 5].
    # Let's try a few values for max_step.
    # If we don't specify max_step, DOP853 might take large steps if the solution is smooth.
    # For Lotka-Volterra, it's periodic/oscillatory.
    
    # Let's try a range of max_step values to find the sweet spot.
    # A rough estimate: if we want ~1000 calls, and DOP853 uses ~12 stages per step,
    # that's ~83 steps. h = 5/83 ~ 0.06.
    # If we want ~100 calls, that's ~8 steps. h ~ 0.6.
    
    # Let's try max_step values: 0.1, 0.05, 0.01, 0.005
    # And tight tolerances to ensure accuracy.
    
    configs = [
        (1e-11, 1e-13, 0.1),
        (1e-11, 1e-13, 0.05),
        (1e-11, 1e-13, 0.01),
        (1e-11, 1e-13, 0.005),
        (1e-12, 1e-14, 0.05),
        (1e-12, 1e-14, 0.01),
    ]

    for rtol, atol, max_step in configs:
        if elapsed() > deadline - 10:
            break
            
        try:
            sol = solve_ivp(
                counting_f, 
                t_span, 
                y0, 
                method='DOP853', 
                rtol=rtol, 
                atol=atol, 
                max_step=max_step,
                dense_output=False
            )
            
            if sol.success:
                y_end = sol.y[:, -1]
                if np.all(np.isfinite(y_end)):
                    # Estimate score to pick the best one if we have multiple
                    # We don't know rel yet, but we can assume lower calls is better
                    # if accuracy is sufficient.
                    # Let's just take the first successful one with reasonable calls?
                    # Or run all and pick the one with min calls assuming rel is good enough.
                    # Given the method matches the reference, rel should be very small.
                    
                    # Let's store the result with the fewest calls among successful runs
                    # assuming they are all accurate enough (rel < 1e-14).
                    if best_y is None or call_count[0] < best_calls:
                        best_y = y_end
                        best_calls = call_count[0]
                        
        except Exception as e:
            continue

    if best_y is not None:
        return best_y

    # Fallback if solve_ivp fails or times out
    # Use a simple RK4 with fixed steps
    def rk4(f, t0, t1, y0, steps):
        h = (t1 - t0) / steps
        t = t0
        y = y0.copy()
        for _ in range(steps):
            k1 = f(t, y)
            k2 = f(t + h/2, y + h/2 * k1)
            k3 = f(t + h/2, y + h/2 * k2)
            k4 = f(t + h, y + h * k3)
            y = y + h/6.0 * (k1 + 2*k2 + 2*k3 + k4)
            t += h
            if not np.all(np.isfinite(y)):
                return None
        return y

    # Try a few step counts for RK4
    for steps in [1000, 5000, 10000]:
        if elapsed() > deadline - 5:
            break
        y_end = rk4(counting_f, t0, t1, y0, steps)
        if y_end is not None and np.all(np.isfinite(y_end)):
            return y_end

    # Final fallback: return y0 (should not happen)
    return y0

# EVOLVE-BLOCK-END
