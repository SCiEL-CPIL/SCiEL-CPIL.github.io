import numpy as np
import time

def integrate(f, t_span, y0):
    """
    Integrate y' = f(t, y) from t_span[0] to t_span[1].
    
    Strategy: 
    Use a fixed-step 4th-order Runge-Kutta method (RK4).
    The number of steps is chosen to balance accuracy and the evaluation penalty.
    Score = -log10(relative_error) - 0.5 * log10(calls)
    
    For RK4, local error ~ h^5, global error ~ h^4.
    If N is the number of steps, h = T/N.
    Global error ~ (T/N)^4.
    Calls ~ 4*N.
    
    We want to maximize: -log10(C * (T/N)^4) - 0.5 * log10(4*N)
    = -log10(C) - 4*log10(T) + 4*log10(N) - 0.5*(log10(4) + log10(N))
    = Constant + 3.5 * log10(N)
    
    This suggests that increasing N always increases the score, limited only by the
    truncation error constant C and the maximum allowed calls.
    However, the truncation error approximation breaks down if h is too small
    due to floating point precision, or if the problem is stiff.
    But for a simple convection-diffusion problem on 100 points over [0, 0.5],
    it is likely non-stiff or mildly stiff.
    
    The constraint is "at most 2000000 calls".
    So 4*N <= 2,000,000 => N <= 500,000.
    
    Let's test if N=500,000 is feasible within the time limit.
    500,000 steps * 4 calls/step = 2,000,000 calls.
    Each call involves vector operations on size 100.
    In Python, a loop of 500,000 iterations might be slow.
    Let's estimate: 1000 iterations of RK4 on n=100 takes ~0.05s?
    500,000 iterations would take ~25s. This is well within the 300s limit.
    
    However, if the error constant C is large, we might not need that many steps.
    But since the score increases with N (due to the 3.5 * log10(N) term dominating
    the error reduction), we should push N as high as possible, up to the call limit
    or until floating point errors dominate.
    
    Let's use N = 250,000 (1,000,000 calls) to be safe on time, or even higher.
    Actually, let's check the time for 10,000 steps first to calibrate.
    
    Wait, the score formula is -log10(error) - 0.5*log10(calls).
    If we double N, error decreases by 16 (factor 2^4), calls double.
    Change in score: -log10(error/16) - 0.5*log10(2*calls)
    = (-log10(error) + log10(16)) - (0.5*log10(calls) + 0.5*log10(2))
    = Score_old + log10(16) - 0.5*log10(2)
    = Score_old + 1.204 - 0.1505 = Score_old + 1.05.
    
    So doubling N increases the score by ~1.05. This is a strong incentive to use
    the maximum number of steps allowed by the call limit.
    
    Max calls = 2,000,000.
    Calls per step = 4.
    Max steps = 500,000.
    
    Let's use N = 500,000.
    """
    t0, t1 = t_span
    y = np.asarray(y0, dtype=float).copy()
    dt_total = t1 - t0
    
    if dt_total == 0:
        return y
    
    # Determine number of steps
    # Use max steps allowed by call limit
    max_calls = 2000000
    calls_per_step = 4
    max_steps = max_calls // calls_per_step
    
    # Also consider time limit.
    # Let's do a quick timing test with a small number of steps to estimate time per step.
    start_time = time.time()
    
    # Use a small test to estimate speed
    test_steps = 100
    h_test = dt_total / test_steps
    t_test = t0
    y_test = y.copy()
    
    for _ in range(test_steps):
        k1 = f(t_test, y_test)
        k2 = f(t_test + 0.5*h_test, y_test + 0.5*h_test*k1)
        k3 = f(t_test + 0.5*h_test, y_test + 0.5*h_test*k2)
        k4 = f(t_test + h_test, y_test + h_test*k3)
        y_test = y_test + (h_test/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t_test += h_test
        
    time_per_step = (time.time() - start_time) / test_steps
    
    # We want to finish within 290 seconds (safety margin)
    time_limit = 290
    max_steps_by_time = int(time_limit / time_per_step)
    
    # Take the minimum of max_steps_by_call_limit and max_steps_by_time
    N = min(max_steps, max_steps_by_time)
    
    # Ensure N is at least 1
    N = max(1, N)
    
    h = dt_total / N
    t = t0
    
    for _ in range(N):
        # Check time limit periodically
        if (time.time() - start_time) > 290:
            break
            
        k1 = f(t, y)
        k2 = f(t + 0.5*h, y + 0.5*h*k1)
        k3 = f(t + 0.5*h, y + 0.5*h*k2)
        k4 = f(t + h, y + h*k3)
        
        y = y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        
    return y

