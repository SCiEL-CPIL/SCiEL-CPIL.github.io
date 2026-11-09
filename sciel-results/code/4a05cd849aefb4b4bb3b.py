"""
Bermudan Put Exercise Policy via Put-Call Parity and High-Resolution Grid.

Strategy:
1. Leverage Put-Call Parity for Bermudan options:
   V_put(S, t) = V_call(S, t) - S + K * exp(-r * (T - t))
   where V_call is the Bermudan call value.
   This allows us to compute the put's continuation value using the call's value function,
   which is smoother and more regular, potentially leading to more accurate boundary detection.
2. Construct a high-resolution log-spaced grid of spot prices.
3. Perform backward induction for the Bermudan call, computing V_call at each step.
4. Derive V_put from V_call using parity.
5. Determine the optimal exercise policy for the put by comparing payoff vs. continuation.
6. Use high-precision root-finding (Brent's method) to find the exact exercise boundary
   at each time step, decoupling threshold accuracy from grid resolution.
7. The resulting policy is a set of precise thresholds, one for each exercise date.
"""
import numpy as np
import time
from scipy.optimize import brentq
from scipy.special import roots_hermite

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    
    # Determine number of exercise dates
    if 'n_steps' in contract:
        n_exercise = contract['n_steps']
    elif 'n_exercise' in contract:
        n_exercise = contract['n_exercise']
    else:
        try:
            n_cols = training_paths.shape[1]
            n_exercise = n_cols - 1
        except:
            n_exercise = 12

    dt = T / n_exercise
    
    # Grid construction: high-resolution log-spaced grid
    n_grid = 500
    log_S_min = np.log(5.0)
    log_S_max = np.log(500.0)
    log_S_grid = np.linspace(log_S_min, log_S_max, n_grid)
    S_grid = np.exp(log_S_grid)
    
    # Quadrature setup for expectation
    n_quad = 64
    x_quad, w_quad = roots_hermite(n_quad)
    quad_z = np.sqrt(2) * x_quad
    quad_w = w_quad / np.sqrt(np.pi)
    
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    disc = np.exp(-r * dt)
    
    # Backward induction for Bermudan CALL
    # V_call at maturity = max(S - K, 0)
    V_call_next = np.maximum(S_grid - K, 0.0)
    
    # Store call values at each step for parity
    call_values = [None] * n_exercise
    call_values[-1] = V_call_next.copy()
    
    for i in range(n_exercise - 2, -1, -1):
        if time.time() > deadline - 10.0:
            # Fallback: assume no exercise for remaining steps
            for k in range(i, -1, -1):
                call_values[k] = V_call_next.copy()
            break
            
        # Compute expected call value at next step
        log_S_next = np.log(S_grid)[:, None] + drift + vol * quad_z[None, :]
        S_next = np.exp(log_S_next)
        
        # Interpolate V_call_next in log-S space
        log_S_next_flat = log_S_next.flatten()
        V_call_next_interp = np.interp(log_S_next_flat, log_S_grid, V_call_next)
        V_call_next_matrix = V_call_next_interp.reshape(n_grid, n_quad)
        
        E_V_call_next = np.sum(V_call_next_matrix * quad_w[None, :], axis=1)
        
        # Continuation value
        cont_val = disc * E_V_call_next
        
        # Payoff for call
        pay = np.maximum(S_grid - K, 0.0)
        
        # Optimal call value
        V_call_curr = np.maximum(pay, cont_val)
        
        call_values[i] = V_call_curr
        V_call_next = V_call_curr
    
    # Now compute put policy using Put-Call Parity
    # V_put(S, t_i) = V_call(S, t_i) - S + K * exp(-r * (T - t_i))
    # Exercise if payoff_put >= continuation_put
    # payoff_put = max(K - S, 0)
    # continuation_put = V_put - payoff_put if we continue, but actually:
    # V_put = max(payoff_put, disc * E[V_put_next])
    # So we exercise if payoff_put >= disc * E[V_put_next]
    # Using parity: V_put = V_call - S + K*exp(-r*(T-t))
    # So continuation_put = V_call - S + K*exp(-r*(T-t)) - payoff_put  (if we continue)
    # But actually, the continuation value is just V_put when we don't exercise.
    # So we exercise if payoff_put >= V_put (where V_put includes the max)
    # Wait, V_put is the optimal value, so V_put >= payoff_put always.
    # We exercise if payoff_put == V_put, i.e., if continuing is not better.
    # Equivalent: exercise if payoff_put >= disc * E[V_put_next]
    
    # Let's compute the put continuation value directly from call values
    # V_put_cont(S, t_i) = V_call(S, t_i) - S + K * exp(-r * (T - t_i))
    # But this is the total put value, not just continuation.
    # Actually, V_put(S, t_i) = max(payoff_put, V_put_cont)
    # where V_put_cont is the value if we continue, which equals V_call - S + K*exp(-r*(T-t_i))
    # So we exercise if payoff_put >= V_call - S + K*exp(-r*(T-t_i))
    
    thresholds = [None] * n_exercise
    
    for i in range(n_exercise):
        if time.time() > deadline - 5.0:
            # Fallback: use simple ITM threshold
            for k in range(i, n_exercise):
                thresholds[k] = K
            break
        
        t_i = (i + 1) * dt
        time_to_maturity = T - t_i
        K_disc = K * np.exp(-r * time_to_maturity)
        
        # V_call at time t_i on the grid
        V_call_i = call_values[i]
        
        # Put continuation value on the grid
        V_put_cont_grid = V_call_i - S_grid + K_disc
        
        # Payoff put on the grid
        payoff_put_grid = np.maximum(K - S_grid, 0.0)
        
        # Difference: payoff - continuation
        diff_grid = payoff_put_grid - V_put_cont_grid
        
        # Find the boundary where diff_grid = 0
        # The boundary is where we switch from exercise (diff > 0) to continue (diff < 0)
        
        # Check if always exercise or never exercise
        if np.all(diff_grid >= 0):
            thresholds[i] = S_grid[-1]  # Always exercise
            continue
        if np.all(diff_grid <= 0):
            thresholds[i] = S_grid[0]  # Never exercise
            continue
        
        # Find sign changes
        sign_changes = np.where(np.diff(np.sign(diff_grid)))[0]
        
        if len(sign_changes) == 0:
            # Shouldn't happen if we checked above, but fallback
            thresholds[i] = K
            continue
        
        # Use the first sign change (lowest S where we switch to continue)
        idx = sign_changes[0]
        
        # Brent's method for high-precision root finding
        # Define function to evaluate diff at arbitrary S
        def diff_func(S):
            if S <= S_grid[0] or S >= S_grid[-1]:
                # Extrapolate using nearest grid point
                if S <= S_grid[0]:
                    V_call_val = V_call_i[0]
                else:
                    V_call_val = V_call_i[-1]
                V_put_cont = V_call_val - S + K_disc
                payoff = max(K - S, 0.0)
                return payoff - V_put_cont
            
            # Interpolate V_call in log-S space
            log_S = np.log(S)
            V_call_val = np.interp(log_S, log_S_grid, V_call_i)
            V_put_cont = V_call_val - S + K_disc
            payoff = max(K - S, 0.0)
            return payoff - V_put_cont
        
        # Bracket the root
        a = S_grid[idx]
        b = S_grid[idx + 1]
        
        try:
            threshold = brentq(diff_func, a, b, xtol=1e-10, rtol=1e-12)
        except:
            threshold = 0.5 * (a + b)
        
        thresholds[i] = threshold
    
    # Ensure thresholds are monotonic (non-increasing for put)
    for i in range(1, n_exercise):
        if thresholds[i] > thresholds[i-1]:
            thresholds[i] = thresholds[i-1]
    
    class ExercisePolicy:
        def __init__(self, thresholds):
            self.thresholds = thresholds
            self.n_exercise = len(thresholds)
            
        def exercise(self, current_state, time):
            if time < 0 or time >= self.n_exercise:
                return np.zeros(len(current_state), dtype=bool)
            
            threshold = self.thresholds[time]
            return current_state <= threshold

    return ExercisePolicy(thresholds)
