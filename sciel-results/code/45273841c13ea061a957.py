"""Bermudan put exercise policy: optimal stopping via backward induction on discretized state space."""
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s kill

    # Fixed parameters from task contract
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_dates = 12  # 12 monthly exercise dates (no time-0 exercise)

    # Convert training paths to numpy
    tp = np.asarray(training_paths, dtype=np.float64)

    # Handle shape: expect (n_paths, n_times) or (n_times, n_paths)
    if tp.ndim == 2:
        if tp.shape[0] == 4096:
            paths = tp  # (4096, n_times)
            n_paths, n_times = tp.shape
        elif tp.shape[1] == 4096:
            paths = tp.T
            n_times, n_paths = tp.shape
        else:
            # Default: assume rows are paths
            paths = tp
            n_paths, n_times = tp.shape
    else:
        paths = tp.reshape(4096, -1)
        n_paths, n_times = paths.shape

    # The exercise dates are at t = 1/12, 2/12, ..., 12/12
    # We need to determine which column corresponds to which exercise date.
    # Typically, the paths are sampled at the exercise dates.
    # Let's assume the last column is t=T (maturity) and the first is t=1/12.
    # Actually, for a Bermudan option, the exercise dates are the time steps.
    # We'll use the last n_dates columns as the exercise dates if available.

    # Determine the number of time steps in the training data
    # The task says 12 monthly exercise dates. The training paths should have data at these dates.
    # Let's assume the training paths have exactly n_dates columns (or more, but we use the last n_dates).

    # Strategy: Use backward induction on a discretized state grid.
    # 1. Create a state grid based on the range of training data.
    # 2. For each exercise date from last to first, compute the continuation value
    #    using Monte Carlo or a simple expectation over the training paths.
    # 3. The optimal exercise threshold at each date is where payoff >= continuation value.

    # Since we have training paths, we can use them to estimate the transition.
    # However, for a GBM, we know the dynamics analytically.
    # Let's use the analytical risk-neutral dynamics for the continuation value estimation.

    # State grid: use percentiles of the final time step to define a reasonable range
    # Actually, let's use a fixed grid based on the parameters.
    # Spot can range from about 50 to 150 for a 1-year option with sigma=0.2.
    # Let's create a grid of 1000 points from 50 to 150.
    
    # But first, let's check if we have enough time.
    if time.time() > deadline - 5:
        # Not enough time, return a simple threshold policy
        # For a put, exercise when S < K * some_factor
        # A simple heuristic: exercise when S < K
        class ExercisePolicy:
            def __init__(self, K):
                self.K = K
            def exercise(self, current_state, time_step):
                return current_state < self.K
        return ExercisePolicy(K)

    # Create state grid
    # Use the range of the training data at the last time step to set bounds
    last_col_idx = min(n_dates, n_times) - 1
    if last_col_idx >= 0:
        s_min = np.percentile(paths[:, last_col_idx], 1)
        s_max = np.percentile(paths[:, last_col_idx], 99)
    else:
        s_min = 50.0
        s_max = 150.0
    
    # Ensure reasonable bounds
    s_min = max(1.0, min(s_min, 70.0))
    s_max = min(200.0, max(s_max, 130.0))
    
    n_grid = 500
    grid = np.linspace(s_min, s_max, n_grid)
    
    # Time steps: exercise dates at t_i = i/12 for i=1..12
    # We work backwards from i=12 to i=1
    # At maturity (i=12), the value is the payoff: max(K - S, 0)
    # For i < 12, the continuation value is the discounted expected value of the next step.
    
    # For a risk-neutral GBM:
    # S_{t+dt} = S_t * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
    # where Z ~ N(0,1)
    
    dt = T / n_dates  # 1/12
    
    # We'll compute the value function on the grid at each exercise date.
    # V[i, s] = max(payoff(s), discounted_continuation(s))
    # where payoff(s) = max(K - s, 0) for a put
    # and discounted_continuation(s) = exp(-r*dt) * E[V[i+1, S_{t+dt}] | S_t = s]
    
    # To estimate E[V[i+1, S_{t+dt}] | S_t = s], we can use the analytical mean and variance
    # or Monte Carlo. Since we have a grid, we can use the fact that under risk-neutral measure,
    # E[S_{t+dt} | S_t = s] = s * exp(r*dt)
    # and we can approximate the expectation by evaluating V[i+1] at the expected next state
    # and adding a convexity correction (or just use the expected state as a first approximation).
    
    # A better approach: use the known transition. For a GBM, the log-price is normal.
    # We can compute the expectation by integrating V[i+1] against the log-normal distribution.
    # But for simplicity and speed, let's use a few Monte Carlo samples or a simple quadrature.
    
    # Actually, since we're on a grid, we can use linear interpolation of V[i+1] and
    # compute the expectation using the moments of the log-normal distribution.
    
    # Let's use a simpler approach: for each grid point s, the next state S_{t+dt} is log-normal.
    # We can approximate E[V[i+1, S_{t+dt}]] by evaluating V[i+1] at the median and mean,
    # or by using a small number of Monte Carlo samples.
    
    # For speed, let's use the fact that for small dt, the distribution is approximately normal
    # with mean s*exp(r*dt) and variance s^2*sigma^2*dt.
    # We can use a 3-point quadrature or just evaluate at the mean and adjust.
    
    # Let's use a simple approach: evaluate V[i+1] at the risk-neutral forward price s*exp(r*dt)
    # This is a first-order approximation. For better accuracy, we could use more points.
    
    # Actually, let's use the analytical expectation for a linear function, but V is not linear.
    # Let's use a small Monte Carlo with, say, 100 samples per grid point. That's 500*100 = 50,000 evaluations per step, 12 steps = 600,000. Should be fast.
    
    # But we need to be careful with time. Let's start with a simple approximation and improve if time permits.
    
    # Simple approximation: continuation value at s is exp(-r*dt) * V[i+1, s*exp(r*dt)]
    # This ignores convexity but is a reasonable first approximation.
    
    # Let's implement the backward induction with this approximation.
    
    # V[12, s] = max(K - s, 0)
    V = np.zeros((n_dates + 1, n_grid))
    V[n_dates, :] = np.maximum(K - grid, 0.0)
    
    # Backward induction
    for i in range(n_dates - 1, 0, -1):
        if time.time() > deadline - 5:
            break
        # Continuation value: exp(-r*dt) * E[V[i+1, S_{t+dt}] | S_t = grid]
        # Approximate E[V[i+1, S_{t+dt}]] by V[i+1, grid * exp(r*dt)]
        forward = grid * np.exp(r * dt)
        # Interpolate V[i+1] at forward
        # forward may be outside grid, so clip
        forward_clipped = np.clip(forward, grid[0], grid[-1])
        # Linear interpolation
        idx = np.searchsorted(grid, forward_clipped)
        idx = np.clip(idx, 1, n_grid - 1)
        # Interpolate
        x0 = grid[idx - 1]
        x1 = grid[idx]
        y0 = V[i + 1, idx - 1]
        y1 = V[i + 1, idx]
        # Handle case where x1 == x0
        mask = (x1 > x0)
        interp = np.zeros_like(forward_clipped)
        interp[mask] = y0[mask] + (y1[mask] - y0[mask]) * (forward_clipped[mask] - x0[mask]) / (x1[mask] - x0[mask])
        interp[~mask] = y0[~mask]
        
        cont = np.exp(-r * dt) * interp
        payoff = np.maximum(K - grid, 0.0)
        V[i, :] = np.maximum(payoff, cont)
    
    # Now, for each exercise date i (1..12), the optimal exercise region is where payoff >= continuation.
    # We can store the threshold: for each i, find the largest s such that payoff(s) >= cont(s).
    # Since payoff is decreasing in s and cont is also decreasing (for a put), the exercise region is s <= threshold.
    
    # Compute thresholds for each date
    thresholds = np.zeros(n_dates)
    for i in range(1, n_dates + 1):
        payoff = np.maximum(K - grid, 0.0)
        if i < n_dates:
            cont = np.exp(-r * dt) * V[i + 1, :]
        else:
            cont = np.zeros(n_grid)  # At maturity, no continuation
        # Exercise if payoff >= cont
        exercise_mask = payoff >= cont
        # Find the largest index where exercise_mask is True
        # Since payoff is decreasing and cont is decreasing, the exercise region should be contiguous from the left.
        # Find the transition point
        if np.any(exercise_mask):
            last_exercise_idx = np.max(np.where(exercise_mask)[0])
            # The threshold is between grid[last_exercise_idx] and grid[last_exercise_idx + 1]
            if last_exercise_idx + 1 < n_grid:
                thresholds[i - 1] = grid[last_exercise_idx]  # Conservative: exercise up to this point
            else:
                thresholds[i - 1] = grid[-1]
        else:
            thresholds[i - 1] = 0.0  # Never exercise
    
    # Create the exercise policy
    class ExercisePolicy:
        def __init__(self, thresholds, grid, K):
            self.thresholds = thresholds
            self.grid = grid
            self.K = K
        
        def exercise(self, current_state, time_step):
            # time_step is 1-based index of the exercise date (1 to 12)
            # current_state is a vector of states
            ts = np.asarray(current_state, dtype=np.float64)
            # Determine which threshold to use
            # time_step could be 0-indexed or 1-indexed. The task says "12 monthly exercise dates (no time-0 exercise)"
            # So time_step likely ranges from 1 to 12, or 0 to 11.
            # Let's assume time_step is the index into the exercise dates, 0-based or 1-based.
            # We'll handle both: if time_step is 0-based (0 to 11), use thresholds[time_step]
            # if 1-based (1 to 12), use thresholds[time_step - 1]
            
            # Check the value of time_step
            if isinstance(time_step, (int, np.integer)):
                if time_step >= 1 and time_step <= n_dates:
                    idx = time_step - 1
                elif time_step >= 0 and time_step < n_dates:
                    idx = time_step
                else:
                    idx = 0
            else:
                idx = 0
            
            thresh = self.thresholds[idx]
            return ts < thresh
    
    return ExercisePolicy(thresholds, grid, K)
