"""Bermudan put exercise policy via backward induction with piecewise-linear value approximation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Extract parameters from contract
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)

    # Exercise dates: 12 monthly, no time-0
    dt = T / n_exercise
    exercise_times = np.arange(1, n_exercise + 1) * dt

    # Build a log-space grid for spot prices
    # We want to cover a reasonable range of prices
    log_min = np.log(0.5)   # S = 0.5*K
    log_max = np.log(2.5)   # S = 2.5*K
    n_grid = 500
    log_grid = np.linspace(log_min, log_max, n_grid)
    grid = np.exp(log_grid)

    # Discount factor per step
    disc = np.exp(-r * dt)

    # Terminal value at the last exercise date (European put payoff at T)
    # At time T, the value is max(K - S, 0)
    V = np.maximum(K - grid, 0.0)

    # Backward induction from last exercise date to first
    # For each exercise date from n_exercise-1 down to 1:
    #   1. Compute continuation value: discount expected future value under risk-neutral measure
    #   2. Exercise value: max(K - S, 0)
    #   3. Value = max(continuation, exercise)

    # Precompute risk-neutral transition probabilities using log-normal distribution
    # Under risk-neutral measure: S_{t+dt} = S_t * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
    # In log space: ln(S_{t+dt}) = ln(S_t) + (r - sigma^2/2)*dt + sigma*sqrt(dt)*Z
    # So ln(S_{t+dt}) | ln(S_t) = x ~ N(x + (r - sigma^2/2)*dt, sigma^2*dt)

    drift_log = (r - 0.5 * sigma**2) * dt
    vol_log = sigma * np.sqrt(dt)

    # For each grid point, we need E[V(ln(S_{t+dt})) | ln(S_t) = x_i]
    # where ln(S_{t+dt}) ~ N(x_i + drift_log, vol_log^2)
    # We approximate this expectation by numerical integration over the grid

    # Precompute the grid in log space for integration
    # E[f(X)] where X ~ N(mu, sigma2) can be approximated by summing f(x_j) * phi((x_j - mu)/sigma) * sigma / (sqrt(2*pi))
    # But more efficiently, we can use the fact that for a normal distribution,
    # E[f(X)] ≈ sum_j f(x_j) * (1/(sigma*sqrt(2*pi))) * exp(-(x_j - mu)^2/(2*sigma^2)) * dx

    # Actually, let's use a simpler approach: for each grid point, compute the conditional
    # expectation by weighting the grid points with the normal pdf centered at x_i + drift_log

    # Precompute the normal pdf weights for each grid point relative to each other
    # This is a convolution-like operation. Let's do it efficiently.

    # For each i, E[V_{t+dt}(S_{t+dt}) | S_t = grid[i]] = 
    #   sum_j V_{t+dt}(grid[j]) * phi((log_grid[j] - (log_grid[i] + drift_log)) / vol_log) * (1/vol_log) * dlog
    # where phi is the standard normal pdf and dlog is the grid spacing in log space

    dlog = log_grid[1] - log_grid[0]

    # Precompute the kernel: for each pair (i, j), the weight is
    # w(i,j) = (1/(vol_log*sqrt(2*pi))) * exp(-(log_grid[j] - log_grid[i] - drift_log)^2 / (2*vol_log^2)) * dlog
    # This is a Toeplitz-like matrix. We can compute it as:
    # diff[j,i] = log_grid[j] - log_grid[i] - drift_log
    # w[i,j] = (1/(vol_log*sqrt(2*pi))) * exp(-diff[j,i]^2 / (2*vol_log^2)) * dlog

    # To save memory and time, we'll compute the continuation value step by step.

    for step in range(n_exercise - 1, 0, -1):
        # Compute continuation value for each grid point
        # cont[i] = disc * E[V(ln(S_{t+dt})) | ln(S_t) = log_grid[i]]
        
        # diff[j, i] = log_grid[j] - log_grid[i] - drift_log
        # We can compute this as an outer difference
        diff = log_grid[:, None] - log_grid[None, :] - drift_log  # shape (n_grid, n_grid)
        diff_sq = diff ** 2
        
        # Normal pdf: (1/(vol_log*sqrt(2*pi))) * exp(-diff_sq/(2*vol_log^2))
        norm_const = 1.0 / (vol_log * np.sqrt(2.0 * np.pi))
        weights = norm_const * np.exp(-diff_sq / (2.0 * vol_log**2)) * dlog
        
        # Continuation value: disc * weights @ V
        cont = disc * (weights @ V)
        
        # Exercise value
        exercise_val = np.maximum(K - grid, 0.0)
        
        # Value at this step
        V = np.maximum(cont, exercise_val)
        
        # Check time
        if time.time() > deadline - 10.0:
            break

    # Now V is the value function at the first exercise date (time = dt)
    # The policy: at each exercise date, exercise if exercise_val >= cont
    # We need to store the exercise decision for each step

    # Re-run backward induction to store exercise decisions
    # Actually, we can just recompute. But let's be efficient.

    # Let's re-do the backward induction storing exercise masks
    V = np.maximum(K - grid, 0.0)  # terminal value
    exercise_masks = [None] * n_exercise  # index 1 to n_exercise-1 (0-indexed for steps 1..11)

    for step in range(n_exercise - 1, 0, -1):
        diff = log_grid[:, None] - log_grid[None, :] - drift_log
        diff_sq = diff ** 2
        weights = norm_const * np.exp(-diff_sq / (2.0 * vol_log**2)) * dlog
        cont = disc * (weights @ V)
        exercise_val = np.maximum(K - grid, 0.0)
        V = np.maximum(cont, exercise_val)
        exercise_masks[step] = (exercise_val >= cont)
        
        if time.time() > deadline - 5.0:
            break

    # Now we have the value function at the first exercise date and exercise masks for each step
    # The policy object: given current_state (a vector of spot prices) and time (which exercise date),
    # return a boolean vector indicating whether to exercise at each path

    class ExercisePolicy:
        def __init__(self, grid, log_grid, exercise_masks, n_exercise):
            self.grid = grid
            self.log_grid = log_grid
            self.exercise_masks = exercise_masks
            self.n_exercise = n_exercise

        def exercise(self, current_state, time):
            # time is the exercise date index (1 to n_exercise)
            # current_state is a vector of spot prices
            # Return boolean vector: True if should exercise
            
            # Determine which step we're at
            # time could be the actual time value or the index
            # Based on the contract, there are 12 exercise dates at t = dt, 2dt, ..., T
            # Let's assume time is the index (1-based) or the time value
            
            # Convert time to step index
            if isinstance(time, (int, float)):
                if time <= 1.0:
                    # If time is given as a time value
                    step_idx = int(round(time / dt))
                else:
                    # If time is already an index
                    step_idx = int(time)
            else:
                step_idx = 1

            step_idx = max(1, min(step_idx, self.n_exercise))

            if step_idx >= self.n_exercise:
                # Last exercise date: always exercise if in the money
                return np.maximum(K - current_state, 0.0) > 0

            mask = self.exercise_masks[step_idx]
            if mask is None:
                # Fallback: exercise if in the money
                return np.maximum(K - current_state, 0.0) > 0

            # Interpolate: for each current_state, find the grid index and check the mask
            # Use nearest neighbor or linear interpolation
            log_state = np.log(current_state)
            # Find nearest grid points
            idx = np.searchsorted(self.log_grid, log_state)
            idx = np.clip(idx, 1, len(self.log_grid) - 1)
            
            # Linear interpolation of the exercise decision
            # Actually, the exercise mask is a boolean, so we just take the nearest
            # Or we can interpolate the value and compare
            # For simplicity, use nearest neighbor
            decision = mask[idx - 1]
            
            # Also handle boundaries
            decision = np.where(log_state <= self.log_grid[0], mask[0], decision)
            decision = np.where(log_state >= self.log_grid[-1], mask[-1], decision)
            
            return decision

    return ExercisePolicy(grid, log_grid, exercise_masks, n_exercise)

# EVOLVE-BLOCK-END
