"""Bermudan put exercise policy: backward induction with high-fidelity quadrature and refined grid."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction.
    
    Parameters:
    -----------
    training_paths : array-like, shape (n_paths, n_time_steps)
        Simulated spot price paths. n_time_steps = 13 (t=0..12).
    contract : dict
        Contract parameters: K, r, sigma, T, n_steps (12 exercise dates).
        
    Returns:
    --------
    An object with method exercise(current_state, time) -> boolean array.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    # Extract contract parameters
    K = float(contract.get('K', 100.0))
    r = float(contract.get('r', 0.05))
    sigma = float(contract.get('sigma', 0.2))
    T = float(contract.get('T', 1.0))
    n_steps = int(contract.get('n_steps', 12))

    # Exercise dates: 12 monthly dates, no time-0 exercise
    dt = T / n_steps
    exercise_times = np.array([k * dt for k in range(1, n_steps + 1)])

    # Discount factors
    disc = np.exp(-r * dt)

    # We'll use a grid-based approach with backward induction
    tp = np.asarray(training_paths, dtype=np.float64)
    if tp.ndim == 1:
        tp = tp.reshape(1, -1)
    n_paths, n_ts = tp.shape

    # Strategy: Use a fine grid of spot prices and compute continuation
    # values via backward induction.

    # Build grid from training data range, but extend for safety
    # Focus on low end for puts
    all_spots = tp[:, 1:]
    min_s = max(0.5, np.min(all_spots) * 0.1) # Extend low end significantly
    max_s = max(K * 4.0, np.max(all_spots) * 2.0)

    # Use a logarithmic grid for better resolution at low prices
    n_grid = 4096
    log_min = np.log(min_s)
    log_max = np.log(max_s)
    log_grid = np.linspace(log_min, log_max, n_grid)
    grid = np.exp(log_grid)

    # Backward induction
    def payoff(s):
        return np.maximum(K - s, 0.0)

    mu = r - 0.5 * sigma * sigma
    vol_step = sigma * np.sqrt(dt)

    # Use Gauss-Hermite quadrature with 128 points for better tail coverage
    n_quad = 128
    x_gh, w_gh = np.polynomial.hermite.hermgauss(n_quad)
    
    # Transform to standard normal
    z_quad = np.sqrt(2.0) * x_gh
    w_quad = w_gh / np.sqrt(np.pi)

    n_ex = n_steps
    cont_at_date = [None] * n_ex

    # V at the last exercise date is the payoff
    V_next = payoff(grid)

    # Process dates from last to first
    for k_idx in range(n_ex - 1, -1, -1):
        if k_idx == n_ex - 1:
            # Last exercise date: no continuation
            cont_at_date[k_idx] = np.zeros(n_grid)
            V_current = V_next
        else:
            # Compute continuation: disc * E[V_next(S_{k+1}) | S_k = s]
            # log_S_next = log_grid[:, np.newaxis] + mu * dt + vol_step * z_quad[np.newaxis, :]
            log_S_next = log_grid[:, np.newaxis] + mu * dt + vol_step * z_quad[np.newaxis, :]
            
            # Interpolate V_next on the log grid
            # Flatten for np.interp
            log_S_next_flat = log_S_next.ravel()
            V_next_flat = np.interp(log_S_next_flat, log_grid, V_next)
            V_next_interp = V_next_flat.reshape(n_grid, n_quad)

            # Continuation value
            cont_vals = disc * np.sum(w_quad[np.newaxis, :] * V_next_interp, axis=1)
            cont_at_date[k_idx] = cont_vals

            # V_current = max(payoff, continuation)
            V_current = np.maximum(payoff(grid), cont_vals)

        V_next = V_current

        # Check time
        if time.time() > deadline - 5.0:
            break

    # Build the exercise policy object
    class ExercisePolicy:
        def __init__(self, grid, log_grid, cont_at_date, payoff_func, K_val):
            self.grid = grid
            self.log_grid = log_grid
            self.cont_at_date = cont_at_date
            self.payoff_func = payoff_func
            self.K = K_val
            self.n_grid = n_grid

        def exercise(self, current_state, time_idx):
            s = np.asarray(current_state, dtype=np.float64)
            
            k_idx = int(time_idx) - 1
            if k_idx < 0 or k_idx >= n_ex:
                return np.zeros(len(s), dtype=bool)

            cont_grid = self.cont_at_date[k_idx]
            if cont_grid is None:
                return np.zeros(len(s), dtype=bool)

            log_s = np.log(np.maximum(s, 1e-10))
            cont_vals = np.interp(log_s, self.log_grid, cont_grid)

            payoffs = np.maximum(self.K - s, 0.0)
            return payoffs >= cont_vals

    return ExercisePolicy(grid, log_grid, cont_at_date, payoff, K)


# EVOLVE-BLOCK-END
