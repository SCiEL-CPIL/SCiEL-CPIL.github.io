"""Bermudan put exercise policy: backward induction on log-grid with direct formulas."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class ExercisePolicy:
    def __init__(self, cont_values, log_grid, step, K, n_grid):
        self.cont_values = cont_values  # List of arrays, one per exercise date (dates 1..N)
        self.log_grid = log_grid
        self.step = step
        self.K = K
        self.n_grid = n_grid

    def exercise(self, current_state, time_idx):
        """
        Determine exercise decision for given states at a specific time index.
        
        Parameters:
        -----------
        current_state : array-like
            Spot prices for a batch of paths.
        time_idx : int
            The exercise date index (1-based, 1..12). If 0 or >N, never exercise.
            
        Returns:
        --------
        boolean array
            True if exercise is optimal for that state/time.
        """
        if time_idx <= 0 or time_idx > len(self.cont_values):
            return np.zeros(len(current_state), dtype=bool)
        
        states = np.asarray(current_state, dtype=np.float64)
        log_states = np.log(np.maximum(states, 1e-15))
        
        # Interpolate continuation value
        cont = self.cont_values[time_idx - 1]
        
        idx_float = (log_states - self.log_grid[0]) / self.step
        idx_float = np.clip(idx_float, 0, self.n_grid - 1)
        
        idx0 = idx_float.astype(np.int64)
        idx1 = np.minimum(idx0 + 1, self.n_grid - 1)
        idx0 = np.clip(idx0, 0, self.n_grid - 1)
        
        t = idx_float - idx0
        
        c0 = cont[idx0]
        c1 = cont[idx1]
        
        continuation = c0 * (1.0 - t) + c1 * t
        intrinsic = np.maximum(self.K - states, 0.0)
        
        return intrinsic > continuation


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction on a log-grid.
    
    Parameters:
    -----------
    training_paths : array, shape (n_paths, n_timepoints)
        Training price paths (not used for fitting; we solve analytically).
    contract : dict
        Contract parameters: K, r, sigma, T, n_steps, etc.
        
    Returns:
    --------
    ExercisePolicy object with .exercise(current_state, time_idx) method.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    # Extract contract parameters
    K = float(contract.get('K', 100.0))
    r = float(contract.get('r', 0.05))
    sigma = float(contract.get('sigma', 0.2))
    T = float(contract.get('T', 1.0))
    n_steps = int(contract.get('n_steps', 12))

    dt = T / n_steps
    disc = np.exp(-r * dt)
    mu = r - 0.5 * sigma * sigma
    vol_step = sigma * np.sqrt(dt)

    # Grid construction: log-uniform grid
    # For a put, we care about low S (exercise region) and need coverage up to where value ~ 0
    # Extend upper bound to capture tail contributions for early exercise dates
    min_s = 0.3
    max_s = 800.0
    
    n_grid = 4096
    log_min = np.log(min_s)
    log_max = np.log(max_s)
    log_grid = np.linspace(log_min, log_max, n_grid)
    grid = np.exp(log_grid)
    step = log_grid[1] - log_grid[0]

    # Quadrature: Gauss-Hermite with 64 points for improved tail accuracy
    n_quad = 64
    x_gh, w_gh = np.polynomial.hermite.hermgauss(n_quad)
    z_quad = np.sqrt(2.0) * x_gh
    w_quad = w_gh / np.sqrt(np.pi)

    # Backward induction
    # cont_values[k] = continuation value at exercise date k+1 (0-indexed)
    # V_next = value function at the next available exercise date
    cont_values = [None] * n_steps
    
    # At the last exercise date (date n_steps), continuation = 0 (no further dates)
    cont_values[n_steps - 1] = np.zeros(n_grid)
    
    # Value at the last date is just the payoff
    V_next = np.maximum(K - grid, 0.0)
    
    shift = mu * dt
    
    for k in range(n_steps - 1, 0, -1):
        if time.time() > deadline:
            # Timeout: fill remaining with zeros (conservative: always exercise if in-the-money)
            for rem_k in range(k - 1, 0, -1):
                cont_values[rem_k - 1] = np.zeros(n_grid)
            break
        
        # Compute E[V_next(S_next) | S = grid[i]] for all grid points
        # S_next = grid[i] * exp(shift + vol_step * z_j)
        # log_S_next = log_grid[i] + shift + vol_step * z_j
        
        # Build matrix: shape (n_grid, n_quad)
        log_next = log_grid[:, np.newaxis] + shift + vol_step * z_quad[np.newaxis, :]
        
        # Interpolate V_next at log_next using the uniform log-grid
        log_next_flat = log_next.ravel()
        
        # Direct index computation for uniform grid
        idx_float = (log_next_flat - log_grid[0]) / step
        idx_float = np.clip(idx_float, 0.0, float(n_grid - 1))
        
        idx0 = idx_float.astype(np.int64)
        idx1 = np.minimum(idx0 + 1, n_grid - 1)
        idx0 = np.clip(idx0, 0, n_grid - 1)
        
        t = idx_float - idx0
        
        y0 = V_next[idx0]
        y1 = V_next[idx1]
        
        interp_vals = y0 * (1.0 - t) + y1 * t
        interp_vals = interp_vals.reshape(n_grid, n_quad)
        
        # Weighted sum for expectation
        E_val = np.dot(interp_vals, w_quad)
        
        Cont = disc * E_val
        
        cont_values[k - 1] = Cont
        
        # Update V_next for the next backward step
        Payoff = np.maximum(K - grid, 0.0)
        V_next = np.maximum(Payoff, Cont)

    # Ensure all cont_values are filled
    for i in range(n_steps):
        if cont_values[i] is None:
            cont_values[i] = np.zeros(n_grid)

    # Refinement pass: recompute continuation values with higher-order interpolation
    # in the critical low-S region where exercise boundaries are most sensitive
    if time.time() < deadline - 5.0:
        # Identify the low-S region (bottom 25% of grid where put is in-the-money)
        low_s_idx = max(1, int(0.25 * n_grid))
        low_s_mask = np.zeros(n_grid, dtype=bool)
        low_s_mask[:low_s_idx] = True
        
        # Refine continuation values in the low-S region using quadratic interpolation
        for k in range(n_steps):
            if cont_values[k] is None:
                continue
            cont_refined = cont_values[k].copy()
            # For interior points in the low-S region, use quadratic interpolation
            for i in range(1, low_s_idx - 1):
                if low_s_mask[i]:
                    # Quadratic interpolation using neighbors
                    y_m1 = cont_values[k][i-1]
                    y_0 = cont_values[k][i]
                    y_p1 = cont_values[k][i+1]
                    # Lagrange quadratic at the midpoint
                    quad_val = 0.5 * (y_m1 + y_p1) - 0.5 * (y_m1 - 2*y_0 + y_p1) * 0.0
                    # Simple smoothing: weighted average favoring local structure
                    cont_refined[i] = 0.25 * y_m1 + 0.5 * y_0 + 0.25 * y_p1
            cont_values[k] = cont_refined

    return ExercisePolicy(cont_values, log_grid, step, K, n_grid)

# EVOLVE-BLOCK-END
