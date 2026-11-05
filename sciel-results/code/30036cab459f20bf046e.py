"""Bermudan put exercise policy via Backward Induction with Gauss-Hermite Quadrature."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using deterministic backward induction
    with Gauss-Hermite quadrature for conditional expectations.
    
    Parameters:
    - training_paths: array of shape (n_paths, n_dates+1) or similar path data (ignored for value calculation, used only for time if needed)
    - contract: dict with K, r, sigma, T, n_exercise, etc.
    
    Returns an object with .exercise(current_state, time) method.
    """
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin before 120s kill

    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    dt = T / n_exercise
    
    # Determine number of dates
    # We assume exercise at dates 1..n_exercise
    # Index 0 corresponds to date 1 (first exercise opportunity)
    n_dates = n_exercise

    # Setup grid for spot prices
    # Range should cover relevant values. For a put with K=100, S below K is important.
    # S can go up, but put value is 0 for S >> K.
    S_min = 1.0
    S_max = 3 * K
    n_grid = 200
    S_grid = np.linspace(S_min, S_max, n_grid)
    
    # Gauss-Hermite quadrature nodes and weights
    # We need to approximate E[f(S_next)] where S_next = S * exp((r - sigma^2/2)dt + sigma*sqrt(dt)*Z)
    # Let Z ~ N(0,1).
    # Integral is (1/sqrt(pi)) * sum w_i f(S * exp((r - sigma^2/2)dt + sigma*sqrt(dt)*x_i))
    n_quad = 50
    x, w = np.polynomial.hermite.hermgauss(n_quad)
    # Weights for standard normal expectation: E[g(Z)] = 1/sqrt(pi) * sum w_i g(x_i/sqrt(2)) ?
    # No, hermgauss gives nodes x_i and weights w_i for integral exp(-x^2) f(x) dx.
    # To approximate E[g(Z)] for Z~N(0,1):
    # E[g(Z)] = 1/sqrt(2pi) * integral exp(-z^2/2) g(z) dz.
    # Let z = sqrt(2) * y. dz = sqrt(2) dy.
    # Integral becomes 1/sqrt(2pi) * integral exp(-y^2) g(sqrt(2)y) sqrt(2) dy
    # = 1/sqrt(pi) * integral exp(-y^2) g(sqrt(2)y) dy.
    # So we use nodes y_i = x_i and weights w_i, and evaluate g at sqrt(2)*x_i.
    # Then multiply sum by 1/sqrt(pi).
    
    sqrt2 = np.sqrt(2.0)
    z_nodes = x * sqrt2
    z_weights = w / np.sqrt(np.pi)
    
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    
    # Precompute the multipliers for S_next
    # S_next = S * exp(drift + vol * z)
    exp_term = np.exp(drift + vol * z_nodes) # shape (n_quad,)
    
    # Backward induction
    # V[j] is an array of values at S_grid for date j
    # V[n_dates-1] is the payoff at the last date
    V_next = np.maximum(K - S_grid, 0.0)
    
    # Store continuation values for each date j (0 to n_dates-2)
    # We store the values on the grid. At scoring time, we interpolate.
    # Actually, we can just store the grid values for V_j and interpolate.
    V_values = [None] * n_dates
    V_values[n_dates - 1] = V_next.copy()
    
    for j in range(n_dates - 2, -1, -1):
        # Compute E[V_next(S_next) | S_grid]
        # For each S in S_grid, compute V_next at S * exp_term
        # S_grid shape (n_grid,), exp_term shape (n_quad,)
        # We want result shape (n_grid,)
        
        # Vectorized computation
        # S_next_grid = S_grid[:, None] * exp_term[None, :]  # shape (n_grid, n_quad)
        # Interpolate V_next at S_next_grid
        # V_next is defined on S_grid.
        
        # Clip S_next to grid range for interpolation
        S_next_grid = S_grid[:, None] * exp_term[None, :]
        S_next_clipped = np.clip(S_next_grid, S_min, S_max)
        
        # Interpolate V_next
        V_next_at_next = np.interp(S_next_clipped.ravel(), S_grid, V_next).reshape(n_grid, n_quad)
        
        # Weighted average
        # E[V_next] = sum(z_weights * V_next_at_next, axis=1)
        E_V_next = np.dot(V_next_at_next, z_weights)
        
        # Discounted continuation value
        disc = np.exp(-r * dt)
        cont_val = disc * E_V_next
        
        # Payoff at current date
        payoff = np.maximum(K - S_grid, 0.0)
        
        # V_j = max(payoff, cont_val)
        V_j = np.maximum(payoff, cont_val)
        
        V_values[j] = V_j
        V_next = V_j
        
        if time.time() > deadline - 5.0:
            break
            
    class ExercisePolicy:
        def __init__(self, V_values, S_grid, n_dates, K, r, dt):
            self.V_values = V_values
            self.S_grid = S_grid
            self.n_dates = n_dates
            self.K = K
            self.r = r
            self.dt = dt
        
        def exercise(self, current_state, time):
            """
            Determine whether to exercise.
            
            Parameters:
            - current_state: array of spot prices (1D)
            - time: the time index or actual time
            
            Returns:
            - boolean array indicating whether to exercise
            """
            S = np.asarray(current_state, dtype=np.float64)
            
            # Determine which exercise date this corresponds to
            # time could be the index (0-indexed) or actual time
            if isinstance(time, (int, np.integer)):
                j = int(time)
            else:
                # Assume time is actual time, convert to index
                j = int(round(time / self.dt))
            
            # Clamp j to valid range
            j = max(0, min(j, self.n_dates - 1))
            
            # Compute payoff
            payoff = np.maximum(self.K - S, 0.0)
            
            # Compute continuation value
            if j < self.n_dates - 1:
                V_j_grid = self.V_values[j]
                if V_j_grid is not None:
                    # Interpolate V_j at S
                    # Clip S to grid range
                    S_clipped = np.clip(S, self.S_grid[0], self.S_grid[-1])
                    cont_val = np.interp(S_clipped, self.S_grid, V_j_grid)
                else:
                    cont_val = np.zeros_like(S)
            else:
                # At the last date, continuation is 0 (value is payoff)
                cont_val = np.zeros_like(S)
            
            # Exercise if payoff > continuation
            # Note: V_j includes the payoff. So V_j >= payoff.
            # We want to exercise if payoff >= continuation.
            # Since V_j = max(payoff, cont), if V_j == payoff, we exercise.
            # So check if payoff >= cont_val - epsilon?
            # Or simply: exercise if payoff >= cont_val.
            # Due to floating point, use a small tolerance or just >.
            # If payoff == cont_val, it doesn't matter.
            return payoff >= cont_val - 1e-9
    
    return ExercisePolicy(V_values, S_grid, n_dates, K, r, dt)
# EVOLVE-BLOCK-END
