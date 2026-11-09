"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Extract parameters from contract
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_steps = contract.get('n_steps', 12)
    
    dt = T / n_steps
    df = np.exp(-r * dt)
    sqrt_dt = np.sqrt(dt)
    
    # Use a log-space grid for better resolution at low spot values (important for puts)
    # Grid in log-space, then exponentiate
    # For a put with K=100, we want good resolution from ~5 to ~200
    log_s_min = np.log(5.0)
    log_s_max = np.log(250.0)
    n_grid = 500
    log_grid = np.linspace(log_s_min, log_s_max, n_grid)
    grid = np.exp(log_grid)
    
    # V[i, k] = value of the option at exercise date i when spot is grid[k]
    V = np.zeros((n_steps, n_grid))
    
    # Terminal condition: at the last exercise date, value is intrinsic
    V[-1, :] = np.maximum(K - grid, 0.0)
    
    # Use Gauss-Hermite quadrature for accurate expectation computation
    # E[f(X)] where X ~ N(mu, sigma^2) can be computed via Gauss-Hermite
    # But we need E[V(S_{i+1}) | S_i = s] where S_{i+1} = s * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
    # So ln(S_{i+1}) ~ N(ln(s) + (r - sigma^2/2)*dt, sigma^2*dt)
    
    # Use Gauss-Hermite quadrature with 64 points for good accuracy
    n_quad = 64
    # Gauss-Hermite nodes and weights for standard normal integral
    # scipy.special.roots_hermite gives nodes/weights for exp(-x^2) integral
    # We need E[g(Z)] where Z ~ N(0,1)
    # Use scipy.stats.norm.ppf with quantiles, or use roots_hermite
    from scipy.special import roots_hermite
    gh_nodes, gh_weights = roots_hermite(n_quad)
    # For standard normal: Z = sqrt(2) * x where x are GH nodes
    # E[g(Z)] = (1/sqrt(pi)) * sum w_i * g(sqrt(2)*x_i)
    z_nodes = np.sqrt(2.0) * gh_nodes
    z_weights = gh_weights / np.sqrt(np.pi)
    
    # Precompute the drift and diffusion terms
    drift = (r - sigma**2 / 2.0) * dt
    diffusion = sigma * sqrt_dt
    
    # Work backwards
    for i in range(n_steps - 2, -1, -1):
        if time.time() > deadline - 5.0:
            break
            
        continuation = np.zeros(n_grid)
        
        # Vectorized computation: for each grid point s, compute E[V_{i+1}(s_next)]
        # s_next = s * exp(drift + diffusion * Z)
        # log(s_next) = log(s) + drift + diffusion * Z
        
        # For each grid point k, log(s_k) + drift + diffusion * z_j
        # We can do this in a matrix form but that would be n_grid x n_quad
        # Instead, loop over grid points but vectorize over quadrature points
        
        for k in range(n_grid):
            log_s = log_grid[k]
            log_s_next = log_s + drift + diffusion * z_nodes
            s_next = np.exp(log_s_next)
            
            # Interpolate V[i+1] at s_next
            s_next_clipped = np.clip(s_next, grid[0], grid[-1])
            v_next = np.interp(s_next_clipped, grid, V[i + 1, :])
            
            continuation[k] = df * np.sum(z_weights * v_next)
        
        # Exercise value
        exercise_val = np.maximum(K - grid, 0.0)
        
        # Optimal value
        V[i, :] = np.maximum(exercise_val, continuation)
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, K, r, sigma, T, n_steps, grid, log_grid, V, z_nodes, z_weights, drift, diffusion, df):
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.n_steps = n_steps
            self.grid = grid
            self.log_grid = log_grid
            self.V = V
            self.z_nodes = z_nodes
            self.z_weights = z_weights
            self.drift = drift
            self.diffusion = diffusion
            self.df = df
            
        def exercise(self, current_state, time_idx):
            # Determine the exercise date index
            if isinstance(time_idx, (int, np.integer)):
                idx = int(time_idx)
            else:
                # time_idx is a float in [0, T]
                dt_local = self.T / self.n_steps
                idx = int(time_idx / dt_local)
                idx = min(idx, self.n_steps - 1)
                idx = max(idx, 0)
            
            s_arr = np.array(current_state, dtype=float)
            
            # At the last exercise date, always exercise if in the money
            if idx == self.n_steps - 1:
                return s_arr < self.K
            
            # For earlier dates, compare exercise value to continuation value
            # Exercise value
            exercise_val = np.maximum(self.K - s_arr, 0.0)
            
            # Continuation value: df * E[V[idx+1](S_{idx+1}) | S_i = s]
            continuation = np.zeros(len(s_arr))
            
            for j in range(len(s_arr)):
                s = s_arr[j]
                if s <= 0:
                    continuation[j] = 0
                    continue
                log_s = np.log(s)
                log_s_next = log_s + self.drift + self.diffusion * self.z_nodes
                s_next = np.exp(log_s_next)
                s_next_clipped = np.clip(s_next, self.grid[0], self.grid[-1])
                v_next = np.interp(s_next_clipped, self.grid, self.V[idx + 1, :])
                continuation[j] = self.df * np.sum(self.z_weights * v_next)
            
            return exercise_val >= continuation
    
    return ExercisePolicy(K, r, sigma, T, n_steps, grid, log_grid, V, z_nodes, z_weights, drift, diffusion, df)
# EVOLVE-BLOCK-END
