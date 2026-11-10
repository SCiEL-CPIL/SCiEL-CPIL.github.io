"""Bermudan exercise-policy research: backward induction on a fine grid with interpolation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction on a state grid.
    
    Parameters
    ----------
    training_paths : array-like, shape (N, T+1) or (N, T)
        Training paths of the underlying asset.
    contract : dict
        Contract parameters: K, r, sigma, T, num_steps, etc.
    
    Returns
    -------
    object with .exercise(current_state, time) method returning boolean vector.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    num_steps = contract.get('num_steps', 12)
    
    # Determine shape of training_paths
    training_paths = np.asarray(training_paths, dtype=np.float64)
    if training_paths.ndim == 1:
        training_paths = training_paths.reshape(1, -1)
    
    N, total_cols = training_paths.shape
    
    # Determine which columns are exercise dates
    # If total_cols == num_steps + 1, index 0 is t=0, 1..num_steps are exercise dates
    # If total_cols == num_steps, index 0..num_steps-1 are exercise dates t=1..num_steps
    if total_cols == num_steps + 1:
        # Columns 1..num_steps are exercise dates
        exercise_cols = list(range(1, num_steps + 1))
    elif total_cols == num_steps:
        # Columns 0..num_steps-1 are exercise dates
        exercise_cols = list(range(num_steps))
    else:
        # Fallback: use last num_steps columns
        exercise_cols = list(range(total_cols - num_steps, total_cols))
    
    # Build a fine grid for backward induction
    # Grid in [0, 1.5*K] with high resolution
    grid_max = 1.5 * K
    # Use 20000 points for good accuracy
    n_grid = 20000
    grid = np.linspace(0.0, grid_max, n_grid)
    
    dt = T / num_steps
    discount = np.exp(-r * dt)
    
    # Payoff function for Bermudan put
    def payoff(S):
        return np.maximum(K - S, 0.0)
    
    # Backward induction
    # V[j] is the value at step j (j=0 is last exercise date, j=num_steps-1 is first)
    # We work backwards from the last exercise date
    
    # At the last exercise date (j = num_steps-1 in forward time, index 0 in backward),
    # the value is just the immediate exercise payoff (no continuation value)
    V_next = payoff(grid)
    
    # Store exercise boundaries for each step
    # exercise_boundary[j] is the threshold below which we exercise at step j
    # j=0 corresponds to the first exercise date (t=dt), j=num_steps-1 to the last (t=T)
    boundaries = np.zeros(num_steps)
    
    # For the last step, we always exercise if payoff > 0 (i.e., S < K)
    # But we need to be more careful: at the last step, continuation = 0, so exercise if payoff > 0
    boundaries[num_steps - 1] = K  # Exercise if S < K at last step
    
    # Work backwards from second-to-last to first exercise date
    for j in range(num_steps - 2, -1, -1):
        # Check deadline
        if time.time() > deadline:
            break
        
        # Continuation value: discount * expected future value
        # For GBM: S_{t+1} = S_t * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        # E[V(S_{t+1}) | S_t] can be approximated by interpolating V_next on the grid
        # and taking the expectation over the lognormal distribution
        
        # The mean of S_{t+1} given S_t is S_t * exp(r*dt)
        # The variance of log(S_{t+1}) is sigma^2 * dt
        
        # We'll compute the continuation value by numerical integration
        # using the fact that S_{t+1} is lognormal
        
        # For each S_t on the grid, compute E[V(S_{t+1}) | S_t]
        # S_{t+1} = S_t * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        
        # Let mu = (r - 0.5*sigma^2)*dt, vol = sigma*sqrt(dt)
        # Then log(S_{t+1}/S_t) ~ N(mu, vol^2)
        
        # We can compute E[V(S_{t+1})] by integrating V(S_t * exp(mu + vol*Z)) * phi(Z) dZ
        # where phi is the standard normal pdf
        
        # For efficiency, we'll use a quadrature approach
        # Use Gauss-Hermite or simple midpoint rule on Z
        
        # Alternative: since we have a grid, we can use the fact that
        # E[V(S_{t+1}) | S_t] = integral V(s) * f(s|S_t) ds
        # where f is the lognormal density
        
        # Let's use a simpler approach: for each S_t, the distribution of S_{t+1} is
        # lognormal with mean S_t*exp(r*dt) and appropriate variance.
        # We'll approximate the expectation by evaluating V_next at a few points
        # and weighting them.
        
        # Actually, let's use a more direct numerical approach:
        # For each S_t on the grid, compute the expected value by integrating
        # V_next(s) * lognormal_pdf(s; S_t, dt) ds
        
        # This is expensive for 20000 grid points. Let's use a smarter approach.
        
        # Since V_next is defined on a uniform grid, and the lognormal distribution
        # has a known form, we can use the following:
        # E[V(S_{t+1}) | S_t] = (1/sqrt(2*pi)) * integral V(S_t * exp(mu + vol*z)) * exp(-z^2/2) dz
        
        # We'll use a fixed set of quadrature points for z
        # Use 50 quadrature points for good accuracy
        
        n_quad = 50
        # Gauss-Hermite quadrature would be ideal, but let's use simple uniform
        # on a reasonable range of z, say [-4, 4]
        z_quad = np.linspace(-4.0, 4.0, n_quad)
        # Weights for uniform quadrature
        dz = z_quad[1] - z_quad[0]
        w_quad = np.full(n_quad, dz)
        # Normalize weights
        w_quad /= np.sum(w_quad)
        
        mu = (r - 0.5 * sigma**2) * dt
        vol = sigma * np.sqrt(dt)
        
        # For each S_t on the grid, compute E[V(S_{t+1}) | S_t]
        # S_{t+1} = S_t * exp(mu + vol * z_quad)
        
        # To vectorize: for each z_quad[i], S_{t+1} = grid * exp(mu + vol * z_quad[i])
        # Then interpolate V_next at these points and average over z_quad with weights
        
        # But this is 50 * 20000 = 1,000,000 interpolations, which might be slow.
        # Let's try a different approach: use the fact that we can precompute
        # the expected value more efficiently.
        
        # Actually, let's just do it directly with numpy interpolation
        # This should be fast enough with 20000 grid points
        
        # Compute S_{t+1} for each quadrature point
        # shape: (n_quad, n_grid)
        S_next_quad = grid[np.newaxis, :] * np.exp(mu + vol * z_quad[:, np.newaxis])
        
        # Interpolate V_next at S_next_quad
        # V_next is defined on 'grid', so we need to interpolate
        # Clip to grid range
        S_next_clipped = np.clip(S_next_quad, grid[0], grid[-1])
        V_interp = np.interp(S_next_clipped, grid, V_next)
        
        # Weighted average over quadrature points
        continuation = np.sum(w_quad[:, np.newaxis] * V_interp, axis=0)
        
        # Total value at step j
        V_current = np.maximum(payoff(grid), discount * continuation)
        
        # Exercise boundary: largest S where we exercise (payoff >= continuation)
        # Find where payoff > discount * continuation
        exercise_mask = payoff(grid) > discount * continuation
        if np.any(exercise_mask):
            # Find the boundary: last index where we exercise
            idx = np.where(exercise_mask)[0]
            boundary_idx = idx[-1]
            # Interpolate between boundary_idx and boundary_idx+1
            if boundary_idx + 1 < n_grid:
                # Linear interpolation
                S1 = grid[boundary_idx]
                S2 = grid[boundary_idx + 1]
                P1 = payoff(S1)
                P2 = discount * continuation[boundary_idx + 1]
                C1 = discount * continuation[boundary_idx]
                C2 = discount * continuation[boundary_idx + 1]
                # Find where P(S) = C(S)
                # P(S) = K - S for S < K
                # C(S) is approximately linear between the two points
                # K - S1 - C1 = (K - S2 - C2) * (S - S1) / (S2 - S1) + (K - S1 - C1)
                # Let's just use the midpoint for simplicity
                boundaries[j] = (S1 + S2) / 2.0
            else:
                boundaries[j] = grid[boundary_idx]
        else:
            boundaries[j] = 0.0
        
        V_next = V_current
    
    # Check deadline before creating policy
    if time.time() > deadline:
        # Return a simple policy
        class SimplePolicy:
            def __init__(self, K, boundaries):
                self.K = K
                self.boundaries = boundaries
            
            def exercise(self, current_state, time):
                current_state = np.asarray(current_state, dtype=np.float64)
                # time is 1-indexed for exercise dates (1 to num_steps)
                # Map to boundary index
                t_idx = int(time) - 1
                if t_idx < 0 or t_idx >= len(self.boundaries):
                    return np.zeros(len(current_state), dtype=bool)
                threshold = self.boundaries[t_idx]
                return current_state < threshold
        
        return SimplePolicy(K, boundaries)
    
    # Create the policy object
    class BermudanPolicy:
        def __init__(self, K, boundaries, num_steps):
            self.K = K
            self.boundaries = boundaries
            self.num_steps = num_steps
        
        def exercise(self, current_state, time):
            """
            Determine whether to exercise at the given time.
            
            Parameters
            ----------
            current_state : array-like
                Current state of the underlying asset.
            time : int or float
                Current time step (1-indexed for exercise dates).
            
            Returns
            -------
            boolean array
                True if we should exercise at this time.
            """
            current_state = np.asarray(current_state, dtype=np.float64)
            t_idx = int(time) - 1
            if t_idx < 0 or t_idx >= self.num_steps:
                return np.zeros(len(current_state), dtype=bool)
            threshold = self.boundaries[t_idx]
            return current_state < threshold
    
    return BermudanPolicy(K, boundaries, num_steps)
# EVOLVE-BLOCK-END
