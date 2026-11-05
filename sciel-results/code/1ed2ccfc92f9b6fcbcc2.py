"""Bermudan exercise-policy research: backward induction with grid-based boundaries."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.special import roots_hermite


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction.
    
    Args:
        training_paths: array of shape (n_paths, n_times) with simulated asset prices
        contract: dict with contract parameters
        
    Returns:
        An object with exercise(current_state, time) method returning boolean vector.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    # Exercise times: no time-0 exercise, so at T/n_exercise, 2T/n_exercise, ..., T
    dt = T / n_exercise
    
    # Build a focused grid for backward induction
    # For a put with K=100, the relevant S range is roughly [0, 1.5*K]
    # The exercise boundary is always below K, so we focus there
    S_max = 1.5 * K
    n_grid = 1500
    
    S_grid = np.linspace(0.0, S_max, n_grid)
    
    # Precompute Gauss-Hermite quadrature
    n_quad = 60
    x_quad, w_quad = roots_hermite(n_quad)
    Z_quad = np.sqrt(2.0) * x_quad
    w_quad_norm = w_quad / np.sqrt(np.pi)
    
    # Under risk-neutral GBM:
    sqrt_dt = np.sqrt(dt)
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * sqrt_dt
    exp_factor = np.exp(drift + vol * Z_quad)  # shape (n_quad,)
    
    # Backward induction
    # V_j(S) = optimal value at exercise date j (j=0 is first at t=dt, j=n_exercise-1 is last at t=T)
    # Start at maturity: V_{n_exercise-1}(S) = max(K - S, 0)
    V = np.maximum(K - S_grid, 0.0)
    
    # Store exercise boundaries for each date
    # boundary[j] = S value above which we continue (exercise if S <= boundary[j])
    # We'll store the decision arrays and extract boundaries later
    exercise_decisions = []  # Will be indexed by exercise date (0 = first, n_exercise-2 = second-to-last)
    
    # We need decisions for dates 0 through n_exercise-2 (the last date n_exercise-1 is always exercise if ITM)
    # Backward induction: for j from n_exercise-2 down to 0
    for j in range(n_exercise - 2, -1, -1):
        if time.time() > deadline - 10.0:
            break
        
        # Compute continuation value: exp(-r*dt) * E[V_{j+1}(S_next) | S]
        # S_next = S * exp(drift + vol * Z)
        
        # Vectorized computation in chunks
        chunk_size = 300
        V_new = np.zeros(n_grid)
        exercise_here = np.zeros(n_grid, dtype=bool)
        
        for chunk_start in range(0, n_grid, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_grid)
            S_chunk = S_grid[chunk_start:chunk_end]
            
            # S_next for each S in chunk and each quadrature point
            S_next = S_chunk[:, None] * exp_factor[None, :]  # (chunk_size, n_quad)
            
            # Clip to grid range
            S_next_clipped = np.clip(S_next, 0.0, S_max)
            
            # Interpolate V at S_next_clipped
            V_interp = np.interp(S_next_clipped.ravel(), S_grid, V).reshape(chunk_size, n_quad)
            
            # Expected continuation value
            expected_cont = V_interp @ w_quad_norm  # (chunk_size,)
            
            # Discounted continuation value
            disc_cont = np.exp(-r * dt) * expected_cont
            
            # Exercise value
            exercise_val = np.maximum(K - S_chunk, 0.0)
            
            # Exercise if exercise_val >= disc_cont
            ex_here = exercise_val >= disc_cont
            exercise_here[chunk_start:chunk_end] = ex_here
            
            # Optimal value
            V_new[chunk_start:chunk_end] = np.maximum(exercise_val, disc_cont)
        
        V = V_new
        exercise_decisions.append(exercise_here)
    
    # exercise_decisions was appended in order j=n_exercise-2, n_exercise-3, ..., 0
    # Reverse to get index by exercise date: exercise_decisions[i] is for exercise date i
    exercise_decisions = exercise_decisions[::-1]
    
    # Extract exercise boundaries: for each date, find the largest S where exercise is True
    boundaries = []
    for decisions in exercise_decisions:
        if not np.any(decisions):
            boundaries.append(0.0)
        elif np.all(decisions):
            boundaries.append(S_max)
        else:
            true_indices = np.where(decisions)[0]
            boundary_idx = true_indices[-1]
            boundaries.append(S_grid[boundary_idx])
    
    # For the last exercise date (n_exercise-1, at T), always exercise if in-the-money
    # This is implicit in the policy
    
    # Now create the policy object
    class ExercisePolicy:
        def __init__(self, boundaries, K, dt, n_exercise, S_grid, S_max):
            self.boundaries = boundaries  # boundaries[i] for exercise date i (0-indexed)
            self.K = K
            self.dt = dt
            self.n_exercise = n_exercise
            self.S_grid = S_grid
            self.S_max = S_max
        
        def exercise(self, current_state, time):
            """
            Determine whether to exercise at the given time and state.
            
            Args:
                current_state: array of asset prices
                time: current time (float)
                
            Returns:
                Boolean array: True if should exercise
            """
            if time <= 0:
                return np.zeros(len(current_state), dtype=bool)
            
            # Find the exercise index
            # t_i = (i+1) * dt for i=0,...,n_exercise-1
            idx = int(round(time / self.dt)) - 1
            
            if idx < 0 or idx >= self.n_exercise:
                return np.zeros(len(current_state), dtype=bool)
            
            # Last exercise date: always exercise if in-the-money
            if idx == self.n_exercise - 1:
                return current_state < self.K
            
            # For earlier dates, use the fitted boundary
            if idx >= len(self.boundaries):
                # Fallback: use a conservative boundary
                return current_state <= self.K * 0.8
            
            S_boundary = self.boundaries[idx]
            return current_state <= S_boundary
    
    return ExercisePolicy(boundaries, K, dt, n_exercise, S_grid, S_max)
# EVOLVE-BLOCK-END
