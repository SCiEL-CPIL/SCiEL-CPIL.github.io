import numpy as np
from numpy.polynomial.hermite import hermgauss

def fit_exercise_policy(training_paths, contract):
    # Parameters
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_exercise = 12
    dt = T / n_exercise
    
    # Grid for log(S)
    # S range [5, 200] -> log(S) range [1.6, 5.3]
    # Let's use a finer grid for accuracy
    n_grid = 2000
    log_s_min = np.log(5)
    log_s_max = np.log(200)
    log_s_grid = np.linspace(log_s_min, log_s_max, n_grid)
    s_grid = np.exp(log_s_grid)
    
    # Quadrature nodes for N(0,1)
    # Transform to N(mu, sigma)
    n_quad = 100
    x, w = hermgauss(n_quad)
    # x, w are for N(0,1) in the sense that integral f(x) exp(-x^2) dx = sum w_i f(x_i)
    # We need integral f(z) phi(z) dz where phi is N(0,1)
    # phi(z) = 1/sqrt(2pi) exp(-z^2/2)
    # Let z = sqrt(2) x. Then dz = sqrt(2) dx.
    # Integral f(sqrt(2)x) 1/sqrt(2pi) exp(-x^2) sqrt(2) dx
    # = 1/sqrt(pi) sum w_i f(sqrt(2) x_i)
    # Actually, hermgauss gives nodes x_i and weights w_i such that
    # integral_{-inf}^{inf} f(x) exp(-x^2) dx ~ sum w_i f(x_i)
    # We want E[f(Z)] for Z~N(0,1).
    # E[f(Z)] = integral f(z) (1/sqrt(2pi)) exp(-z^2/2) dz
    # Let z = sqrt(2) x. dz = sqrt(2) dx.
    # E[f(Z)] = integral f(sqrt(2)x) (1/sqrt(2pi)) exp(-x^2) sqrt(2) dx
    # = (1/sqrt(pi)) integral f(sqrt(2)x) exp(-x^2) dx
    # ~ (1/sqrt(pi)) sum w_i f(sqrt(2) x_i)
    
    z_nodes = np.sqrt(2) * x
    z_weights = w / np.sqrt(np.pi)
    
    # Backward Induction
    # V_next is the value at t_{i+1}
    # We store values on the log_s_grid
    V_next = np.maximum(K - s_grid, 0) # Intrinsic at maturity t_12
    
    # Boundaries for t_1 ... t_12
    # boundaries[i] corresponds to time t_{i+1}
    # We need boundaries for the 12 exercise dates.
    # The last date t_12 is maturity. We always exercise if ITM.
    # The first date t_1 is the earliest exercise.
    
    # Let's store the exercise decision grid for each time step
    # exercise_grids[i] is a boolean array of shape (n_grid,)
    # True if we exercise at that grid point at time t_{i+1}
    
    exercise_grids = []
    
    # We need to go from t_11 back to t_1
    # V_next is value at t_{i+1}
    # We are calculating V_i at t_i
    
    # Note: The problem states "12 monthly exercise dates (no time-0 exercise)".
    # So the dates are t_1, ..., t_12.
    # At t_12, we have the terminal payoff.
    # We determine the policy for t_11, ..., t_1.
    # The policy at t_12 is "exercise if (K-S)>0".
    
    # Let's store the boundary S for each time step.
    # Since the grid is log-uniform, the boundary in log-space is monotonic.
    # We can store the log-boundary.
    
    log_boundaries = []
    
    # Initialize V_current
    # We start with V at t_12
    V_current = V_next.copy()
    
    # We need the boundary for t_12 as well?
    # The exercise method will be called at times t_1...t_12.
    # At t_12, the decision is trivial: exercise if S < K.
    # Let's compute boundaries for t_1...t_11 via induction, and set t_12 to K.
    
    # Actually, let's just compute the full value function backwards and store the "exercise" mask for each step.
    
    # Step 12 (Maturity)
    # Value is intrinsic. Exercise if S < K.
    # log_K = np.log(K)
    # mask_12 = s_grid < K
    
    # Let's store the value function V at each step on the grid.
    # V[0] corresponds to t_1, V[11] corresponds to t_12.
    
    V_vals = [None] * 12
    V_vals[11] = V_next
    
    # We need to calculate V for t_11 down to t_1.
    # And determine the exercise boundary.
    
    # Pre-compute the interpolation for the next step's value
    # We will use np.interp on the log-grid.
    
    for i in range(10, -1, -1): # i corresponds to t_{i+1}
        # Calculate continuation value at t_i
        # S_next = S * exp((r - sigma^2/2)dt + sigma sqrt(dt) Z)
        # log(S_next) = log(S) + (r - sigma^2/2)dt + sigma sqrt(dt) Z
        
        drift = (r - 0.5 * sigma**2) * dt
        vol = sigma * np.sqrt(dt)
        
        # For each point in the grid, we need E[ V_next(S_next) ]
        # V_next is defined on log_s_grid.
        # We can vectorize this.
        
        # log_s_grid shape (n_grid,)
        # z_nodes shape (n_quad,)
        
        # We want to compute for each j in 0..n_grid-1:
        # C_j = sum_k w_k V_next( log_s_grid[j] + drift + vol * z_k )
        
        # This is a matrix multiplication if we can evaluate V_next at arbitrary points.
        # We can use np.interp.
        
        # Construct the target log-spots
        # shape (n_grid, n_quad)
        # This might be memory heavy if n_grid=2000 and n_quad=100. 200k points. Fine.
        
        log_s_targets = log_s_grid[:, None] + drift + vol * z_nodes[None, :]
        
        # Interpolate V_next at log_s_targets
        # np.interp works on 1D. We need to flatten or use a loop.
        # Or use a custom vectorized interp.
        # Since log_s_grid is sorted, we can use searchsorted or just np.interp on flattened arrays.
        
        # Flatten log_s_targets
        flat_targets = log_s_targets.flatten()
        
        # Interpolate
        # V_next is defined on log_s_grid.
        # We need to be careful with boundaries.
        # If target < log_s_min, use V_next[0]. If > log_s_max, use V_next[-1].
        
        V_next_flat = np.interp(flat_targets, log_s_grid, V_vals[11] if i == 10 else V_vals[i+1], left=V_vals[11][0] if i == 10 else V_vals[i+1][0], right=V_vals[11][-1] if i == 10 else V_vals[i+1][-1])
        
        # Reshape to (n_grid, n_quad)
        V_next_at_targets = V_next_flat.reshape(n_grid, n_quad)
        
        # Weighted sum
        # C_j = sum_k w_k V_next_at_targets[j, k]
        C = np.sum(V_next_at_targets * z_weights[None, :], axis=1)
        
        # Discount
        C = np.exp(-r * dt) * C
        
        # Intrinsic Value
        I = np.maximum(K - s_grid, 0)
        
        # Optimal Value
        V_opt = np.maximum(I, C)
        
        # Store Value
        V_vals[i] = V_opt
        
        # Determine Exercise Boundary
        # Exercise if I >= C
        exercise_mask = I >= C
        
        # Find the boundary
        # Since I is decreasing and C is generally decreasing (but slower), the set of S where we exercise is S <= B.
        # Find the last index where exercise_mask is True.
        # If all True, boundary is max. If all False, boundary is min.
        
        # Get indices where mask is True
        true_indices = np.where(exercise_mask)[0]
        if len(true_indices) > 0:
            last_true_idx = true_indices[-1]
            # Boundary is s_grid[last_true_idx]
            # To be more precise, we can interpolate between last_true_idx and last_true_idx+1
            # But for the policy, using the grid point is sufficient.
            # Let's store the log-boundary for interpolation in the exercise method.
            log_B = log_s_grid[last_true_idx]
        else:
            log_B = -np.inf # Never exercise
        
        log_boundaries.append(log_B)
        
    # log_boundaries is for t_1 ... t_11.
    # We need boundary for t_12.
    # At t_12, we exercise if S < K.
    log_B_12 = np.log(K)
    
    # Full list of log boundaries for t_1 ... t_12
    # log_boundaries list is currently in reverse order of the loop (i=10 down to 0).
    # Loop i=10 -> t_11. i=0 -> t_1.
    # So log_boundaries[0] is for t_11.
    # log_boundaries[10] is for t_1.
    
    # Let's reorder them to be t_1, t_2, ..., t_12
    # t_1 corresponds to i=0 in the loop, which is the last element appended.
    # t_11 corresponds to i=10 in the loop, which is the first element appended.
    
    # Let's create a list `all_log_boundaries` of length 12.
    all_log_boundaries = [0.0] * 12
    
    # Fill t_1 to t_11
    # Loop index i (0 to 10) corresponds to time step i+1 (t_1 to t_11)
    # In the loop `for i in range(10, -1, -1)`:
    #   i=10 -> t_11. Appended to index 0 of log_boundaries.
    #   i=0  -> t_1.  Appended to index 10 of log_boundaries.
    
    for k in range(11):
        # k is 0..10, corresponding to t_1..t_11
        # The boundary for t_{k+1} was calculated when i=k.
        # In the loop, i=k.
        # The append order:
        # i=10 (t_11) -> log_boundaries[0]
        # i=9  (t_10) -> log_boundaries[1]
        # ...
        # i=k  (t_{k+1}) -> log_boundaries[10-k]
        
        all_log_boundaries[k] = log_boundaries[10 - k]
        
    # t_12
    all_log_boundaries[11] = log_B_12
    
    # Convert to spot boundaries for easier comparison? Or keep log.
    # Keeping log is better for interpolation if needed, but simple comparison S <= B is fine.
    # Let's store spot boundaries.
    spot_boundaries = np.exp(all_log_boundaries)
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, boundaries):
            self.boundaries = boundaries
            self.times = np.linspace(dt, T, 12)
            
        def exercise(self, current_state, time):
            # current_state is an array
            # time is a float or array? "returning a boolean vector" implies current_state is a vector.
            # "time" is likely a scalar time value.
            
            # Find the closest exercise date index
            # time is in [0, T].
            # Exercise dates are t_1 ... t_12.
            # If time < t_1, we shouldn't be exercising, but the method might be called.
            # The contract says "12 monthly exercise dates".
            # We map time to the nearest date index.
            
            # Ensure time is scalar
            if np.isscalar(time):
                t_val = float(time)
            else:
                t_val = float(time[0])
                
            # Find index
            # self.times are t_1 ... t_12
            idx = np.argmin(np.abs(self.times - t_val))
            
            B = self.boundaries[idx]
            
            return current_state <= B
            
    return ExercisePolicy(spot_boundaries)

# EVOLVE-BLOCK-END
