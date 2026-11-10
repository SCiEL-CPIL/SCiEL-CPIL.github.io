"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fits the exercise policy for a Bermudan put option using Longstaff-Schwartz 
    regression and local boundary optimization.
    
    Strategy:
    1. Use Longstaff-Schwartz regression on training_paths to estimate continuation values.
    2. Derive initial exercise boundaries from the regression-based optimal policy.
    3. Refine boundaries via local grid search to maximize empirical payoff on training paths.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin within 120s limit
    
    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12
    dt = T / n_exercises
    disc = np.exp(-r * dt)
    
    # training_paths shape: (n_paths, n_exercises) or (n_paths, n_exercises+1)?
    # Contract says "12 monthly exercise dates (no time-0 exercise)".
    # Usually paths include time 0. Let's assume shape is (n_paths, n_exercises+1) 
    # or (n_paths, n_exercises). 
    # "4096 training paths". 
    # Let's inspect typical structure. If it's just the exercise dates, it's 12 columns.
    # If it includes t=0, it's 13.
    # The prompt says "12 monthly exercise dates".
    # Let's assume the input `training_paths` contains the stock prices at the exercise dates.
    # If shape[1] == 12, indices 0..11 are the exercise dates.
    # If shape[1] == 13, indices 1..12 are the exercise dates (0 is t=0).
    # We need to be careful. Let's assume the standard convention for such tasks:
    # Often `training_paths` is the full path including t=0.
    # However, the interface `exercise(current_state, time)` takes `time` as index.
    # If there are 12 exercise dates, `time` likely ranges 0..11.
    # Let's check the number of columns.
    
    paths = np.asarray(training_paths)
    if paths.ndim != 2:
        raise ValueError("training_paths must be 2D")
        
    n_paths, n_cols = paths.shape
    
    # Determine offset for exercise dates
    # If n_cols == 12, exercise dates are 0..11
    # If n_cols == 13, exercise dates are 1..12 (assuming t=0 is first col)
    # The contract says "12 monthly exercise dates".
    # Let's assume the columns correspond to the 12 exercise dates directly.
    # If n_cols > 12, we might need to slice.
    # Let's assume the standard case where the provided paths are exactly the states at exercise times.
    # If n_cols == 13, we drop the first column (t=0) as it's not an exercise date.
    if n_cols == 13:
        exercise_paths = paths[:, 1:13]
    elif n_cols == 12:
        exercise_paths = paths[:, 0:12]
    else:
        # Fallback: assume last 12 columns
        exercise_paths = paths[:, -12:]
        
    # 1. Longstaff-Schwartz Regression
    # We regress the continuation value at step j on the state S_j.
    # V_j(S_j) = max(Intrinsic_j, disc * E[V_{j+1}(S_{j+1}) | S_j])
    # We approximate E[V_{j+1} | S_j] using regression on the in-the-money paths at step j.
    
    # Start from the last exercise date (index n_exercises - 1)
    # At the last date, the option value is the intrinsic value.
    # V_last = max(K - S_last, 0)
    
    # We store the "continuation value" estimates or the "optimal value" for regression.
    # Actually, LSM usually regresses the *continuation* part to decide exercise.
    # Let's compute the "Americanized" payoff backwards.
    
    # payoff_at_maturity = max(K - S_T, 0)
    # For j from T-1 down to 0:
    #   Regress discounted future payoff on S_j for paths where S_j is in the money.
    #   Continuation_j = Regressed Value
    #   Optimal_Value_j = max(Intrinsic_j, Continuation_j)
    #   The "future payoff" for the next regression step is Optimal_Value_j.
    
    # We need to store the optimal values for each step to use as targets for the previous step.
    # Since we only have the paths, we can compute the optimal values on the paths.
    
    # Initialize the "future value" target for the last step (which is just intrinsic)
    # Note: The "future value" passed to step j is the value at step j+1.
    # So for step j, we need the value at step j+1.
    
    # Let's store the optimal values for all steps to use in boundary optimization later.
    optimal_values = np.zeros((n_paths, n_exercises))
    
    # Last step (j = n_exercises - 1)
    optimal_values[:, -1] = np.maximum(K - exercise_paths[:, -1], 0.0)
    
    # Polynomial degree for regression
    poly_deg = 4
    # Minimum number of in-the-money paths required for stable regression
    min_paths_for_reg = 100
    
    # Backward induction steps
    # We will estimate the continuation value function C_j(S_j) for each j.
    # We store the coefficients or the predicted values.
    # For boundary finding, we need C_j(S_j) at the grid points or on the paths.
    # For boundary optimization, we need to simulate the policy on the paths.
    
    # Let's store the regression coefficients for each step.
    # X = [1, S, S^2, S^3, S^4]
    # We regress disc * V_{j+1} on X_j for paths where S_j < K (in the money).
    
    regression_coeffs = []
    # We also need the predicted continuation values on the training paths for the current step
    # to determine the initial boundaries.
    cont_values_on_paths = np.zeros((n_paths, n_exercises))
    
    # Last step continuation is 0 (no future)
    cont_values_on_paths[:, -1] = 0.0
    
    # Work backwards
    for j in range(n_exercises - 2, -1, -1):
        if time.time() > deadline - 10.0:
            # Emergency break
            break
            
        S_j = exercise_paths[:, j]
        S_next = exercise_paths[:, j+1]
        V_next = optimal_values[:, j+1]
        
        # Target for regression: discounted future value
        target = disc * V_next
        
        # Filter for in-the-money paths at step j
        # For a put, ITM if S_j < K
        it_mask = S_j < K
        n_it = np.sum(it_mask)
        
        if n_it < min_paths_for_reg:
            # Not enough data, assume continuation is 0 (exercise always)
            coeffs = np.zeros(poly_deg + 1)
            regression_coeffs.append(coeffs)
            cont_values_on_paths[:, j] = 0.0
        else:
            S_it = S_j[it_mask]
            target_it = target[it_mask]
            
            # Build polynomial features
            # Use normalized S to improve conditioning?
            # S is around 100. S^4 is 1e8. 
            # Let's normalize S by K.
            S_norm = S_it / K
            X = np.column_stack([S_norm**i for i in range(poly_deg + 1)])
            
            # Solve least squares
            try:
                coeffs, residuals, rank, sv = np.linalg.lstsq(X, target_it, rcond=None)
            except:
                coeffs = np.zeros(poly_deg + 1)
            
            regression_coeffs.append(coeffs)
            
            # Predict continuation values for ALL paths at step j
            S_all_norm = S_j / K
            X_all = np.column_stack([S_all_norm**i for i in range(poly_deg + 1)])
            cont_vals = X_all @ coeffs
            # Ensure non-negative? Continuation value should be non-negative.
            cont_vals = np.maximum(cont_vals, 0.0)
            cont_values_on_paths[:, j] = cont_vals
    
    # If we broke early, fill remaining
    while len(regression_coeffs) < n_exercises - 1:
        regression_coeffs.append(np.zeros(poly_deg + 1))
        # We need to align indices. 
        # regression_coeffs[j] corresponds to step j.
        # If we broke at j, we have coeffs for j+1 ... n-2.
        # We need coeffs for 0 ... j.
        # This logic is tricky if we break. Let's assume we finish the loop.
        # If time is tight, we will skip optimization.

    # 2. Determine Initial Boundaries
    # Boundary B_j is where Intrinsic_j == Continuation_j.
    # Intrinsic = K - S. Continuation = C_j(S).
    # We can find the boundary by searching on a fine grid using the regression models.
    
    boundaries = np.zeros(n_exercises)
    
    # For the last step, boundary is K.
    boundaries[-1] = K
    
    # Grid for boundary search
    S_min = 1e-6
    S_max = 2.0 * K
    n_grid = 1000
    S_grid = np.geomspace(S_min, S_max, n_grid)
    
    for j in range(n_exercises - 2, -1, -1):
        if time.time() > deadline - 5.0:
            boundaries[j] = K
            continue
            
        # Get coefficients for step j
        # Note: regression_coeffs list is built backwards? 
        # In the loop `for j in range(n_exercises - 2, -1, -1)`, we appended coeffs.
        # So regression_coeffs[0] corresponds to j = n_exercises - 2.
        # regression_coeffs[k] corresponds to j = n_exercises - 2 - k.
        # Let's create a dictionary or list aligned to j.
        pass
        
    # Let's reconstruct coeffs aligned to index j
    coeffs_by_j = {}
    # The loop ran j from n_exercises-2 down to 0.
    # It appended coeffs in that order.
    # So coeffs_list[0] is for j = n_exercises-2.
    # coeffs_list[1] is for j = n_exercises-3.
    # ...
    # coeffs_list[n_exercises-2] is for j = 0.
    
    # Let's just re-calculate or store properly.
    # I'll re-implement the storage to be cleaner.
    
    # Re-doing storage logic for clarity and correctness
    coeffs_by_j = {}
    cont_values_on_paths = np.zeros((n_paths, n_exercises))
    optimal_values = np.zeros((n_paths, n_exercises))
    
    optimal_values[:, -1] = np.maximum(K - exercise_paths[:, -1], 0.0)
    cont_values_on_paths[:, -1] = 0.0
    coeffs_by_j[-1] = None # No regression needed for last step
    
    for j in range(n_exercises - 2, -1, -1):
        if time.time() > deadline - 10.0:
            break
            
        S_j = exercise_paths[:, j]
        V_next = optimal_values[:, j+1]
        target = disc * V_next
        
        it_mask = S_j < K
        n_it = np.sum(it_mask)
        
        if n_it < min_paths_for_reg:
            coeffs = np.zeros(poly_deg + 1)
            cont_vals = np.zeros(n_paths)
        else:
            S_it = S_j[it_mask]
            target_it = target[it_mask]
            S_norm = S_it / K
            X = np.column_stack([S_norm**i for i in range(poly_deg + 1)])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, target_it, rcond=None)
            except:
                coeffs = np.zeros(poly_deg + 1)
            
            S_all_norm = S_j / K
            X_all = np.column_stack([S_all_norm**i for i in range(poly_deg + 1)])
            cont_vals = X_all @ coeffs
            cont_vals = np.maximum(cont_vals, 0.0)
            
        coeffs_by_j[j] = coeffs
        cont_values_on_paths[:, j] = cont_vals
        optimal_values[:, j] = np.maximum(K - S_j, cont_vals)

    # If loop broke early, fill remaining j with K boundary
    for j in range(n_exercises - 1):
        if j not in coeffs_by_j:
            coeffs_by_j[j] = np.zeros(poly_deg + 1)
            cont_values_on_paths[:, j] = 0.0
            optimal_values[:, j] = np.maximum(K - exercise_paths[:, j], 0.0)

    # Compute initial boundaries from regression
    for j in range(n_exercises - 1):
        coeffs = coeffs_by_j[j]
        # Find S where K - S = C(S)
        # K - S = sum(coeffs[i] * (S/K)^i)
        # We solve for S on a grid.
        
        S_grid = np.geomspace(1e-6, 2.0 * K, 1000)
        S_norm_grid = S_grid / K
        C_grid = np.zeros_like(S_grid)
        for i in range(len(coeffs)):
            C_grid += coeffs[i] * (S_norm_grid**i)
        C_grid = np.maximum(C_grid, 0.0)
        
        intrinsic_grid = K - S_grid
        
        # Exercise if Intrinsic >= Continuation
        # Find the largest S where this is true.
        ex_mask = intrinsic_grid >= C_grid
        if np.any(ex_mask):
            # Get indices
            idx = np.where(ex_mask)[0]
            max_idx = idx[-1]
            boundaries[j] = S_grid[max_idx]
        else:
            boundaries[j] = 0.0
            
        # Ensure monotonicity? 
        # For puts, boundaries should be non-decreasing as j increases (approaching maturity).
        # boundaries[0] <= boundaries[1] ... <= boundaries[11] = K
        # We will enforce this after optimization or now.
        pass
    
    # Enforce monotonicity on initial boundaries
    for j in range(1, n_exercises):
        if boundaries[j] < boundaries[j-1]:
            boundaries[j] = boundaries[j-1]

    # 3. Local Optimization of Boundaries
    # We want to maximize the mean discounted payoff on the training paths.
    # Payoff for a path p:
    #   Find the first j where S_p,j <= B_j.
    #   If found, payoff = (K - S_p,j) * disc^(j) ... wait.
    #   The payoff is received at time j.
    #   The objective is "Maximize mean discounted payoff".
    #   Discounting to t=0: Payoff_p = (K - S_p,j_ex) * exp(-r * t_j)
    #   If never exercised, payoff = 0.
    
    # We will perform a coordinate ascent on the boundaries.
    # For each boundary B_j, try a few values in a local range and pick the best.
    
    def compute_payoff(bnds):
        # bnds: array of shape (n_exercises,)
        # Returns mean discounted payoff
        total_payoff = 0.0
        # Vectorized computation
        # For each path, we need to find the first j where S <= B_j.
        # This is hard to vectorize easily with different B_j per step.
        # Loop over paths might be slow for 4096 paths * 12 steps * iterations.
        # 4096 * 12 = ~50k ops per eval. 
        # If we do 100 evals, that's 5M ops. Fine.
        
        # However, we can be smarter.
        # Let's just loop.
        S_paths = exercise_paths
        n_p = S_paths.shape[0]
        
        # Precompute discount factors for each step
        # t_j = (j+1) * dt ? 
        # Contract: "12 monthly exercise dates".
        # Usually t_1 = T/12, t_12 = T.
        # So t_j = j * dt if 1-indexed? 
        # Let's assume index 0 is t=dt, index 11 is t=T.
        t_steps = np.arange(1, n_exercises + 1) * dt
        disc_factors = np.exp(-r * t_steps)
        
        payoffs = np.zeros(n_p)
        exercised = np.zeros(n_p, dtype=bool)
        
        # Iterate steps
        for j in range(n_exercises):
            # Paths not yet exercised
            mask = ~exercised
            if not np.any(mask):
                break
            
            S_j = S_paths[mask, j]
            B_j = bnds[j]
            
            ex_mask = S_j <= B_j
            
            # Mark these as exercised
            # We need to update the 'exercised' array
            # Get indices in original array
            # This is inefficient. Let's use a different approach.
            
            # Alternative: 
            # For each path, find the first j.
            # Since n_exercises is small (12), we can loop over paths.
            # 4096 paths.
            # Let's try vectorizing the "first hit" logic.
            
            # Create a boolean matrix: (n_paths, n_exercises)
            # hit[p, j] = (S[p, j] <= B[j]) AND (not hit[p, 0..j-1])
            
            # We can do this iteratively.
            remaining = np.ones(n_p, dtype=bool)
            
            for j in range(n_exercises):
                if not np.any(remaining):
                    break
                S_j = S_paths[remaining, j]
                B_j = bnds[j]
                hit = S_j <= B_j
                if np.any(hit):
                    # Get indices of remaining paths that hit
                    rem_indices = np.where(remaining)[0]
                    hit_indices = rem_indices[hit]
                    payoffs[hit_indices] = (K - S_paths[hit_indices, j]) * disc_factors[j]
                    remaining[hit_indices] = False
        
        return np.mean(payoffs)

    # Optimization Loop
    # We have 12 boundaries.
    # We will do a few passes of coordinate ascent.
    # For each boundary, try values: B - 0.1, B - 0.05, B, B + 0.05, B + 0.1
    # Or relative: B * [0.9, 0.95, 1.0, 1.05, 1.1]
    
    current_bnds = boundaries.copy()
    current_val = compute_payoff(current_bnds)
    
    max_iterations = 20
    for _ in range(max_iterations):
        if time.time() > deadline - 2.0:
            break
            
        improved = False
        for j in range(n_exercises - 1): # Last boundary is K, don't change
            # Local search around current_bnds[j]
            # Range: +/- 10% of K? Or +/- 5% of current B?
            # Let's use absolute steps relative to K.
            # K=100. Steps: 1, 2, 5.
            candidates = [
                current_bnds[j] - 5.0,
                current_bnds[j] - 2.0,
                current_bnds[j] - 1.0,
                current_bnds[j],
                current_bnds[j] + 1.0,
                current_bnds[j] + 2.0,
                current_bnds[j] + 5.0
            ]
            
            # Clip to valid range [0, K]
            candidates = [np.clip(c, 0.0, K) for c in candidates]
            
            best_val = current_val
            best_bnd = current_bnds[j]
            
            for c in candidates:
                if c == current_bnds[j]:
                    continue
                test_bnds = current_bnds.copy()
                test_bnds[j] = c
                
                # Enforce monotonicity for this candidate
                # B[j-1] <= B[j] <= B[j+1]
                # If j > 0: B[j-1] <= c
                # If j < n-1: c <= B[j+1]
                if j > 0 and c < current_bnds[j-1]:
                    continue
                if j < n_exercises - 1 and c > current_bnds[j+1]:
                    continue
                
                val = compute_payoff(test_bnds)
                if val > best_val:
                    best_val = val
                    best_bnd = c
            
            if best_bnd != current_bnds[j]:
                current_bnds[j] = best_bnd
                current_val = best_val
                improved = True
        
        if not improved:
            break
            
    # Final monotonicity check
    for j in range(1, n_exercises):
        if current_bnds[j] < current_bnds[j-1]:
            current_bnds[j] = current_bnds[j-1]

    # Create the policy object
    class ExercisePolicy:
        def __init__(self, boundaries, n_exercises):
            self.boundaries = boundaries
            self.n_exercises = n_exercises
            
        def exercise(self, current_state, time):
            """
            current_state: array of stock prices for all paths at this time step.
            time: the time step index (0-based, 0 to 11 for 12 dates)
            """
            if time < 0 or time >= self.n_exercises:
                return np.zeros(len(current_state), dtype=bool)
            
            B = self.boundaries[time]
            return current_state <= B
    
    return ExercisePolicy(current_bnds, n_exercises)
# EVOLVE-BLOCK-END
