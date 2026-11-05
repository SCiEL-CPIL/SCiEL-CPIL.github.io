"""Bermudan exercise-policy research: Monte Carlo Regression (Longstaff-Schwartz) stopping rule."""
# EVOLVE-BLOCK-START
import numpy as np

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using Monte Carlo Regression (Longstaff-Schwartz method).
    
    Parameters:
    -----------
    training_paths : ndarray
        Training price paths of shape (n_paths, n_steps + 1).
    contract : dict or object
        Contract specification with fields like K, r, sigma, T, n_steps, etc.
    
    Returns:
    --------
    object with exercise(current_state, time) method returning boolean vector.
    """
    # Extract contract parameters
    K = float(getattr(contract, 'K', 100.0))
    r = float(getattr(contract, 'r', 0.05))
    sigma = float(getattr(contract, 'sigma', 0.2))
    T = float(getattr(contract, 'T', 1.0))
    
    # Determine n_steps from training paths or contract
    if isinstance(training_paths, np.ndarray):
        n_paths, n_total_steps = training_paths.shape
        # Assume training_paths includes t=0, so exercise dates are 1..n_total_steps-1
        n_steps = n_total_steps - 1
    else:
        n_steps = int(getattr(contract, 'n_steps', 12))
        n_paths = 0 # Fallback if no paths, but task guarantees paths

    dt = T / n_steps
    discount = np.exp(-r * dt)

    # Ensure training_paths is a numpy array
    paths = np.asarray(training_paths, dtype=np.float64)
    
    # We need to handle the case where training_paths might not be perfectly aligned or if we need to generate them
    # The task says "4096 training paths". We assume paths shape is (N, n_steps+1).
    
    # If n_steps inferred from paths doesn't match contract, we trust paths shape for indices
    # But we use contract params for discounting and K.
    
    # Initialize arrays for regression coefficients and continuation values
    # We will store the fitted continuation value function for each step.
    # Since the state is 1D (Price), we can use polynomial regression.
    # Degree 3 is usually sufficient for GBM put options in short horizons.
    deg = 3
    
    # Store coefficients for each step k (1 to n_steps)
    # coeffs[k] will be used to evaluate continuation value at step k
    # We store them in a list, indexed by step (1-based for clarity, 0-based for array)
    step_coeffs = [None] * (n_steps + 1) # 0 is unused, 1..n_steps
    
    # We need the "optimal value" at step k+1 to regress for step k.
    # Let V_opt[k] be the optimal value at step k for all paths.
    # V_opt[n_steps] = max(K - S_{n_steps}, 0)
    
    # We'll maintain an array `V_next` of shape (n_paths,) representing the optimal value at the next step.
    # And a boolean mask `alive` indicating which paths have not been exercised yet.
    
    # Initialize for the last step (maturity)
    S_last = paths[:, -1]
    V_next = np.maximum(K - S_last, 0.0)
    alive = np.ones(n_paths, dtype=bool)
    
    # We will iterate backwards from step n_steps-1 down to 1.
    # Wait, the exercise dates are 1..n_steps.
    # At step n_steps (maturity), exercise is forced (or optimal if ITM).
    # So V_opt at step n_steps is the intrinsic value.
    
    # We need to determine the policy for steps 1, 2, ..., n_steps-1.
    # For step k, we look at S_k and compare with Continuation Value estimated from V_opt at k+1.
    
    # Let's refine the loop:
    # We are at step k. We have V_opt for step k+1 (stored in V_next).
    # We regress V_next on S_k (for alive paths).
    # This gives us C_k(S_k).
    # Then V_opt_k = max(Intrinsic_k, C_k(S_k)).
    # We store the coefficients for C_k.
    # Then we set V_next = V_opt_k for the next iteration (step k-1).
    
    # We need to be careful with the "alive" mask.
    # In Longstaff-Schwartz, we regress only on paths that are alive at step k.
    # A path is alive at step k if it was alive at step k+1 AND it did not exercise at step k+1?
    # No, standard LS: 
    # 1. Regress V_{k+1} on S_k for all paths that are alive at k.
    # 2. Compute C_k(S_k).
    # 3. Determine exercise at k: if Intrinsic_k > C_k(S_k), then Exercise.
    # 4. V_k = max(Intrinsic_k, C_k(S_k)).
    # 5. Update alive mask for step k: alive_k = alive_{k+1} & (Intrinsic_k <= C_k(S_k))? 
    #    Actually, for the *training* of the previous step, we only care about paths that reach step k.
    #    If a path exercises at k+1, its value is fixed. If it continues, it contributes to the regression at k.
    #    So, `alive_k` should be True for paths that did NOT exercise at k+1, k+2, etc.
    #    Wait, if a path exercises at k+1, it doesn't exist at k. So it shouldn't be in the regression for k.
    #    Therefore, the mask for regression at step k is: paths that are "alive" at step k.
    #    A path is alive at step k if it did not exercise in any future step (k+1...n_steps).
    
    # Let's re-structure:
    # Start from the end.
    # V_opt at n_steps is known.
    # Mask `alive` at n_steps is all True (everyone reaches maturity).
    # 
    # Loop k from n_steps-1 down to 1:
    #   S_k = paths[:, k]
    #   Regress V_next (which is V_opt at k+1) on S_k using only indices where `alive` is True.
    #   Note: V_next contains values for all paths, but we only use the ones that are alive.
    #   Fit polynomial.
    #   Compute C_k(S_k) for all paths.
    #   Intrinsic_k = max(K - S_k, 0)
    #   Exercise_mask_k = Intrinsic_k > C_k(S_k)
    #   V_opt_k = np.where(Exercise_mask_k, Intrinsic_k, C_k(S_k))
    #   
    #   Update `alive` for the previous step (k-1):
    #   A path is alive at k-1 if it was alive at k AND it did not exercise at k.
    #   alive = alive & ~Exercise_mask_k
    #   
    #   Store coefficients for step k.
    #   Set V_next = V_opt_k.
    
    # Edge case: Step n_steps.
    # The policy at step n_steps is simply exercise if ITM.
    # We don't need a regression for step n_steps, we just know the rule.
    # But we need to set up the state for step n_steps-1.
    # So, initialize:
    # V_next = V_opt at n_steps = max(K - S_{n_steps}, 0)
    # alive = all True (everyone is "alive" at n_steps in the sense that they reach it)
    # But wait, if someone exercises at n_steps, they don't go to n_steps-1.
    # So for the regression at n_steps-1, we only use paths that did NOT exercise at n_steps.
    # Exercise at n_steps happens if K - S_{n_steps} > 0.
    # So, initial `alive` for the loop (which will process k=n_steps-1) should be:
    # alive = (K - S_{n_steps}) <= 0  i.e., S_{n_steps} >= K.
    # Because if S_{n_steps} < K, they exercised, so they are not alive to be regressed at n_steps-1.
    
    S_n = paths[:, -1]
    intrinsic_n = np.maximum(K - S_n, 0.0)
    V_next = intrinsic_n.copy()
    
    # Paths that exercised at n_steps are NOT alive for step n_steps-1 regression.
    # Paths that continued at n_steps (S >= K) are alive.
    alive = (S_n >= K)
    
    # If no paths are alive, we have a problem, but with 4096 paths and ATM option, this is unlikely.
    
    for k in range(n_steps - 1, 0, -1):
        S_k = paths[:, k]
        
        # Filter for alive paths
        mask = alive
        if np.sum(mask) < deg + 1:
            # Not enough data points, fallback to simple estimate or skip
            # If we can't fit, assume continuation is 0 (always exercise if ITM) or use previous?
            # Let's just fit with whatever we have, or use a constant if too few.
            if np.sum(mask) == 0:
                # No alive paths, this shouldn't happen in normal scenarios
                break
            
        X = S_k[mask]
        Y = V_next[mask]
        
        # Polynomial Regression: Y ~ X^0 + X^1 + X^2 + X^3
        # Using np.polyfit
        try:
            coeffs = np.polyfit(X, Y, deg)
        except Exception:
            # Fallback to linear if polyfit fails
            coeffs = np.polyfit(X, Y, 1)
            deg = 1 # Update deg for this step? No, coeffs shape changes. 
            # Let's stick to fixed deg=3 and handle errors by reducing degree.
            # Simpler: use a fixed basis like [1, S, S^2, S^3]
            
        # Let's use explicit basis for stability and consistency
        # Basis: 1, S, S^2, S^3
        X_design = np.column_stack([np.ones_like(X), X, X**2, X**3])
        
        try:
            # Least squares
            coeffs, residuals, rank, s = np.linalg.lstsq(X_design, Y, rcond=None)
        except np.linalg.LinAlgError:
            # Fallback to simple average if matrix is singular
            coeffs = np.array([np.mean(Y), 0, 0, 0])
            
        step_coeffs[k] = coeffs
        
        # Evaluate continuation value for ALL paths (to determine exercise for all)
        S_all = S_k
        X_all_design = np.column_stack([np.ones_like(S_all), S_all, S_all**2, S_all**3])
        C_k = X_all_design @ coeffs
        
        # Ensure C_k is non-negative (continuation value of a put can't be negative)
        C_k = np.maximum(C_k, 0.0)
        
        Intrinsic_k = np.maximum(K - S_all, 0.0)
        
        # Determine exercise
        Exercise_mask_k = Intrinsic_k > C_k
        
        # Optimal Value at step k
        V_opt_k = np.where(Exercise_mask_k, Intrinsic_k, C_k)
        
        # Update alive mask for the next (earlier) step
        # A path is alive at k-1 if it was alive at k AND it did not exercise at k
        alive = alive & ~Exercise_mask_k
        
        # Update V_next for the next iteration
        V_next = V_opt_k
        
    # Now we have step_coeffs for k=1 to n_steps-1.
    # We also need the policy for step n_steps.
    # At step n_steps, exercise if K - S > 0.
    # We can represent this as a boundary K.
    # Or we can just handle it in the exercise method.
    
    # For consistency, let's store a "boundary" or just the coeffs.
    # The exercise method will use coeffs for k < n_steps and intrinsic logic for k = n_steps.
    
    class ExercisePolicy:
        def __init__(self, coeffs_list, n_steps, K, T):
            self.coeffs_list = coeffs_list
            self.n_steps = n_steps
            self.K = K
            self.dt = T / n_steps
            
        def exercise(self, current_state, time):
            """
            Determine whether to exercise.
            
            Parameters:
            -----------
            current_state : ndarray of shape (n_eval_paths,)
                Current prices.
            time : int or float
                Time index or time value.
            
            Returns:
            --------
            ndarray of bools, shape (n_eval_paths,)
            """
            S = np.asarray(current_state, dtype=np.float64)
            
            # Convert time to exercise date index (1-based)
            if isinstance(time, (int, np.integer)):
                k = int(time)
            else:
                k = int(round(float(time) / self.dt))
            
            # Clamp k to [1, n_steps]
            k = max(1, min(k, self.n_steps))
            
            if k == self.n_steps:
                # Maturity: exercise if ITM
                return S < self.K
            
            # For k < n_steps, use the regression coefficients
            coeffs = self.coeffs_list[k]
            if coeffs is None:
                # Fallback if fitting failed for some reason
                return S < self.K
            
            # Evaluate continuation value
            X_design = np.column_stack([np.ones_like(S), S, S**2, S**3])
            C_k = X_design @ coeffs
            C_k = np.maximum(C_k, 0.0)
            
            Intrinsic_k = np.maximum(self.K - S, 0.0)
            
            return Intrinsic_k > C_k

    return ExercisePolicy(step_coeffs, n_steps, K, T)
# EVOLVE-BLOCK-END
