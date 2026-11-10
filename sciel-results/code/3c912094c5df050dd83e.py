"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fits the exercise policy for a Bermudan put option using Monte Carlo 
    Longstaff-Schwarz (LSM) regression.
    
    This approach uses the provided training paths to estimate the continuation 
    value at each exercise date by regressing the discounted future payoff on 
    the current state. It is a data-driven method that adapts to the specific 
    training ensemble, unlike grid-based methods which assume a specific grid 
    structure.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12
    
    # Training paths shape: (n_paths, n_steps + 1) typically, or (n_paths, n_exercises + 1)
    # The contract says "12 monthly exercise dates". 
    # Usually training_paths includes time 0. So shape is likely (4096, 13) or (4096, 12) depending on definition.
    # "no time-0 exercise" means exercise decisions are made at t_1 ... t_12.
    # Let's assume training_paths contains prices at t_0, t_1, ..., t_12.
    
    if training_paths is None or len(training_paths) == 0:
        # Fallback to simple intrinsic if no data
        class FallbackPolicy:
            def exercise(self, current_state, time):
                return current_state < K
        return FallbackPolicy()

    paths = np.asarray(training_paths, dtype=float)
    
    # Determine number of time steps in data
    # If paths are (N, T+1), then T+1 is the number of columns.
    # We have 12 exercise dates. Let's assume the data aligns with these.
    # If the data has 13 columns (t0..t12), we use indices 1..12 for decisions.
    # If the data has 12 columns (t1..t12), we use indices 0..11.
    
    n_paths, n_cols = paths.shape
    
    # Heuristic to align indices:
    # If n_cols == 13, assume t0..t12. Exercise at steps 1..12.
    # If n_cols == 12, assume t1..t12. Exercise at steps 0..11.
    # The prompt says "12 monthly exercise dates".
    # Standard Bermudan setup: t_1, ..., t_12.
    
    if n_cols >= 13:
        # Assume last column is maturity t=1.0
        # We need values at t_1 ... t_12
        # Indices: 1 to 12
        start_idx = 1
        end_idx = 12
    else:
        # Assume indices 0 to 11 correspond to t_1 ... t_12
        start_idx = 0
        end_idx = 11
        
    # Extract prices at exercise dates
    # S_j for j in 0..11 corresponding to t_{j+1}
    S_ex = paths[:, start_idx:end_idx+1]  # Shape (n_paths, 12)
    
    # Payoff at maturity (last date)
    # The last column in S_ex is the price at t_12.
    # Payoff = max(K - S, 0)
    payoff = np.maximum(K - S_ex[:, -1], 0.0)
    
    # We need to compute the continuation value V_j(S_j) for j=0..10
    # V_j(S_j) = E[ disc^(1) * V_{j+1}(S_{j+1}) | S_j ]
    # where V_11(S_11) = max(K - S_11, 0)
    
    # LSM Approach:
    # 1. Simulate paths (we have them).
    # 2. At each step j from 10 down to 0:
    #    a. Determine which paths are still "alive" (not exercised in future steps).
    #    b. For alive paths, compute the realized continuation value (discounted future value).
    #    c. Regress this continuation value on S_j (and possibly other features) for alive paths.
    #    d. The regression function gives the estimated continuation value for ALL paths at step j.
    #    e. Compare intrinsic value at step j with estimated continuation value.
    #    f. Update "alive" status: if intrinsic > continuation, exercise (stop being alive).
    
    # Discount factor per step
    dt = T / n_exercises
    disc = np.exp(-r * dt)
    
    # Features for regression: S, S^2, S^3, 1/S, log(S), etc.
    # For a put, the value is convex and decreases with S.
    # Let's use a polynomial basis in S and 1/S to handle the asymmetry near 0.
    # Basis: [1, S, S^2, S^3, 1/S, 1/S^2] might be unstable if S is very small.
    # Better: [1, S, S^2, S^3, log(S)]
    # Or just use S and S^2 and S^3.
    
    def build_features(S):
        # Avoid log(0)
        S_safe = np.maximum(S, 1e-8)
        f1 = np.ones_like(S)
        f2 = S
        f3 = S**2
        f4 = S**3
        f5 = np.log(S_safe)
        return np.column_stack([f1, f2, f3, f4, f5])
    
    # Initialize continuation values
    # V_next is the value at step j+1
    V_next = payoff.copy()  # Value at maturity for all paths
    
    # Alive mask: initially all paths are alive
    alive = np.ones(n_paths, dtype=bool)
    
    # Store boundaries or the regression coefficients for each step?
    # The interface requires exercise(current_state, time) -> bool vector.
    # We can store the regression coefficients for each step and use them at inference.
    # Or we can store the optimal boundary if we assume a threshold policy.
    # LSM gives a function V(S). The exercise rule is S < B(S) where B is the root of Intrinsic(S) = V(S).
    # However, the optimal policy for a Bermudan put is indeed a threshold policy (exercise if S < B_j).
    # So we can fit the regression, compute the implied boundary for each step, and store that.
    # This is much faster and cleaner for the `exercise` call.
    
    boundaries = np.zeros(n_exercises)
    
    # We iterate backwards from step 10 (second to last) to 0 (first exercise date)
    # Note: S_ex columns 0..11 correspond to dates 1..12.
    # Date 12 is maturity. No decision there (or decision is trivial: exercise if ITM).
    # Decisions are at dates 1..11 (indices 0..10 in S_ex).
    
    for j in range(n_exercises - 2, -1, -1):
        if time.time() > deadline:
            # Emergency fallback: assume intrinsic value (exercise if ITM) for remaining
            for k in range(j, -1, -1):
                boundaries[k] = K
            break
        
        # Current prices at step j
        S_j = S_ex[:, j]
        
        # Intrinsic value at step j
        intrinsic_j = np.maximum(K - S_j, 0.0)
        
        # Continuation value for paths that are currently alive
        # We need to estimate E[ disc * V_{j+1} | S_j ]
        # V_{j+1} is known for all paths (from previous iteration or maturity payoff).
        # But in LSM, we only use the realized values for paths that WOULD HAVE continued.
        # Since we are going backwards, we know which paths were exercised in the future.
        # The `alive` mask tells us which paths survived up to step j.
        # Wait, the standard LSM algorithm:
        # 1. Start at maturity.
        # 2. At step j, consider only paths that have NOT been exercised in steps j+1..N.
        # 3. For these "alive" paths, the realized continuation value is disc * V_{j+1}(S_{j+1}).
        # 4. Regress this on S_j.
        # 5. Use the regression to estimate continuation value for ALL paths at step j.
        # 6. Compare with intrinsic. If intrinsic > cont, exercise. Update alive mask.
        
        # Get indices of alive paths
        alive_indices = np.where(alive)[0]
        
        if len(alive_indices) < 10:
            # Not enough data to regress, assume exercise if ITM
            boundaries[j] = K
            # Update alive: only those who are ITM? No, if we can't regress, we assume we exercise if ITM.
            # So paths that are ITM at j are exercised.
            new_alive = alive & (intrinsic_j > 0)
            alive = new_alive
            continue
        
        # Realized continuation value for alive paths
        # V_next is the value at step j+1.
        # The realized value is disc * V_next[alive_indices]
        realized_cont = disc * V_next[alive_indices]
        
        # Features for alive paths
        S_alive = S_j[alive_indices]
        X = build_features(S_alive)
        
        # Ridge Regression to avoid overfitting
        # y = X @ beta
        # Use least squares with regularization
        try:
            # Add small regularization
            alpha = 1.0
            beta = np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ realized_cont)
        except np.linalg.LinAlgError:
            # Fallback to simple average or intrinsic
            beta = np.zeros(X.shape[1])
            beta[0] = np.mean(realized_cont)
            
        # Estimate continuation value for ALL paths at step j
        X_all = build_features(S_j)
        cont_val_all = X_all @ beta
        
        # Ensure continuation value is non-negative
        cont_val_all = np.maximum(cont_val_all, 0.0)
        
        # Determine exercise decision for all paths
        # Exercise if intrinsic >= continuation
        exercise_mask = intrinsic_j >= cont_val_all
        
        # Find the boundary
        # Since the policy is a threshold, we can find the max S where exercise is optimal.
        # However, due to noise, it might not be perfectly monotonic.
        # We can find the largest S such that intrinsic > cont.
        # Sort S_j and find the crossover.
        
        # Simple approach: Find the max S in the exercise_mask
        if np.any(exercise_mask):
            max_S_exercise = np.max(S_j[exercise_mask])
            # To be safe, maybe take a slightly conservative boundary?
            # Or just use the max.
            boundaries[j] = max_S_exercise
        else:
            boundaries[j] = 0.0
            
        # Update alive mask for next iteration (j-1)
        # Paths that are exercised at j are no longer alive for step j-1.
        # Also, paths that were already not alive remain not alive.
        alive = alive & ~exercise_mask
        
        # Update V_next? 
        # In the next iteration (j-1), we need V_j for the alive paths.
        # V_j for a path is:
        # - If exercised: Intrinsic
        # - If not exercised: Continuation (estimated by regression)
        # But for the regression at j-1, we only care about paths that are ALIVE at j-1.
        # The realized continuation value for a path alive at j-1 is disc * V_j(S_j).
        # If the path was exercised at j, it wouldn't be alive at j-1? 
        # No, the logic is:
        # At step j-1, a path is "alive" if it has not been exercised in steps j..N.
        # So the set of alive paths at j-1 is a subset of the set of alive paths at j.
        # For these paths, they were NOT exercised at j.
        # Therefore, their V_j is the continuation value (which we estimated as cont_val_all).
        # So we can set V_next = cont_val_all for the next iteration?
        # Wait. V_next in the loop represents the value at step j+1.
        # After processing step j, we move to j-1.
        # The value at step j for the paths that continue (i.e., were not exercised at j) is the continuation value.
        # So for the next iteration (j-1), the "realized continuation" is disc * V_j.
        # V_j for the alive paths is cont_val_all[alive].
        # So we should update V_next to be the value at step j for ALL paths?
        # Actually, we only use V_next for the indices that are `alive`.
        # So we can just update V_next to be the continuation values for the paths that continued.
        # But `alive` will be updated to `alive & ~exercise_mask`.
        # So for the new `alive` set, V_next should be `cont_val_all`.
        
        V_next = cont_val_all.copy()

    # Ensure boundaries are monotonic? 
    # For puts, boundaries should increase as we approach maturity.
    # boundaries[0] <= boundaries[1] ... <= boundaries[11]=K
    for j in range(1, n_exercises):
        if boundaries[j] < boundaries[j-1]:
            boundaries[j] = boundaries[j-1]
            
    # Final check: boundary at last step (j=10, which is t_11) should be close to K.
    # The last decision is at t_11. Maturity is t_12.
    # boundaries[10] is the boundary for the 11th month.
    # boundaries[11] is not used in the loop (loop goes 10 down to 0).
    # We set boundaries[11] = K implicitly by the monotonicity or we should set it explicitly.
    # The `exercise` function will be called with time 0..11.
    # If time=11, it's the last date. We should exercise if S < K.
    boundaries[11] = K
    
    class ExercisePolicy:
        def __init__(self, boundaries, n_exercises):
            self.boundaries = boundaries
            self.n_exercises = n_exercises
            
        def exercise(self, current_state, time):
            """
            current_state: array of stock prices.
            time: step index 0..11.
            """
            if time < 0 or time >= self.n_exercises:
                return np.zeros(len(current_state), dtype=bool)
            
            B = self.boundaries[time]
            return current_state <= B
    
    return ExercisePolicy(boundaries, n_exercises)
# EVOLVE-BLOCK-END
