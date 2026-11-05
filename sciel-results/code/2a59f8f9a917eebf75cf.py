"""Bermudan exercise-policy research: High-resolution binomial tree for optimal boundaries."""
# EVOLVE-BLOCK-START
import numpy as np

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using a high-resolution Binomial Tree.
    
    This method uses a fine-grained time grid (e.g., 1200 steps) to accurately
    compute the continuation values at the 12 discrete exercise dates. This
    significantly reduces discretization error compared to a coarse 12-step tree.
    
    Parameters:
    -----------
    training_paths : ndarray
        Training price paths (ignored for computation).
    contract : dict or object
        Contract specification with fields like K, r, sigma, T, n_steps, etc.
    
    Returns:
    --------
    object with exercise(current_state, time) method returning boolean vector.
    """
    # Extract contract parameters
    K = getattr(contract, 'K', 100.0)
    r = getattr(contract, 'r', 0.05)
    sigma = getattr(contract, 'sigma', 0.2)
    T = getattr(contract, 'T', 1.0)
    n_exercise_dates = getattr(contract, 'n_steps', 12)
    
    # Ensure numeric types
    K = float(K)
    r = float(r)
    sigma = float(sigma)
    T = float(T)
    n_exercise_dates = int(n_exercise_dates)
    
    # If n_exercise_dates is not provided or is 0, try to infer from training_paths
    if n_exercise_dates <= 0:
        if isinstance(training_paths, np.ndarray):
            n_exercise_dates = training_paths.shape[1] - 1
        else:
            n_exercise_dates = 12
            
    # High-resolution parameters
    # Use 100 steps between each exercise date for a total of 1200 steps
    steps_per_period = 100
    total_steps = n_exercise_dates * steps_per_period
    dt = T / total_steps
    
    # Binomial Tree Parameters (CRR)
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)
    
    # We need to compute the optimal exercise boundary for each of the n_exercise_dates.
    # Exercise occurs at steps: steps_per_period, 2*steps_per_period, ..., total_steps.
    # Let's index exercise dates 0 to n_exercise_dates - 1.
    # Exercise date i corresponds to tree step i*steps_per_period + 1? 
    # Usually step 0 is t=0. Step 1 is t=dt.
    # Exercise dates are at t = T/n, 2T/n, ..., T.
    # So exercise date i (0-indexed) is at tree step (i+1)*steps_per_period.
    
    # We will perform backward induction from the final step (total_steps) back to 0.
    # We only need to store the values for the next step.
    # However, we need the boundary at the exercise dates.
    
    # To save memory, we can just iterate backwards.
    # We need to know the price at each node to determine the boundary.
    # The price at step k, node j is S0 * u^j * d^(k-j).
    # Since the boundary is a price level, and the tree is scale-invariant with respect to S0?
    # No, the boundary depends on K.
    # Let's set S0 = 1.0 for the tree construction.
    # The calculated boundary B will be in units of S0=1.
    # Wait, if S0=1 and K=100, the put is deeply ITM.
    # The boundary will be close to K.
    # The policy receives actual S.
    # So we just use the absolute price level calculated from the tree with S0=1.
    # Is this correct?
    # Value(S, t) = S * v(K/S, t).
    # Boundary B(t) is such that S = B(t).
    # If we build the tree with S0=1, the price at node j is P_j.
    # The condition is P_j <= B.
    # Since P_j = 1 * u^j * d^(k-j), the boundary B is an absolute number.
    # This number is valid for any S0 because the ratio K/S determines the decision.
    # Actually, the boundary B(t) is a function of K, r, sigma, and time-to-maturity.
    # It is an absolute price.
    # So yes, calculating it with S0=1 gives the correct absolute price level.
    
    S0 = 1.0
    
    # Initialize arrays for the final step
    # We need values for all nodes at the final step to compute the previous step.
    # But we only need to keep the values for the current step.
    # Let's use a list of arrays? No, just two arrays: next_vals and curr_vals.
    # We also need to identify which steps are exercise steps.
    
    # next_vals will hold the values for step k+1.
    # We start at step total_steps.
    
    # Prices at step total_steps
    # j ranges from 0 to total_steps
    # S_j = S0 * u^j * d^(total_steps - j)
    
    # To avoid computing u**j and d****(total_steps-j) repeatedly which is slow,
    # we can precompute log prices or just use vectorized operations.
    # log_S_j = log(S0) + j*log(u) + (total_steps-j)*log(d)
    # log_u = sigma*sqrt(dt)
    # log_d = -sigma*sqrt(dt)
    
    log_u = np.log(u)
    log_d = np.log(d)
    
    # We will iterate k from total_steps down to 1.
    # At each step k, we compute values for nodes j=0..k.
    
    # We need to store the boundary for exercise dates.
    # Exercise dates are at steps: steps_per_period, 2*steps_per_period, ..., total_steps.
    # Let's map exercise date index i (0 to n_exercise_dates-1) to step k = (i+1)*steps_per_period.
    
    boundaries = np.zeros(n_exercise_dates)
    
    # Initialize for step total_steps
    k = total_steps
    j_range = np.arange(k + 1)
    log_prices = j_range * log_u + (k - j_range) * log_d
    prices = np.exp(log_prices)
    
    # Values at maturity (step total_steps)
    # This is also an exercise date (the last one)
    vals = np.maximum(K - prices, 0.0)
    
    # Determine boundary for the last exercise date (index n_exercise_dates - 1)
    # Exercise if K - S >= 0 => S <= K
    # Find the largest j such that prices[j] <= K
    # Since prices are increasing with j, we can use searchsorted
    # We want the last index where price <= K
    # indices where price <= K are 0 to j_max
    j_max = np.searchsorted(prices, K, side='right') - 1
    if j_max >= 0:
        # Interpolate boundary between j_max and j_max+1 if possible for better accuracy?
        # The true boundary is where Immed Payoff = Continuation.
        # At maturity, Continuation = 0. So Boundary = K.
        # So for the last step, boundary is exactly K.
        boundaries[n_exercise_dates - 1] = K
    else:
        boundaries[n_exercise_dates - 1] = 0.0
        
    # Now iterate backwards from k = total_steps - 1 down to 1
    for k in range(total_steps - 1, 0, -1):
        # Current step is k. Nodes j=0..k.
        # Next step is k+1. Nodes j=0..k+1.
        # We have `vals` for step k+1 (length k+2) and `prices` for step k+1?
        # Wait, I only stored `vals` for step k+1. I need `prices` for step k to calculate payoffs.
        # I can calculate prices for step k from scratch or update them.
        # Let's calculate prices for step k.
        
        j_range_curr = np.arange(k + 1)
        log_prices_curr = j_range_curr * log_u + (k - j_range_curr) * log_d
        prices_curr = np.exp(log_prices_curr)
        
        # Calculate continuation value for each node j in step k
        # From node j at step k, we go to j at step k+1 (down) with prob (1-p) and j+1 at step k+1 (up) with prob p.
        # Wait, standard binomial:
        # S_{k+1, j} = S_{k, j} * d
        # S_{k+1, j+1} = S_{k, j} * u
        # So from node j at step k, the next nodes are j (down) and j+1 (up) in step k+1.
        # The value at step k+1, node j is `vals[j]`.
        # The value at step k+1, node j+1 is `vals[j+1]`.
        
        # `vals` currently holds values for step k+1.
        # Length of `vals` is k+2.
        # `vals[j]` and `vals[j+1]` are valid for j=0..k.
        
        cont_vals = discount * (p * vals[1:] + (1 - p) * vals[:-1])
        
        # Immediate Payoff
        imm_payoff = np.maximum(K - prices_curr, 0.0)
        
        # Optimal Value
        opt_vals = np.maximum(imm_payoff, cont_vals)
        
        # Check if this step k is an exercise date
        # Exercise dates are at k = m * steps_per_period for m = 1..n_exercise_dates
        # Note: The last exercise date was handled in initialization (k=total_steps).
        # So we check for m = 1 to n_exercise_dates - 1.
        if k % steps_per_period == 0:
            # This is an exercise date.
            # Determine the index of the exercise date.
            # k = m * steps_per_period => m = k // steps_per_period
            # Exercise date index i = m - 1 (0-based)
            m = k // steps_per_period
            i = m - 1
            
            # Find the boundary
            # Exercise if imm_payoff >= cont_vals
            exercise_mask = imm_payoff >= cont_vals
            if np.any(exercise_mask):
                # Find the last index where it is True
                # Since prices increase with j, and for a put, we exercise at low S (low j),
                # the exercise region is a prefix of the array.
                # So we find the last True index.
                idxs = np.where(exercise_mask)[0]
                j_star = idxs[-1]
                
                # Interpolate the boundary between j_star and j_star+1
                # At j_star, we exercise. At j_star+1, we continue (usually).
                # The boundary is where Immed Payoff = Continuation.
                # Immed(S) = max(K-S, 0)
                # Cont(S) is linearly interpolated between Cont(j_star) and Cont(j_star+1)
                # We can solve for S where K-S = Cont_interp(S)
                
                S1 = prices_curr[j_star]
                S2 = prices_curr[j_star + 1] if j_star + 1 < len(prices_curr) else S1 * u
                
                C1 = cont_vals[j_star]
                C2 = cont_vals[j_star + 1] if j_star + 1 < len(cont_vals) else C1
                
                # Immed Payoff at S1: K - S1
                # Immed Payoff at S2: K - S2
                # Continuation is linear: C(S) = C1 + (C2 - C1) * (S - S1) / (S2 - S1)
                # We want S such that K - S = C(S)
                # K - S = C1 + (C2 - C1) * (S - S1) / (S2 - S1)
                # Let slope = (C2 - C1) / (S2 - S1)
                # K - S = C1 + slope * (S - S1)
                # K - S = C1 + slope * S - slope * S1
                # K - C1 + slope * S1 = S + slope * S
                # K - C1 + slope * S1 = S * (1 + slope)
                # S = (K - C1 + slope * S1) / (1 + slope)
                
                if S2 > S1:
                    slope = (C2 - C1) / (S2 - S1)
                    if 1 + slope != 0:
                        B_interp = (K - C1 + slope * S1) / (1 + slope)
                        # Clamp B_interp between S1 and S2
                        B_interp = max(S1, min(B_interp, S2))
                    else:
                        B_interp = S1
                else:
                    B_interp = S1
                    
                boundaries[i] = B_interp
            else:
                boundaries[i] = 0.0
                
        # Update for next iteration
        vals = opt_vals
        # We don't need to update prices, we calculate them in the next loop iteration.
        # But we need to make sure `vals` corresponds to the current step `k`.
        # Yes, `opt_vals` is the value at step `k`.
        # So in the next iteration (k-1), `vals` will be the values for step `k` (which is `k_next+1`).
        # This is correct.
        
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, boundaries, n_steps, T, n_exercise_dates):
            self.boundaries = boundaries
            self.n_steps = n_steps
            self.T = T
            self.n_exercise_dates = n_exercise_dates
            self.dt = T / n_exercise_dates
            
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
            # Convert time to exercise date index (0-based)
            if isinstance(time, (int, np.integer)):
                # Assume time is the exercise date index (1-based)
                idx = int(time) - 1
            else:
                # Assume time is the actual time value
                idx = int(round(float(time) / self.dt)) - 1
            
            # Clamp index
            idx = max(0, min(idx, self.n_exercise_dates - 1))
            
            boundary = self.boundaries[idx]
            return np.array(current_state) <= boundary
    
    return ExercisePolicy(boundaries, n_steps, T, n_exercise_dates)
# EVOLVE-BLOCK-END
