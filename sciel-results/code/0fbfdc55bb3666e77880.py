# EVOLVE-BLOCK-START
import time
import numpy as np
import math

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction.
    
    Args:
        training_paths: array of shape (n_paths, n_dates) with simulated spot prices
        contract: dict with contract parameters
    
    Returns:
        An object with exercise(current_state, time) method returning boolean vector
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    # Extract parameters from contract
    spot0 = contract.get('spot', 100.0)
    K = contract.get('strike', contract.get('K', 100.0))
    r = contract.get('rate', contract.get('r', 0.05))
    sigma = contract.get('volatility', contract.get('sigma', 0.2))
    T = contract.get('maturity', contract.get('T', 1.0))
    n_exercise = contract.get('n_exercise', contract.get('n_dates', 12))
    
    # Determine number of exercise dates from training paths
    if training_paths is not None and len(training_paths.shape) >= 2:
        n_dates = training_paths.shape[1]
    else:
        n_dates = n_exercise
    
    # Time step
    dt = T / n_dates
    
    # Discount factor per period
    disc = math.exp(-r * dt)
    
    # Generate exercise times (no time-0 exercise)
    # Exercise dates are at dt, 2*dt, ..., n_dates*dt
    # We'll work with indices 0 to n_dates-1 representing exercise at time (i+1)*dt
    
    # Backward induction using training paths
    # V[i] = optimal continuation value at exercise date i (0-indexed, exercise at time (i+1)*dt)
    
    n_paths = training_paths.shape[0] if training_paths is not None else 4096
    
    # Initialize: at the last exercise date, exercise if in-the-money
    # Payoff at last date: max(K - S, 0)
    
    # We'll estimate continuation values using regression on spot prices
    # For each exercise date i, we need E[disc * max(V[i+1], K-S) | S_i]
    
    # Use piecewise linear regression with bins for speed
    n_bins = 50
    
    # Determine price range for binning
    all_prices = training_paths.flatten()
    price_min = max(0.01, all_prices.min() * 0.9)
    price_max = all_prices.max() * 1.1
    
    # Create bin edges
    bin_edges = np.linspace(price_min, price_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    # Backward induction
    # cont_value[i] = estimated continuation value at date i given spot S
    # We store this as a function approximated by bin averages
    
    # Start from the last exercise date
    # At last date (index n_dates-1), the value is max(K - S, 0)
    # For earlier dates, continuation = disc * E[max(cont_value[i+1], K-S_{i+1}) | S_i]
    
    # Store continuation value estimates as arrays of bin averages
    # cont_bins[i] = array of shape (n_bins,) with estimated continuation value in each bin
    
    # Initialize from the last date
    # At the last exercise date, the holder will exercise if K > S
    # So the value at the last date is max(K - S, 0)
    
    # For backward induction, we need to estimate E[payoff_at_next | S_current]
    # where payoff_at_next = max(continuation_value_next, K - S_next)
    
    # Let's do this step by step
    
    # cont_value_at_date[i] as a function of S_i
    # Represent as piecewise constant with n_bins
    
    # Last date: value = max(K - S, 0)
    last_value = np.maximum(K - bin_centers, 0.0)
    
    # We'll store continuation values for each date
    # cont_values[i] = array of shape (n_bins,) representing E[future payoff | S in bin i]
    # Actually, let's think of it differently:
    # At date i, given S_i, the holder chooses to exercise (get K-S_i) or continue (get disc * E[V_{i+1} | S_i])
    # The optimal value V_i(S_i) = max(K - S_i, disc * E[V_{i+1}(S_{i+1}) | S_i])
    
    # We need to estimate E[V_{i+1}(S_{i+1}) | S_i] from the training paths
    
    # For each bin of S_i, compute the average of V_{i+1}(S_{i+1}) for paths where S_i is in that bin
    
    # Start: V_{n_dates-1}(S) = max(K - S, 0)
    # For i from n_dates-2 down to 0:
    #   E[V_{i+1}(S_{i+1}) | S_i in bin] = mean of V_{i+1}(S_{i+1}) for paths with S_i in bin
    #   V_i(S_i) = max(K - S_i, disc * E[V_{i+1} | S_i])
    
    # Let's compute V_{i+1}(S_{i+1}) for each path first
    
    # V_last(S) = max(K - S, 0)
    V_next = np.maximum(K - training_paths[:, -1], 0.0)  # shape (n_paths,)
    
    # Store V_i for each date
    # V_values[i] = array of shape (n_paths,) with V_i(S_i) for each path
    V_values = [None] * n_dates
    V_values[-1] = V_next
    
    # Also store the exercise decision for each path at each date
    exercise_decisions = [None] * n_dates
    exercise_decisions[-1] = V_next > 0  # Exercise if in-the-money at last date
    
    # Backward induction
    for i in range(n_dates - 2, -1, -1):
        if time.time() > deadline - 10:
            break
        
        S_i = training_paths[:, i]
        S_next = training_paths[:, i + 1]
        
        # For each bin of S_i, compute mean of V_next (which is V_{i+1}(S_{i+1}))
        # Bin the S_i values
        bin_indices = np.digitize(S_i, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Compute mean V_next for each bin
        bin_sums = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        np.add.at(bin_sums, bin_indices, V_next)
        np.add.at(bin_counts, bin_indices, 1)
        
        # Avoid division by zero
        bin_counts_safe = np.where(bin_counts > 0, bin_counts, 1)
        bin_means = bin_sums / bin_counts_safe
        # For bins with no data, use the overall mean or interpolate
        overall_mean = np.mean(V_next)
        bin_means = np.where(bin_counts > 0, bin_means, overall_mean)
        
        # Get the continuation value for each path based on its S_i bin
        cont_value = bin_means[bin_indices]
        
        # Optimal value: max(K - S_i, disc * cont_value)
        exercise_payoff = np.maximum(K - S_i, 0.0)
        continue_value = disc * cont_value
        
        V_i = np.maximum(exercise_payoff, continue_value)
        
        # Exercise decision: exercise if exercise_payoff >= continue_value
        exercise_decisions[i] = exercise_payoff >= continue_value
        
        V_values[i] = V_i
        V_next = V_i
    
    # Now we have the exercise decisions for each path at each date
    # We need to create a policy function that takes current_state (array of spots) and time (date index)
    # and returns a boolean vector of whether to exercise
    
    # Build a lookup table or use the binning approach
    # For each date i and each bin, store whether we exercise
    
    exercise_lookup = [None] * n_dates
    for i in range(n_dates):
        if exercise_decisions[i] is None:
            # Default: exercise if in-the-money
            exercise_lookup[i] = np.ones(n_bins, dtype=bool)
        else:
            S_i = training_paths[:, i]
            bin_indices = np.digitize(S_i, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            
            # For each bin, determine if we exercise (use majority vote)
            bin_exercise = np.zeros(n_bins, dtype=bool)
            bin_exercise_counts = np.zeros(n_bins)
            bin_total_counts = np.zeros(n_bins)
            np.add.at(bin_total_counts, bin_indices, 1)
            # Count exercises in each bin
            exercise_flat = exercise_decisions[i].astype(float)
            np.add.at(bin_exercise_counts, bin_indices, exercise_flat)
            
            # Majority vote
            bin_exercise = bin_exercise_counts > 0.5 * bin_total_counts
            # For bins with no data, default to exercise if in-the-money
            no_data = bin_total_counts == 0
            if np.any(no_data):
                # Use threshold: exercise if K - bin_center > 0
                bin_exercise[no_data] = (K - bin_centers[no_data]) > 0
            
            exercise_lookup[i] = bin_exercise
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, exercise_lookup, bin_edges, bin_centers, K, n_dates):
            self.exercise_lookup = exercise_lookup
            self.bin_edges = bin_edges
            self.bin_centers = bin_centers
            self.K = K
            self.n_dates = n_dates
        
        def exercise(self, current_state, time):
            """
            Determine whether to exercise for each path.
            
            Args:
                current_state: array of spot prices, shape (n_eval_paths,)
                time: exercise date index (0 to n_dates-1)
            
            Returns:
                boolean array of shape (n_eval_paths,)
            """
            if time >= self.n_dates:
                return np.zeros(len(current_state), dtype=bool)
            
            if self.exercise_lookup[time] is None:
                # Default: exercise if in-the-money
                return current_state < self.K
            
            # Bin the current states
            bin_indices = np.digitize(current_state, self.bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, len(self.bin_edges) - 2)
            
            return self.exercise_lookup[time][bin_indices]
    
    policy = ExercisePolicy(exercise_lookup, bin_edges, bin_centers, K, n_dates)
    return policy
# EVOLVE-BLOCK-END
