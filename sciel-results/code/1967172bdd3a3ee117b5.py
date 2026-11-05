"""Bermudan put exercise-policy research with binned backward induction."""
# EVOLVE-BLOCK-START
import numpy as np
import time as _time


def fit_exercise_policy(training_paths, contract):
    """
    Fit an exercise policy for a Bermudan put option.
    
    Parameters:
        training_paths: array of shape (n_dates, n_paths) with risk-neutral GBM paths
        contract: dict with keys S0, K, r, sigma, T, n_dates, n_paths, n_eval_paths
    
    Returns:
        An object with method exercise(current_state, time) -> boolean vector
    """
    start_time = _time.time()
    deadline = 110.0  # safety margin under 120s limit

    # Extract contract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    # training_paths shape: (n_dates, n_paths)
    # Exercise dates are at t = T/n_dates, 2*T/n_dates, ..., T
    # No exercise at time 0
    n_paths = training_paths.shape[1]
    dt = T / n_dates
    
    # We have n_dates exercise opportunities (at dates 1..n_dates in the path)
    # training_paths[i] gives prices at date i+1 (i=0..n_dates-1)
    
    # Binned backward induction:
    # 1. At each date, bin the state space into intervals
    # 2. For each bin, compute the continuation value (discounted expected value from next date)
    # 3. Exercise if immediate payoff > continuation value
    
    # Define bins: use quantile-based or fixed bins
    # We'll use a grid of price levels for binning
    n_bins = 200
    
    # Determine price range from training data
    all_prices = training_paths.flatten()
    p_min = np.percentile(all_prices, 0.5)
    p_max = np.percentile(all_prices, 99.5)
    
    # Ensure reasonable bounds
    p_min = max(p_min, 0.1)
    p_max = min(p_max, 3.0 * S0)
    
    # Create bin edges
    bin_edges = np.linspace(p_min, p_max, n_bins + 1)
    
    # For each date from last to first, compute optimal exercise decision per bin
    # continuation_value[date, bin] = expected discounted continuation value
    # We work backwards from the last exercise date
    
    # At the last exercise date (date index n_dates-1), continuation = 0 (option expires)
    # So we always exercise if in the money
    
    # Store decision boundaries: for each date, a threshold per bin
    # Actually, we'll store for each date and each bin: should we exercise?
    # But for the exercise() method, we need a function of (state, time)
    
    # Let's store: for each date index d (0..n_dates-1), a set of thresholds
    # where for each bin, we know if exercise is optimal
    
    # Better approach: store continuation values per bin, and at exercise time,
    # look up the bin for the current state and compare payoff vs continuation
    
    # continuation[d, b] = continuation value at date d for state in bin b
    # We compute backwards:
    #   d = n_dates-1: continuation = 0
    #   d = n_dates-2 down to 0: 
    #     For each bin b at date d:
    #       Find which bins at date d+1 the paths in bin b transition to
    #       continuation[d, b] = exp(-r*dt) * mean(continuation[d+1, b'] for paths in bin b)
    #       where b' is the bin at date d+1 for those paths
    #     Exercise if (K - S) > continuation[d, b]
    
    # Build bin indices for each date
    # bin_idx[d, p] = bin index for path p at date d
    bin_indices = np.zeros((n_dates, n_paths), dtype=np.int32)
    for d in range(n_dates):
        prices_d = training_paths[d]
        bin_indices[d] = np.searchsorted(bin_edges, prices_d, side='right') - 1
        bin_indices[d] = np.clip(bin_indices[d], 0, n_bins - 1)
    
    # Continuation values: shape (n_dates, n_bins)
    # At the last date, continuation = 0
    continuation = np.zeros((n_dates, n_bins))
    
    # For the last exercise date (index n_dates-1), continuation is 0
    # For earlier dates, compute backwards
    
    # We also need to track, for each date d and bin b, the average next-state bin
    # to compute continuation values
    
    # Work backwards from date n_dates-2 down to 0
    for d in range(n_dates - 2, -1, -1):
        if _time.time() - start_time > deadline - 5:
            break  # Safety margin
        
        # For each bin at date d, find the paths in that bin
        # and compute the average continuation value from date d+1
        for b in range(n_bins):
            # Find paths in bin b at date d
            mask = (bin_indices[d] == b)
            if not np.any(mask):
                continuation[d, b] = 0.0
                continue
            
            # Get the bin indices at date d+1 for these paths
            next_bins = bin_indices[d + 1, mask]
            
            # Average continuation value from next date
            avg_cont = np.mean(continuation[d + 1, next_bins])
            
            # Discount
            continuation[d, b] = np.exp(-r * dt) * avg_cont
    
    # Now determine exercise decisions per date and bin
    # exercise_decision[d, b] = True if we should exercise at date d for state in bin b
    exercise_decision = np.zeros((n_dates, n_bins), dtype=bool)
    for d in range(n_dates):
        # Immediate payoff for a state in bin b: (K - S) where S is representative of bin b
        # Use the midpoint of the bin as representative price
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        payoffs = np.maximum(K - bin_midpoints, 0.0)
        
        # Exercise if payoff > continuation value
        exercise_decision[d] = payoffs > continuation[d]
    
    # Create the exercise policy object
    class ExercisePolicy:
        def __init__(self, exercise_decision, bin_edges, n_dates, dt, T):
            self.exercise_decision = exercise_decision
            self.bin_edges = bin_edges
            self.n_dates = n_dates
            self.dt = dt
            self.T = T
        
        def exercise(self, current_state, time):
            """
            Determine exercise decision for given states and time.
            
            Parameters:
                current_state: array of shape (n_states,) with current prices
                time: current time (scalar)
            
            Returns:
                Boolean array of shape (n_states,) indicating exercise decisions
            """
            # Determine which date index we're at
            # Exercise dates are at t = dt, 2*dt, ..., n_dates*dt = T
            # time is the current time
            
            # Find the date index: date d corresponds to time (d+1)*dt
            # If time is close to (d+1)*dt, we're at exercise date d
            date_idx = int(round(time / self.dt)) - 1
            
            # Clamp to valid range
            date_idx = max(0, min(date_idx, self.n_dates - 1))
            
            # Find bin for each state
            bins = np.searchsorted(self.bin_edges, current_state, side='right') - 1
            bins = np.clip(bins, 0, len(self.bin_edges) - 2)
            
            # Get exercise decisions
            return self.exercise_decision[date_idx, bins]
    
    return ExercisePolicy(exercise_decision, bin_edges, n_dates, dt, T)
# EVOLVE-BLOCK-END
