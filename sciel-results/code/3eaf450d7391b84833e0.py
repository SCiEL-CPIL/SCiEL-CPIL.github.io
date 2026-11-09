"""Bermudan exercise-policy research: fit a policy via backward induction on training paths."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill

    # Extract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)

    # training_paths: shape (N, n_total_steps) where n_total_steps = 13 (t=0 to t=1)
    # Exercise dates are at indices 1..12 (monthly, no time-0 exercise)
    N, n_total_steps = training_paths.shape
    n_exercise = 12  # monthly exercise dates

    # Time steps for exercise dates: t_k = k/12 for k=1..12
    dt = T / n_exercise
    discount = np.exp(-r * dt)

    # Payoff at maturity (last step, index 12): max(K - S, 0)
    # We need to compute continuation values backward from t=12 to t=1

    # Get the states at each exercise date
    # training_paths[:, k] gives the state at time step k
    # Exercise dates are at indices 1, 2, ..., 12

    # Start with terminal payoff at t=12 (index 12)
    # The value at t=12 is just the payoff (must exercise or let it expire)
    # For a Bermudan put, at the last exercise date you get max(K-S, 0)
    # Continuation value at t=12 is the payoff itself (no further exercise dates)

    # Actually, let's think carefully:
    # At t=12 (maturity), the put expires with payoff max(K-S, 0)
    # At t=11, you can exercise (get max(K-S,0)) or continue (get discounted expected value from t=12)
    # ...
    # At t=1, you can exercise or continue

    # The training paths give us the realized states. We use them to estimate
    # the conditional expectation E[continuation_value | S_t].

    # Strategy: For each exercise date t_k (k=1..11), we need to estimate
    # E[discounted value at t_{k+1} | S_t = s]
    # Then the optimal policy is: exercise if payoff > continuation value

    # We'll use a simple approach:
    # 1. Compute terminal payoff at t=12
    # 2. For k=11 down to 1:
    #    a. Estimate E[V_{k+1} | S_k] using the training paths
    #    b. Continuation value at t_k = discount * E[V_{k+1} | S_k]
    #    c. V_k = max(payoff_k, continuation_value_k)
    #    d. Policy: exercise if payoff_k >= continuation_value_k

    # To estimate E[V_{k+1} | S_k], we can use a simple regression or
    # nearest-neighbor / binning approach. Given the time constraint, let's use
    # a piecewise-constant or linear regression approach.

    # Better approach: Use the fact that under risk-neutral GBM, the continuation
    # value is a smooth function of S. We can fit a simple model.

    # Let's use a basis function approach with polynomials in log(S/K)
    # Or simpler: use quantile-based binning

    # Actually, the simplest robust approach: 
    # For each path, we know the full trajectory. We can compute the "implied"
    # continuation value by backward induction on each path, then average.
    # But that's just the Monte Carlo estimate of the continuation value.

    # Let's do proper backward induction:
    # V_12(S_12) = max(K - S_12, 0)  [terminal payoff]
    # For k = 11, ..., 1:
    #   V_k(S_k) = max(K - S_k, 0, discount * E[V_{k+1}(S_{k+1}) | S_k])

    # To compute E[V_{k+1}(S_{k+1}) | S_k], we need a model. 
    # Simple approach: assume V_{k+1} is a function of S_{k+1}, and use
    # the training paths to estimate this conditional expectation.

    # Let's use a simple linear regression: V_{k+1} ≈ a + b * S_{k+1} + c * log(S_{k+1})
    # Or even simpler: use the empirical distribution.

    # Given time constraints, let's use a simple approach:
    # For each k, bin the states S_k into quantiles, and for each bin,
    # compute the average of V_{k+1} for paths in that bin.

    # Actually, let's try a cleaner approach using the structure of the problem.

    # The key insight: under risk-neutral GBM, given S_k, S_{k+1} is lognormal.
    # So E[V_{k+1}(S_{k+1}) | S_k] can be estimated by simulating from S_k,
    # but we don't have that luxury. Instead, we use the training paths.

    # Let's use a simple approach: for each exercise date k (1..11), 
    # we'll fit a simple model to predict V_{k+1} from S_k.

    # Step 1: Compute terminal values
    S_terminal = training_paths[:, 12]
    V = np.zeros((N, n_total_steps))
    V[:, 12] = np.maximum(K - S_terminal, 0.0)

    # Step 2: Backward induction
    # For k = 11 down to 1:
    for k in range(11, 0, -1):
        if time.time() > deadline - 10.0:
            break

        S_k = training_paths[:, k]
        S_next = training_paths[:, k+1]
        V_next = V[:, k+1]

        # Estimate E[discount * V_next | S_k]
        # Use a simple approach: sort by S_k and use a moving average
        # Or fit a simple model

        # Let's use a simple quantile-based approach
        # Divide S_k into bins, compute mean of V_next in each bin
        
        n_bins = min(200, N // 10)  # At least 10 samples per bin
        if n_bins < 2:
            n_bins = 2

        # Create bins based on quantiles of S_k
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(S_k, quantiles)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        # Assign each path to a bin
        bin_indices = np.searchsorted(bin_edges[1:-1], S_k, side='right')

        # Compute mean V_next per bin
        bin_sums = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        for b in range(n_bins):
            mask = bin_indices == b
            bin_sums[b] = np.sum(V_next[mask])
            bin_counts[b] = np.sum(mask)

        # Handle empty bins
        for b in range(n_bins):
            if bin_counts[b] == 0:
                bin_counts[b] = 1
                bin_sums[b] = 0

        bin_means = bin_sums / bin_counts

        # For each path, get the continuation value estimate
        cont_values = bin_means[bin_indices] * discount

        # Optimal value: max(payoff, continuation)
        payoff_k = np.maximum(K - S_k, 0.0)
        V[:, k] = np.maximum(payoff_k, cont_values)

        # Store the policy: exercise if payoff >= continuation
        # We'll store this in a separate structure

    # Now we need to create a policy object
    # The policy needs to take current_state (a vector of states) and time,
    # and return a boolean vector indicating which paths to exercise.

    # We need to store, for each exercise date k (1..12), a function that
    # given S_k, returns whether to exercise.

    # Let's store the bin structure for each exercise date
    policy_data = {}
    
    for k in range(1, 12):
        if time.time() > deadline - 5.0:
            # Fallback: use simple rule
            break
        S_k = training_paths[:, k]
        S_next = training_paths[:, k+1]
        V_next = V[:, k+1]

        n_bins = min(200, N // 10)
        if n_bins < 2:
            n_bins = 2

        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(S_k, quantiles)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        bin_indices = np.searchsorted(bin_edges[1:-1], S_k, side='right')

        bin_sums = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        for b in range(n_bins):
            mask = bin_indices == b
            bin_sums[b] = np.sum(V_next[mask])
            bin_counts[b] = np.sum(mask)

        for b in range(n_bins):
            if bin_counts[b] == 0:
                bin_counts[b] = 1
                bin_sums[b] = 0

        bin_means = bin_sums / bin_counts
        cont_values = bin_means[bin_indices] * discount
        payoff_k = np.maximum(K - S_k, 0.0)
        
        # Exercise if payoff >= continuation value
        exercise_mask = payoff_k >= cont_values

        # For the policy, we need to generalize to new states.
        # Store bin edges and for each bin, whether to exercise (majority vote)
        bin_exercise = np.zeros(n_bins, dtype=bool)
        bin_ex_counts = np.zeros(n_bins)
        for b in range(n_bins):
            mask = bin_indices == b
            if np.sum(mask) > 0:
                bin_exercise[b] = np.sum(exercise_mask[mask]) > np.sum(mask) / 2
                bin_ex_counts[b] = np.sum(mask)

        policy_data[k] = {
            'bin_edges': bin_edges[1:-1],  # Internal edges
            'bin_exercise': bin_exercise
        }

    # For k=12 (last exercise date), always exercise (it's maturity)
    # Actually at maturity you get the payoff, so "exercising" means taking the payoff
    # The policy at k=12 should always return True (take the payoff)

    class Policy:
        def __init__(self, data, K_val):
            self.data = data
            self.K = K_val

        def exercise(self, current_state, time):
            # time is the exercise date index (1-12)
            # current_state is a vector of states
            # Return boolean vector
            if time >= 12:
                # At maturity, always "exercise" (take payoff)
                return np.ones(len(current_state), dtype=bool)
            
            if time not in self.data:
                # Fallback: simple rule
                return np.maximum(self.K - current_state, 0.0) > 0

            bin_edges = self.data[time]['bin_edges']
            bin_exercise = self.data[time]['bin_exercise']
            
            # Assign each state to a bin
            bin_indices = np.searchsorted(bin_edges, current_state, side='right')
            # Clip to valid range
            bin_indices = np.clip(bin_indices, 0, len(bin_exercise) - 1)
            
            return bin_exercise[bin_indices]

    return Policy(policy_data, K)

# EVOLVE-BLOCK-END
