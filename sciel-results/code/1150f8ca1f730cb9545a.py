"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_steps = contract.get('n_steps', 12)
    
    # Time grid: 12 monthly exercise dates (no time-0 exercise)
    dt = T / n_steps
    # Exercise times are at t_1, t_2, ..., t_12
    # t_i = i * dt for i = 1, ..., 12
    
    # Number of training paths
    n_paths = training_paths.shape[0] if hasattr(training_paths, 'shape') else len(training_paths)
    
    # Convert to numpy array if needed
    training_paths = np.asarray(training_paths, dtype=np.float64)
    
    # training_paths shape: (n_paths, n_time_steps)
    # n_time_steps should be 13 (time 0 through 12)
    # Exercise decisions at indices 1 through 12
    
    # Piecewise-linear value function with knots
    # Use quantiles of the training data to define knots
    n_knots = 50
    
    # Collect all spot prices at exercise times for knot determination
    all_spots = []
    for t_idx in range(1, n_steps + 1):
        spots_t = training_paths[:, t_idx]
        all_spots.append(spots_t)
    all_spots = np.concatenate(all_spots)
    
    # Define knots using quantiles
    knots = np.unique(np.quantile(all_spots, np.linspace(0, 1, n_knots)))
    
    # Ensure knots span the range
    min_spot = all_spots.min()
    max_spot = all_spots.max()
    knots = np.concatenate(([min_spot], knots, [max_spot]))
    knots = np.unique(knots)
    
    # Backward induction
    # V[t](S) = value at time t for spot S
    # At exercise time t_i, V[i](S) = max(K - S, discount * E[V[i+1](S_{i+1}) | S_i = S])
    
    # Initialize terminal value (at t_12)
    # V[12](S) = max(K - S, 0)  # European put payoff
    
    # We'll store value functions as piecewise linear
    # For each time step, store knots and values at knots
    
    # Start from the last exercise date and work backward
    # V_next will hold the value function for the next time step
    
    # At t_12 (last exercise), the continuation value is 0 (no more exercise dates)
    # So V[12](S) = max(K - S, 0)
    
    # We'll represent value functions as arrays of values at the knots
    # For simplicity, use the same knots for all time steps
    
    n_knots_actual = len(knots)
    
    # Initialize: value at last exercise date
    # V[12](S) = max(K - S, 0)
    V_next = np.maximum(K - knots, 0.0)
    
    # Store exercise decisions for each time step
    # exercise_decision[t_idx, knot_idx] = True if exercise at that knot
    exercise_decisions = {}
    
    # Backward induction from t_11 down to t_1
    for t_idx in range(n_steps - 1, 0, -1):
        if time.time() > deadline - 10.0:
            break
        
        # For each knot, compute continuation value
        # Continuation value at knot j = discount * E[V_next(S_{t_idx+1}) | S_{t_idx} = knots[j]]
        
        # Use training paths to estimate the conditional expectation
        # For each path, if S_{t_idx} is close to knots[j], use S_{t_idx+1}
        
        # More efficient: for each path, find which knot it's closest to at time t_idx
        spots_t = training_paths[:, t_idx]
        spots_t_next = training_paths[:, t_idx + 1]
        
        # Find nearest knot index for each path
        # Use searchsorted for efficiency
        knot_indices = np.searchsorted(knots, spots_t)
        knot_indices = np.clip(knot_indices, 0, n_knots_actual - 1)
        
        # For each knot, collect the continuation values from paths near that knot
        # Use a simple binning approach
        continuation_values = np.zeros(n_knots_actual)
        counts = np.zeros(n_knots_actual)
        
        # For each path, evaluate V_next at S_{t_idx+1} and assign to the knot bin
        # First, evaluate V_next at all S_{t_idx+1} values
        # V_next is piecewise linear, so we can interpolate
        
        # Interpolate V_next at spots_t_next
        V_next_at_next = np.interp(spots_t_next, knots, V_next)
        
        # Assign to bins
        np.add.at(continuation_values, knot_indices, V_next_at_next)
        np.add.at(counts, knot_indices, 1)
        
        # Average for each knot
        valid = counts > 0
        continuation_values[valid] /= counts[valid]
        
        # For knots with no data, use interpolation or set to 0
        # Simple approach: linear interpolation of continuation values
        if np.any(~valid):
            valid_indices = np.where(valid)[0]
            invalid_indices = np.where(~valid)[0]
            if len(valid_indices) > 0:
                continuation_values[invalid_indices] = np.interp(
                    knots[invalid_indices], knots[valid_indices], continuation_values[valid_indices]
                )
            else:
                continuation_values[~valid] = 0
        
        # Discount factor
        discount = np.exp(-r * dt)
        cont_val = discount * continuation_values
        
        # Exercise value at each knot
        exercise_val = np.maximum(K - knots, 0.0)
        
        # Optimal decision: exercise if exercise_val >= cont_val
        exercise_at_knot = exercise_val >= cont_val
        
        # Store the decision
        exercise_decisions[t_idx] = exercise_at_knot
        
        # Current value function: max of exercise and continuation
        V_current = np.maximum(exercise_val, cont_val)
        
        # Update for next iteration
        V_next = V_current
    
    # Now we have exercise_decisions for each time step
    # Create the exercise policy object
    
    class ExercisePolicy:
        def __init__(self, knots, exercise_decisions, n_steps, K):
            self.knots = knots
            self.exercise_decisions = exercise_decisions
            self.n_steps = n_steps
            self.K = K
        
        def exercise(self, current_state, time):
            """
            current_state: array of spot prices for each path
            time: integer time index (1-based, corresponding to exercise date)
            Returns: boolean array indicating whether to exercise
            """
            if time < 1 or time > self.n_steps:
                return np.zeros(len(current_state), dtype=bool)
            
            if time not in self.exercise_decisions:
                # If no decision stored, use simple rule: exercise if S < K
                return current_state < self.K
            
            exercise_at_knot = self.exercise_decisions[time]
            
            # Interpolate the exercise decision
            # For each spot, find if we should exercise based on the knot-based decision
            # Simple approach: exercise if S is below the threshold where decision changes
            
            # Find the threshold: the highest knot where we exercise
            # If exercise_at_knot is [True, True, False, False, ...], threshold is between knot[1] and knot[2]
            
            # More robust: for each spot, check if it's in a region where we exercise
            # Use the knot indices
            knot_indices = np.searchsorted(self.knots, current_state)
            knot_indices = np.clip(knot_indices, 0, len(self.knots) - 1)
            
            # Get the decision for each knot
            decisions = exercise_at_knot[knot_indices]
            
            return decisions
    
    policy = ExercisePolicy(knots, exercise_decisions, n_steps, K)
    return policy
# EVOLVE-BLOCK-END
