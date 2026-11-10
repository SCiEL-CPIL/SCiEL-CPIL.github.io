"""Rewritten program: Bermudan put exercise policy via high-resolution binomial tree backward induction."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using high-resolution binomial tree
    backward induction to compute optimal exercise thresholds at each of the
    12 monthly exercise dates.
    
    The policy: exercise at date i if current spot < threshold_i.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Parse contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    # Exercise dates: 12 monthly dates over T=1 year (no time-0 exercise)
    # Exercise at t = 1/12, 2/12, ..., 12/12
    exercise_times = np.linspace(T / n_exercise, T, n_exercise)
    
    # Use a high-resolution binomial tree for accurate thresholds
    # We need the tree to have nodes at each exercise date
    # Use a fine grid: let's use n_steps total such that each month has many sub-steps
    # For accuracy, use 1200 steps total (100 per month) as referenced in the task
    n_steps = 1200
    dt = T / n_steps
    
    # Binomial tree parameters
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)
    
    # Map exercise dates to tree step indices
    # Exercise at time t_i corresponds to step i * n_steps / n_exercise
    steps_per_exercise = n_steps // n_exercise
    exercise_steps = np.arange(1, n_exercise + 1) * steps_per_exercise
    
    # Backward induction: compute optimal value at each node
    # V[j] at step n is max(K - S_j, 0) for a put
    # We work backward from maturity
    
    # Final step: payoff
    # Number of nodes at step n is n+1
    # S at step n, node j: S0 * u^j * d^(n-j)
    # For efficiency, work with log-space or just track the boundary
    
    # Actually, for a put, the optimal exercise boundary at each step is a single
    # threshold: exercise if S < B. We can find B by backward induction.
    
    # Let's do full backward induction on the tree, storing the value at each node.
    # But with 1200 steps, that's 1201 nodes at the final step, manageable.
    
    # Value at maturity (step n_steps): max(K - S, 0)
    # S at step n, node j: S0 * u^j * d^(n-j)
    
    S0 = 100.0  # Reference spot for tree construction
    
    # Build the tree in log-space for numerical stability
    # log(S) at step n, node j = log(S0) + j*log(u) + (n-j)*log(d)
    log_u = np.log(u)
    log_d = np.log(d)
    log_S0 = np.log(S0)
    
    # Work backward from step n_steps to step 0
    # At each step, value[j] = max(payoff, discount * (p*V_next[j+1] + (1-p)*V_next[j]))
    # For a put: payoff = max(K - S, 0)
    
    # Start at maturity
    n = n_steps
    j = np.arange(n + 1)
    log_S = log_S0 + j * log_u + (n - j) * log_d
    S = np.exp(log_S)
    V = np.maximum(K - S, 0.0)
    
    # Store thresholds at exercise dates
    # Threshold at a given step: the maximum S at which it's optimal to exercise
    # i.e., the largest S where payoff >= continuation value
    
    thresholds = {}
    
    # Work backward
    for step in range(n_steps - 1, -1, -1):
        # Check if this is an exercise date
        for ei, estep in enumerate(exercise_steps):
            if step == estep - 1:  # We're about to transition to step estep
                # Actually, let me restructure: at each step, after computing V,
                # check if it's an exercise step
                pass
        
        # Compute values at this step
        # V at step, node j: max(payoff, discount * (p * V_next[j+1] + (1-p) * V_next[j]))
        # V_next is the array from step+1, which has step+2 elements
        # V at step has step+1 elements
        
        j_curr = np.arange(step + 1)
        log_S_curr = log_S0 + j_curr * log_u + (step - j_curr) * log_d
        S_curr = np.exp(log_S_curr)
        
        # Continuation value
        # V_next has step+2 elements (from step+1)
        # For node j at step: continuation = discount * (p * V_next[j+1] + (1-p) * V_next[j])
        cont = discount * (p * V[1:] + (1 - p) * V[:-1])
        
        # Payoff
        payoff = np.maximum(K - S_curr, 0.0)
        
        # Optimal value
        V_curr = np.maximum(payoff, cont)
        
        # If this step corresponds to an exercise date, record the threshold
        for ei, estep in enumerate(exercise_steps):
            if step == estep:
                # Find the threshold: largest S where payoff >= cont
                # i.e., K - S >= cont  =>  S <= K - cont
                # But we need the boundary in terms of the node index
                # The threshold is the S value at the boundary
                # Exercise if payoff > cont, i.e., K - S > cont, i.e., S < K - cont
                # Find the crossover point
                diff = payoff - cont
                # Find where diff changes from positive to negative
                # For a put, diff is positive for low S, negative for high S
                # Find the last index where diff >= 0
                positive = diff >= 0
                if np.any(positive):
                    last_pos = np.where(positive)[0][-1]
                    # Threshold is between S_curr[last_pos] and S_curr[last_pos+1] if exists
                    if last_pos < len(S_curr) - 1:
                        # Interpolate
                        # diff[last_pos] >= 0, diff[last_pos+1] < 0
                        # Linear interpolation in log-S space
                        if diff[last_pos] != diff[last_pos + 1]:
                            frac = diff[last_pos] / (diff[last_pos] - diff[last_pos + 1])
                            log_S_thresh = log_S_curr[last_pos] + frac * (log_S_curr[last_pos + 1] - log_S_curr[last_pos])
                        else:
                            log_S_thresh = log_S_curr[last_pos]
                        thresholds[ei] = np.exp(log_S_thresh)
                    else:
                        thresholds[ei] = S_curr[last_pos]
                else:
                    # Never exercise at this date (unlikely for a put)
                    thresholds[ei] = 0.0
        
        V = V_curr
        
        # Time check
        if time.time() > deadline - 5.0:
            break
    
    # Ensure all thresholds are set
    for ei in range(n_exercise):
        if ei not in thresholds:
            thresholds[ei] = K * 0.9  # Fallback
    
    # Convert to list in order
    threshold_list = [thresholds[ei] for ei in range(n_exercise)]
    
    # Exercise dates for reference
    ex_dates = exercise_times
    
    class ExercisePolicy:
        def __init__(self, thresholds, exercise_times, K, r, T, n_exercise):
            self.thresholds = np.array(thresholds)
            self.exercise_times = np.array(exercise_times)
            self.K = K
            self.r = r
            self.T = T
            self.n_exercise = n_exercise
        
        def exercise(self, current_state, time):
            """
            Determine whether to exercise at each path given current states and time.
            
            current_state: array of shape (n_paths,) - current spot prices
            time: scalar or array - current time(s)
            
            Returns: boolean array of shape (n_paths,) - True if exercise
            """
            if isinstance(time, (int, float, np.integer, np.floating)):
                t = float(time)
            else:
                t = float(np.asarray(time).flatten()[0])
            
            # Find which exercise date this corresponds to
            # Exercise dates are at 1/12, 2/12, ..., 12/12
            # Find the closest exercise date
            idx = np.argmin(np.abs(self.exercise_times - t))
            
            # Check if we're actually at an exercise date (within tolerance)
            if abs(self.exercise_times[idx] - t) > 1e-6:
                # Not at an exercise date, don't exercise
                return np.zeros(len(current_state), dtype=bool)
            
            # Exercise if current_state < threshold
            return current_state < self.thresholds[idx]
    
    return ExercisePolicy(threshold_list, ex_dates, K, r, T, n_exercise)
# EVOLVE-BLOCK-END
