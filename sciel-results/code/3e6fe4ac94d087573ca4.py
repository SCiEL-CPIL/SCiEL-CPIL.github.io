"""Bermudan put exercise policy via dense binomial tree backward induction."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using dense binomial tree backward induction.
    
    Returns an object with .exercise(current_state, time) method that
    returns a boolean vector indicating whether to exercise at each path.
    """
    t_start = time.time()
    deadline = t_start + 108.0  # Safety margin before 120s kill

    # Extract parameters from contract
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    # Build a dense CRR binomial tree for accurate backward induction
    n_tree_steps = 6000
    dt_tree = T / n_tree_steps
    
    # CRR parameters
    u = np.exp(sigma * np.sqrt(dt_tree))
    d = np.exp(-sigma * np.sqrt(dt_tree))
    p = (np.exp(r * dt_tree) - d) / (u - d)
    discount_tree = np.exp(-r * dt_tree)
    
    # Determine S0
    S0 = np.median(training_paths[:, 0])
    
    # For efficiency, work with log-price grid
    log_S0 = np.log(S0)
    log_u = np.log(u)
    log_d = np.log(d)
    log_step = log_u - log_d
    
    # Exercise dates in tree steps
    exercise_tree_steps = [int(round(n_tree_steps * i / n_exercise)) for i in range(1, n_exercise + 1)]
    
    # Initialize the value at maturity
    j_max = n_tree_steps
    j_vals = np.arange(j_max + 1)
    log_S_final = log_S0 + n_tree_steps * log_d + j_vals * log_step
    S_final = np.exp(log_S_final)
    
    # Value at maturity: put payoff
    V = np.maximum(K - S_final, 0.0)
    
    # We need to track the exercise boundary at each exercise date
    boundaries = [None] * n_exercise
    
    # Map from tree step to exercise index
    step_to_ex_idx = {}
    for k, ts in enumerate(exercise_tree_steps):
        step_to_ex_idx[ts] = k
    
    # Backward induction
    for i in range(n_tree_steps - 1, -1, -1):
        # Continuation value
        V_cont = discount_tree * (p * V[1:] + (1.0 - p) * V[:-1])
        
        # Stock prices at current step
        j_curr = np.arange(i + 1)
        log_S_curr = log_S0 + i * log_d + j_curr * log_step
        S_curr = np.exp(log_S_curr)
        
        # Payoff at current step
        payoff = np.maximum(K - S_curr, 0.0)
        
        # Exercise if payoff > continuation
        exercise_mask = payoff > V_cont
        
        # Check if this is an exercise date
        if i in step_to_ex_idx:
            ex_idx = step_to_ex_idx[i]
            # Find the crossing point where payoff = V_cont via interpolation
            diff = payoff - V_cont
            sign_change_idx = np.where(np.diff(np.sign(diff)) != 0)[0]
            if len(sign_change_idx) > 0:
                idx = sign_change_idx[0]
                S1, S2 = S_curr[idx], S_curr[idx + 1]
                d1, d2 = diff[idx], diff[idx + 1]
                if abs(d2 - d1) > 1e-15:
                    boundary = S1 + (S2 - S1) * d1 / (d1 - d2)
                else:
                    boundary = S1
            elif np.any(exercise_mask):
                boundary = float(np.max(S_curr[exercise_mask]))
            else:
                boundary = 0.0
            boundaries[ex_idx] = float(boundary)
        
        # Update V
        V = np.maximum(payoff, V_cont)
        
        # Deadline check
        if time.time() > deadline:
            break
    
    # Fill any remaining None boundaries with reasonable defaults
    for k in range(n_exercise):
        if boundaries[k] is None:
            boundaries[k] = K * 0.95 * (1.0 - 0.01 * k)
    
    # Ensure boundaries are monotonically non-decreasing
    for k in range(n_exercise - 1):
        if boundaries[k] > boundaries[k + 1]:
            boundaries[k] = boundaries[k + 1]
    
    # Also ensure no boundary exceeds K (no point exercising OTM)
    for k in range(n_exercise):
        if boundaries[k] > K:
            boundaries[k] = K
    
    # Create the exercise policy
    class ExercisePolicy:
        def __init__(self, K, boundaries, n_exercise, T):
            self.K = K
            self.boundaries = boundaries
            self.n_exercise = n_exercise
            self.T = T
            self.exercise_times = np.array([i / n_exercise for i in range(1, n_exercise + 1)])
        
        def exercise(self, current_state, time):
            """
            current_state: array of S values (shape: (n_paths,))
            time: current time t
            Returns boolean array: True if should exercise at this path
            """
            n_paths = len(current_state)
            # Determine which exercise date we're at based on time
            best_ex_idx = int(np.argmin(np.abs(self.exercise_times - time)))
            threshold = self.boundaries[best_ex_idx]
            result = (current_state <= threshold)
            
            return result
    
    policy = ExercisePolicy(K, boundaries, n_exercise, T)
    return policy

# EVOLVE-BLOCK-END
