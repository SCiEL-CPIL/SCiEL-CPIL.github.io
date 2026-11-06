import numpy as np
import time

class ExercisePolicy:
    def __init__(self, critical_prices):
        self.critical_prices = critical_prices
        self.exercise_times = np.linspace(1/12.0, 1.0, 12)

    def exercise(self, current_state, time):
        current_state = np.asarray(current_state)
        k_float = time * 12.0
        k_int = int(round(k_float))
        
        if k_int < 1 or k_int > 12:
            return np.zeros_like(current_state, dtype=bool)
            
        crit_price = self.critical_prices[k_int - 1]
        return current_state <= crit_price

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    
    n_ex = 12
    
    paths_raw = np.asarray(training_paths)
    
    # The error indicated shape (4096, 13). 
    # The task says 12 exercise dates, no time-0 exercise.
    # This implies the paths likely include S_0 at index 0 and S_1..S_12 at indices 1..12.
    # We need the states at the 12 exercise dates.
    if paths_raw.ndim == 1:
        paths_raw = paths_raw.reshape(1, -1)
        
    if paths_raw.shape[1] == 13:
        # Assume columns 0 is S0, 1-12 are exercise dates
        paths = paths_raw[:, 1:13]
    elif paths_raw.shape[1] == 12:
        # Assume columns 0-11 are exercise dates
        paths = paths_raw[:, 0:12]
    else:
        # Fallback: take last 12 columns
        paths = paths_raw[:, -12:]
        
    M = paths.shape[0]
    
    # Discount factors for the 12 exercise dates
    t_steps = np.arange(1, 13) / 12.0
    discount_factors = np.exp(-r * t_steps)
    
    # Initial guess from a coarse binomial tree
    # Use a relatively coarse tree to keep it fast
    steps_per_month = 20
    N_tree = n_ex * steps_per_month
    dt_tree = T / N_tree
    u_tree = np.exp(sigma * np.sqrt(dt_tree))
    d_tree = 1.0 / u_tree
    p_tree = (np.exp(r * dt_tree) - d_tree) / (u_tree - d_tree)
    disc_tree = np.exp(-r * dt_tree)
    
    S0_ref = 100.0
    i_vals = np.arange(N_tree + 1)
    log_S_N = np.log(S0_ref) + N_tree * np.log(d_tree) + i_vals * (np.log(u_tree) - np.log(d_tree))
    S_N = np.exp(log_S_N)
    V = np.maximum(K - S_N, 0.0)
    
    critical_prices = np.zeros(12)
    S_curr = S_N.copy()
    
    # Backward induction
    for j in range(N_tree, 0, -1):
        V_cont = disc_tree * (p_tree * V[0:j] + (1 - p_tree) * V[1:j+1])
        S_prev = S_curr[0:j] / d_tree
        V_ex = np.maximum(K - S_prev, 0.0)
        V_new = np.maximum(V_cont, V_ex)
        
        step_num = j - 1
        # Check if this step corresponds to an exercise date boundary
        # We are going backwards from N_tree to 1.
        # Exercise dates are at steps: steps_per_month, 2*steps_per_month, ..., 12*steps_per_month
        # In the backward loop, j goes from N_tree down to 1.
        # The state S_prev is at step j-1.
        # We want to record the critical price when the step index (j-1) is a multiple of steps_per_month.
        if step_num % steps_per_month == 0 and step_num > 0:
            k_idx = step_num // steps_per_month # 1 to 12
            exercise_mask = V_ex >= V_cont
            if np.any(exercise_mask):
                # Find the highest price where exercise is optimal
                last_true_idx = np.where(exercise_mask)[0][-1]
                critical_prices[k_idx - 1] = S_prev[last_true_idx]
            else:
                critical_prices[k_idx - 1] = 0.0
        
        V = V_new
        S_curr = S_curr[0:j] / d_tree
        
        if time.time() > deadline - 30.0:
            break
    
    critical_prices = np.clip(critical_prices, 0, K)
    
    # Refine using coordinate descent on the training paths
    def evaluate_policy(c_prices):
        """Evaluate the mean discounted payoff for a given policy."""
        # Vectorized check for exercise
        # W[i, k] is True if path i exercises at step k
        W = paths <= c_prices[np.newaxis, :]
        
        # Find the first exercise step for each path
        # We want the smallest k where W[i, k] is True.
        # If no exercise, payoff is 0.
        
        # Create an array of indices 0..11
        k_indices = np.arange(12)
        
        # Mask out already exercised paths? 
        # Actually, the policy is "exercise at the first opportunity where S <= c".
        # So for path i, exercise step is min{k | S_i[k] <= c_k}.
        
        # We can compute this efficiently:
        # For each path, we need the first True in the row of W.
        # np.argmax returns 0 if no True, so we need to be careful.
        # Let's use a cumulative product or similar trick, or just a loop if M is small? 
        # M=4096, 12 steps. A loop is fine but vectorized is better.
        
        # Trick: 
        # If we flip W, the first True becomes the last True in the flipped array? No.
        # Let's just use a simple loop over the 12 steps to determine the exercise step for all paths simultaneously.
        
        exercised = np.zeros(M, dtype=bool)
        exercise_step = np.full(M, -1, dtype=int)
        
        for k in range(12):
            not_exercised = ~exercised
            if not np.any(not_exercised):
                break
            cond = W[:, k] & not_exercised
            exercised[cond] = True
            exercise_step[cond] = k
        
        payoffs = np.zeros(M)
        if np.any(exercised):
            idx = np.arange(M)
            ex_idx = idx[exercised]
            ex_steps = exercise_step[exercised]
            S_ex = paths[ex_idx, ex_steps]
            disc_ex = discount_factors[ex_steps]
            payoffs[ex_idx] = disc_ex * (K - S_ex)
        
        return np.mean(payoffs)
    
    # Coordinate descent
    best_c = critical_prices.copy()
    best_val = evaluate_policy(best_c)
    
    # For each coordinate, do a line search
    # To save time, we will do fewer iterations if time is running out
    max_iter = 20
    
    for iteration in range(max_iter):
        if time.time() > deadline - 10.0:
            break
        
        improved = False
        for k in range(12):
            # Try to improve coordinate k
            c_old = best_c[k]
            
            # Bisection search for the best c_k
            # The function is likely unimodal or has a flat region.
            # We search in [0, K]
            lo, hi = 0.0, K
            best_c_k = c_old
            best_val_k = best_val
            
            # Perform 8 iterations of bisection
            for _ in range(8):
                if time.time() > deadline - 5.0:
                    break
                mid = (lo + hi) / 2
                test_c = best_c.copy()
                test_c[k] = mid
                val = evaluate_policy(test_c)
                
                if val > best_val_k:
                    best_val_k = val
                    best_c_k = mid
                    lo = mid
                else:
                    hi = mid
            
            if best_val_k > best_val + 1e-10:
                best_c[k] = best_c_k
                best_val = best_val_k
                improved = True
        
        if not improved:
            break
    
    critical_prices = best_c
    critical_prices = np.clip(critical_prices, 0, K)
    
    return ExercisePolicy(critical_prices)
