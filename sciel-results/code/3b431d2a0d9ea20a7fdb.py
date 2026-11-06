import numpy as np
import time

class ExercisePolicy:
    def __init__(self, critical_prices):
        self.critical_prices = critical_prices
        self.exercise_times = np.linspace(1/12.0, 1.0, 12)

    def exercise(self, current_state, time):
        # current_state: array-like
        # time: float
        # Determine exercise index
        # time is expected to be one of the 12 monthly dates
        k_float = time * 12.0
        k_int = int(round(k_float))
        
        # Validate index
        if k_int < 1 or k_int > 12:
            return np.zeros_like(current_state, dtype=bool)
            
        # Optional: check if time is close to expected date
        # t_expected = k_int / 12.0
        # if abs(time - t_expected) > 1e-4:
        #     return np.zeros_like(current_state, dtype=bool)
        
        crit_price = self.critical_prices[k_int - 1]
        return np.asarray(current_state) <= crit_price

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin

    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    
    # Number of exercise dates
    n_ex = 12
    
    # Tree parameters
    # Use 100 steps per month for high accuracy
    N = n_ex * 100 
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount_factor = np.exp(-r * dt)
    
    # We compute the critical prices using a reference S0
    # The boundary is in absolute price terms, so S0 choice doesn't change the boundary curve
    S0_ref = 100.0 
    
    # Initialize value array at maturity
    # Size N+1
    V = np.zeros(N + 1)
    
    # Maturity payoff: Put
    # S at maturity (step N) for node i is S0_ref * u^i * d^(N-i)
    # Note: i goes from 0 to N.
    # S_i = S0_ref * u^i * d^(N-i)
    i_vals = np.arange(N + 1)
    S_N = S0_ref * (u ** i_vals) * (d ** (N - i_vals))
    V = np.maximum(K - S_N, 0.0)
    
    # Critical prices for each exercise date
    # Exercise dates are at steps: N, N-100, N-200, ..., 100
    # Wait, backward induction goes from N down to 1.
    # Step N is T. Step N-100 is T - 1/12.
    # Exercise dates are at t_k = k/12.
    # t=1.0 is k=12 (step N)
    # t=11/12 is k=11 (step N-100)
    # ...
    # t=1/12 is k=1 (step 100)
    
    critical_prices = np.zeros(12)
    # Map exercise index k (1..12) to step j
    # k=12 -> j=N
    # k=11 -> j=N-100
    # k -> j = N - (12-k)*100 = N - 1200 + 100k = 100k (since N=1200)
    # So exercise date k corresponds to step j=100*k.
    
    # We will store the critical price when we hit the step
    # Note: Step 0 is t=0. We don't exercise at t=0.
    # Steps 1..1200.
    # Exercise at steps 100, 200, ..., 1200.
    
    for j in range(N, 0, -1):
        # Current step j. Previous step j-1.
        # V[j-1 ... j] corresponds to nodes i=0..j-1 at step j-1?
        # No, standard binomial:
        # Step j has nodes i=0..j.
        # Step j-1 has nodes i=0..j-1.
        # V[i] at step j-1 comes from V[i] and V[i+1] at step j.
        
        # Calculate continuation values for step j-1
        # V_cont[i] = discount * (p * V[i] + (1-p) * V[i+1]) for i=0..j-1
        V_cont = discount_factor * (p * V[0:j] + (1 - p) * V[1:j+1])
        
        # Calculate stock prices at step j-1
        # S[i] = S0_ref * u^i * d^(j-1-i)
        i_vals_curr = np.arange(j) # 0 to j-1
        S_curr = S0_ref * (u ** i_vals_curr) * (d ** ((j - 1) - i_vals_curr))
        
        # Exercise value at step j-1
        V_ex = np.maximum(K - S_curr, 0.0)
        
        # Determine optimal value and exercise decision
        V_new = np.maximum(V_cont, V_ex)
        
        # Check if this step is an exercise date
        # Exercise dates are steps 100, 200, ..., 1200.
        # We are currently computing values for step j-1.
        # So if j-1 is a multiple of 100, we are at an exercise date.
        step_num = j - 1
        if step_num % 100 == 0 and step_num > 0:
            # This is an exercise date.
            # Which one?
            # k = step_num / 100
            # k=1 corresponds to t=1/12
            k_idx = step_num // 100
            # k_idx is 1..12
            
            # Find critical price
            # We exercise if V_ex >= V_cont
            # Since S increases with i, and Put payoff decreases with S,
            # we exercise for small S (small i).
            # We want the largest i where exercise is optimal.
            # Or simply, the price at the boundary.
            
            exercise_mask = V_ex >= V_cont
            
            if np.any(exercise_mask):
                # Find the last True index
                # Since S increases with i, the last True is the highest price where we exercise.
                last_true_idx = np.where(exercise_mask)[0][-1]
                
                # Interpolate between last_true_idx and last_true_idx+1 for sub-grid accuracy
                if last_true_idx < j - 1:
                    # We have S_curr[last_true_idx] and S_curr[last_true_idx+1]
                    # V_ex[last_true_idx] >= V_cont[last_true_idx]
                    # V_ex[last_true_idx+1] < V_cont[last_true_idx+1]
                    
                    # Linear interpolation of the boundary
                    # Let x = S_curr[last_true_idx], y = S_curr[last_true_idx+1]
                    # Let a = V_ex[last_true_idx] - V_cont[last_true_idx] >= 0
                    # Let b = V_ex[last_true_idx+1] - V_cont[last_true_idx+1] < 0
                    # We want S such that V_ex(S) - V_cont(S) = 0
                    # Assuming linearity in S between these two points:
                    # diff(S) = a + (b - a) * (S - x) / (y - x)
                    # 0 = a + (b - a) * (S - x) / (y - x)
                    # S - x = -a * (y - x) / (b - a)
                    # S = x - a * (y - x) / (b - a)
                    
                    x = S_curr[last_true_idx]
                    y = S_curr[last_true_idx + 1]
                    a = V_ex[last_true_idx] - V_cont[last_true_idx]
                    b = V_ex[last_true_idx + 1] - V_cont[last_true_idx + 1]
                    
                    if abs(b - a) > 1e-12:
                        S_crit = x - a * (y - x) / (b - a)
                    else:
                        S_crit = x
                else:
                    S_crit = S_curr[last_true_idx]
            else:
                # Never exercise (e.g. deep in the money? No, Put always exercise if S->0)
                # If no exercise, S_crit = 0
                S_crit = 0.0
                
            critical_prices[k_idx - 1] = S_crit
            
        # Update V for next iteration (backward)
        V = V_new
        
        # Time check
        if time.time() > deadline - 5.0:
            # If we are running out of time, we might have incomplete critical prices
            # But N=1200 is very fast. This guard is just in case.
            break
            
    # If loop broke early, some critical prices might be 0 (default).
    # We should handle this, but it's unlikely.
    
    return ExercisePolicy(critical_prices)
