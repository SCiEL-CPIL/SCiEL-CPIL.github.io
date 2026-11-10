import numpy as np
import time

class ExercisePolicy:
    def __init__(self, boundaries, exercise_times):
        self.boundaries = boundaries
        self.exercise_times = exercise_times
        self.n_ex = len(exercise_times)
        
    def exercise(self, current_state, time):
        # current_state: array of stock prices
        # time: current time float
        # Check if time is an exercise date
        # We assume time is passed as the current simulation time
        # Exercise dates are fixed: 1/12, 2/12, ..., 12/12
        
        # Find if time matches any exercise date
        # Use a tolerance
        tol = 1e-9
        for k in range(self.n_ex):
            if abs(time - self.exercise_times[k]) < tol:
                # Exercise if S <= Boundary
                return current_state <= self.boundaries[k]
        
        # If not at an exercise date, do not exercise
        return np.zeros_like(current_state, dtype=bool)

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = 150.0 - 10.0  # 140s safety margin
    
    # Extract parameters
    # Based on task description: Spot 90/100/110, K 100, r 0.05, sigma 0.2, T 1
    # The contract object might contain these, but they are fixed in the task description.
    # I will try to extract from contract, fallback to defaults.
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_ex = 12
    
    # Try to extract from contract if possible
    if hasattr(contract, 'params'):
        params = contract.params
        if isinstance(params, dict):
            K = params.get('K', K)
            r = params.get('r', r)
            sigma = params.get('sigma', sigma)
            T = params.get('T', T)
            n_ex = params.get('n_ex', n_ex)
    elif isinstance(contract, dict):
        K = contract.get('K', K)
        r = contract.get('r', r)
        sigma = contract.get('sigma', sigma)
        T = contract.get('T', T)
        n_ex = contract.get('n_ex', n_ex)
        
    # Exercise times: 12 monthly dates
    # t_k = k * T / n_ex for k=1..n_ex
    exercise_times = np.linspace(T/n_ex, T, n_ex)
    
    # Grid parameters
    # Use a fine grid to ensure accuracy
    # Steps per month M
    M = 200 
    N = n_ex * M # Total steps = 2400
    
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Initialize value array
    # V[j] represents the value at node j in the current time step
    # At T, S = S0 * u^j * d^(N-j)
    # We need S0? The task says "Spot 90/100/110". 
    # The policy must be valid for any spot? 
    # "Candidate receives current states only during scoring".
    # The training_paths likely start from the specific spots.
    # However, the exercise policy is a function of (S, t).
    # The optimal boundary B_k is independent of S0 if we define it in terms of S.
    # Wait, the binomial tree nodes depend on S0.
    # But the decision rule is S <= B_k.
    # B_k is a specific price level.
    # The tree construction usually assumes S0.
    # If we build the tree with S0=100, the boundaries B_k are absolute prices.
    # This is correct because the optimal exercise boundary for a put is a function of S and t, 
    # and for GBM, it scales. But since K is fixed, the boundary is an absolute value.
    
    S0 = 100.0 # Reference spot for grid construction
    
    # Array of stock prices at each step
    # We only need to store V.
    V = np.zeros(N + 1)
    
    # Initialize at T
    # j = 0..N
    # S_j = S0 * u^j * d^(N-j)
    # Payoff = max(K - S_j, 0)
    
    # Vectorized initialization
    j = np.arange(N + 1)
    S_T = S0 * (u ** j) * (d ** (N - j))
    V[:] = np.maximum(K - S_T, 0.0)
    
    # Store boundaries
    boundaries = np.zeros(n_ex)
    
    # Backward induction
    # Step i goes from N-1 down to 0
    # At step i, we are at time t = (i+1)*dt? No.
    # Let's define step index n from 0 to N.
    # n=0 is t=0. n=N is t=T.
    # We iterate n from N-1 down to 0.
    # The value at node j at time n is V[j].
    # The children are at n+1.
    # V_up = V[j+1], V_down = V[j] (if V is shifted)
    
    # Let's use a 2D array or careful shifting.
    # V_next[j] = value at n+1, node j
    # V_curr[j] = value at n, node j
    
    # Initialize V_next with values at T
    V_next = np.zeros(N + 1)
    V_next[:] = np.maximum(K - S_T, 0.0)
    
    for n in range(N - 1, -1, -1):
        # Calculate continuation value for nodes 0..n
        # V_curr[j] = exp(-r dt) * (p * V_next[j+1] + (1-p) * V_next[j])
        # j goes from 0 to n
        V_curr = np.zeros(n + 1)
        
        # Vectorized calculation
        # V_next has indices 0..N
        # For node j at time n, children are j and j+1 at time n+1
        V_curr[:] = np.exp(-r * dt) * (p * V_next[1:n+2] + (1-p) * V_next[0:n+1])
        
        # Check if this step corresponds to an exercise date
        # Exercise dates are at n = k * M for k=1..n_ex
        # Note: n=0 is t=0 (no exercise). n=M is t=T/n_ex (first exercise).
        # So we check if n % M == 0 and n > 0.
        if n % M == 0 and n > 0:
            # Determine which exercise date index k
            k = n // M - 1 # 0-indexed for boundaries array
            
            # Calculate stock prices at this step
            j_idx = np.arange(n + 1)
            S_curr = S0 * (u ** j_idx) * (d ** (n - j_idx))
            Payoff = np.maximum(K - S_curr, 0.0)
            
            # Exercise if Payoff >= V_curr
            exercise_mask = Payoff >= V_curr
            V_curr[exercise_mask] = Payoff[exercise_mask]
            
            # Find the highest stock price where exercise is optimal
            # Since Put payoff decreases with S, and V_curr increases with S,
            # the exercise region is S <= B.
            # Find the largest j where exercise_mask is True.
            # If all are True, B is the highest S.
            # If none are True, B is 0 (or very low).
            
            if np.any(exercise_mask):
                # Find the last index where exercise is True
                # Note: exercise_mask is True for low S (low j) and False for high S (high j)
                # So we want the largest j such that exercise_mask[j] is True.
                # Since it's a contiguous block from 0 to j_max, we can find the first False and subtract 1.
                # Or simply find the max index.
                # Using argmax on the reversed mask?
                # Let's just find the boundary index.
                # The mask is True, True, ..., True, False, False...
                # Find the first False.
                false_indices = np.where(~exercise_mask)[0]
                if len(false_indices) > 0:
                    j_max = false_indices[0] - 1
                else:
                    j_max = n
                
                if j_max >= 0:
                    B_k = S0 * (u ** j_max) * (d ** (n - j_max))
                    boundaries[k] = B_k
                else:
                    boundaries[k] = 0.0
            else:
                boundaries[k] = 0.0
        
        # Update V_next for next iteration (moving back in time)
        # The length of V_curr is n+1.
        # We need to align it for the next step (n-1).
        # In the next step, we will access V_next[0:n+1] and V_next[1:n+2].
        # So we just assign V_curr to V_next, but we need to handle the array size.
        # It's easier to keep V_next as a fixed size N+1 and only use the first n+1 elements.
        V_next[:n+1] = V_curr
        # The rest of V_next is ignored in the next iteration's slice.
        
        # Time check
        if time.time() > deadline:
            # Fallback: use current boundaries, fill rest with 0 or heuristic
            break
            
    # If we broke early, we might have incomplete boundaries.
    # But N=2400 is fast.
    
    return ExercisePolicy(boundaries, exercise_times)
