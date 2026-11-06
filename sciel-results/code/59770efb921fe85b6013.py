import numpy as np

class BermudanPolicy:
    def __init__(self, K, r, sigma, T, n_steps):
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.n_steps = n_steps
        self.dt = T / n_steps
        
        # CRR parameters
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = np.exp(-self.sigma * np.sqrt(self.dt))
        self.p = (np.exp(self.r * self.dt) - self.d) / (self.u - self.d)
        self.discount = np.exp(-self.r * self.dt)
        
        self.critical_boundaries = self._compute_boundaries()
        
    def _compute_boundaries(self):
        """
        Compute the critical stock price for each exercise date using backward induction.
        The tree has n_steps+1 time points: 0, 1, ..., n_steps.
        Exercise is allowed at times 1, ..., n_steps.
        """
        n = self.n_steps
        # We only need the boundaries for each time step.
        # At each time step k (1 to n), there are k+1 nodes in a binomial tree if we start from 1 node.
        # However, since we are dealing with a policy function of the state (price), 
        # and the tree is recombining, the value at a node depends only on the price.
        # Actually, for a European-style grid or a recombining tree, the value function V(S, t) 
        # is determined by the price S. 
        # We can compute the value function on a grid of prices.
        
        # Let's define a grid of prices for each time step.
        # To ensure we cover the range of prices seen in training/evaluation, 
        # we can use a wide range. However, for a put with K=100, prices > 200 are never exercised.
        # Prices < 0 are 0.
        
        # Instead of a fixed grid, we can use the tree nodes directly.
        # But the input to 'exercise' is an arbitrary 'current_state'.
        # The value function V(S, k) is piecewise linear or smooth.
        # A robust approach is to compute the value function on a dense grid of prices 
        # for each time step k, then interpolate or just use the grid if we assume 
        # the input states are close to tree nodes? No, states are arbitrary.
        
        # Better approach: 
        # The optimal stopping boundary S_crit(k) is the unique price where 
        # V_ex(S) = V_cont(S).
        # We can find this boundary for each k by computing the value function 
        # on a fine grid of prices.
        
        # Let's create a grid of prices.
        # Max price relevant: S0 * u^n. Min price: 0.
        # Let's use a log-spaced or linear grid.
        # For a put, we care about S in [0, K * some_factor].
        # Let's use a grid from 0 to max_price.
        
        max_price = 150.0 # Sufficiently high for K=100, boundary is < K
        min_price = 0.0
        grid_size = 5000 # Higher resolution for precise boundaries
        
        S_grid = np.linspace(min_price, max_price, grid_size)
        
        # Value function at maturity (step n)
        # V_n(S) = (K - S)^+
        V_next = np.maximum(self.K - S_grid, 0.0)
        
        boundaries = [None] * (n + 1) # Index 0 to n. We care about 1 to n.
        
        for k in range(n, 0, -1):
            # Compute continuation value at step k-1 given values at step k
            # The price at step k is either S*u or S*d from step k-1.
            # Wait, backward induction on a grid:
            # V_k(S) = max( (K-S)^+, discount * [p * V_{k-1}(S/u) + (1-p) * V_{k-1}(S/d)] )
            # Note: The transition is from S at time k to S*u or S*d at time k+1?
            # Standard CRR: S_{t+1} = S_t * u or S_t * d.
            # So to find V_k(S), we need V_{k+1}(S*u) and V_{k+1}(S*d).
            # We are iterating backwards from n to 1.
            # We have V_next which corresponds to time k (since we start with V_n).
            # Let's rename: V_curr corresponds to time k. V_next corresponds to time k+1.
            # We want to find V_k.
            # V_k(S) = max( Payoff(S), discount * E[V_{k+1}(S') | S] )
            # S' = S*u with prob p, S*d with prob 1-p.
            
            # Interpolate V_next at S*u and S*d
            S_up = S_grid * self.u
            S_down = S_grid * self.d
            
            V_up = np.interp(S_up, S_grid, V_next)
            V_down = np.interp(S_down, S_grid, V_next)
            
            V_cont = self.discount * (self.p * V_up + (1 - self.p) * V_down)
            V_ex = np.maximum(self.K - S_grid, 0.0)
            
            V_curr = np.maximum(V_ex, V_cont)
            
            # Find the boundary where V_ex >= V_cont
            # For a put, V_ex is decreasing, V_cont is typically increasing (or less decreasing).
            # The exercise region is S <= S_crit.
            # We want the largest S such that V_ex(S) >= V_cont(S).
            # diff = V_ex - V_cont
            diff = V_ex - V_cont
            # Find the last index where diff >= 0
            # Since S_grid is increasing, and exercise is optimal for low S,
            # the region S <= S_crit will have diff >= 0.
            # We find the crossover point.
            
            # If all diff >= 0, boundary is max_price (or infinity, but capped)
            # If all diff < 0, boundary is 0 (never exercise)
            
            # Find indices where diff >= 0
            valid_indices = np.where(diff >= 0)[0]
            
            if len(valid_indices) > 0:
                # The largest S in the exercise region
                idx = valid_indices[-1]
                # Interpolate to find the exact S_crit
                # diff[idx] >= 0, diff[idx+1] < 0 (if idx < len-1)
                if idx < len(S_grid) - 1:
                    # Linear interpolation between idx and idx+1
                    s1 = S_grid[idx]
                    s2 = S_grid[idx+1]
                    d1 = diff[idx]
                    d2 = diff[idx+1]
                    # d = d1 + (d2-d1)*(s-s1)/(s2-s1) = 0
                    # s = s1 - d1*(s2-s1)/(d2-d1)
                    denom = d2 - d1
                    if denom != 0:
                        s_crit = s1 - d1 * (s2 - s1) / denom
                    else:
                        s_crit = s1
                else:
                    s_crit = S_grid[idx]
                boundaries[k] = s_crit
            else:
                boundaries[k] = 0.0
                
            V_next = V_curr
            
        return boundaries

    def exercise(self, current_state, time):
        """
        current_state: array-like of spot prices.
        time: scalar or array-like of current time in years.
        Returns: boolean vector.
        """
        current_state = np.asarray(current_state)
        time = np.asarray(time)
        
        # Handle scalar time
        if time.ndim == 0:
            time = np.full_like(current_state, time, dtype=float)
        
        # Determine the step index k
        # time is in years. dt = 1/12.
        # k = round(time / dt) = round(time * 12)
        k = np.round(time * 12).astype(int)
        
        # Clamp k to [1, self.n_steps]
        # If time is 0, k=0. But exercise is not allowed at t=0.
        # The problem says "12 monthly exercise dates (no time-0 exercise)".
        # So valid k are 1..12.
        # If k < 1, we cannot exercise. Return False.
        # If k > 12, we are past maturity? Or at maturity?
        # Usually, at maturity (t=1), you must exercise if ITM.
        # Our boundaries array is indexed 1..12.
        # boundaries[12] is the boundary at t=1.
        
        exercise_mask = np.zeros_like(current_state, dtype=bool)
        
        # Vectorized lookup
        # We need to handle different k values for different paths if time is an array.
        # But typically in this task, `time` might be a scalar for the current step 
        # or an array if evaluating multiple times at once.
        # Given "exercise(current_state, time)", and "returning a boolean vector",
        # it implies we might be evaluating a batch of states at a specific time, 
        # or states at different times.
        
        # Let's assume `time` can be an array.
        # We iterate over unique times to be safe and efficient.
        unique_times = np.unique(time)
        
        for t_val in unique_times:
            k_raw = int(np.round(t_val * 12))
            
            # No exercise at time 0
            if k_raw < 1:
                mask_t = (time == t_val)
                exercise_mask[mask_t] = False
                continue
                
            k_val = np.clip(k_raw, 1, self.n_steps)
            
            # Get the critical boundary for this step
            s_crit = self.critical_boundaries[k_val]
            
            # Find paths with this time
            mask_t = (time == t_val)
            
            # Exercise if S <= s_crit
            exercise_mask[mask_t] = current_state[mask_t] <= s_crit
            
        return exercise_mask

def fit_exercise_policy(training_paths, contract):
    """
    Fits the exercise policy using the contract parameters.
    """
    # Extract parameters from contract
    # Based on the task description:
    # Spot 90/100/110, K 100, r 0.05, sigma 0.2, T 1
    # 12 monthly exercise dates
    
    # We assume the contract object has attributes or methods to get these.
    # Common attributes: K, r, sigma, T, n_steps or exercise_dates.
    # Let's try to access them robustly.
    
    K = getattr(contract, 'K', 100.0)
    r = getattr(contract, 'r', 0.05)
    sigma = getattr(contract, 'sigma', 0.2)
    T = getattr(contract, 'T', 1.0)
    
    # Determine number of steps
    # "12 monthly exercise dates"
    n_steps = 12
    
    # If contract has explicit dates, use them
    if hasattr(contract, 'exercise_dates'):
        dates = contract.exercise_dates
        # Assume dates are in years
        # Count the number of dates
        n_steps = len(dates)
        # We might need to adjust dt if it's not uniform, but CRR assumes uniform.
        # For simplicity, we stick to the 12-month assumption if dates are monthly.
        
    policy = BermudanPolicy(K, r, sigma, T, n_steps)
    return policy
