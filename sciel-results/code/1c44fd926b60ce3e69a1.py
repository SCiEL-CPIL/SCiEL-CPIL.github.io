"""Bermudan put exercise policy via binomial tree with backward induction."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using a binomial tree with backward induction.
    
    The policy is a threshold-based rule: exercise at time t if spot <= threshold[t].
    We use a high-resolution binomial tree to compute optimal exercise boundaries.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin under 150s limit
    
    # Extract contract parameters
    K = contract.get('strike', 100.0)
    r = contract.get('rate', 0.05)
    sigma = contract.get('volatility', 0.2)
    T = contract.get('maturity', 1.0)
    n_exercise = contract.get('n_exercise_dates', 12)
    
    # Time grid: 12 monthly exercise dates (no time-0 exercise)
    # Exercise dates are at T/12, 2T/12, ..., 12T/12 = T
    dt = T / n_exercise
    
    # Build binomial tree parameters
    # Use a high number of steps for accuracy
    # We need to resolve the exercise dates, so use n_steps = n_exercise * steps_per_period
    steps_per_period = 100
    n_steps = n_exercise * steps_per_period
    
    # Risk-neutral parameters
    u = np.exp(sigma * np.sqrt(dt / steps_per_period))
    d = 1.0 / u
    p = (np.exp(r * dt / steps_per_period) - d) / (u - d)
    discount = np.exp(-r * dt / steps_per_period)
    
    # We'll do backward induction on the tree
    # The tree has n_steps+1 levels, each with increasing number of nodes
    # But for efficiency, we work with log-space or use a fixed grid
    
    # Alternative: Use a fixed grid in log-price space for each exercise date
    # This is more efficient and accurate for threshold finding
    
    # Let's use a different approach: simulate the optimal policy via
    # backward induction on a discretized state space
    
    # State space: log(S/K) from some min to max
    # At maturity, put payoff = max(K-S, 0)
    # We discretize S on a grid
    
    # Grid size - use enough points to resolve the threshold accurately
    n_grid = 5000
    s_max = 2.0 * K  # reasonable upper bound
    s_min = 0.01 * K  # reasonable lower bound
    S_grid = np.linspace(s_min, s_max, n_grid)
    
    # Initialize continuation value at maturity
    # At the last exercise date (T), value = max(K - S, 0)
    V = np.maximum(K - S_grid, 0.0)
    
    # Store thresholds for each exercise date
    # thresholds[t] = S such that we exercise if S <= thresholds[t]
    thresholds = np.zeros(n_exercise)
    
    # Backward induction from last exercise date to first
    # Exercise dates are at times: dt, 2*dt, ..., n_exercise*dt = T
    # We go backwards: from T to dt
    
    for t_idx in range(n_exercise - 1, -1, -1):
        time_to_maturity = (n_exercise - 1 - t_idx) * dt
        
        if time_to_maturity == 0:
            # This is the last exercise date; V is already the payoff
            # Find threshold: largest S where K-S >= V (which is K-S, so always exercise if K>S)
            # Actually at maturity, exercise if K > S
            thresholds[t_idx] = K
            continue
        
        # Discount the continuation value from next step
        # We need to compute E[discount * V(S_next)] where S_next follows GBM
        # Using the binomial tree: S_next = S * u or S * d with probs p and 1-p
        # But our grid is in S-space, not aligned with u,d steps
        
        # Better approach: use the analytical transition
        # For a small time step, we can use the fact that under risk-neutral measure:
        # ln(S_next/S) ~ N((r - sigma^2/2)*dt, sigma^2*dt)
        
        # For each S in grid, compute expected discounted future value
        # This is a convolution-like operation
        
        # Use a simpler approach: since we're on a uniform grid in S,
        # and the transition is multiplicative, we can interpolate
        
        # Actually, let's use a proper binomial tree aligned with the grid
        # Or better: use the PDE approach with finite differences
        
        # For now, use a simple approach: approximate the expectation using
        # the fact that for GBM, we can compute the expectation analytically
        # if V is piecewise linear (which it approximately is)
        
        # Simpler: use a large number of Monte Carlo paths for the expectation
        # But that's slow. Let's use the analytical Black-Scholes for European
        # as a lower bound, then add the exercise premium
        
        # Actually, the cleanest approach: use a binomial tree with the same
        # time steps as exercise dates, but with a fine price grid
        
        # Let me reconsider: use a recombining binomial tree
        # n_steps total steps, n_exercise exercise dates
        # At each exercise date, we have (n_steps_so_far + 1) nodes
        
        # This is getting complex. Let's use a different strategy:
        # Use the Longstaff-Schwartz style but with a deterministic grid
        
        # For simplicity and speed, let's use the following:
        # At each exercise date, compute the continuation value using
        # the Black-Scholes formula for the remaining horizon, adjusted
        # for the fact that we can exercise at intermediate dates
        
        # This is an approximation. For a better approach, let's do
        # proper backward induction on a binomial tree.
        
        # Let's restart with a cleaner binomial tree approach
        break
    
    # Clean approach: build a full binomial tree
    # n_total_steps = n_exercise * steps_per_period
    # But this creates a huge tree. Instead, use a compact representation.
    
    # Use the fact that for a recombining binomial tree, the price at step i
    # and node j is S0 * u^(j - i/2) * d^(i/2 - j/2) = S0 * u^j * d^(i-j)
    
    # For a put option, we care about the threshold S* where exercise is optimal.
    # Let's work in terms of m = log(S/K)
    
    # Define the grid in log-space
    m_min = np.log(s_min / K)
    m_max = np.log(s_max / K)
    n_m = 3000
    m_grid = np.linspace(m_min, m_max, n_m)
    S_vals = K * np.exp(m_grid)
    
    # Initialize at maturity
    V = np.maximum(K - S_vals, 0.0)
    
    thresholds = np.zeros(n_exercise)
    
    # Backward induction
    for t_idx in range(n_exercise - 1, -1, -1):
        remaining_steps = n_exercise - 1 - t_idx
        time_to_exercise = remaining_steps * dt
        
        if remaining_steps == 0:
            # Last exercise date: exercise if K > S
            # Threshold is K
            thresholds[t_idx] = K
            continue
        
        # Compute continuation value: E[exp(-r*time_to_exercise) * V(S_{t+1})]
        # where S_{t+1} = S * exp((r - sigma^2/2)*time_to_exercise + sigma*sqrt(time_to_exercise)*Z)
        
        # For each S in S_vals, we need to compute this expectation.
        # This is a convolution in log-space.
        
        # Use a quadrature approach: for each S, the distribution of ln(S_next)
        # is normal with mean ln(S) + (r - sigma^2/2)*time_to_exercise
        # and variance sigma^2*time_to_exercise
        
        # We can compute this using the fact that V is defined on our grid.
        # For each S_i, E[V(S_next)] = integral V(S_next) * pdf(ln(S_next) | S_i) d(ln(S_next))
        
        # This is a Gaussian convolution. We can compute it efficiently.
        
        # For each point in m_grid, the mean of next m is:
        # m_next_mean = m + (r - sigma^2/2)*time_to_exercise
        # m_next_std = sigma * sqrt(time_to_exercise)
        
        # We need to compute: for each m_i, E[V(K*exp(m_next))] where m_next ~ N(m_i + drift, var)
        
        # This is equivalent to: (V * gaussian_kernel)(m_i) where the kernel is
        # centered at m_i + drift
        
        # Let's compute this using interpolation and a sum
        # For efficiency, we can use the fact that the Gaussian has limited support
        
        drift = (r - 0.5 * sigma**2) * time_to_exercise
        vol = sigma * np.sqrt(time_to_exercise)
        
        # For each grid point m_i, the expected value is:
        # sum_j V(m_j) * N(m_j | m_i + drift, vol^2) * dm
        # where N is the normal pdf
        
        # This is a convolution. We can compute it by, for each m_j,
        # adding V(m_j) * N(m_i | m_j - drift, vol^2) for all m_i
        
        # Or more efficiently, for each m_i, find the relevant m_j's
        
        # For speed, let's use a simpler approximation:
        # Use the Black-Scholes formula for the European put as a baseline,
        # then add a correction for the exercise option
        
        # European put price at S with time_to_exercise remaining:
        T_rem = time_to_exercise
        if T_rem > 0:
            sqrt_T = np.sqrt(T_rem)
            d1 = (np.log(S_vals / K) + (r + 0.5 * sigma**2) * T_rem) / (sigma * sqrt_T)
            d2 = d1 - sigma * sqrt_T
            from scipy.stats import norm
            N_cdf = norm.cdf
            eu_put = K * np.exp(-r * T_rem) * N_cdf(-d2) - S_vals * N_cdf(-d1)
        else:
            eu_put = np.maximum(K - S_vals, 0.0)
        
        # The Bermudan put value is >= European put value
        # For a first approximation, use the European value as continuation
        # Then the optimal exercise threshold is where K-S >= continuation
        
        # But this underestimates the continuation value (ignores intermediate exercise)
        # For a better estimate, we can iterate
        
        # For now, let's use this as a starting point and refine
        
        # Actually, let's do proper backward induction with a coarser but exact method
        # Use a binomial tree with n_exercise steps (one per exercise date)
        
        # This is simpler and exact for the discretized problem
        break
    
    # Let's use a clean binomial tree with one step per exercise date
    # This gives us the exact optimal policy for the 12-date Bermudan put
    # on the binomial tree
    
    # Parameters
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    
    # Tree: at step i (0 to n_exercise), there are i+1 nodes
    # Node j at step i: S = S0 * u^j * d^(i-j)
    # But we don't know S0. Instead, we work with the ratio S/K or just S
    
    # For a put, the value depends on S. Let's define the value function V_i(S)
    # at step i (time i*dt) as a function of S.
    
    # At step n_exercise (maturity): V_n(S) = max(K - S, 0)
    
    # At step i: V_i(S) = max(K - S, disc * [p * V_{i+1}(S*u) + (1-p) * V_{i+1}(S*d)])
    
    # We represent V_i on a grid of S values.
    
    # Grid: from S_min to S_max
    S_min = K * 0.01
    S_max = K * 3.0
    n_S = 2000
    S_grid = np.linspace(S_min, S_max, n_S)
    
    # Initialize at maturity
    V_next = np.maximum(K - S_grid, 0.0)
    
    thresholds = np.zeros(n_exercise)
    
    for i in range(n_exercise - 1, -1, -1):
        # Compute continuation value at step i for each S in S_grid
        # S_up = S * u, S_down = S * d
        S_up = S_grid * u
        S_down = S_grid * d
        
        # Interpolate V_next at S_up and S_down
        # V_next is defined on S_grid (uniform in S)
        # Use linear interpolation
        
        def interp_S(S_query, S_ref, V_ref):
            """Linear interpolation of V_ref (defined on S_ref) at S_query."""
            # Clip to range
            S_q = np.clip(S_query, S_ref[0], S_ref[-1])
            # Find indices
            idx = np.searchsorted(S_ref, S_q)
            idx = np.clip(idx, 1, len(S_ref) - 1)
            x0 = S_ref[idx - 1]
            x1 = S_ref[idx]
            y0 = V_ref[idx - 1]
            y1 = V_ref[idx]
            # Linear interpolation
            if x1 == x0:
                return y0
            frac = (S_q - x0) / (x1 - x0)
            return y0 + frac * (y1 - y0)
        
        V_up = interp_S(S_up, S_grid, V_next)
        V_down = interp_S(S_down, S_grid, V_next)
        
        # Continuation value
        cont = disc * (p * V_up + (1 - p) * V_down)
        
        # Exercise value
        ex = np.maximum(K - S_grid, 0.0)
        
        # Optimal value
        V_curr = np.maximum(ex, cont)
        
        # Find threshold: largest S where ex >= cont
        # i.e., where K - S >= cont (and S < K)
        # Exercise region: S <= threshold
        diff = ex - cont
        # Find the largest S where diff >= 0
        # diff is non-negative for small S, becomes negative for large S
        
        # Find the crossing point
        # Look for the last index where diff >= 0
        mask = diff >= 0
        if np.any(mask):
            last_idx = np.where(mask)[0][-1]
            if last_idx < n_S - 1:
                # Interpolate between last_idx and last_idx+1
                S_a = S_grid[last_idx]
                S_b = S_grid[last_idx + 1]
                d_a = diff[last_idx]
                d_b = diff[last_idx + 1]
                if d_a - d_b != 0:
                    threshold = S_a + (0 - d_a) * (S_b - S_a) / (d_a - d_b)
                else:
                    threshold = S_a
            else:
                threshold = S_grid[-1]
        else:
            threshold = S_grid[0]
        
        thresholds[i] = threshold
        
        V_next = V_curr
    
    # Check time
    if time.time() > deadline:
        # Return a simple policy
        pass
    
    # Create the exercise policy object
    class ExercisePolicy:
        def __init__(self, thresholds, n_exercise, T, K):
            self.thresholds = thresholds
            self.n_exercise = n_exercise
            self.T = T
            self.K = K
            self.dt = T / n_exercise
        
        def exercise(self, current_state, time):
            """
            Determine whether to exercise at the given time.
            
            current_state: array of spot prices
            time: current time (continuous)
            
            Returns: boolean array, True if exercise
            """
            # Determine which exercise date we're at
            # Exercise dates are at dt, 2*dt, ..., n_exercise*dt
            # time is given as a continuous value
            
            # Find the nearest exercise date
            # The exercise dates are at k*dt for k=1,...,n_exercise
            k = np.round(time / self.dt)
            k = np.clip(k, 1, self.n_exercise)
            
            # k is an integer (or array if time is an array)
            # thresholds is indexed 0 to n_exercise-1, corresponding to k=1 to n_exercise
            # So threshold index is k-1
            
            if isinstance(k, np.ndarray):
                idx = k.astype(int) - 1
                idx = np.clip(idx, 0, self.n_exercise - 1)
                thresh = self.thresholds[idx]
            else:
                idx = int(k) - 1
                idx = max(0, min(idx, self.n_exercise - 1))
                thresh = self.thresholds[idx]
            
            # Exercise if S <= threshold
            return current_state <= thresh
    
    return ExercisePolicy(thresholds, n_exercise, T, K)
# EVOLVE-BLOCK-END
