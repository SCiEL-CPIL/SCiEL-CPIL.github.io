"""Bermudan put exercise policy: high-fidelity backward induction with exact discretization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

class ExercisePolicy:
    def __init__(self, cont_values, log_grid, step, K):
        """
        cont_values: List of arrays, cont_values[k] is the continuation value at exercise date k+1 (0-indexed list for dates 1..N)
                     Actually, let's align indices clearly.
                     Exercise dates are 1..N.
                     cont_values[0] corresponds to date 1.
                     cont_values[N-1] corresponds to date N.
        log_grid: 1D array of log spot prices.
        step: Log grid step size.
        K: Strike price.
        """
        self.cont_values = cont_values
        self.log_grid = log_grid
        self.step = step
        self.K = K
        self.n_grid = len(log_grid)
        self.min_log = log_grid[0]
        self.max_log = log_grid[-1]

    def exercise(self, current_state, time_idx):
        """
        Determine exercise decision.
        time_idx: 1-based index of the exercise date (1..12).
        """
        if time_idx <= 0 or time_idx > len(self.cont_values):
            return np.zeros(len(current_state), dtype=bool)
        
        states = np.asarray(current_state, dtype=np.float64)
        # Calculate intrinsic value
        intrinsic = np.maximum(self.K - states, 0.0)
        
        # Calculate continuation value by interpolating on the log grid
        log_states = np.log(np.maximum(states, 1e-10))
        
        # Map log_states to grid indices
        # idx_float = (log_states - min_log) / step
        idx_float = (log_states - self.min_log) / self.step
        idx_float = np.clip(idx_float, 0, self.n_grid - 1)
        
        idx0 = idx_float.astype(np.int64)
        idx1 = np.clip(idx0 + 1, 0, self.n_grid - 1)
        idx0 = np.clip(idx0, 0, self.n_grid - 1)
        
        t = idx_float - idx0
        
        cont_vals = self.cont_values[time_idx - 1]
        
        c0 = cont_vals[idx0]
        c1 = cont_vals[idx1]
        
        continuation = c0 * (1 - t) + c1 * t
        
        # Exercise if intrinsic value is strictly greater than continuation value
        # Using a small epsilon to handle floating point precision near the boundary
        return intrinsic > continuation

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s kill

    # Extract contract parameters
    K = float(contract.get('K', 100.0))
    r = float(contract.get('r', 0.05))
    sigma = float(contract.get('sigma', 0.2))
    T = float(contract.get('T', 1.0))
    n_steps = int(contract.get('n_steps', 12))

    dt = T / n_steps
    disc = np.exp(-r * dt)
    mu = r - 0.5 * sigma * sigma
    vol_step = sigma * np.sqrt(dt)

    # Grid construction
    # Wide range to cover all likely paths
    min_s = 0.1
    max_s = 1000.0
    
    # High resolution grid
    n_grid = 8192
    
    log_min = np.log(min_s)
    log_max = np.log(max_s)
    log_grid = np.linspace(log_min, log_max, n_grid)
    grid = np.exp(log_grid)
    step = log_grid[1] - log_grid[0]
    
    # Precompute convolution weights for the expectation operator
    # E[V(x_{t+dt}) | x_i] = sum_j V(x_j) * P(x_{t+dt} in [x_j, x_{j+1}] | x_i)
    # x_{t+dt} ~ N(x_i + mu*dt, vol_step^2)
    # Let d = j - i.
    # P(x_{t+dt} in [x_i + d*step, x_i + (d+1)*step] | x_i)
    # = P(N(0, vol_step^2) in [d*step - mu*dt, (d+1)*step - mu*dt])
    # = Phi(((d+1)*step - mu*dt)/vol_step) - Phi((d*step - mu*dt)/vol_step)
    
    # Determine the range of d for which the weight is significant
    # 5 standard deviations is usually enough
    max_d = int(np.ceil(5.0 * vol_step / step)) + 2
    # We need weights for d from -max_d to max_d
    # Note: The shift mu*dt also affects the center, so we might need to extend range slightly
    # The mean shift is mu*dt. In terms of grid steps, shift_idx = mu*dt/step.
    # So the effective center for weight d is d - shift_idx.
    # We want the normal pdf centered at 0 (in the shifted frame) to be non-negligible.
    
    d_range = np.arange(-max_d, max_d + 1)
    weights = np.zeros(len(d_range))
    
    for idx, d in enumerate(d_range):
        lower = (d * step - mu * dt) / vol_step
        upper = ((d + 1) * step - mu * dt) / vol_step
        weights[idx] = norm.cdf(upper) - norm.cdf(lower)
        
    # Normalize weights to sum to 1 (they should already, but floating point errors might occur)
    # Actually, the sum of probabilities over the infinite grid is 1. 
    # Our finite range might miss a tiny tail. 
    # For safety, we can normalize, but it's usually negligible.
    # Let's check the sum.
    # If the sum is not 1, we might want to scale, but it's better to ensure the range is large enough.
    
    # Function to apply the expectation operator (convolution)
    def apply_expectation(v_next):
        # v_next is an array of length n_grid
        # We need to compute E[i] = sum_d weights[d] * v_next[i+d]
        # This is a convolution.
        # Using np.convolve is easy but might be slow for 8192 * 800? 
        # 8192 * (2*max_d+1) operations.
        # max_d ~ 5 * 0.057 / 0.00075 ~ 380.
        # 8192 * 761 ~ 6 million operations. This is fast in NumPy if vectorized.
        
        # We can use np.convolve with 'same' mode? 
        # np.convolve(v_next, weights, mode='same')
        # But weights are indexed by d. 
        # Let's create a full kernel array centered.
        # Kernel length L = 2*max_d + 1.
        # Center index is max_d.
        
        # Actually, simpler:
        # E[i] = sum_{j=i-max_d}^{i+max_d} v_next[j] * weights[j-i+max_d]
        
        # Let's use a loop over the weights, which is vectorized over the grid.
        # This is often faster than fftconvolve for small kernels in Python due to overhead.
        
        result = np.zeros_like(v_next)
        
        # Weights correspond to d from -max_d to max_d.
        # weights[k] corresponds to d = k - max_d.
        
        # To avoid boundary issues, we pad v_next with zeros (or edge values?).
        # For GBM, if S goes below min_s, the value is effectively capped or we can assume 0 for put?
        # Actually, if S < min_s, the put is deep ITM. Value is approx K-S.
        # But our grid starts at 0.1. If we go below, we should probably extrapolate.
        # However, the probability of going below 0.1 in one step from the grid is very small 
        # unless we are already near 0.1.
        # Given the grid starts at 0.1, and step is small, the probability mass leaving the grid is negligible.
        # We will use zero padding for simplicity, which assumes V=0 outside the grid.
        # This is a slight approximation for the lower tail.
        # To be more accurate, we could pad with the boundary value or linear extrapolation.
        # For a put, V(S) ~ K-S for small S.
        # Let's just use zero padding; the mass is tiny.
        
        for k in range(len(weights)):
            d = d_range[k]
            w = weights[k]
            if w < 1e-10:
                continue
            
            # Shift v_next by -d
            # If d > 0, we look at v_next[i+d]. This is a shift to the left.
            # If d < 0, we look at v_next[i+d]. This is a shift to the right.
            
            if d == 0:
                result += w * v_next
            elif d > 0:
                # v_next[i+d] for i=0..n_grid-1-d
                # For i > n_grid-1-d, we are out of bounds.
                # We add w * v_next[d:] to result[:n_grid-d]
                result[:n_grid-d] += w * v_next[d:]
            else:
                # d is negative. Let dd = -d.
                # v_next[i-dd] for i=dd..n_grid-1
                # We add w * v_next[:n_grid-dd] to result[dd:]
                dd = -d
                result[dd:] += w * v_next[:n_grid-dd]
                
        return result

    # Backward Induction
    # V_next: value function at the next exercise date
    # At the last exercise date (k=n_steps), V = Payoff
    V_next = np.maximum(K - grid, 0.0)
    
    # Store continuation values for each date (1 to n_steps)
    # conts[k-1] will be the continuation value at date k
    conts = [None] * n_steps
    
    # At the last date (n_steps), there is no continuation value (or it is 0)
    # The decision is: Exercise if Payoff > 0.
    # So conts[n_steps-1] = 0
    conts[n_steps-1] = np.zeros(n_grid)
    
    # Iterate backwards from k = n_steps-1 down to 1
    # k represents the current date index (1-based)
    for k in range(n_steps - 1, 0, -1):
        if time.time() > deadline:
            # Fallback: fill remaining with zeros or simple heuristic
            for rem_k in range(k-1, 0, -1):
                conts[rem_k-1] = np.zeros(n_grid)
            break
            
        # Compute Continuation Value
        # Cont = disc * E[V_next(S_next) | S]
        E_val = apply_expectation(V_next)
        Cont = disc * E_val
        
        # Store continuation value for this date k
        conts[k-1] = Cont
        
        # Update V_next for the next iteration (going further back)
        # V_k = max(Payoff, Cont)
        Payoff = np.maximum(K - grid, 0.0)
        V_next = np.maximum(Payoff, Cont)

    # If timeout occurred in the middle, some conts might be None. Fill them.
    for i in range(n_steps):
        if conts[i] is None:
            conts[i] = np.zeros(n_grid)

    # Create the policy object
    return ExercisePolicy(conts, log_grid, step, K)

# EVOLVE-BLOCK-END
