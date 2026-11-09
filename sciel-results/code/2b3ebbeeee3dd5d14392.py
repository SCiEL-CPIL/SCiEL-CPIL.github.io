"""Rewritten program for Bermudan exercise-policy research."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction with 
    parametric continuation value approximation.
    
    Parameters:
    -----------
    training_paths : array-like, shape (n_paths, n_time_steps)
        Simulated spot price paths.
        
    contract : dict
        Contract parameters.
        
    Returns:
    --------
    An object with method exercise(current_state, time) -> boolean array.
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_steps = contract.get('n_steps', 12)
    
    # Convert training_paths to numpy array
    training_paths = np.asarray(training_paths, dtype=np.float64)
    n_paths, total_time_steps = training_paths.shape
    
    # Determine exercise dates: 12 monthly exercise dates, no time-0 exercise
    dt = T / n_steps
    discount = np.exp(-r * dt)
    
    # Payoff function for put: max(K - S, 0)
    def payoff(S):
        return np.maximum(K - S, 0.0)
    
    # Determine exercise indices
    if total_time_steps == n_steps + 1:
        exercise_indices = list(range(1, n_steps + 1))
    else:
        exercise_indices = list(range(total_time_steps - n_steps, total_time_steps))
    
    n_exercise_dates = len(exercise_indices)
    
    # Pre-compute all spot prices at each exercise date
    S_at_dates = [training_paths[:, idx] for idx in exercise_indices]
    
    # For each exercise date, we will store the continuation value as a function of S
    # We use a flexible parametric form: V_cont(S) = A * S^p + B * exp(-C * S) + D
    # But for a put, continuation value is 0 for large S and positive for small S.
    # A better form: V_cont(S) = A * (K - S)^p * exp(-B * S) for S < K, else 0
    # Actually, let's use a simpler but effective approach:
    # Fit continuation value using piecewise linear interpolation on a fine grid,
    # but compute the grid values using regression on the training paths.
    
    # Alternative: Use the fact that for a Bermudan put under GBM, the exercise
    # boundary is monotonic. We can fit a threshold S_j(t) for each date t.
    # Exercise if S <= S_j(t).
    
    # Let's use a hybrid: backward induction with regression-based continuation values.
    # For each date j, given V_{j+1}(S), the continuation value is:
    # E[discount * V_{j+1}(S_{j+1}) | S_j = s]
    # Under GBM, S_{j+1} = S_j * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
    # So we can compute the conditional expectation analytically or via Monte Carlo.
    
    # Since we have training paths, we can use them to estimate the conditional expectation
    # via regression. But a cleaner approach for GBM is to use the known transition.
    
    # Let's use a parametric value function approach:
    # V_j(S) = max(payoff(S), E[discount * V_{j+1}(S_{j+1}) | S_j = S])
    
    # For the expectation, under GBM:
    # S_{j+1} = S_j * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z), Z ~ N(0,1)
    # So V_{j+1}(S_{j+1}) = V_{j+1}(S_j * exp(a + b*Z)) where a = (r-sigma^2/2)*dt, b = sigma*sqrt(dt)
    
    # If V_{j+1} is represented as a function, we can compute the expectation numerically.
    
    # Let's represent V_j as a set of (S, V) points and use interpolation.
    # But to avoid grid issues, let's use a polynomial or spline fit.
    
    # Simpler robust approach: Use the training paths directly with a better binning strategy.
    # The key insight: use quantile-based bins to ensure uniform path distribution.
    
    # Let's use a higher resolution with quantile bins.
    all_states = np.concatenate(S_at_dates)
    # Use 2000 quantile bins for better resolution
    n_bins = 2000
    bin_edges = np.quantile(all_states, np.linspace(0, 1, n_bins + 1))
    # Ensure edges are strictly increasing
    for i in range(1, n_bins + 1):
        if bin_edges[i] <= bin_edges[i-1]:
            bin_edges[i] = bin_edges[i-1] + 1e-10
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Value function at each exercise date
    all_V_vals = [None] * n_exercise_dates
    
    # At the last exercise date, V = payoff
    all_V_vals[-1] = payoff(bin_centers)
    
    # Work backwards
    for j in range(n_exercise_dates - 2, -1, -1):
        if time.time() > deadline - 10.0:
            # Not enough time, use simple approximation
            for k in range(j, -1, -1):
                if all_V_vals[k] is None:
                    all_V_vals[k] = payoff(bin_centers)
            break
            
        S_current = S_at_dates[j]
        S_next = S_at_dates[j + 1]
        
        # Evaluate V_next at S_next
        V_next_at_paths = np.interp(S_next, bin_centers, all_V_vals[j + 1])
        cont_vals = discount * V_next_at_paths
        
        # Assign paths to bins for S_current
        bin_indices = np.digitize(S_current, bin_edges, right=False)
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Compute average continuation value per bin
        bin_sums = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        
        np.add.at(bin_sums, bin_indices, cont_vals)
        np.add.at(bin_counts, bin_indices, 1.0)
        
        V_cont_at_bins = np.zeros(n_bins)
        valid = bin_counts > 0
        V_cont_at_bins[valid] = bin_sums[valid] / bin_counts[valid]
        
        # Fill empty bins by interpolation
        if not np.all(valid):
            valid_indices = np.where(valid)[0]
            if len(valid_indices) > 0:
                empty_indices = np.where(~valid)[0]
                V_cont_at_bins[empty_indices] = np.interp(
                    bin_centers[empty_indices],
                    bin_centers[valid_indices],
                    V_cont_at_bins[valid_indices]
                )
            else:
                V_cont_at_bins[:] = payoff(bin_centers)
        
        # Smooth the continuation value to reduce noise
        # Use a simple moving average
        kernel_size = 5
        if n_bins > kernel_size:
            kernel = np.ones(kernel_size) / kernel_size
            V_cont_smooth = np.convolve(V_cont_at_bins, kernel, mode='same')
            # Blend original and smoothed
            V_cont_at_bins = 0.7 * V_cont_at_bins + 0.3 * V_cont_smooth
        
        # Value = max(payoff, continuation)
        payoff_at_bins = payoff(bin_centers)
        all_V_vals[j] = np.maximum(payoff_at_bins, V_cont_at_bins)
    
    # Determine exercise policy: exercise if payoff(S) >= continuation(S)
    # Store continuation values for each date
    cont_vals_list = [None] * n_exercise_dates
    cont_vals_list[-1] = np.zeros(n_bins)  # At maturity, always exercise if ITM
    
    for j in range(n_exercise_dates - 1):
        if time.time() > deadline - 5.0:
            # Fallback: use payoff * 0.95 to slightly encourage exercise
            cont_vals_list[j] = payoff(bin_centers) * 0.95
            continue
        
        S_current = S_at_dates[j]
        S_next = S_at_dates[j + 1]
        
        V_next_at_paths = np.interp(S_next, bin_centers, all_V_vals[j + 1])
        cont_vals = discount * V_next_at_paths
        
        bin_indices = np.digitize(S_current, bin_edges, right=False)
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        bin_sums = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        
        np.add.at(bin_sums, bin_indices, cont_vals)
        np.add.at(bin_counts, bin_indices, 1.0)
        
        cont_at_bins = np.zeros(n_bins)
        valid = bin_counts > 0
        cont_at_bins[valid] = bin_sums[valid] / bin_counts[valid]
        
        if not np.all(valid):
            valid_indices = np.where(valid)[0]
            if len(valid_indices) > 0:
                empty_indices = np.where(~valid)[0]
                cont_at_bins[empty_indices] = np.interp(
                    bin_centers[empty_indices],
                    bin_centers[valid_indices],
                    cont_at_bins[valid_indices]
                )
            else:
                cont_at_bins[:] = payoff(bin_centers)
        
        # Smooth
        kernel_size = 5
        if n_bins > kernel_size:
            kernel = np.ones(kernel_size) / kernel_size
            cont_smooth = np.convolve(cont_at_bins, kernel, mode='same')
            cont_at_bins = 0.7 * cont_at_bins + 0.3 * cont_smooth
        
        cont_vals_list[j] = cont_at_bins
    
    class ExercisePolicy:
        def __init__(self, cont_vals, bin_centers, K, T, n_steps):
            self.cont_vals = cont_vals
            self.bin_centers = bin_centers
            self.K = K
            self.T = T
            self.n_steps = n_steps
            
        def exercise(self, current_state, time):
            current_state = np.asarray(current_state, dtype=np.float64)
            
            dt = self.T / self.n_steps
            if isinstance(time, (int, np.integer)):
                step = int(time)
            else:
                step = int(round(time / dt))
            
            step = max(1, min(step, self.n_steps))
            date_idx = step - 1
            date_idx = max(0, min(date_idx, len(self.cont_vals) - 1))
            
            if date_idx >= len(self.cont_vals):
                return current_state < self.K
            
            cont_at_bins = self.cont_vals[date_idx]
            cont_vals = np.interp(current_state, self.bin_centers, cont_at_bins)
            payoffs = np.maximum(self.K - current_state, 0.0)
            
            return payoffs >= cont_vals
    
    return ExercisePolicy(cont_vals_list, bin_centers, K, T, n_steps)
# EVOLVE-BLOCK-END
