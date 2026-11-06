"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class BermudanExercisePolicy:
    def __init__(self, thresholds):
        """
        thresholds: array of shape (12,) with critical stock price for each exercise date
                     Exercise if S < thresholds[i] at date i
        """
        self.thresholds = np.asarray(thresholds, dtype=np.float64)
    
    def exercise(self, current_state, time):
        """
        current_state: array of shape (n_paths,) with current stock prices
        time: integer or array of integers indicating which exercise date (0-indexed, 0-11)
              or a scalar time in [0, 1]
        Returns: boolean array of shape (n_paths,)
        """
        current_state = np.asarray(current_state, dtype=np.float64)
        
        if np.isscalar(time):
            # Convert continuous time to exercise date index
            # 12 monthly dates: t = 1/12, 2/12, ..., 12/12
            t = float(time)
            if t <= 0:
                return np.zeros(len(current_state), dtype=bool)
            if t > 1.0:
                t = 1.0
            # Map to exercise date index (0-indexed)
            idx = int(np.floor(t * 12))
            if idx >= 12:
                idx = 11
            return current_state < self.thresholds[idx]
        else:
            # time is an array
            time_arr = np.asarray(time)
            result = np.zeros(len(current_state), dtype=bool)
            for i in range(len(current_state)):
                t = float(time_arr[i]) if np.isscalar(time_arr[i]) else float(time_arr[i])
                if t <= 0:
                    result[i] = False
                else:
                    if t > 1.0:
                        t = 1.0
                    idx = int(np.floor(t * 12))
                    if idx >= 12:
                        idx = 11
                    result[i] = current_state[i] < self.thresholds[idx]
            return result

def fit_exercise_policy(training_paths, contract):
    """
    training_paths: array of shape (n_paths, n_time_steps) with stock prices
    contract: dict with parameters
    Returns: object with exercise(current_state, time) method
    """
    start_time = time.time()
    
    # Extract parameters from contract
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = 12  # 12 monthly exercise dates
    
    # Determine the time step from training paths
    # Training paths should have n_exercise + 1 columns (time 0 through time T)
    training_paths = np.asarray(training_paths, dtype=np.float64)
    
    if training_paths.ndim == 1:
        # Single path, reshape
        training_paths = training_paths.reshape(1, -1)
    
    n_paths, n_cols = training_paths.shape
    
    # The training paths should span from t=0 to t=T with n_exercise+1 points
    # Times: 0, T/12, 2T/12, ..., T
    dt = T / n_exercise
    discount = np.exp(-r * dt)
    
    # Payoff at maturity (last exercise date)
    # At the last date, exercise if payoff > 0
    S_final = training_paths[:, -1]
    payoff_final = np.maximum(K - S_final, 0.0)
    
    # Backward induction to compute continuation values
    # V[i] = continuation value at exercise date i (i = 0, ..., n_exercise-1)
    # At date i, we can either exercise (get payoff) or continue (get discounted expected continuation)
    
    # Start from the last date: continuation value is the expected payoff at maturity
    # Actually, for Bermudan, at each date we compare exercise vs continuation
    
    # Let's compute the value at each date
    # V[n_exercise-1] at the last date: max(exercise_payoff, continuation_to_maturity)
    # But since the last date IS maturity for our purposes, V_last = payoff
    
    # More carefully:
    # We have n_exercise dates: t_1 = T/12, t_2 = 2T/12, ..., t_12 = T
    # At t_12 (maturity), value = max(K-S, 0)
    # At t_i (i < 12), value = max(K-S, discount * E[V_{i+1} | S_i])
    
    # Use the training paths to estimate E[V_{i+1} | S_i]
    # For simplicity, use a linear regression or simple averaging
    
    # Let's use a grid-based approach for better accuracy
    # Create price bins and estimate continuation values
    
    n_bins = 50
    price_min = np.percentile(training_paths, 1)
    price_max = np.percentile(training_paths, 99)
    if price_max <= price_min:
        price_max = price_min + 1.0
    bin_edges = np.linspace(price_min, price_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    # Initialize continuation values at maturity
    V_next = np.zeros(n_bins)
    for b in range(n_bins):
        mask = (S_final >= bin_edges[b]) & (S_final < bin_edges[b+1])
        if np.any(mask):
            V_next[b] = np.mean(payoff_final[mask])
        else:
            # Extrapolate: for a put, value decreases with S
            if b == 0:
                V_next[b] = max(K - bin_centers[b], 0.0)
            else:
                V_next[b] = V_next[b-1] * 0.9  # rough decay
    
    # Backward induction
    thresholds = np.zeros(n_exercise)
    
    for i in range(n_exercise - 1, -1, -1):
        # At exercise date i, stock prices are training_paths[:, i+1]
        S_i = training_paths[:, i + 1]
        
        # Compute continuation value at each price level using binning
        V_cont = np.zeros(len(S_i))
        for b in range(n_bins):
            mask = (S_i >= bin_edges[b]) & (S_i < bin_edges[b+1])
            V_cont[mask] = V_next[b]
        
        # Exercise payoff
        payoff_i = np.maximum(K - S_i, 0.0)
        
        # Optimal decision: exercise if payoff_i > V_cont
        exercise_mask = payoff_i > V_cont
        
        # Determine threshold: the critical price where exercise switches
        # Find the highest price where exercise is optimal
        # Sort by price
        sorted_indices = np.argsort(S_i)
        S_sorted = S_i[sorted_indices]
        ex_sorted = exercise_mask[sorted_indices]
        
        # Find the transition point
        threshold = 0.0
        for j in range(len(S_sorted) - 1, -1, -1):
            if ex_sorted[j]:
                threshold = S_sorted[j]
                break
        
        thresholds[i] = threshold
        
        # Update V_next for the next (earlier) date
        # V_next represents the value at date i, which is max(payoff, discount * V_cont_next)
        # We need to recompute V_next as the value function at date i
        V_current = np.maximum(payoff_i, discount * V_cont)
        
        for b in range(n_bins):
            mask = (S_i >= bin_edges[b]) & (S_i < bin_edges[b+1])
            if np.any(mask):
                V_next[b] = np.mean(V_current[mask])
            else:
                V_next[b] = V_next[b]  # keep previous estimate
    
    # Time guard
    elapsed = time.time() - start_time
    if elapsed > 100:
        # Fallback to simple heuristic if we're running out of time
        # For a put, exercise when S is significantly below K
        thresholds = np.array([K * 0.85, K * 0.82, K * 0.80, K * 0.78, K * 0.76, K * 0.74,
                               K * 0.72, K * 0.70, K * 0.68, K * 0.66, K * 0.64, K * 0.62])
    
    return BermudanExercisePolicy(thresholds)
# EVOLVE-BLOCK-END
