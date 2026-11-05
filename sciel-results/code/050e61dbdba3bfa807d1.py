import numpy as np
import time as _time

class ExercisePolicy:
    def __init__(self, thresholds):
        self.thresholds = thresholds
    
    def exercise(self, current_state, time_index):
        if time_index < 0 or time_index >= len(self.thresholds):
            return np.zeros(len(current_state), dtype=bool)
        return current_state < self.thresholds[time_index]

def fit_exercise_policy(training_paths, contract):
    start_time = _time.time()
    deadline = start_time + 110  # Safety margin
    
    paths = np.asarray(training_paths, dtype=np.float64)
    n_paths, n_steps = paths.shape
    n_exercise = 12  # exercise dates at steps 1..12
    
    dt = 1.0 / 12
    r = 0.05
    K = 100.0
    discount = np.exp(-r * dt)
    
    # V[j, i] stores the value of the option at exercise date j (step j+1) for path i
    # We store the "optimal" value (max(payoff, cont)) to use for the previous step's regression target
    V = np.zeros((n_exercise, n_paths))
    
    # Regression coefficients for each step, to be used for threshold extraction
    # We'll use a polynomial of degree 3 for the continuation value
    # Normalize S to [0,1] range for numerical stability
    poly_degree = 3
    coeffs = [None] * n_exercise
    S_norm_scale = 200.0  # S ranges roughly 0-200, normalize by this
    
    # Last exercise date: j = 11 (index in V), step 12 in paths
    # Value is just the payoff
    S_last = paths[:, 12]
    V[11, :] = np.maximum(K - S_last, 0.0)
    
    # We don't strictly need a regression for the last step for the *value* calculation,
    # but we might want it for threshold extraction if we treat it uniformly.
    # However, for the last step, the "continuation" is 0.
    # So threshold is where Payoff(S) > 0 => S < K.
    # So threshold[11] = K.
    # Let's verify if we should exercise earlier.
    # At step 11, we choose max(Payoff, 0). So we exercise if Payoff > 0.
    # Threshold is K.
    
    # Backward induction for j = 10 down to 0
    for j in range(10, -1, -1):
        if _time.time() > deadline:
            break
            
        S_curr = paths[:, j+1]
        S_next = paths[:, j+2] # Wait, V[j+1] is already computed for step j+2?
        # No, V[j+1, i] is the value at step (j+1)+1 = j+2?
        # Let's re-verify indices.
        # Exercise date 0 is at step 1.
        # Exercise date 11 is at step 12.
        # V[j, i] is value at exercise date j (step j+1).
        # Continuation at j is E[ V[j+1] | S_j ].
        # V[j+1, i] is the value at step (j+1)+1 = j+2 for path i.
        # So yes, we use V[j+1, :] as the target for regression against S_curr (which is step j+1).
        
        # Target for regression: Discounted value of next step
        # Note: In LSMC, we regress the *discounted* next-step values on current states.
        targets = discount * V[j+1, :]
        
        # Only use in-the-money paths for regression to improve stability?
        # Standard practice for American options: regress on ITM paths.
        # Payoff at j is max(K - S_curr, 0).
        # ITM if S_curr < K.
        itm_mask = S_curr < K
        # To avoid division by zero or small sample issues, if too few ITM, use all
        if np.sum(itm_mask) < 50:
            itm_mask = np.ones(n_paths, dtype=bool)
            
        S_reg = S_curr[itm_mask]
        T_reg = targets[itm_mask]
        
        # Fit polynomial on normalized S: T = c0 + c1*(S/scale) + c2*(S/scale)^2 + c3*(S/scale)^3
        # This improves numerical conditioning
        S_reg_norm = S_reg / S_norm_scale
        try:
            # Use degree 3. If n_samples < degree+1, reduce degree.
            deg = min(poly_degree, len(S_reg_norm) - 1)
            if deg < 0:
                 deg = 0
            c = np.polyfit(S_reg_norm, T_reg, deg)
            coeffs[j] = c
        except:
            # Fallback to mean if fit fails
            coeffs[j] = np.array([np.mean(T_reg)])
            deg = 0
            
        # Evaluate continuation value for all paths at current step
        # np.polyval takes coeffs and x (normalized)
        S_curr_norm = S_curr / S_norm_scale
        cont_values = np.polyval(coeffs[j], S_curr_norm)
        
        # Current Payoff
        payoff = np.maximum(K - S_curr, 0.0)
        
        # Optimal Value
        V[j, :] = np.maximum(payoff, cont_values)
        
    # Extract Thresholds
    thresholds = np.zeros(n_exercise)
    
    # For the last step (j=11), we didn't compute a regression in the loop above.
    # But the logic is: Exercise if Payoff > 0.
    # So threshold is K.
    thresholds[11] = K
    
    for j in range(10, -1, -1):
        if coeffs[j] is None:
            thresholds[j] = K
            continue
            
        c = coeffs[j]
        # We need to find S such that K - S = P(S) for S < K
        # i.e., K - S - P(S) = 0
        # Let f(S) = K - S - polyval(c, S/scale)
        # We want the largest root in [0, K]
        
        # Create a fine grid for initial bracket
        S_grid = np.linspace(0, K, 1000)
        S_grid_norm = S_grid / S_norm_scale
        f_vals = K - S_grid - np.polyval(c, S_grid_norm)
        
        # Find sign changes
        signs = np.sign(f_vals)
        signs[signs == 0] = 1 
        
        changes = np.where(np.diff(signs) != 0)[0]
        
        if len(changes) > 0:
            idx = changes[-1]
            # Linear interpolation for initial estimate
            x1, x2 = S_grid[idx], S_grid[idx+1]
            y1, y2 = f_vals[idx], f_vals[idx+1]
            if y1 == y2:
                thresh = x1
            else:
                thresh = x1 - y1 * (x2 - x1) / (y2 - y1)
            
            # Newton refinement for higher precision
            # f(S) = K - S - polyval(c, S/scale)
            # f'(S) = -1 - polyval_derivative(c, S/scale) * (1/scale)
            for _ in range(20):
                if thresh <= 0 or thresh >= K:
                    break
                S_n = thresh / S_norm_scale
                f_val = K - thresh - np.polyval(c, S_n)
                # Derivative of polyval with respect to S_n
                # np.polyval(c, x) where c = [c_n, ..., c_1, c_0]
                # d/dx polyval = c_n*n*x^(n-1) + ... + c_1
                deg_c = len(c) - 1
                if deg_c > 0:
                    f_deriv_sn = np.polyval(np.polyder(c), S_n)
                else:
                    f_deriv_sn = 0.0
                f_deriv_s = -1.0 - f_deriv_sn / S_norm_scale
                if abs(f_deriv_s) < 1e-12:
                    break
                delta = f_val / f_deriv_s
                thresh_new = thresh - delta
                if abs(thresh_new - thresh) < 1e-10:
                    thresh = thresh_new
                    break
                thresh = thresh_new
            
            # Clip to valid range
            thresholds[j] = np.clip(thresh, 0, K)
        else:
            # No sign change? 
            if f_vals[0] < 0:
                thresholds[j] = 0.0
            else:
                thresholds[j] = K
                
    return ExercisePolicy(thresholds)
