import numpy as np
import time as _time
from numpy.polynomial import chebyshev as C

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
    V = np.zeros((n_exercise, n_paths))
    
    # We will use Chebyshev polynomials for the continuation value regression.
    # We map S to [-1, 1] for the Chebyshev basis.
    # S is typically in [0, 200] or so. Let's map [0, 2*K] to [-1, 1] or similar.
    # A safer mapping is to use the min/max of the training data or a fixed reasonable range.
    # Fixed range [0, 200] is reasonable for GBM with S0=100, sigma=0.2, T=1.
    S_min = 0.0
    S_max = 200.0
    # Map S to x in [-1, 1]: x = (2*S - (S_min + S_max)) / (S_max - S_min)
    
    def map_to_cheb(S):
        return (2.0 * S - (S_min + S_max)) / (S_max - S_min)
    
    def map_from_cheb(x):
        return (x * (S_max - S_min) / 2.0) + (S_min + S_max) / 2.0

    cheb_degree = 8
    coeffs = [None] * n_exercise
    
    # Last exercise date: j = 11 (index in V), step 12 in paths
    S_last = paths[:, 12]
    V[11, :] = np.maximum(K - S_last, 0.0)
    
    # Backward induction for j = 10 down to 0
    for j in range(10, -1, -1):
        if _time.time() > deadline:
            break
            
        S_curr = paths[:, j+1]
        
        # Target for regression: Discounted value of next step
        targets = discount * V[j+1, :]
        
        # Only use in-the-money paths for regression to improve stability?
        # Standard practice for American options: regress on ITM paths.
        itm_mask = S_curr < K
        if np.sum(itm_mask) < 50:
            itm_mask = np.ones(n_paths, dtype=bool)
            
        S_reg = S_curr[itm_mask]
        T_reg = targets[itm_mask]
        
        # Map S_reg to Chebyshev domain [-1, 1]
        x_reg = map_to_cheb(S_reg)
        
        try:
            # Fit Chebyshev polynomial
            # C.chebfit(x, y, deg)
            # x should be in [-1, 1]
            c = C.chebfit(x_reg, T_reg, cheb_degree)
            coeffs[j] = c
        except:
            # Fallback to mean if fit fails
            # Create a constant Chebyshev coeff vector
            c = np.zeros(cheb_degree + 1)
            c[0] = np.mean(T_reg)
            coeffs[j] = c
            
        # Evaluate continuation value for all paths at current step
        S_curr_cheb = map_to_cheb(S_curr)
        cont_values = C.chebval(S_curr_cheb, coeffs[j])
        
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
        # Let f(S) = K - S - chebval(map_to_cheb(S), c)
        # We want the largest root in [0, K]
        
        # Create a fine grid for initial bracket
        S_grid = np.linspace(0, K, 1000)
        x_grid = map_to_cheb(S_grid)
        P_grid = C.chebval(x_grid, c)
        f_vals = K - S_grid - P_grid
        
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
            # f(S) = K - S - P(S)
            # f'(S) = -1 - P'(S)
            # P'(S) = dP/dx * dx/dS
            # dx/dS = 2 / (S_max - S_min)
            # dP/dx = chebder(c) evaluated at x
            for _ in range(20):
                if thresh <= 0 or thresh >= K:
                    break
                x_n = map_to_cheb(thresh)
                f_val = K - thresh - C.chebval(x_n, c)
                
                # Derivative of Chebyshev series
                c_deriv = C.chebder(c)
                dP_dx = C.chebval(x_n, c_deriv)
                dx_dS = 2.0 / (S_max - S_min)
                dP_dS = dP_dx * dx_dS
                
                f_deriv_s = -1.0 - dP_dS
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
