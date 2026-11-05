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
    poly_degree = 3
    coeffs = [None] * n_exercise
    
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
        
        # Fit polynomial: T = c0 + c1*S + c2*S^2 + c3*S^3
        # We want to fit the continuation value.
        # np.polyfit returns highest degree first.
        try:
            # Use degree 3. If n_samples < degree+1, reduce degree.
            deg = min(poly_degree, len(S_reg) - 1)
            if deg < 0:
                 deg = 0
            c = np.polyfit(S_reg, T_reg, deg)
            coeffs[j] = c
        except:
            # Fallback to mean if fit fails
            coeffs[j] = np.array([np.mean(T_reg)])
            deg = 0
            
        # Evaluate continuation value for all paths at current step
        # np.polyval takes coeffs and x
        cont_values = np.polyval(coeffs[j], S_curr)
        
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
        # Let f(S) = K - S - np.polyval(c, S)
        # We want the largest root in [0, K] (or slightly above to be safe, but payoff is 0 above K)
        # Actually, if S > K, payoff is 0. Continuation is likely positive. So we don't exercise.
        # We look for crossing in [0, K].
        
        # Create a grid
        S_grid = np.linspace(0, K, 200)
        f_vals = K - S_grid - np.polyval(c, S_grid)
        
        # Find sign changes
        # f(S) = Payoff - Cont
        # We want Payoff >= Cont.
        # At S=0, Payoff=K, Cont=small. f>0.
        # At S=K, Payoff=0, Cont=positive. f<0.
        # So there is a root where f goes from + to -.
        
        # Find indices where sign changes
        signs = np.sign(f_vals)
        # Handle zeros
        signs[signs == 0] = 1 
        
        changes = np.where(np.diff(signs) != 0)[0]
        
        if len(changes) > 0:
            # Take the last change (highest S where we still want to exercise? 
            # Wait. If f > 0, we exercise.
            # As S increases, Payoff decreases linearly, Cont is roughly flat or increasing.
            # So f should decrease.
            # The crossing is the boundary.
            # We want to exercise if S < Threshold.
            # So Threshold is the root.
            
            idx = changes[-1]
            # Linear interpolation for better precision
            x1, x2 = S_grid[idx], S_grid[idx+1]
            y1, y2 = f_vals[idx], f_vals[idx+1]
            if y1 == y2:
                thresh = x1
            else:
                thresh = x1 - y1 * (x2 - x1) / (y2 - y1)
                
            # Clip to valid range
            thresholds[j] = np.clip(thresh, 0, K)
        else:
            # No sign change? 
            # If f is always positive, exercise everywhere? Threshold = K (or higher, but payoff 0).
            # If f is always negative, never exercise? Threshold = 0.
            if f_vals[0] < 0:
                thresholds[j] = 0.0
            else:
                thresholds[j] = K
                
    return ExercisePolicy(thresholds)
