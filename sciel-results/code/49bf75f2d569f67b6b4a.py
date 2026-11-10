"""Bermudan exercise-policy research: direct regression for continuation value."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using regression on training paths.
    
    Parameters:
    -----------
    training_paths : ndarray of shape (N, 13) where columns are S_0, S_1, ..., S_12
                     (13 columns: initial spot + 12 monthly steps)
    contract : dict with keys 'K', 'r', 'sigma', 'T', 'num_steps' (optional)
    
    Returns:
    --------
    An object with method exercise(current_state, time) -> boolean vector
    """
    start_time = time.time()
    deadline = start_time + 100.0  # 100s safety margin out of 120s
    
    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    num_steps = contract.get('num_steps', 12)
    
    # Ensure training_paths is 2D
    training_paths = np.asarray(training_paths, dtype=np.float64)
    if training_paths.ndim == 1:
        training_paths = training_paths.reshape(1, -1)
    
    N, total_cols = training_paths.shape
    
    # Determine indexing: expect 13 columns (S_0 through S_12)
    # Exercise dates are at steps 1..12 (indices 1..12 in the array)
    # If we have 12 columns, they might be S_1..S_12
    if total_cols == num_steps + 1:
        # Columns: 0=S_0, 1=S_1, ..., 12=S_12
        s0 = training_paths[:, 0]
        s_ex = training_paths[:, 1:num_steps+1]  # shape (N, 12)
    elif total_cols == num_steps:
        # Columns: 0=S_1, 1=S_2, ..., 11=S_12
        s_ex = training_paths  # shape (N, 12)
    else:
        # Fallback: use last 12 columns
        s_ex = training_paths[:, -num_steps:]
    
    # Time step in years
    dt = T / num_steps
    discount = np.exp(-r * dt)
    
    # We'll compute continuation values backward using regression
    # For each exercise date i (from 12 down to 1), we want to decide:
    # Exercise if intrinsic >= continuation value
    
    # Strategy: Use a flexible function approximator (polynomial regression)
    # to estimate the continuation value at each step.
    
    # For each step i, the continuation value is:
    # C_i(S_i) = E[discount * max(intrinsic_{i+1}, C_{i+1}(S_{i+1})) | S_i]
    
    # We'll do this by:
    # 1. Start from the last step: C_12(S_12) = intrinsic_12(S_12) = max(K - S_12, 0)
    # 2. For step i, we need to estimate E[discount * max(intrinsic_{i+1}, C_{i+1}(S_{i+1})) | S_i]
    #    We can do this by regressing the discounted future payoff on S_i
    
    # Let's work with the training paths to estimate these expectations.
    # For each path n and step i, we know S_i and S_{i+1}.
    # We'll use the fact that under GBM, S_{i+1} is conditionally lognormal given S_i.
    
    # Better approach: Use the training data to directly estimate the optimal
    # stopping boundary via regression.
    
    # Let's define the payoff at each step:
    # intrinsic_i(S_i) = max(K - S_i, 0)
    
    # We'll compute the continuation values backward:
    # At step 12 (last exercise date): 
    #   The value if we hold past step 12 is 0 (or we exercise)
    #   So V_12(S_12) = max(K - S_12, 0)
    
    # At step i:
    #   V_i(S_i) = max(K - S_i, E[discount * V_{i+1}(S_{i+1}) | S_i])
    
    # To estimate E[discount * V_{i+1}(S_{i+1}) | S_i], we can use regression.
    # For each training path, we have (S_i, S_{i+1}) pairs.
    # We can regress V_{i+1}(S_{i+1}) on S_i using a flexible model.
    
    # Let's use polynomial regression with a reasonable degree.
    # We'll use features: [1, S_i, S_i^2, S_i^3, S_i^4, S_i^5]
    
    def make_features(s):
        """Create polynomial features up to degree 5."""
        s = np.asarray(s, dtype=np.float64)
        return np.column_stack([
            np.ones_like(s),
            s,
            s**2,
            s**3,
            s**4,
            s**5
        ])
    
    # Start from the last step
    # V_12(S_12) = max(K - S_12, 0)
    v_next = np.maximum(K - s_ex[:, -1], 0.0)  # shape (N,)
    
    # Store the decision boundaries
    # We'll store the continuation value function as a polynomial fit
    # For each step i, we store the coefficients of the polynomial
    # that approximates E[discount * V_{i+1}(S_{i+1}) | S_i]
    
    # Actually, we need to be more careful. Let's think about this differently.
    
    # At each step i, given S_i, we want to know:
    # - Intrinsic value: max(K - S_i, 0)
    # - Continuation value: E[discount * V_{i+1}(S_{i+1}) | S_i]
    
    # We'll estimate the continuation value by regressing the discounted
    # future values on S_i.
    
    # Let's store the polynomial coefficients for each step's continuation value
    # poly_coeffs[i] = coefficients for step i (0-indexed for steps 1..12)
    
    poly_coeffs = []
    
    # Process from last step to first
    for i in range(num_steps - 1, -1, -1):
        if time.time() > deadline:
            # Return a simple heuristic policy
            break
        
        # At step i, we have S_i = s_ex[:, i]
        # The next step's value is v_next (which is V_{i+1}(S_{i+1}))
        # We want to estimate E[discount * v_next | S_i]
        
        # Use the training data: for each path n, we have (S_i[n], v_next[n])
        s_i = s_ex[:, i]
        target = discount * v_next
        
        # Fit polynomial regression
        X = make_features(s_i)
        
        # Use least squares
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(X, target, rcond=None)
        except:
            coeffs = np.zeros(X.shape[1])
        
        poly_coeffs.append(coeffs)
        
        # Compute the continuation value at step i for each training path
        cont_val = X @ coeffs
        
        # The value at step i is max(intrinsic, continuation)
        intrinsic = np.maximum(K - s_i, 0.0)
        v_current = np.maximum(intrinsic, cont_val)
        
        # For the next iteration, v_next should be v_current
        v_next = v_current
    
    # Now we have poly_coeffs for each step
    # poly_coeffs[0] is for step 1 (first exercise date)
    # poly_coeffs[11] is for step 12 (last exercise date)
    
    # Wait, I need to be more careful with the indexing.
    # Let me redo this.
    
    # Actually, let me reconsider the approach. The issue is that at the last step,
    # there's no future to continue to. So:
    # - At step 12: V_12(S_12) = max(K - S_12, 0) (must exercise or get 0)
    # - At step 11: V_11(S_11) = max(K - S_11, E[discount * V_12(S_12) | S_11])
    # - ...
    # - At step 1: V_1(S_1) = max(K - S_1, E[discount * V_2(S_2) | S_1])
    
    # Let me redo the backward induction properly.
    
    poly_coeffs = [None] * num_steps  # index 0 for step 1, index 11 for step 12
    
    # Start from the last step (step 12, index 11)
    # At step 12, the continuation value is 0 (no future)
    # So V_12(S_12) = max(K - S_12, 0)
    # For the exercise decision at step 12, we always exercise if intrinsic > 0
    
    # For step 11 (index 10):
    # We need E[discount * V_12(S_12) | S_11]
    # V_12(S_12) = max(K - S_12, 0)
    
    s_ex_last = s_ex[:, -1]  # S_12
    v_step12 = np.maximum(K - s_ex_last, 0.0)
    
    # For each step from 11 down to 1:
    for i in range(num_steps - 1, 0, -1):
        if time.time() > deadline:
            break
        
        # i is the step index (1-based), so s_ex[:, i-1] is S_i
        # The next step is i+1, so s_ex[:, i] is S_{i+1}
        
        s_i = s_ex[:, i-1]  # S_i
        s_next = s_ex[:, i]  # S_{i+1}
        
        # v_next is V_{i+1}(S_{i+1}), which we computed in the previous iteration
        # For i=11, v_next = v_step12
        # For i<11, v_next was computed at the end of the previous iteration
        
        if i == num_steps - 1:
            v_next = v_step12
        else:
            # v_next was computed at the end of the previous iteration
            pass
        
        # Estimate E[discount * v_next | S_i] by regressing on S_i
        target = discount * v_next
        X = make_features(s_i)
        
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(X, target, rcond=None)
        except:
            coeffs = np.zeros(X.shape[1])
        
        poly_coeffs[i-1] = coeffs  # Store for step i (0-indexed as i-1)
        
        # Compute continuation value at step i
        cont_val = X @ coeffs
        
        # Intrinsic value at step i
        intrinsic = np.maximum(K - s_i, 0.0)
        
        # Value at step i
        v_current = np.maximum(intrinsic, cont_val)
        
        # For the next iteration (step i-1), v_next should be v_current
        v_next = v_current
    
    # Handle step 1 separately if not done
    # Actually, the loop goes from i=num_steps-1 down to i=1, so step 1 is handled.
    
    # For the exercise decision at step 12 (last step), we always exercise if intrinsic > 0
    # For steps 1-11, we exercise if intrinsic >= continuation value
    
    class BermudanPolicy:
        def __init__(self, K, r, sigma, T, num_steps, poly_coeffs, s_ex, deadline):
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.num_steps = num_steps
            self.poly_coeffs = poly_coeffs
            self.s_ex = s_ex
            self.deadline = deadline
            
        def exercise(self, current_state, time):
            """
            Determine whether to exercise the Bermudan put.
            
            Parameters:
            -----------
            current_state : ndarray of shape (M,) - current spot prices
            time : int or float - current time step (1-based, 1 to 12)
            
            Returns:
            --------
            boolean vector of shape (M,) - True if should exercise
            """
            current_state = np.asarray(current_state, dtype=np.float64)
            
            # time is 1-based (1 to 12)
            if time is None:
                return np.zeros(len(current_state), dtype=bool)
            
            t = int(time)
            
            if t < 1 or t > self.num_steps:
                return np.zeros(len(current_state), dtype=bool)
            
            # At the last step (t=12), always exercise if intrinsic > 0
            if t == self.num_steps:
                intrinsic = np.maximum(self.K - current_state, 0.0)
                return intrinsic > 0
            
            # For steps 1-11, compare intrinsic to continuation value
            intrinsic = np.maximum(self.K - current_state, 0.0)
            
            # Get the polynomial coefficients for this step
            coeffs = self.poly_coeffs[t-1]
            
            if coeffs is None:
                # Fallback: exercise if intrinsic > 0
                return intrinsic > 0
            
            # Compute continuation value
            X = make_features(current_state)
            cont_val = X @ coeffs
            
            # Exercise if intrinsic >= continuation value
            return intrinsic >= cont_val
    
    # Check if we have valid coefficients
    if all(c is None for c in poly_coeffs):
        # Fallback to a simple heuristic
        class SimplePolicy:
            def __init__(self, K, num_steps):
                self.K = K
                self.num_steps = num_steps
            
            def exercise(self, current_state, time):
                current_state = np.asarray(current_state, dtype=np.float64)
                t = int(time)
                if t < 1 or t > self.num_steps:
                    return np.zeros(len(current_state), dtype=bool)
                intrinsic = np.maximum(self.K - current_state, 0.0)
                return intrinsic > 0
        
        return SimplePolicy(K, num_steps)
    
    return BermudanPolicy(K, r, sigma, T, num_steps, poly_coeffs, s_ex, deadline)

# EVOLVE-BLOCK-END
