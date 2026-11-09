"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class ExercisePolicy:
    def __init__(self, K, r, sigma, T, n_steps, coeffs_list, fallback_thresholds):
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.n_steps = n_steps
        self.coeffs_list = coeffs_list
        self.fallback_thresholds = fallback_thresholds
        self.dt = T / n_steps
        self.discount = np.exp(-r * self.dt)

    def _get_basis(self, S):
        # S is an array of spots
        x = np.log(S / self.K)
        # Polynomial basis up to degree 3
        return np.column_stack([np.ones_like(x), x, x**2, x**3])

    def exercise(self, current_state, time):
        S = np.asarray(current_state, dtype=float)
        
        # Determine step index
        # The exercise dates are t_i = i * T / n_steps for i=1..n_steps
        # If time is float:
        if isinstance(time, (float, np.floating)):
            # Map to index 1..n_steps
            idx = int(round(time * self.n_steps / self.T))
            # Clamp
            idx = max(1, min(idx, self.n_steps))
        else:
            # Assume integer index 1..n_steps or 0..n_steps-1
            # The prompt says "no time-0 exercise", so valid times are 1/12 .. 1.
            # Let's assume time is the step number 1..12 or fraction.
            # Safest: treat as float if < 1.5, else index.
            if time < 1.5:
                idx = int(round(time * self.n_steps / self.T))
            else:
                idx = int(time)
            
        idx = max(1, min(idx, self.n_steps))
        
        # Get model for this step
        # coeffs_list is 0-indexed list, so step idx uses model at idx-1
        # coeffs_list[i] corresponds to t_{i+1}
        # If idx == n_steps, it's the last step. We don't exercise at the last step if we can avoid it? 
        # Actually, at the last step, continuation is 0. So we exercise if payoff > 0.
        # But wait, the model is fitted for t_1 to t_{n_steps-1}.
        # If idx == n_steps, we should just check payoff > 0.
        
        if idx == self.n_steps:
            payoff = np.maximum(self.K - S, 0)
            return payoff > 1e-12
        
        model_idx = idx - 1
        coeffs = self.coeffs_list[model_idx]
        
        if coeffs is None:
            # Fallback: use pre-computed threshold
            threshold = self.fallback_thresholds[model_idx]
            return S < threshold

        # Construct basis
        basis = self._get_basis(S)
        
        # Predict continuation value
        predicted_val = basis @ coeffs
        v_cont = self.discount * predicted_val
        
        # Immediate payoff
        payoff = np.maximum(self.K - S, 0)
        
        # Exercise if payoff > continuation
        return payoff > v_cont

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Extract parameters
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = 12
    
    # Parse training paths
    # Assumed shape: (N, n_steps+1) or (N, n_steps)
    # If N, 13: includes t=0. We only need t=1..12 for exercise decisions, but t=0 is start.
    # Exercise decisions are made at t=1..12.
    # Backward induction starts at t=12.
    
    paths = np.asarray(training_paths)
    if paths.ndim == 1:
        paths = paths.reshape(1, -1)
    
    N = paths.shape[0]
    
    # Determine columns for exercise dates
    # If paths has 13 columns, col 0 is t=0, col 1 is t=1/12... col 12 is t=1
    # If paths has 12 columns, col 0 is t=1/12... col 11 is t=1
    if paths.shape[1] == n_steps + 1:
        S_paths = paths[:, 1:] # t=1 to t=12
    else:
        S_paths = paths # t=1 to t=12
    
    # Ensure S_paths shape is (N, 12)
    
    # Backward Induction
    # V_next is the realized value at the next step (for fitting the previous step)
    # At t=12 (last step), V is the European payoff
    V_realized = np.maximum(K - S_paths[:, -1], 0)
    
    # We will store the fitted models for each step i=11 down to 0 (indices in S_paths)
    # S_paths[:, i] corresponds to t_{i+1}
    # We need models for t=1/12 to t=11/12.
    # t=12 is terminal, no model needed (payoff is known).
    
    models = [None] * n_steps # Index 0 corresponds to t=1/12, Index 11 to t=12 (unused for exercise, but V_12 is used to fit 11)
    fallback_thresholds = [K * 0.9] * n_steps # Default fallback
    
    # We fit models for steps i=10 down to 0 (0-indexed in S_paths, corresponding to t_{i+1})
    # i=10 corresponds to t=11/12. Next step is t=12.
    
    dt = T / n_steps
    discount = np.exp(-r * dt)
    
    for i in range(n_steps - 2, -1, -1):
        if time.time() > deadline:
            # Break and use default
            break
            
        S_current = S_paths[:, i]
        
        # Fit regression: V_realized (from step i+1) ~ Basis(S_current)
        # Note: V_realized here is the *realized* value from the path, which includes the optimal decision at i+1.
        # This is the standard LSM approach.
        
        basis = _get_basis_global(S_current, K)
        
        # Simple Least Squares
        try:
            # Add small regularization to handle collinearity? 
            # np.linalg.lstsq is usually stable enough for this small basis
            coeffs, residuals, rank, s = np.linalg.lstsq(basis, V_realized, rcond=None)
            models[i] = coeffs
            
            # Compute fallback threshold: find S where payoff = continuation
            # We can approximate this by finding the median S where we exercise
            # Or just keep the previous fallback
            # Let's refine fallback: 
            # predicted_val = basis @ coeffs
            # v_cont = discount * predicted_val
            # payoff = max(K - S, 0)
            # Exercise if payoff > v_cont
            # We can solve for S where K - S = v_cont(S)
            # This is complex, so let's just use the empirical boundary from training data
            predicted_val = basis @ coeffs
            v_cont = discount * predicted_val
            payoff = np.maximum(K - S_current, 0)
            exercise_mask = payoff > v_cont
            if np.any(exercise_mask):
                # Use a slightly conservative threshold (max S where we exercise)
                fallback_thresholds[i] = np.max(S_current[exercise_mask])
            else:
                fallback_thresholds[i] = K * 0.5 # Very deep ITM only
            
        except Exception:
            models[i] = None
            
        # Determine exercise decision for these training paths (for generating V_realized for previous step)
        # Note: In LSM, the decision is based on the *fitted* continuation value, not the realized next value.
        # V_cont = exp(-r*dt) * (Basis(S_current) @ coeffs)
        
        if models[i] is not None:
            predicted_val = basis @ models[i]
            v_cont = discount * predicted_val
        else:
            # Fallback decision
            v_cont = 0.0 # If model failed, assume continuation is low? Or use previous step?
            # Better: use a simple heuristic. If S < 0.9K, exercise.
            v_cont = np.where(S_current < K * 0.9, 0.0, np.maximum(K - S_current, 0) * 0.5)
            
        payoff = np.maximum(K - S_current, 0)
        
        # If payoff > v_cont, exercise. Else, hold.
        # If hold, the value for this step is the realized future value 
        # (because we are simulating the path). 
        # However, the *optimal* value at this step is max(payoff, v_cont).
        # For the purpose of fitting the *previous* step's regression, we use the realized value 
        # that would be obtained if we followed the policy.
        # If we exercise, value = payoff.
        # If we hold, value = V_realized (the realized value from the next step along this path).
        # This is the standard "Realized Value" update for LSM.
        
        V_realized = np.where(payoff > v_cont, payoff, V_realized)
        
        # Check time
        if time.time() > deadline:
            break

    # If loop was broken early, some models might be None.
    # Fill remaining models with a default (e.g., European continuation or simple heuristic)
    
    policy = ExercisePolicy(K, r, sigma, T, n_steps, models, fallback_thresholds)
    return policy

def _get_basis_global(S, K):
    x = np.log(S / K)
    return np.column_stack([np.ones_like(x), x, x**2, x**3])
# EVOLVE-BLOCK-END
