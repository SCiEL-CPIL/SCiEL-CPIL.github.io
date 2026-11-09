"""Bermudan exercise-policy: Longstaff-Schwartz Least Squares Monte Carlo."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_steps = contract.get('n_steps', 12)
    dt = T / n_steps
    discount = np.exp(-r * dt)

    training_paths = np.asarray(training_paths, dtype=np.float64)
    n_paths = training_paths.shape[0]
    
    # Define basis functions: 1, S, S^2, S^3, S^4
    # We will fit on log(S) or S? S is standard.
    # Let's use S, S^2, S^3, S^4.
    
    # V_t is the value at time t for all paths
    V = np.maximum(K - training_paths[:, -1], 0.0) # Value at T (last step)
    
    # We need to store coefficients for each time step t=1..n_steps-1
    # Note: The problem states 12 monthly exercise dates, no time-0 exercise.
    # So exercise decisions are made at t=1, ..., 12.
    # Backward induction starts from t=11 (index 11 in 0-indexed if T is index 12?)
    # Let's clarify indices.
    # training_paths has shape (n_paths, n_steps + 1)?
    # "12 monthly exercise dates". Usually this implies 12 intervals, so 13 points (0..12).
    # Or 12 points (1..12)?
    # "no time-0 exercise".
    # Let's assume training_paths has shape (n_paths, 13) where index 0 is t=0, index 12 is t=12.
    # Exercise decisions at t=1, ..., 12.
    # Value at t=12 is intrinsic.
    # We iterate t from 11 down to 1.
    
    n_cols = training_paths.shape[1]
    if n_cols != n_steps + 1:
        # Fallback if shape is different, but assume standard
        pass
        
    coeffs = {}
    basis_funcs = {}
    
    # We need to evaluate the policy on evaluation paths.
    # The exercise method takes current_state (vector of S_t for all eval paths) and time t.
    # It returns a boolean vector.
    
    # We will store the coefficients for each t.
    # And we will compute the continuation value as phi(S) @ coeffs[t].
    
    for t in range(n_steps - 1, 0, -1):
        # t is the index of the current time step where we decide.
        # S_t is training_paths[:, t]
        # S_{t+1} is training_paths[:, t+1]
        # V is the value at t+1 (already computed in previous iteration)
        
        S_t = training_paths[:, t]
        
        # Compute features for S_t
        # Use S, S^2, S^3, S^4
        features = np.column_stack([
            np.ones_like(S_t),
            S_t,
            S_t**2,
            S_t**3,
            S_t**4
        ])
        
        # Target: Discounted V_{t+1}
        target = discount * V
        
        # Mask: Only fit on paths where exercise is possible (S_t < K for put)
        # If S_t >= K, payoff is 0, and we never exercise (since V_cont >= 0).
        # So we can exclude them from fitting to improve accuracy near boundary.
        mask = S_t < K
        
        if np.sum(mask) > 50: # Need enough points
            X_fit = features[mask]
            y_fit = target[mask]
            
            # Solve least squares: X @ c = y
            # Use np.linalg.lstsq
            c, residuals, rank, s = np.linalg.lstsq(X_fit, y_fit, rcond=None)
            coeffs[t] = c
        else:
            # Fallback: constant approximation
            c = np.zeros(5)
            c[0] = np.mean(target[mask]) if np.sum(mask) > 0 else 0.0
            coeffs[t] = c
            
        # Update V to be the value at time t
        # V_t = max(Payoff_t, Continuation_t)
        # Payoff_t = max(K - S_t, 0)
        payoff = np.maximum(K - S_t, 0.0)
        
        # Compute continuation value for ALL paths using the fitted coeffs
        # This is important because even if we didn't fit on out-of-money paths,
        # we need a continuation value for them to determine if they should be exercised
        # (though they won't be, it's good for consistency).
        cont = features @ coeffs[t]
        
        V = np.maximum(payoff, cont)
        
        # Store the decision logic?
        # We don't store decisions, we store coeffs.
        # The exercise method will recompute cont and compare to payoff.

    class ExercisePolicy:
        def __init__(self, coeffs, n_steps, K):
            self.coeffs = coeffs
            self.n_steps = n_steps
            self.K = K

        def exercise(self, current_state, time):
            if time < 1 or time > self.n_steps:
                return np.zeros(len(current_state), dtype=bool)
            
            cs = np.asarray(current_state, dtype=np.float64)
            t = int(time)
            
            if t not in self.coeffs:
                # Should not happen
                return np.zeros(len(cs), dtype=bool)
            
            # Compute features
            features = np.column_stack([
                np.ones_like(cs),
                cs,
                cs**2,
                cs**3,
                cs**4
            ])
            
            # Compute continuation value
            cont = features @ self.coeffs[t]
            
            # Compute payoff
            payoff = np.maximum(self.K - cs, 0.0)
            
            # Exercise if payoff >= continuation
            return payoff >= cont

    return ExercisePolicy(coeffs, n_steps, K)
# EVOLVE-BLOCK-END
