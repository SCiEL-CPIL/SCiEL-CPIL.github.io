"""Bermudan put exercise policy via Monte Carlo Longstaff-Schwartz regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    # Extract parameters from contract
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    # Exercise dates: 12 monthly, no time-0
    dt = T / n_exercise
    exercise_times = np.arange(1, n_exercise + 1) * dt
    
    # Get training paths
    # training_paths shape: (n_paths, n_timesteps) or similar
    # We need to extract states at exercise dates
    n_paths = training_paths.shape[0]
    
    # Determine which columns correspond to exercise dates
    # Assume training_paths has columns for each time step including time 0
    # Exercise dates are at indices 1, 2, ..., n_exercise (0-indexed from time 0)
    # If paths have n_exercise+1 columns (time 0 through T), use indices 1..n_exercise
    
    # Extract states at exercise dates
    # training_paths shape should be (n_paths, n_time_points)
    # Exercise at time points 1, 2, ..., n_exercise (0-indexed)
    
    # Get the states at each exercise date
    # Index 0 is time 0, index 1 is first exercise, etc.
    exercise_states = []
    for i in range(1, n_exercise + 1):
        if i < training_paths.shape[1]:
            exercise_states.append(training_paths[:, i])
        else:
            # If not enough columns, use last available
            exercise_states.append(training_paths[:, -1])
    
    exercise_states = np.array(exercise_states)  # shape: (n_exercise, n_paths)
    
    # Discount factors
    discount_factors = np.exp(-r * exercise_times)
    
    # Payoff at maturity (last exercise date)
    final_state = exercise_states[-1]
    final_payoff = np.maximum(K - final_state, 0.0)
    
    # Backward induction using Longstaff-Schwartz
    # Continue value starts at final payoff
    continue_value = final_payoff.copy()
    
    # Store exercise decisions
    exercise_decisions = np.zeros((n_exercise, n_paths), dtype=bool)
    
    # Polynomial basis functions for regression
    def get_basis_features(S):
        """Create polynomial basis features for regression."""
        # Use log-transformed spot and polynomial terms
        S_log = np.log(S + 1e-10)
        S_scaled = S / K
        features = np.column_stack([
            np.ones_like(S),
            S_scaled,
            S_log,
            S_scaled ** 2,
            S_log ** 2,
            S_scaled * S_log,
            S_scaled ** 3,
            S_log ** 3,
        ])
        return features
    
    # Backward induction from T-1 to 1
    for i in range(n_exercise - 1, 0, -1):
        if time.time() > deadline - 5.0:
            break
        
        current_state = exercise_states[i]
        intrinsic_value = np.maximum(K - current_state, 0.0)
        
        # Discounted continue value from next step
        disc_continue = np.exp(-r * dt) * continue_value
        
        # Determine which paths are "in the money" (exercise candidates)
        in_money_mask = intrinsic_value > 0
        
        if np.sum(in_money_mask) < 10:
            # Not enough in-the-money paths, use all paths for regression
            train_mask = np.ones(n_paths, dtype=bool)
        else:
            # Use in-the-money paths for regression
            train_mask = in_money_mask
        
        # Fit regression: continue_value ~ features(current_state)
        # Only fit on paths that will continue (not exercised)
        # For LS method: fit on in-the-money paths, predict for all
        
        X_train = get_basis_features(current_state[train_mask])
        y_train = disc_continue[train_mask]
        
        # Ridge regression for stability
        alpha = 1.0  # regularization parameter
        try:
            # Solve (X^T X + alpha*I) beta = X^T y
            XTX = X_train.T @ X_train + alpha * np.eye(X_train.shape[1])
            XTy = X_train.T @ y_train
            beta = np.linalg.solve(XTX, XTy)
        except np.linalg.LinAlgError:
            # Fallback: use simple threshold
            beta = None
        
        if beta is not None:
            X_all = get_basis_features(current_state)
            predicted_continue = X_all @ beta
        else:
            # Fallback: use average continue value
            predicted_continue = np.full(n_paths, np.mean(disc_continue))
        
        # Exercise decision: exercise if intrinsic > predicted continue
        exercise_now = intrinsic_value > predicted_continue
        exercise_decisions[i] = exercise_now
        
        # Update continue value for next iteration
        # If exercised, value is intrinsic; if not, value is predicted continue
        continue_value = np.where(exercise_now, intrinsic_value, predicted_continue)
    
    # At time 0, we don't exercise (no time-0 exercise per contract)
    # So the value at time 0 is the expected discounted continue value
    
    # Create exercise policy object
    class ExercisePolicy:
        def __init__(self, exercise_decisions, exercise_states, K, r, sigma, T, n_exercise):
            self.exercise_decisions = exercise_decisions
            self.exercise_states = exercise_states
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.n_exercise = n_exercise
            self.dt = T / n_exercise
            self.exercise_times = np.arange(1, n_exercise + 1) * self.dt
            
            # Learn a simple threshold-based policy for evaluation
            # For each exercise date, learn the threshold spot price
            self.thresholds = self._learn_thresholds()
        
        def _learn_thresholds(self):
            """Learn exercise thresholds from training data."""
            thresholds = []
            for i in range(self.n_exercise):
                if i >= len(self.exercise_decisions):
                    break
                
                decisions = self.exercise_decisions[i]
                states = self.exercise_states[i]
                
                # For a put, exercise when S < threshold
                # Find the threshold as the maximum S where we exercise
                if np.any(decisions):
                    exercising_states = states[decisions]
                    threshold = np.max(exercising_states)
                else:
                    threshold = 0.0
                
                thresholds.append(threshold)
            
            # If no thresholds learned, use intrinsic value heuristic
            if not thresholds:
                thresholds = [self.K * 0.8] * self.n_exercise
            
            return thresholds
        
        def exercise(self, current_state, time):
            """
            Determine exercise decision given current state and time.
            
            Args:
                current_state: array of spot prices
                time: current time (continuous)
            
            Returns:
                boolean vector indicating which paths to exercise
            """
            # Determine which exercise date we're at
            # Exercise dates are at dt, 2*dt, ..., T
            # Find the closest exercise date
            
            if time <= 0:
                # No exercise at time 0
                return np.zeros(len(current_state), dtype=bool)
            
            # Find which exercise period we're in
            # time is in [0, T], exercise dates at dt, 2dt, ..., T
            idx = int(np.floor(time / self.dt))
            idx = min(max(idx, 0), self.n_exercise - 1)
            
            # Use the learned threshold for this exercise date
            if idx < len(self.thresholds):
                threshold = self.thresholds[idx]
                # For a put, exercise when S < threshold
                decision = current_state < threshold
            else:
                # Fallback: exercise if in the money
                decision = current_state < self.K
            
            return decision
    
    return ExercisePolicy(exercise_decisions, exercise_states, K, r, sigma, T, n_exercise)
# EVOLVE-BLOCK-END
