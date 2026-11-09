import numpy as np
import time
from scipy.optimize import brentq

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Extract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    # Ensure training_paths is a numpy array
    # Shape should be (n_paths, n_times) or similar
    # The task says "4096 training paths". 
    # Usually training_paths is (n_paths, n_steps) where steps include t0?
    # "12 monthly exercise dates (no time-0 exercise)"
    # So we likely have 13 time points: t0, t1, ..., t12.
    # Or just the 12 exercise points?
    # Let's assume training_paths has shape (n_paths, n_time_steps).
    # We need to identify which columns correspond to exercise dates.
    
    paths = np.asarray(training_paths)
    if paths.ndim == 1:
        # Assume it's a single path or flat? Unlikely.
        pass
        
    n_paths, n_times = paths.shape
    
    # Determine exercise time indices
    # If n_times == 13, indices 1..12 are exercise dates.
    # If n_times == 12, indices 0..11 are exercise dates?
    # Standard convention: t0 is initial, t1..t12 are exercise.
    # Let's assume the input includes t0.
    # If n_times > n_exercise, we assume the first column is t0 and the next n_exercise are the exercise dates.
    # If n_times == n_exercise, we assume all are exercise dates (t1..t12).
    
    if n_times > n_exercise:
        exercise_indices = list(range(1, n_exercise + 1))
    else:
        exercise_indices = list(range(n_exercise))
        
    dt = T / n_exercise
    discount = np.exp(-r * dt)
    
    # V will store the value at each exercise date
    # V[i] corresponds to exercise date i (0-indexed in our list, which is t_{i+1} in real time if t0 is index 0)
    # Let's use 0-indexed for the exercise dates in our internal logic.
    # V[0] is value at t1, V[11] is value at t12.
    
    V = np.zeros((n_paths, n_exercise))
    
    # Terminal condition at t12 (index n_exercise-1)
    S_terminal = paths[:, exercise_indices[-1]]
    V[:, -1] = np.maximum(K - S_terminal, 0)
    
    # Thresholds for each exercise date
    thresholds = [0.0] * n_exercise
    
    # Backward induction
    for i in range(n_exercise - 2, -1, -1):
        S_current = paths[:, exercise_indices[i]]
        S_next = paths[:, exercise_indices[i+1]]
        
        # Continuation value: discounted expected future value
        # We regress V[:, i+1] on features of S_current
        # Features: S, S^2, S^3
        X = np.column_stack([S_current, S_current**2, S_current**3])
        
        # Add intercept
        X = np.column_stack([np.ones_like(S_current), X])
        
        # Least Squares
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, V[:, i+1], rcond=None)
            C_current = X @ coeffs
        except:
            # Fallback to simple mean if regression fails
            C_current = np.full_like(S_current, np.mean(V[:, i+1]))
            
        C_current = C_current * discount
        
        # Exercise value
        E_current = np.maximum(K - S_current, 0)
        
        # Optimal value
        V[:, i] = np.maximum(E_current, C_current)
        
        # Determine threshold
        # Exercise if E_current > C_current
        exercise_mask = E_current > C_current
        
        if np.any(exercise_mask) and np.any(~exercise_mask):
            S_ex = S_current[exercise_mask]
            S_cont = S_current[~exercise_mask]
            
            # The boundary is between max(S_ex) and min(S_cont)
            max_ex = np.max(S_ex)
            min_cont = np.min(S_cont)
            
            # Use the midpoint as a robust estimate
            threshold = (max_ex + min_cont) / 2.0
        elif np.all(exercise_mask):
            # Always exercise, threshold is infinity (or very large)
            threshold = np.inf
        elif np.all(~exercise_mask):
            # Never exercise, threshold is 0
            threshold = 0.0
        else:
            threshold = 0.0
            
        thresholds[i] = threshold
        
        if time.time() > deadline - 10:
            break
            
    # Handle the last step (t12) threshold
    # At t12, we always exercise if ITM.
    # The threshold is K.
    # But our loop went from n_exercise-2 down to 0.
    # So thresholds[0] is for t1, ..., thresholds[n_exercise-2] is for t_{n_exercise-1}.
    # We need a threshold for t12 as well?
    # The exercise function is called for any t.
    # If t is close to t12, we should use the t12 logic.
    # Actually, the policy at t12 is simply: Exercise if S < K.
    # So threshold for t12 is K.
    
    # Let's align thresholds with the exercise_indices
    # thresholds[0] corresponds to exercise_indices[0] (t1)
    # ...
    # thresholds[n_exercise-2] corresponds to exercise_indices[n_exercise-2] (t_{n_exercise-1})
    # We need a threshold for exercise_indices[n_exercise-1] (t_{n_exercise})
    # Which is K.
    
    # Let's create a list of (time, threshold) pairs
    time_thresholds = []
    for i in range(n_exercise):
        t_val = (i + 1) * dt  # Assuming t0=0, t1=dt, etc.
        if i < n_exercise - 1:
            t_val_threshold = thresholds[i]
        else:
            t_val_threshold = K
            
        time_thresholds.append((t_val, t_val_threshold))
        
    # Sort by time
    time_thresholds.sort(key=lambda x: x[0])
    
    class ExercisePolicy:
        def __init__(self, time_thresholds, K):
            self.time_thresholds = time_thresholds
            self.K = K
            
        def exercise(self, current_state, time):
            # current_state is a vector
            # time is a scalar
            # Find the appropriate threshold
            # We need to find the exercise date that corresponds to 'time'
            # The problem says "12 monthly exercise dates".
            # If 'time' is between exercise dates, the policy should be the same as the *next* exercise date?
            # Or the *current* one?
            # Usually, for Bermudan, you can only exercise at the dates.
            # The evaluator likely calls exercise(S, t) where t is one of the exercise dates.
            # If t is not exactly an exercise date, we should pick the nearest one?
            # Or the one that is "active"?
            # Given the contract, it's likely t is one of the 12 dates.
            
            # Find the closest exercise time
            # Or simply: if t is close to t_i, use threshold_i.
            
            # Let's find the index i such that t is closest to t_i
            times = [tt[0] for tt in self.time_thresholds]
            thresholds = [tt[1] for tt in self.time_thresholds]
            
            # Map time to index
            # If time is exactly t_i, use i.
            # If time is between t_i and t_{i+1}, the option is not exercisable.
            # But the function must return a boolean vector.
            # If it's not an exercise date, should it return False?
            # "exercise(current_state, time) returning a boolean vector"
            # "Bermudan put... 12 monthly exercise dates"
            # If time is not an exercise date, the policy is "do not exercise".
            # However, the evaluator might pass the exact dates.
            # Let's assume time is one of the dates.
            
            # Find the index
            min_diff = np.inf
            idx = 0
            for i, t_i in enumerate(times):
                diff = abs(time - t_i)
                if diff < min_diff:
                    min_diff = diff
                    idx = i
                    
            # If the time is not close to any exercise date, return False
            if min_diff > 1e-5:
                return np.zeros_like(current_state, dtype=bool)
                
            thresh = thresholds[idx]
            
            if np.isinf(thresh):
                return np.ones_like(current_state, dtype=bool)
            elif thresh == 0:
                return np.zeros_like(current_state, dtype=bool)
            else:
                return current_state < thresh

    return ExercisePolicy(time_thresholds, K)
