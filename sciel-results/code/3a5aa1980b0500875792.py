import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fits a Bermudan exercise policy using Monte Carlo stochastic gradient ascent
    on exercise threshold parameters.
    Returns an object with an 'exercise' method.
    """
    start_time = time.time()
    deadline = 140.0  # 140s safety margin

    # Extract parameters from contract
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_exercises = 12
    
    if hasattr(contract, 'K'): K = contract.K
    if hasattr(contract, 'r'): r = contract.r
    if hasattr(contract, 'sigma'): sigma = contract.sigma
    if hasattr(contract, 'T'): T = contract.T
    if hasattr(contract, 'n_exercises'): n_exercises = contract.n_exercises
    if hasattr(contract, 'exercise_dates'):
        exercise_dates = np.array(contract.exercise_dates)
    else:
        exercise_dates = np.linspace(0, T, n_exercises + 1)[1:]

    exercise_dates = np.sort(exercise_dates)
    n_paths = training_paths.shape[0]

    # Initialize boundaries with a reasonable guess
    # For a put, boundaries increase over time (exercise region shrinks as maturity approaches)
    t_norm = exercise_dates / T
    boundaries = K * (0.55 + 0.40 * t_norm)

    # Precompute discount factors for each exercise date
    disc_factors = np.exp(-r * exercise_dates)

    # Fast vectorized payoff computation
    def compute_payoff(bnds):
        payoffs = np.zeros(n_paths)
        not_exercised = np.ones(n_paths, dtype=bool)
        
        for i in range(n_exercises):
            if not np.any(not_exercised):
                break
            S_i = training_paths[:, i]
            exercise_now = (S_i < bnds[i]) & not_exercised
            if np.any(exercise_now):
                payoffs[exercise_now] = (K - S_i[exercise_now]) * disc_factors[i]
                not_exercised[exercise_now] = False
        
        if np.any(not_exercised):
            S_T = training_paths[:, -1]
            payoff_T = np.maximum(K - S_T, 0.0)
            payoffs[not_exercised] = payoff_T[not_exercised] * disc_factors[-1]
        
        return payoffs

    # Stochastic gradient ascent with central differences
    lr = 1.0
    n_iterations = 50
    eps = 0.5

    for it in range(n_iterations):
        if time.time() > deadline:
            break
        
        grads = np.zeros(n_exercises)
        
        for i in range(n_exercises):
            if time.time() > deadline:
                break
            
            bnds_plus = boundaries.copy()
            bnds_plus[i] += eps
            bnds_minus = boundaries.copy()
            bnds_minus[i] -= eps
            
            mean_plus = np.mean(compute_payoff(bnds_plus))
            mean_minus = np.mean(compute_payoff(bnds_minus))
            
            grads[i] = (mean_plus - mean_minus) / (2.0 * eps)
        
        boundaries += lr * grads
        boundaries = np.clip(boundaries, 1.0, K * 1.5)

    # Create the policy object
    class ExercisePolicy:
        def __init__(self, boundaries, exercise_dates, K, r, T):
            self.boundaries = boundaries
            self.exercise_dates = exercise_dates
            self.K = K
            self.r = r
            self.T = T
            
        def exercise(self, current_state, time):
            idx = np.argmin(np.abs(self.exercise_dates - time))
            if np.abs(self.exercise_dates[idx] - time) < 1e-6:
                boundary = self.boundaries[idx]
                return current_state < boundary
            else:
                return np.zeros(len(current_state), dtype=bool)

    return ExercisePolicy(boundaries, exercise_dates, K, r, T)
