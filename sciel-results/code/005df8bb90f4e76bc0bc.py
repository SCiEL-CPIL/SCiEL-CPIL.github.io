"""Rewritten program for Bermudan exercise-policy research.
Uses a parametric smooth boundary fitted via high-resolution grid search and local refinement.
Decouples policy representation from fixed-grid backward induction bias.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using a parametric smooth boundary.
    
    Approach:
    - Represent the exercise boundary $B_i$ at exercise date $i$ as a smooth parametric function.
    - We use a flexible form: $B_i = K \cdot (1 - (1 - i/N)^\beta)$.
      This ensures $B_N = K$ and allows for a wide range of concave/convex shapes.
    - Fit the parameter $\beta$ via a coarse grid search followed by local refinement
      using Nelder-Mead to maximize the mean discounted payoff on training paths.
    """
    # Extract parameters from contract
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise_dates = contract.get('n_exercise_dates', 12)
    
    # Time step
    dt = T / n_exercise_dates
    
    # Assume training_paths is a 2D array of shape (n_paths, n_exercise_dates + 1)
    # where column 0 is the initial spot and columns 1 to n_exercise_dates are the
    # spot prices at each exercise date.
    paths = np.asarray(training_paths)
    
    if paths.ndim == 1:
        # Single path, reshape
        paths = paths.reshape(1, -1)
    
    n_paths, n_cols = paths.shape
    
    # Extract spot prices at exercise dates (columns 1 to n_exercise_dates)
    S = paths[:, 1:n_exercise_dates + 1]  # shape: (n_paths, n_exercise_dates)
    
    # Discount factors
    discount_factors = np.exp(-r * np.arange(1, n_exercise_dates + 1) * dt)
    
    # Precompute payoff
    payoff = np.maximum(K - S, 0)
    
    # Precompute normalized time indices
    i_indices = np.arange(1, n_exercise_dates + 1, dtype=float)
    normalized_i = i_indices / n_exercise_dates
    
    def compute_score(beta):
        """Compute mean discounted payoff for a given beta."""
        # B_i = K * (1 - (1 - i/N)^beta)
        # Ensure boundaries are within [0, K]
        boundaries = K * (1.0 - (1.0 - normalized_i) ** beta)
        boundaries = np.clip(boundaries, 0, K)
        
        # Exercise mask: S <= boundaries
        exercise_mask = S <= boundaries[None, :]
        
        # Find first exercise using vectorized cumsum
        cum_exercise = np.cumsum(exercise_mask, axis=1)
        first_exercise_mask = (cum_exercise == 1)
        
        # Realized payoff
        # Only count payoff if exercised for the first time at that step
        realized_payoff = payoff * first_exercise_mask * discount_factors[None, :]
        
        # Mean discounted payoff
        mean_payoff = np.mean(realized_payoff.sum(axis=1))
        return mean_payoff

    # 1. Coarse Grid Search
    # Search beta in [0.1, 10.0]
    beta_vals = np.linspace(0.1, 10.0, 100)
    
    best_score = -np.inf
    best_beta = 0.5
    
    for beta in beta_vals:
        score = compute_score(beta)
        if score > best_score:
            best_score = score
            best_beta = beta
            
    # 2. Local Refinement using Nelder-Mead
    # Define negative score for minimization
    def neg_score(beta_arr):
        beta = beta_arr[0]
        # Handle invalid beta if optimizer goes out of bounds (though bounded)
        if beta <= 0:
            return 1e9
        return -compute_score(beta)
        
    # Initial guess is the best from grid search
    x0 = np.array([best_beta])
    
    # Bounds: beta should be positive
    bounds = [(0.01, 20.0)]
    
    try:
        res = minimize(neg_score, x0, method='Nelder-Mead', bounds=bounds, options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6})
        if res.success:
            best_beta = res.x[0]
            # Verify the score improved
            new_score = compute_score(best_beta)
            if new_score > best_score:
                best_score = new_score
    except Exception:
        pass # Keep grid search result if optimization fails

    # Create the exercise policy
    boundaries = K * (1.0 - (1.0 - normalized_i) ** best_beta)
    boundaries = np.clip(boundaries, 0, K)
    
    class ExercisePolicy:
        def __init__(self, boundaries, n_exercise_dates, dt):
            self.boundaries = boundaries
            self.n_exercise_dates = n_exercise_dates
            self.dt = dt
        
        def exercise(self, current_state, time):
            """
            Determine whether to exercise the option.
            
            Parameters:
            - current_state: array of spot prices
            - time: current time (fraction of T) or exercise date index
            
            Returns:
            - boolean vector indicating whether to exercise
            """
            current_state = np.asarray(current_state)
            
            # Determine the exercise date index from time
            # Assume time is the fraction of T (0 to 1)
            # If time > 1, assume it's an index (1 to N)
            if isinstance(time, (int, float)):
                if time <= 1.0:
                    # time is fraction of T
                    # t_i = i * dt, so i = t / dt
                    idx = int(round(time / self.dt))
                else:
                    # time is already the index
                    idx = int(time)
            else:
                # time is an array
                time_arr = np.asarray(time)
                if np.max(time_arr) <= 1.0:
                    idx_arr = np.round(time_arr / self.dt).astype(int)
                else:
                    idx_arr = time_arr.astype(int)
                # If time is an array, we might need to handle vectorized logic,
                # but typically exercise() is called with a scalar time for a batch of states.
                # If time is an array, we assume it matches the shape of current_state?
                # The task says "exercise(current_state, time) returning a boolean vector".
                # Usually time is scalar for a specific step.
                # If time is array, we will try to handle it, but scalar is more likely.
                # For safety, if time is array, we return a boolean array where each element
                # is compared against the boundary for its specific time.
                # But boundaries are indexed by time.
                # Let's assume time is scalar for now as per common interface.
                # If we must support array time:
                # This is complex. Let's stick to scalar time assumption for robustness.
                # If time is array, we'll treat it as a single time step for all? No.
                # Let's assume the caller passes scalar time.
                raise ValueError("Vectorized time not supported, please pass scalar time")
            
            # Clamp index to valid range (1 to N)
            idx = np.clip(idx, 1, self.n_exercise_dates)
            
            # Get the boundary for this exercise date
            # boundaries is 0-indexed array of length N, corresponding to i=1..N
            # So boundary for index i is boundaries[i-1]
            boundary = self.boundaries[idx - 1]
            
            # Exercise if current_state <= boundary
            return current_state <= boundary

    policy = ExercisePolicy(boundaries, n_exercise_dates, dt)
    return policy

# EVOLVE-BLOCK-END
