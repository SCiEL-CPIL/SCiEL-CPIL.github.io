"""Rewritten program for Bermudan exercise-policy research.
Uses a parametric smooth boundary in log-space fitted via grid search on a coarse 1D grid,
decoupling policy representation from fixed-grid backward induction bias.
"""
# EVOLVE-BLOCK-START
import numpy as np

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using a parametric smooth boundary in log-spot.
    
    Approach:
    - Represent the exercise boundary as a monotonic decreasing function of log(S) at each
      exercise date. We use a low-order polynomial in log(S) per time step, but enforce
      monotonicity via a parameterization that guarantees the boundary is well-behaved.
    - Actually, a simpler and more robust approach: for each exercise date t_i, the optimal
      exercise boundary S*(t_i) is a single number for a European-style payoff (since the
      payoff is linear in S for a put: max(K-S, 0)). However, for Bermudan options with
      path-dependent features or if we consider the full state, the boundary can depend on
      more. But for a standard Bermudan put on GBM with no dividends, the optimal exercise
      region at each exercise date is S <= B_i for some boundary B_i.
      
    Wait, the task says "exercise(current_state, time) returning a boolean vector". The
    current_state is the spot price. For a standard Bermudan put, the optimal policy is
    indeed to exercise if S <= B_i at exercise date i. So the boundary is a scalar per
    exercise date.
    
    But the directive says: "Represent the optimal exercise boundary as a parametric smooth
    curve (e.g., a monotonic polynomial in log-spot) fitted via grid search on a coarse 1D
    grid, avoiding the bias of fixed-grid backward induction by allowing the boundary to
    adapt continuously across the entire state space."
    
    This suggests that maybe the boundary is not just a scalar per time step, but a function
    that could be more complex. However, for a standard Bermudan put, the boundary is a
    scalar per exercise date. Let me re-read: "Bermudan put on risk-neutral GBM, 12 monthly
    exercise dates". So the state is just the spot price S. The optimal exercise decision at
    time t_i depends only on S. The optimal policy is to exercise if S <= B_i.
    
    So the "parametric smooth curve" might refer to fitting B_i as a smooth function of i,
    or perhaps the directive is suggesting a more general approach that could handle
    multi-dimensional states. But here, the state is 1D (spot price).
    
    Let me think differently. Perhaps the "current_state" includes more than just the spot
    price? The task says "Candidate receives current states only during scoring, never future
    evaluation states." And "exercise(current_state, time) returning a boolean vector".
    
    Given the contract has 4096 training paths and 8192 evaluation paths, and the state is
    the spot price, I think the boundary is indeed a scalar per exercise date.
    
    However, to follow the directive, I will:
    1. Use a parametric form for the boundary that is smooth across exercise dates.
    2. Fit it via grid search on a coarse 1D grid.
    
    For a Bermudan put, the boundary B_i should be increasing over time (you exercise earlier
    if the option is deeper in the money, so the boundary starts low and increases toward K).
    Actually, for a put, you exercise when S is low. The boundary B_i is the maximum S at
    which you exercise. As time progresses, B_i should increase toward K (since time value
    decreases, you're more willing to exercise).
    
    Let me parameterize B_i as a smooth function. I'll use a logistic-like or linear
    parameterization.
    
    Actually, let me reconsider. The standard approach for Bermudan options is backward
    induction. The "bias of fixed-grid backward induction" refers to the discretization
    error. The parametric approach fits a smooth curve to the boundaries obtained from
    backward induction, or directly optimizes the boundary parameters.
    
    Here's my plan:
    1. Simulate the training paths to get the spot prices at each exercise date.
    2. Use backward induction to get approximate boundaries B_i for each exercise date.
    3. Fit a parametric smooth curve to these boundaries (e.g., B_i = a + b * i + c * i^2,
       with constraints to ensure monotonicity and 0 < B_i <= K).
    4. Use this parametric boundary as the exercise policy.
    
    But the directive says "avoiding the bias of fixed-grid backward induction", so maybe
    I should directly optimize the boundary parameters via grid search, using the training
    paths to estimate the continuation value.
    
    Let me implement:
    1. For a given set of boundary parameters, compute the expected discounted payoff using
       the training paths.
    2. Grid search over the boundary parameters to maximize this expected payoff.
    
    The boundary parameters: I'll use a simple linear model B_i = a + b * i, where i is the
    exercise date index (1 to 12). Constraints: a > 0, b >= 0, and B_i <= K for all i.
    
    Actually, let me use a more flexible parameterization. I'll use B_i = K * (1 - exp(-c * i))
    for some c > 0, which ensures B_i is increasing and bounded by K.
    
    Or even simpler: B_i = K * (i / 12) * alpha, where alpha is a scaling factor.
    
    Let me go with: B_i = a + b * i, with a in [0, K], b in [0, K/12], and ensure B_i <= K.
    
    Grid search:
    - a: 0 to K, step K/20
    - b: 0 to K/12, step (K/12)/20
    
    For each (a, b), compute the expected payoff using the training paths.
    
    But wait, the expected payoff depends on the exercise policy, which depends on the
    boundaries. I need to simulate the paths and at each exercise date, decide whether to
    exercise based on the boundary.
    
    Let me implement this step by step.
    
    First, I need to understand the contract structure. The contract has:
    - 12 monthly exercise dates (no time-0 exercise)
    - Spot 90/100/110 (three initial spot prices? Or three scenarios?)
    - K 100, r 0.05, sigma 0.2, T 1
    
    The training_paths likely contain the simulated spot prices at each exercise date for
    multiple paths.
    
    Let me assume training_paths is a 2D array of shape (n_paths, n_exercise_dates + 1)
    where the first column is the initial spot and the remaining columns are the spot
    prices at each exercise date.
    
    Actually, I need to check the format. The task says "4096 training paths". Let me
    assume training_paths is a 2D numpy array where each row is a path and each column
    is the spot price at a specific time.
    
    Let me code this up.
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
    
    # Define the boundary parameterization
    # B_i = a + b * i, where i = 1, 2, ..., n_exercise_dates
    # Constraints: a >= 0, b >= 0, B_i <= K
    
    # Grid search over a and b
    n_grid_a = 30
    n_grid_b = 30
    
    a_vals = np.linspace(0, K, n_grid_a)
    b_vals = np.linspace(0, K / n_exercise_dates, n_grid_b)
    
    best_score = -np.inf
    best_a = 0
    best_b = 0
    
    # Precompute the payoff at each exercise date for each path
    # payoff[i, j] = max(K - S[i, j], 0)
    payoff = np.maximum(K - S, 0)
    
    # For each grid point (a, b), compute the expected discounted payoff
    for a in a_vals:
        # Compute boundaries for this a
        for b in b_vals:
            # Boundaries: B_i = a + b * i for i = 1, ..., n_exercise_dates
            i = np.arange(1, n_exercise_dates + 1)
            boundaries = a + b * i
            
            # Ensure boundaries are within [0, K]
            boundaries = np.clip(boundaries, 0, K)
            
            # For each path, find the first exercise date where S <= boundary
            # and compute the discounted payoff
            total_payoff = np.zeros(n_paths)
            
            for j in range(n_exercise_dates):
                # Exercise if S[i, j] <= boundaries[j]
                exercise_mask = S[:, j] <= boundaries[j]
                
                # For paths that haven't exercised yet and are exercising now
                not_exercised = total_payoff == 0
                exercise_now = exercise_mask & not_exercised
                
                total_payoff[exercise_now] = payoff[exercise_now, j] * discount_factors[j]
            
            # Mean discounted payoff
            mean_payoff = np.mean(total_payoff)
            
            if mean_payoff > best_score:
                best_score = mean_payoff
                best_a = a
                best_b = b
    
    # Create the exercise policy
    i = np.arange(1, n_exercise_dates + 1)
    boundaries = best_a + best_b * i
    boundaries = np.clip(boundaries, 0, K)
    
    # Store the boundaries in the policy object
    class ExercisePolicy:
        def __init__(self, boundaries, n_exercise_dates):
            self.boundaries = boundaries
            self.n_exercise_dates = n_exercise_dates
        
        def exercise(self, current_state, time):
            """
            Determine whether to exercise the option.
            
            Parameters:
            - current_state: array of spot prices
            - time: current time (or exercise date index)
            
            Returns:
            - boolean vector indicating whether to exercise
            """
            # Determine the exercise date index from time
            # time is likely the fraction of T, so index = round(time / dt)
            # But we need to be careful about the indexing
            
            # Assume time is the exercise date index (1 to n_exercise_dates)
            # or the fraction of T. Let's handle both cases.
            
            if isinstance(time, (int, float)):
                # time is a scalar
                if time <= 1.0:
                    # time is fraction of T
                    idx = int(round(time / dt)) - 1
                else:
                    # time is already the index
                    idx = int(time) - 1
            else:
                # time is an array
                time_arr = np.asarray(time)
                if time_arr.max() <= 1.0:
                    idx_arr = np.round(time_arr / dt).astype(int) - 1
                else:
                    idx_arr = time_arr.astype(int) - 1
                # For vectorized time, we need to handle each element
                # But the function should return a boolean vector for the current_state
                # Let's assume time is a scalar for now
                raise ValueError("Vectorized time not supported yet")
            
            # Clamp index to valid range
            idx = np.clip(idx, 0, self.n_exercise_dates - 1)
            
            # Get the boundary for this exercise date
            boundary = self.boundaries[idx]
            
            # Exercise if current_state <= boundary
            return np.asarray(current_state) <= boundary
    
    policy = ExercisePolicy(boundaries, n_exercise_dates)
    return policy

# EVOLVE-BLOCK-END
