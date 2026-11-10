import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fits a Bermudan exercise policy using Backward Induction on a price grid.
    This computes the optimal exercise boundaries for the given contract parameters.
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
    if hasattr(contract, 'sigma'): contract.sigma
    if hasattr(contract, 'T'): T = contract.T
    if hasattr(contract, 'n_exercises'): n_exercises = contract.n_exercises
    if hasattr(contract, 'exercise_dates'):
        exercise_dates = np.array(contract.exercise_dates)
    else:
        exercise_dates = np.linspace(0, T, n_exercises + 1)[1:]

    exercise_dates = np.sort(exercise_dates)
    n_paths = training_paths.shape[0]
    
    # Grid parameters for Backward Induction
    n_grid = 2000
    # Ensure grid covers potential price range. For a put, prices > K are never exercised.
    # Max price is unlikely to exceed 2*K significantly in this time frame with sigma=0.2.
    # Let's use a range [0, 2*K] to be safe.
    S_max = 2.0 * K
    S_grid = np.linspace(0, S_max, n_grid)
    
    # Time steps for the lattice
    # We need to align the lattice steps with the exercise dates.
    # The exercise dates are monthly (12 steps in 1 year).
    # dt = T / n_exercises
    dt = T / n_exercises
    
    # Risk neutral probabilities for the binomial tree on the grid
    # We approximate the GBM transition using the mean and variance over dt.
    # However, standard CRR binomial tree parameters are:
    # u = exp(sigma * sqrt(dt))
    # d = exp(-sigma * sqrt(dt))
    # p = (exp(r*dt) - d) / (u - d)
    # 
    # But we are using a fixed grid, not a tree that expands.
    # We can compute the expectation of S_{t+dt} given S_t using the grid.
    # Or simpler: Use the fact that E[S_{t+dt} | S_t] = S_t * exp(r*dt).
    # We can perform a linear interpolation of the value function at the expected future price?
    # No, that's not correct for convex payoffs.
    # We need to approximate the expectation of V(S_{t+dt}) given S_t.
    #
    # A common approach for grid-based backward induction is to use the transition probabilities.
    # Let's use a simple approximation:
    # The distribution of S_{t+dt} is lognormal.
    # We can approximate the expectation by summing over the grid points with weights.
    # Or, we can use the property that for a binomial tree, the expected value is p*V(S*u) + (1-p)*V(S*d).
    # Since we have a fixed grid, we can interpolate V at S*u and S*d.
    
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Interpolation function for the value function on the grid
    def interp_value(V, S):
        # V is defined on S_grid
        # S is an array of prices
        # We want V at S
        # Use np.interp for linear interpolation
        # Handle out of bounds: assume 0 for S > S_max, and extrapolate or clamp for S < 0 (not needed as S>=0)
        return np.interp(S, S_grid, V, left=V[0], right=V[-1])
    
    # Initialize value function at maturity (last exercise date is T)
    # The value at T is the payoff if not exercised earlier.
    # Note: The last exercise date is T.
    # We iterate backwards from T to the first exercise date.
    
    # We store the value function at each step to determine boundaries
    # V[i] corresponds to the value at exercise_dates[i]
    # We need boundaries for i=0 to n_exercises-1
    
    # Let's store the value function at each time step
    # V_current is the value at the next time step (continuation value source)
    # We start at T (index n_exercises-1 in exercise_dates, which is the last date)
    
    # Wait, exercise_dates has n_exercises elements.
    # exercise_dates[0] is the first opportunity (t=1/12)
    # exercise_dates[n_exercises-1] is the last opportunity (t=1.0)
    
    # At t=T (last date), the value is the payoff.
    # V_T(S) = max(K - S, 0)
    
    V_next = np.maximum(K - S_grid, 0.0)
    
    boundaries = np.zeros(n_exercises)
    
    # We need to compute boundaries for all dates.
    # Let's iterate backwards from the last date to the first.
    
    # At the last date (index n_exercises - 1), the continuation value is 0 (maturity).
    # So we just exercise if payoff > 0.
    # But the value function V_next already holds the payoff at maturity.
    # So for the last date, the optimal value is max(Payoff, 0) = Payoff (since put payoff is non-negative).
    # The boundary is where Payoff > 0 => K - S > 0 => S < K.
    # So boundary at last date is K.
    
    # Let's be more systematic.
    # V[i] is the value at exercise_dates[i] assuming optimal exercise from i to end.
    # V[n_exercises-1] = Payoff(S) at T.
    # Boundary[i] is the max S such that Payoff(S) >= ContinuationValue(S).
    
    # We compute V[i] from V[i+1].
    # ContinuationValue(S) = exp(-r*dt) * E[V[i+1](S_{t+dt}) | S_t=S]
    
    # We need V[i+1] to compute V[i].
    # So we start with V[n_exercises-1] = Payoff(S_grid).
    
    V_current = V_next.copy()
    
    # We will store the boundaries in reverse order or just compute them on the fly.
    # Let's compute boundaries for i from n_exercises-1 down to 0.
    
    # For i = n_exercises - 1 (last date, T):
    # Continuation value is 0 (no future).
    # Exercise if Payoff > 0.
    # Payoff = K - S.
    # Boundary is K.
    boundaries[n_exercises - 1] = K
    
    # For i = n_exercises - 2 down to 0:
    for i in range(n_exercises - 2, -1, -1):
        if time.time() > deadline:
            break
            
        # Compute Continuation Value for S_grid
        # E[V_next(S_{t+dt})] approximated by p * V_next(S*u) + (1-p) * V_next(S*d)
        S_up = S_grid * u
        S_down = S_grid * d
        
        # Interpolate V_next at S_up and S_down
        V_up = interp_value(V_next, S_up)
        V_down = interp_value(V_next, S_down)
        
        E_V_next = p * V_up + (1 - p) * V_down
        ContVal = np.exp(-r * dt) * E_V_next
        
        # Exercise Value
        ExVal = np.maximum(K - S_grid, 0.0)
        
        # Optimal Value
        V_opt = np.maximum(ContVal, ExVal)
        
        # Determine Boundary
        # Exercise if ExVal > ContVal
        # This happens for S < Boundary
        # Find the largest S where ExVal > ContVal
        # Since ExVal is linear decreasing and ContVal is convex-ish, the region is likely [0, B]
        
        # Find indices where exercise is optimal
        exercise_mask = ExVal >= ContVal
        
        if np.any(exercise_mask):
            # Get the max price where exercise is optimal
            # S_grid is sorted
            # Find the last index where mask is True
            # Note: It's possible the mask is not contiguous if there are numerical issues,
            # but for a put, it should be.
            idxs = np.where(exercise_mask)[0]
            max_idx = idxs[-1]
            # Interpolate between max_idx and max_idx+1 for better precision?
            # Or just take the grid point.
            # Let's interpolate for higher precision.
            if max_idx < n_grid - 1:
                S1 = S_grid[max_idx]
                S2 = S_grid[max_idx + 1]
                # We want the point where ExVal = ContVal
                # ExVal(S) = K - S
                # ContVal(S) is linearly interpolated between ContVal[S1] and ContVal[S2]
                # Let f(S) = ExVal(S) - ContVal(S)
                # f(S1) > 0, f(S2) < 0 (assuming boundary is between)
                # f(S) = (K - S) - (ContVal1 + (ContVal2 - ContVal1)*(S - S1)/(S2 - S1))
                # Set f(S) = 0
                # K - S = C1 + (C2 - C1)*(S - S1)/h
                # K - C1 - (C2 - C1)*(-S1/h) = S * (1 + (C2 - C1)/h)
                # This is getting complicated. Simple linear interpolation of the boundary index is often sufficient.
                # Or just use the grid point S_grid[max_idx].
                # Given n_grid=2000, error is small.
                boundary = S_grid[max_idx]
            else:
                boundary = S_grid[max_idx]
        else:
            # No exercise, boundary is 0 (or negative)
            boundary = 0.0
            
        boundaries[i] = boundary
        
        # Update V_next for the next iteration (which is now the current V_opt)
        V_next = V_opt
        
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, boundaries, exercise_dates, K, r, T):
            self.boundaries = boundaries
            self.exercise_dates = exercise_dates
            self.K = K
            self.r = r
            self.T = T
            
        def exercise(self, current_state, time):
            # Find the corresponding exercise date index
            # time is a scalar
            # exercise_dates is an array
            # Find the index with minimum difference
            idx = np.argmin(np.abs(self.exercise_dates - time))
            
            # Check if it's a valid exercise date
            if np.abs(self.exercise_dates[idx] - time) < 1e-6:
                boundary = self.boundaries[idx]
                return current_state < boundary
            else:
                # If time is not an exercise date, do not exercise
                return np.zeros(len(current_state), dtype=bool)

    return ExercisePolicy(boundaries, exercise_dates, K, r, T)
