import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using Least Squares Monte Carlo (LSMC).
    
    Parameters:
    training_paths: Array of shape (n_paths, n_exercise + 1) containing simulated price paths.
    contract: Dictionary containing contract parameters.
    
    Returns:
    An object with an .exercise(current_state, time) method.
    """
    t_start = time.time()
    deadline = t_start + 100.0  # Safety margin
    
    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    # Ensure n_exercise matches the data if possible, or derive from data
    # The task says "12 monthly exercise dates (no time-0 exercise)".
    # training_paths usually includes time 0 to T.
    # If training_paths has shape (N, 13), it includes t=0.
    # The exercise dates are t_1, ..., t_12.
    
    N_paths, N_steps = training_paths.shape
    # Assume training_paths[0] is S_0, ..., training_paths[12] is S_T
    # Exercise dates correspond to indices 1 to 12 in the array if 0-indexed and 13 columns.
    # Let's verify dimensions.
    if N_steps != n_exercise + 1:
        # Fallback: assume last n_exercise steps are exercise dates
        pass 

    # Discount factors for each exercise date
    # Time step dt = T / n_exercise
    dt = T / n_exercise
    # Exercise times: t_i = i * dt for i = 1..n_exercise
    # Discount factor to time 0: exp(-r * t_i)
    # Discount factor from t_i to t_{i+1}: exp(-r * dt)
    
    # We work backward from T (index n_exercise) to t_1 (index 1).
    # Value at T: V_T = max(K - S_T, 0)
    
    # Basis functions: We will use polynomials in log(S).
    # Let x = log(S). Basis: 1, x, x^2, x^3, x^4
    # We need to handle S <= 0, but GBM is strictly positive.
    
    def make_basis(S):
        """Create basis matrix for regression."""
        S = np.maximum(S, 1e-10)
        log_S = np.log(S)
        B = np.column_stack([
            np.ones_like(log_S),
            log_S,
            log_S**2,
            log_S**3,
            log_S**4
        ])
        return B

    # Initialize values
    # V[i] is the value at exercise date i for path j.
    # We store the regression coefficients for each step.
    # coeffs[i] is a vector of shape (5,) for step i.
    
    coeffs = [None] * (n_exercise + 1)
    # V_next will hold the values from the next time step, discounted to current.
    V_next = np.maximum(K - training_paths[:, n_exercise], 0.0)
    
    # Backward induction
    # i goes from n_exercise - 1 down to 0? 
    # Exercise dates are indices 1 to n_exercise in the path array.
    # Let's align indices:
    # Path array index 0: t=0
    # Path array index 1: t_1
    # ...
    # Path array index n_exercise: t_n_exercise = T
    
    # We need to decide at t_1, ..., t_{n_exercise-1}. 
    # At T, we must exercise (or payoff is realized).
    
    # So we loop i from n_exercise - 1 down to 1.
    # At step i, we use S[:, i] to predict V[:, i+1] (discounted).
    
    for i in range(n_exercise - 1, 0, -1):
        if time.time() > deadline:
            break
            
        S_i = training_paths[:, i]
        S_next = training_paths[:, i + 1]
        
        # Discount V_next from t_{i+1} to t_i
        disc = np.exp(-r * dt)
        V_next_disc = V_next * disc
        
        # Intrinsic value at t_i
        intrinsic = np.maximum(K - S_i, 0.0)
        
        # We regress V_next_disc on S_i for paths that were NOT exercised at t_{i+1}... 
        # Actually, standard LSMC: 
        # 1. Determine exercise set at t_{i+1} based on previous iteration? 
        # No, we are working backward. 
        # At step i, we have already determined the optimal policy for steps i+1...n.
        # So V_next represents the optimal value at t_{i+1}.
        # We want to estimate E[V_next | S_i].
        
        # To avoid look-ahead bias in the regression (using paths that exercised later),
        # we should ideally only use paths that are still "alive". 
        # However, in a simple backward induction, V_next already accounts for the optimal stopping.
        # The standard LSMC approach uses ALL paths for the regression at step i, 
        # because V_next is the "continuation value" which is 0 if we exercised? 
        # No, V_next is the value of the option if we continue. 
        # If we exercised at t_{i+1}, the value is the intrinsic value at t_{i+1}.
        # The regression should predict the expected future payoff given S_i.
        
        # Let's use all paths for the regression.
        B = make_basis(S_i)
        
        # Solve least squares: B * coeffs = V_next_disc
        # Use ridge regression for stability
        lambda_ridge = 1e-4
        A = B.T @ B + lambda_ridge * np.eye(B.shape[1])
        b = B.T @ V_next_disc
        try:
            coeffs_i = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            coeffs_i = np.linalg.lstsq(B, V_next_disc, rcond=None)[0]
            
        coeffs[i] = coeffs_i
        
        # Estimated continuation value
        V_cont = B @ coeffs_i
        
        # Exercise if intrinsic > continuation
        exercise = intrinsic > V_cont
        
        # Value at t_i is max(intrinsic, V_cont)
        V_i = np.where(exercise, intrinsic, V_cont)
        
        # Update V_next for the next iteration (which is step i-1)
        V_next = V_i

    # Now we have coeffs[1] to coeffs[n_exercise-1].
    # We need to define the exercise policy object.
    
    class ExercisePolicy:
        def __init__(self, coeffs, K, r, sigma, T, n_exercise):
            self.coeffs = coeffs
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.n_exercise = n_exercise
            self.dt = T / n_exercise
            
        def exercise(self, current_state, time):
            """
            current_state: Array of shape (N_eval,) or (N_eval, 1)
            time: float, the current time.
            Returns: boolean array of shape (N_eval,)
            """
            # Determine which exercise step this time corresponds to
            # time is continuous, but exercise happens at discrete dates.
            # The evaluator likely calls this at the specific exercise times.
            # We need to map 'time' to the index i.
            
            # Find the closest exercise index
            # Exercise times: i * dt for i=1..n_exercise
            # If time is close to i*dt, use coeffs[i].
            
            # Handle edge cases: if time is T, we must exercise if in the money? 
            # Actually at T, the option expires. The payoff is intrinsic.
            # The policy should return True if exercising is optimal.
            # At T, continuation is 0. So exercise if K > S.
            
            # Identify the step index
            # time is in [0, T]
            # i = round(time / dt)
            i = int(round(time / self.dt))
            
            # Clamp i to valid range [1, n_exercise]
            if i < 1:
                i = 1
            if i > self.n_exercise:
                i = self.n_exercise
            
            S = np.asarray(current_state).flatten()
            S = np.maximum(S, 1e-10)
            
            if i == self.n_exercise:
                # At maturity, exercise if intrinsic > 0
                return S < self.K
            
            # Get coefficients for this step
            c = self.coeffs[i]
            if c is None:
                # Fallback: use intrinsic only if no coeff (should not happen)
                return S < self.K
                
            B = make_basis(S)
            V_cont = B @ c
            intrinsic = np.maximum(self.K - S, 0.0)
            
            return intrinsic > V_cont

    return ExercisePolicy(coeffs, K, r, sigma, T, n_exercise)
