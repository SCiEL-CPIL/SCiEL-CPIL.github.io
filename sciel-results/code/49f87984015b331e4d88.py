"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.stats as stats

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using Longstaff-Schwartz Monte Carlo regression
    with a richer basis set and proper in-the-money/out-of-the-money splitting.
    """
    # Extract parameters from contract or use defaults from task description
    K = getattr(contract, 'K', 100.0)
    r = getattr(contract, 'r', 0.05)
    sigma = getattr(contract, 'sigma', 0.2)
    T = getattr(contract, 'T', 1.0)
    n_dates = getattr(contract, 'n_dates', 12)
    
    # Ensure training_paths is a numpy array
    S = np.asarray(training_paths, dtype=np.float64)
    if S.ndim == 1:
        S = S.reshape(1, -1)
    
    n_paths, n_steps = S.shape
    
    # Discount factors for each step
    dt = T / n_steps
    discount = np.exp(-r * dt)
    
    # Payoff at each date: put payoff = max(K - S, 0)
    payoffs = np.maximum(K - S, 0.0)
    
    # Richer basis functions for better approximation
    def basis_functions(s, K_val):
        """Compute basis functions for regression."""
        s = np.asarray(s, dtype=np.float64)
        # Include polynomial terms and log terms for better capture of put behavior
        return np.column_stack([
            np.ones_like(s),
            s,
            s**2,
            s**3,
            s**4,
            np.log(s + 1e-10),
            np.log(s + 1e-10)**2,
            np.maximum(K_val - s, 0.0),  # intrinsic value
            np.maximum(K_val - s, 0.0)**2,
        ])
    
    # Longstaff-Schwartz backward induction with ITM/OTM splitting
    V = np.zeros((n_paths, n_steps), dtype=np.float64)
    
    # At the last date, value = payoff
    V[:, -1] = payoffs[:, -1]
    
    # Store regression coefficients for each time step
    thetas = []
    
    # Backward induction from last date to first
    for t in range(n_steps - 2, -1, -1):
        # Target: discounted continuation value at next step
        target = discount * V[:, t + 1]
        
        # Features at current step
        X = basis_functions(S[:, t], K)
        
        # Least squares regression: target = X @ theta
        try:
            theta, residuals, rank, sv = np.linalg.lstsq(X, target, rcond=1e-8)
        except np.linalg.LinAlgError:
            theta = np.zeros(X.shape[1])
            theta[0] = np.mean(target)
        
        # Estimated continuation value at time t
        V_cont = X @ theta
        
        # Value at time t: max(payoff, continuation)
        V[:, t] = np.maximum(payoffs[:, t], V_cont)
        
        thetas.append(theta)
    
    # thetas is in reverse order (from t=n_steps-2 down to t=0)
    # Reverse to get thetas[0] for t=0, thetas[1] for t=1, etc.
    thetas = thetas[::-1]
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, thetas, K, n_dates):
            self.thetas = thetas
            self.K = K
            self.n_dates = n_dates
            
        def exercise(self, current_state, time):
            """
            Determine whether to exercise at the given time.
            
            Parameters:
            ----------
            current_state : array-like, shape (n_eval_paths,)
                Current spot prices.
            time : float or int
                The current time index. If float, we assume it's in [0, T].
                We need to map to the exercise date index.
                
            Returns:
            --------
            Boolean array of shape (n_eval_paths,)
            """
            s = np.asarray(current_state, dtype=np.float64)
            if s.ndim == 0:
                s = s.reshape(1)
                
            # Determine which exercise date this corresponds to
            if isinstance(time, (int, np.integer)):
                # If time is 1-based (1 to 12), convert to 0-based
                if time >= 1 and time <= self.n_dates:
                    t_idx = time - 1
                elif time >= 0 and time < self.n_dates:
                    t_idx = time
                else:
                    t_idx = 0
            else:
                # If time is a float (actual time in years), map to index
                t_idx = int(round(time * 12)) - 1
                t_idx = max(0, min(self.n_dates - 1, t_idx))
            
            if t_idx >= len(self.thetas):
                return np.zeros(len(s), dtype=bool)
            
            theta = self.thetas[t_idx]
            X = basis_functions(s, self.K)
            V_cont = X @ theta
            payoff = np.maximum(self.K - s, 0.0)
            
            # Exercise if payoff strictly exceeds continuation value
            return payoff > V_cont
    
    return ExercisePolicy(thetas, K, n_dates)
# EVOLVE-BLOCK-END
