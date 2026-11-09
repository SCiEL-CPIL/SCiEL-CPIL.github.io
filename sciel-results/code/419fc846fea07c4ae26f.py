"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.stats as stats

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using Longstaff-Schwartz Monte Carlo regression
    with in-the-money subset regression and rich basis functions.
    
    Parameters:
    -----------
    training_paths : array-like, shape (n_paths, n_dates)
        Simulated spot price paths. n_dates = 12 (monthly, no time-0).
        The contract specifies: Spot 90/100/110, K=100, r=0.05, sigma=0.2, T=1.
        12 monthly exercise dates.
    
    contract : object or dict
        Contains contract parameters. We assume it has fields or can be accessed
        to get K, r, sigma, T, n_dates, etc. Based on the task description:
        K=100, r=0.05, sigma=0.2, T=1.0, 12 monthly dates.
    
    Returns:
    --------
    An object with an exercise(current_state, time) method returning a boolean vector.
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
    
    # Rich basis functions for better approximation
    def basis_functions(s):
        """Compute rich basis functions for regression."""
        s = np.asarray(s, dtype=np.float64)
        # Use features: 1, S, S^2, S^3, S^4, log(S), |S - K|, (S-K)^2, (S-K)^3
        ones = np.ones_like(s)
        s1 = s
        s2 = s**2
        s3 = s**3
        s4 = s**4
        log_s = np.log(np.maximum(s, 1e-10))
        abs_diff = np.abs(s - K)
        diff_sq = (s - K)**2
        diff_cu = (s - K)**3
        return np.column_stack([ones, s1, s2, s3, s4, log_s, abs_diff, diff_sq, diff_cu])
    
    # Longstaff-Schwartz backward induction
    # V[i] = continuation value at step i, estimated via regression
    V = np.zeros((n_paths, n_steps), dtype=np.float64)
    
    # At the last date, continuation value = payoff
    V[:, -1] = payoffs[:, -1]
    
    # Store regression coefficients for each time step
    thetas = [None] * n_steps
    
    # Backward induction from last date to first
    for t in range(n_steps - 2, -1, -1):
        # Target: discounted V at next step
        target = discount * V[:, t + 1]
        
        # Only use in-the-money paths for regression (standard LS practice)
        itm_mask = payoffs[:, t] > 0
        n_itm = np.sum(itm_mask)
        
        if n_itm < 10:
            # Not enough in-the-money paths, use all paths
            X = basis_functions(S[:, t])
            y = target
        else:
            X = basis_functions(S[itm_mask, t])
            y = target[itm_mask]
        
        # Regularized least squares for stability
        n_features = X.shape[1]
        # Add small ridge regularization
        reg = 1e-6 * np.eye(n_features)
        try:
            theta = np.linalg.solve(X.T @ X + reg, X.T @ y)
        except np.linalg.LinAlgError:
            # Fallback to lstsq
            try:
                theta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
            except np.linalg.LinAlgError:
                theta = np.array([np.mean(y)] + [0.0] * (n_features - 1))
        
        thetas[t] = theta
        
        # Estimated continuation value at time t (for all paths)
        X_all = basis_functions(S[:, t])
        V_cont = X_all @ theta
        
        # Exercise value: max(payoff, continuation)
        V[:, t] = np.maximum(payoffs[:, t], V_cont)
    
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
            # Handle time as index
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
                # Exercise dates are at k/12 for k=1,...,12
                # Find the closest exercise date index
                t_idx = int(round(time * 12)) - 1
                t_idx = max(0, min(self.n_dates - 1, t_idx))
            
            if t_idx >= len(self.thetas) or self.thetas[t_idx] is None:
                return np.zeros(len(s), dtype=bool)
            
            theta = self.thetas[t_idx]
            X = basis_functions(s)
            V_cont = X @ theta
            payoff = np.maximum(self.K - s, 0.0)
            
            # Exercise if payoff strictly exceeds continuation value
            return payoff > V_cont
    
    return ExercisePolicy(thetas, K, n_dates)
# EVOLVE-BLOCK-END
