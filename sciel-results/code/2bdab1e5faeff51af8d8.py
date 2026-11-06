"""Bermudan exercise-policy research with explicit candidate API."""
# EVOLVE-BLOCK-START
import time
import numpy as np

class BermudanExercisePolicy:
    """
    Bermudan put exercise policy using binomial tree backward induction.
    
    This approach uses a fine-grained binomial tree to compute the optimal
    exercise thresholds at each exercise date. This is more accurate than
    polynomial regression approaches because it directly solves the optimal
    stopping problem on a grid.
    """
    
    def __init__(self, training_paths, contract):
        """
        Fit the exercise policy using training paths.
        
        Parameters:
        -----------
        training_paths : array of shape (n_paths, n_time_points)
            Simulated price paths.
        contract : dict or object
            Contract parameters.
        """
        self.t_start = time.time()
        self.deadline = 140.0  # seconds, with safety margin
        
        # Extract contract parameters
        K = self._get_param(contract, 'K', 100.0)
        r = self._get_param(contract, 'r', 0.05)
        sigma = self._get_param(contract, 'sigma', 0.2)
        T = self._get_param(contract, 'T', 1.0)
        
        # Exercise dates: 12 monthly, no time-0
        exercise_times = self._get_exercise_times(contract, T)
        n_ex = len(exercise_times)
        
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.exercise_times = exercise_times
        self.n_ex = n_ex
        
        # Compute critical thresholds via backward induction on a binomial tree
        self._compute_thresholds()
        
        # Store the thresholds for each exercise date
        # self.thresholds[j] = critical price at exercise date j
        # Exercise if S <= thresholds[j]
    
    def _get_param(self, contract, key, default):
        """Extract parameter from contract (dict or object)."""
        if contract is None:
            return default
        if isinstance(contract, dict):
            return contract.get(key, default)
        return getattr(contract, key, default)
    
    def _get_exercise_times(self, contract, T):
        """Get exercise times from contract or default to monthly."""
        if contract is not None:
            if isinstance(contract, dict):
                times = contract.get('exercise_times', None)
                if times is not None:
                    return np.asarray(times, dtype=float)
                n_ex = contract.get('n_exercise_dates', 12)
            else:
                times = getattr(contract, 'exercise_times', None)
                if times is not None:
                    return np.asarray(times, dtype=float)
                n_ex = getattr(contract, 'n_exercise_dates', 12)
        else:
            n_ex = 12
        
        # Default: 12 monthly exercise dates over T
        return np.linspace(T / n_ex, T, n_ex)
    
    def _compute_thresholds(self):
        """
        Compute critical stock price thresholds for each exercise date
        using binomial tree backward induction.
        """
        n_ex = self.n_ex
        K = self.K
        r = self.r
        sigma = self.sigma
        
        # Use a fine grid for stock prices
        # For a put option, the critical price is always below K
        # We'll use a grid from 0 to K (or slightly above)
        n_grid = 2001
        S_max = K * 1.5  # Go a bit above K to be safe
        S_min = 1e-6
        S_grid = np.linspace(S_min, S_max, n_grid)
        
        # Store thresholds for each exercise date
        # thresholds[j] = critical price at exercise date j
        self.thresholds = np.zeros(n_ex)
        
        # Start from the last exercise date
        # At the last exercise date, we always exercise if in the money
        # V_last(S) = max(K - S, 0)
        V = np.maximum(K - S_grid, 0.0)
        self.thresholds[n_ex - 1] = K  # Exercise if S < K at last date
        
        # Now go backwards through exercise dates
        # For each pair of consecutive exercise dates, compute the continuation value
        for i in range(n_ex - 1, 0, -1):
            # Check time
            if time.time() - self.t_start > self.deadline - 5:
                # Not enough time, use simple approximation
                # For remaining dates, use the threshold from the next date
                for k in range(i - 1, -1, -1):
                    self.thresholds[k] = self.thresholds[i]
                return
            
            t_curr = self.exercise_times[i - 1]
            t_next = self.exercise_times[i]
            dt = t_next - t_curr
            
            if dt <= 0:
                self.thresholds[i - 1] = self.thresholds[i]
                continue
            
            # For each S on the grid, compute the expected continuation value
            # S_next = S * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
            # E[V(S_next)] where Z ~ N(0,1)
            
            # Use Gauss-Hermite quadrature for efficiency
            # Or use a simple numerical integration with a grid of Z values
            
            n_z = 101
            z_max = 4.0
            z_vals = np.linspace(-z_max, z_max, n_z)
            # Standard normal PDF weights
            z_weights = np.exp(-z_vals**2 / 2.0) / np.sqrt(2.0 * np.pi)
            dz = z_vals[1] - z_vals[0]
            
            # Drift and volatility terms
            drift = (r - 0.5 * sigma**2) * dt
            vol_term = sigma * np.sqrt(dt)
            
            # For each S on the grid, compute E[V(S_next)]
            # S_next[i, j] = S_grid[i] * exp(drift + vol_term * z_vals[j])
            S_next = S_grid[:, np.newaxis] * np.exp(drift + vol_term * z_vals[np.newaxis, :])
            
            # Interpolate V at S_next
            # V is defined on S_grid, we need V(S_next)
            V_next = np.interp(S_next, S_grid, V, left=0.0, right=0.0)
            
            # Expected continuation value for each S
            cont_val = np.sum(V_next * z_weights[np.newaxis, :] * dz, axis=1)
            
            # Discount
            discount = np.exp(-r * dt)
            cont_val = discount * cont_val
            
            # Exercise value
            exercise_val = np.maximum(K - S_grid, 0.0)
            
            # Optimal value: max(exercise, continuation)
            V = np.maximum(exercise_val, cont_val)
            
            # Find the threshold: the largest S where exercise is optimal
            # i.e., where exercise_val >= cont_val
            exercise_mask = exercise_val >= cont_val
            if np.any(exercise_mask):
                # Find the largest S where exercise is optimal
                threshold_idx = np.max(np.where(exercise_mask)[0])
                self.thresholds[i - 1] = S_grid[threshold_idx]
            else:
                # Never exercise (shouldn't happen for a put)
                self.thresholds[i - 1] = 0.0
    
    def exercise(self, current_state, time):
        """
        Determine exercise decision for given current state and time.
        
        Parameters:
        -----------
        current_state : array of shape (n_states,)
            Current price states.
        time : float
            Current time.
        
        Returns:
        --------
        exercise_decisions : boolean array of shape (n_states,)
            True if exercise, False if continue.
        """
        n_states = len(current_state)
        decisions = np.zeros(n_states, dtype=bool)
        
        # Find which exercise date we're at
        j = self._find_exercise_date(time)
        
        if j is None:
            # Not at an exercise date, don't exercise
            return decisions
        
        # Get the threshold for this exercise date
        if j >= len(self.thresholds):
            # Shouldn't happen, but handle gracefully
            intrinsic = np.maximum(self.K - current_state, 0.0)
            decisions = intrinsic > 0
            return decisions
        
        # Exercise if S <= threshold
        decisions = current_state <= self.thresholds[j]
        
        return decisions
    
    def _find_exercise_date(self, time):
        """Find which exercise date index corresponds to the given time."""
        for j, t in enumerate(self.exercise_times):
            if abs(time - t) < 1e-6:
                return j
        # If not found, find the closest
        if len(self.exercise_times) == 0:
            return None
        diffs = np.abs(self.exercise_times - time)
        min_idx = np.argmin(diffs)
        if diffs[min_idx] < 1e-3:  # Within a small tolerance
            return min_idx
        return None


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan exercise policy using training paths.
    
    Parameters:
    -----------
    training_paths : array of shape (n_paths, n_time_points)
        Simulated price paths.
    contract : dict or object
        Contract parameters.
    
    Returns:
    --------
    policy : BermudanExercisePolicy
        The fitted exercise policy.
    """
    return BermudanExercisePolicy(training_paths, contract)
# EVOLVE-BLOCK-END
