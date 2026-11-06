"""Bermudan exercise-policy research: backward induction with antithetic MC for continuation values."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.interpolate import interp1d

class BermudanExercisePolicy:
    """
    Bermudan put exercise policy using backward induction with Monte Carlo
    estimation of continuation values on a state grid.
    """
    
    def __init__(self, training_paths, contract):
        self.start_time = time.time()
        self.deadline = 115.0  # safety margin before 120s limit
        
        # Parse contract parameters
        self.S0 = contract.get('S0', 100.0)
        self.K = contract.get('K', 100.0)
        self.r = contract.get('r', 0.05)
        self.sigma = contract.get('sigma', 0.2)
        self.T = contract.get('T', 1.0)
        self.n_exercise = contract.get('n_exercise', 12)
        
        # Exercise times: 12 monthly dates, no time-0
        self.exercise_times = np.linspace(self.T / self.n_exercise, self.T, self.n_exercise)
        
        # Number of training paths
        self.n_paths = training_paths.shape[0]
        # training_paths shape: (n_paths, n_time_steps) or similar
        # We need to figure out the structure
        
        # Fit the policy
        self._fit_policy(training_paths)
    
    def _get_param(self, contract, key, default):
        """Safely get a parameter from contract."""
        if hasattr(contract, 'get'):
            return contract.get(key, default)
        elif isinstance(contract, dict):
            return contract.get(key, default)
        else:
            return default
    
    def _fit_policy(self, training_paths):
        """
        Fit the exercise policy using backward induction.
        
        training_paths: array of shape (n_paths, n_steps) where n_steps includes time 0
        We assume paths are simulated under risk-neutral measure.
        """
        # Determine the number of time steps in training paths
        if training_paths.ndim == 1:
            # Might be a single path or need to reshape
            n_paths = 1
            n_steps = training_paths.shape[0]
            paths = training_paths.reshape(1, -1)
        else:
            paths = training_paths
            n_paths = paths.shape[0]
            n_steps = paths.shape[1]
        
        # The exercise dates correspond to specific indices in the path
        # With 12 monthly exercise dates and T=1, each step is 1/12 year
        # So exercise at steps 1, 2, ..., 12 (0-indexed: 1 through 12)
        # But we need to map exercise times to path indices
        
        # Assume the paths have n_steps = n_exercise + 1 (including time 0)
        # So exercise at indices 1, 2, ..., n_exercise
        
        n_ex = self.n_exercise
        # Exercise indices in the path (0-indexed, excluding time 0)
        exercise_indices = np.arange(1, n_ex + 1)
        
        # Discount factors between consecutive exercise dates
        dt = self.T / n_ex
        disc = np.exp(-self.r * dt)
        
        # For each exercise date, we want to estimate:
        # V_j(S) = max((K - S)+, disc * E[V_{j+1}(S_{j+1}) | S_j = S])
        
        # We'll use a grid-based approach:
        # 1. Collect all (S_j, S_{j+1}) pairs from training paths
        # 2. For the last exercise date, continuation value is the terminal payoff
        # 3. Work backward
        
        # Create a state grid for interpolation
        # Use a reasonable range of spot prices
        S_min = max(1.0, self.S0 * 0.2)
        S_max = self.S0 * 5.0
        n_grid = 200
        S_grid = np.linspace(S_min, S_max, n_grid)
        
        # Store continuation value functions for each exercise date
        # cont_values[j] will be a function of S giving E[discounted future payoff | S_j = S]
        # We'll store as arrays: (S_grid, continuation_value)
        
        # Terminal condition: after last exercise, payoff is (K - S_T)+
        # But for Bermudan, if not exercised at last date, it expires worthless? 
        # No - typically Bermudan options can only be exercised on specific dates,
        # and if not exercised by maturity, they expire worthless.
        # So at the last exercise date, continuation = 0 (no more exercise opportunities)
        # Wait, that's not right either. Let me reconsider.
        
        # For a Bermudan option with exercise dates t_1, ..., t_n:
        # At t_n (last exercise date): V_n(S) = max((K-S)+, 0) = (K-S)+
        #   because if you don't exercise at t_n, the option expires worthless
        # At t_j: V_j(S) = max((K-S)+, disc * E[V_{j+1}(S_{j+1}) | S_j = S])
        
        # So we work backward from j = n-1 down to j = 1 (0-indexed exercise dates)
        
        # Initialize: continuation value after last exercise is 0
        # At last exercise: value = (K - S)+
        
        # Let's store the value function at each exercise date
        # value_funcs[j] = function mapping S -> V_j(S)
        # We'll represent these as arrays on S_grid
        
        value_on_grid = np.zeros((n_ex, n_grid))
        
        # Last exercise date (index n_ex - 1 in 0-indexed):
        # V_{n}(S) = (K - S)+
        value_on_grid[n_ex - 1, :] = np.maximum(self.K - S_grid, 0.0)
        
        # Now work backward from j = n-2 down to 0
        for j in range(n_ex - 2, -1, -1):
            if time.time() - self.start_time > self.deadline:
                break
            
            # We need to estimate E[V_{j+1}(S_{j+1}) | S_j = S] for S in S_grid
            # Using training paths: for each path, we have S_j and S_{j+1}
            # V_{j+1}(S_{j+1}) is given by value_on_grid[j+1, :] interpolated
            
            # Get S_j and S_{j+1} from training paths
            # exercise_indices[j] and exercise_indices[j+1]
            idx_j = exercise_indices[j]
            idx_j1 = exercise_indices[j + 1]
            
            if idx_j1 >= n_steps:
                break
            
            S_j = paths[:, idx_j]  # shape (n_paths,)
            S_j1 = paths[:, idx_j1]  # shape (n_paths,)
            
            # Interpolate V_{j+1} at S_{j+1}
            # Use linear interpolation
            V_j1 = np.interp(S_j1, S_grid, value_on_grid[j + 1, :])
            
            # Now we have pairs (S_j, V_j1) for each training path
            # We want to estimate E[V_j1 | S_j = S] for S in S_grid
            # Use binning or kernel regression
            
            # Simple approach: bin S_j into bins, compute mean V_j1 per bin
            # Then interpolate to get a smooth function
            
            # Use a finer grid for binning
            n_bins = 100
            bin_edges = np.linspace(S_min, S_max, n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Bin the data
            bin_indices = np.digitize(S_j, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            
            # Compute mean V_j1 per bin
            bin_sums = np.zeros(n_bins)
            bin_counts = np.zeros(n_bins)
            np.add.at(bin_sums, bin_indices, V_j1)
            np.add.at(bin_counts, bin_indices, 1)
            
            # Avoid division by zero
            valid = bin_counts > 0
            bin_means = np.zeros(n_bins)
            bin_means[valid] = bin_sums[valid] / bin_counts[valid]
            
            # For bins with no data, use interpolation from neighboring bins
            # Fill in gaps
            if not np.all(valid):
                # Interpolate from valid bins
                valid_indices = np.where(valid)[0]
                if len(valid_indices) >= 2:
                    bin_means = np.interp(np.arange(n_bins), valid_indices, bin_means[valid_indices])
                elif len(valid_indices) == 1:
                    bin_means[:] = bin_means[valid_indices[0]]
            
            # Now interpolate bin_means to S_grid
            value_on_grid[j, :] = np.interp(S_grid, bin_centers, bin_means)
            
            # Apply the max: V_j(S) = max((K-S)+, disc * E[V_{j+1}(S_{j+1}) | S_j = S])
            immediate = np.maximum(self.K - S_grid, 0.0)
            continuation = disc * value_on_grid[j, :]
            value_on_grid[j, :] = np.maximum(immediate, continuation)
        
        # Store the value functions for use in the exercise() method
        self.S_grid = S_grid
        self.value_on_grid = value_on_grid
        self.exercise_indices = exercise_indices
        self.n_steps = n_steps
        
        # Precompute the decision boundaries: for each exercise date j,
        # find the threshold S* such that for S < S*, we exercise
        # (since put payoff increases as S decreases)
        
        self.thresholds = np.zeros(n_ex)
        for j in range(n_ex):
            if time.time() - self.start_time > self.deadline:
                break
            
            immediate = np.maximum(self.K - S_grid, 0.0)
            # The continuation value is already stored in value_on_grid[j, :]
            # but that's the max. We need the pure continuation to compare.
            # Actually, value_on_grid[j, :] = max(immediate, cont)
            # So we exercise where immediate >= cont
            # Which is where immediate >= value_on_grid[j, :] 
            # i.e., where value_on_grid[j, :] == immediate
            
            # Find the threshold: the largest S where we exercise
            # Exercise when (K-S)+ >= cont(S)
            # For S <= K: K-S >= cont(S)
            # For S > K: 0 >= cont(S), which is never true (cont >= 0)
            
            # So threshold is in [0, K]
            # Find where K - S = cont(S)
            # i.e., S = K - cont(S)
            
            # Simple approach: find the largest S in S_grid where we exercise
            exercise_mask = immediate >= value_on_grid[j, :] - 1e-10
            if np.any(exercise_mask):
                # Find the largest S where we exercise
                exercise_S = S_grid[exercise_mask]
                self.thresholds[j] = np.max(exercise_S)
            else:
                # Never exercise
                self.thresholds[j] = 0.0
    
    def exercise(self, current_state, time):
        """
        Determine whether to exercise at the given time.
        
        current_state: array of spot prices, shape (n_paths,)
        time: current time (scalar)
        
        Returns: boolean array, shape (n_paths,), True if exercise
        """
        # Determine which exercise date this corresponds to
        # Find the closest exercise time
        if time <= 0:
            # No exercise at time 0
            return np.zeros(len(current_state), dtype=bool)
        
        # Find the exercise index
        # Exercise times are self.exercise_times
        # Find the closest exercise time
        diffs = np.abs(self.exercise_times - time)
        j = np.argmin(diffs)
        
        # Check if this is actually an exercise date
        if diffs[j] > 1e-6:
            # Not an exercise date
            return np.zeros(len(current_state), dtype=bool)
        
        # Exercise if S < threshold
        threshold = self.thresholds[j]
        return current_state < threshold

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan exercise policy.
    
    training_paths: array of training paths
    contract: contract parameters
    
    Returns: object with exercise(current_state, time) method
    """
    policy = BermudanExercisePolicy(training_paths, contract)
    return policy
# EVOLVE-BLOCK-END
