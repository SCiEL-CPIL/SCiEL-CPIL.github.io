# EVOLVE-BLOCK-START
import numpy as np
import time


def _extract_params(contract):
    """Extract model parameters from the contract dictionary."""
    K = float(contract.get('K', 100.0))
    r = float(contract.get('r', 0.05))
    sigma = float(contract.get('sigma', 0.2))
    T = float(contract.get('T', 1.0))
    n_exercise = int(contract.get('n_exercise', 12))
    return K, r, sigma, T, n_exercise


def _build_grid(K, r, sigma, T, n_exercise, n_steps=1200):
    """
    Build a fine-grained binomial tree for a Bermudan put.
    Returns:
        S_grid: 2D array [n_exercise+1, n_steps+1] of stock prices
        V_grid: 2D array [n_exercise+1, n_steps+1] of option values
    """
    dt = T / n_steps
    # Use Cox-Ross-Rubinstein parameters
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Build the stock price grid
    # At time step j, the stock can be at K * u^(i - j) for i = 0..j
    # We'll use a 2D array where row j corresponds to time step j
    # and column i corresponds to the i-th up-move count
    
    # For efficiency, precompute log-spacings
    # S(j, i) = K * u^(i - j) for i in 0..j
    # But this is triangular. Let's use a rectangular grid for simplicity.
    # Max number of up moves at any time is n_steps.
    
    # Actually, let's use a more compact representation.
    # At time step j (0-indexed, so j=0 is t=0), there are j+1 possible states.
    # S_j,i = K * u^(i - j) where i = 0, 1, ..., j
    
    # We'll store values in a 2D array of shape (n_steps+1, n_steps+1)
    # where entry [j, i] is valid only for i <= j.
    
    S_grid = np.zeros((n_steps + 1, n_steps + 1))
    V_grid = np.zeros((n_steps + 1, n_steps + 1))
    
    # Initialize at t=0
    S_grid[0, 0] = K  # S_0,0 = K * u^0 = K
    
    # Fill in stock prices
    for j in range(1, n_steps + 1):
        for i in range(j + 1):
            S_grid[j, i] = K * (u ** (i - j))
    
    # Terminal condition (European put at maturity)
    for i in range(n_steps + 1):
        S_T = S_grid[n_steps, i]
        V_grid[n_steps, i] = max(K - S_T, 0.0)
    
    # Map exercise dates to tree time steps
    # Exercise dates are at times T/k for k = 1, ..., n_exercise
    # In tree steps: step_k = int(round(T/k / dt)) = int(round(n_steps * T/k / T)) = int(round(n_steps/k))
    exercise_steps = []
    for k in range(1, n_exercise + 1):
        step_k = int(round(n_steps / k))
        exercise_steps.append(step_k)
    # Sort in reverse order for backward induction
    exercise_steps.sort(reverse=True)
    
    # Backward induction
    current_ex_idx = 0
    for j in range(n_steps - 1, -1, -1):
        # Check if this is an exercise date
        is_exercise_date = (current_ex_idx < len(exercise_steps) and exercise_steps[current_ex_idx] == j)
        if is_exercise_date:
            current_ex_idx += 1
        
        for i in range(j + 1):
            S_ji = S_grid[j, i]
            intrinsic = max(K - S_ji, 0.0)
            
            if is_exercise_date:
                # Continuation value
                V_up = V_grid[j + 1, i + 1]  # up move: i -> i+1
                V_dn = V_grid[j + 1, i]      # down move: i -> i
                cont = np.exp(-r * dt) * (p * V_up + (1 - p) * V_dn)
                V_grid[j, i] = max(intrinsic, cont)
            else:
                # No exercise, just discounted expected value
                V_up = V_grid[j + 1, i + 1]
                V_dn = V_grid[j + 1, i]
                V_grid[j, i] = np.exp(-r * dt) * (p * V_up + (1 - p) * V_dn)
    
    return S_grid, V_grid, exercise_steps, dt


def _find_thresholds(V_grid, S_grid, exercise_steps, n_exercise):
    """
    Find the exercise threshold at each exercise date.
    At each exercise date j, find the smallest S such that intrinsic >= continuation.
    For a put, we exercise when S is low enough.
    """
    thresholds = []
    for j in exercise_steps:
        # Find the boundary between exercise and continuation regions
        # For a put, we exercise for low S values
        # Find the largest i (highest S) where we still exercise
        threshold_S = 0.0
        for i in range(j + 1):
            S_ji = S_grid[j, i]
            intrinsic = max(100.0 - S_ji, 0.0)  # K=100 assumed, but let's pass K
            # Actually we need K here. Let me restructure.
        thresholds.append(threshold_S)
    return thresholds


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using a fine binomial tree.
    
    Returns an object with .exercise(current_state, time) method that
    returns a boolean vector indicating whether to exercise at each path.
    """
    t_start = time.time()
    deadline = t_start + 105.0  # Safety margin before 120s kill
    
    K, r, sigma, T, n_exercise = _extract_params(contract)
    
    # Use a fine binomial tree
    n_steps = 2000
    
    # Check time budget
    if time.time() > deadline - 30:
        # Fallback to simple threshold approach
        return _simple_policy(K, r, sigma, T, n_exercise)
    
    # Build the tree
    S_grid, V_grid, exercise_steps, dt = _build_grid(K, r, sigma, T, n_exercise, n_steps)
    
    # Find exercise thresholds at each exercise date
    # For each exercise date j, find the critical stock price S* such that
    # for S < S*, we exercise; for S > S*, we continue.
    
    thresholds = np.zeros(n_exercise)
    # exercise_steps is sorted in reverse (from last to first)
    # We need to map exercise dates in chronological order
    # Exercise date k (1-indexed) corresponds to step n_steps/k
    # Let's create a mapping from exercise index to step
    
    # exercise_steps is already sorted reverse: [step_1, step_2, ..., step_12] where step_k = round(n_steps/k)
    # But we need them in chronological order: date 1 (earliest) to date 12 (latest)
    # Date k is at step round(n_steps/k)
    
    chrono_steps = []
    for k in range(1, n_exercise + 1):
        step_k = int(round(n_steps / k))
        chrono_steps.append(step_k)
    
    # For each exercise date in chronological order, find the threshold
    for idx, j in enumerate(chrono_steps):
        # At step j, find the largest S where we exercise
        # For a put, exercise when S is below threshold
        # Find the boundary
        threshold = 0.0
        for i in range(j + 1):
            S_ji = S_grid[j, i]
            intrinsic = max(K - S_ji, 0.0)
            # Get continuation value
            if i + 1 <= j + 1:
                V_up = V_grid[j + 1, i + 1]
            else:
                V_up = 0.0
            V_dn = V_grid[j + 1, i]
            p = (np.exp(r * dt) - 1.0 / np.exp(sigma * np.sqrt(dt))) / (np.exp(sigma * np.sqrt(dt)) - 1.0 / np.exp(sigma * np.sqrt(dt)))
            cont = np.exp(-r * dt) * (p * V_up + (1 - p) * V_dn)
            
            if intrinsic >= cont:
                threshold = S_ji  # Update to this S (we want the largest such S)
            else:
                # Since S increases with i, once we hit continuation, all higher S are continuation
                break
        thresholds[idx] = threshold
    
    # Check time
    if time.time() > deadline - 5:
        return _threshold_policy(thresholds, chrono_steps, K, r, sigma, T, n_exercise)
    
    return _threshold_policy(thresholds, chrono_steps, K, r, sigma, T, n_exercise)


def _threshold_policy(thresholds, chrono_steps, K, r, sigma, T, n_exercise):
    """Create a policy object that uses precomputed thresholds."""
    
    class ExercisePolicy:
        def __init__(self, thresholds, chrono_steps, K, r, sigma, T, n_exercise):
            self.thresholds = thresholds
            self.chrono_steps = chrono_steps
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.n_exercise = n_exercise
            # Precompute exercise date times
            self.exercise_times = np.array([T / k for k in range(1, n_exercise + 1)])
        
        def exercise(self, current_state, time):
            """
            Determine whether to exercise at the given time.
            
            Args:
                current_state: array of stock prices (one per path)
                time: current time (scalar)
            
            Returns:
                Boolean array indicating exercise decisions
            """
            # Find which exercise date this time corresponds to
            # Exercise dates are at T/k for k=1,...,n_exercise
            # Find the closest exercise date
            best_k = -1
            best_dist = np.inf
            for k in range(1, self.n_exercise + 1):
                t_k = self.T / k
                dist = abs(time - t_k)
                if dist < best_dist:
                    best_dist = dist
                    best_k = k - 1  # 0-indexed
            
            if best_k < 0 or best_dist > 1e-6:
                # Not at an exercise date
                return np.zeros(len(current_state), dtype=bool)
            
            threshold = self.thresholds[best_k]
            # For a put, exercise when S < threshold
            # Also need intrinsic value > 0
            return current_state < threshold
    
    return ExercisePolicy(thresholds, chrono_steps, K, r, sigma, T, n_exercise)


def _simple_policy(K, r, sigma, T, n_exercise):
    """Fallback simple policy using heuristic thresholds."""
    
    class ExercisePolicy:
        def __init__(self, K, r, sigma, T, n_exercise):
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.n_exercise = n_exercise
            # Heuristic: for a put, exercise threshold decreases over time
            # Start near K and decrease
            self.thresholds = np.array([K * np.exp(-sigma * np.sqrt(T / k) * 0.5) for k in range(1, n_exercise + 1)])
        
        def exercise(self, current_state, time):
            best_k = -1
            best_dist = np.inf
            for k in range(1, self.n_exercise + 1):
                t_k = self.T / k
                dist = abs(time - t_k)
                if dist < best_dist:
                    best_dist = dist
                    best_k = k - 1
            
            if best_k < 0 or best_dist > 1e-6:
                return np.zeros(len(current_state), dtype=bool)
            
            return current_state < self.thresholds[best_k]
    
    return ExercisePolicy(K, r, sigma, T, n_exercise)
# EVOLVE-BLOCK-END
