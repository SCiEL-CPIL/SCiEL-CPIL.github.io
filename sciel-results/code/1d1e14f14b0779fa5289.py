import numpy as np

class ExercisePolicy:
    def __init__(self, grid, cont_values, K, r, dt):
        self.grid = grid
        self.cont_values = cont_values # List of arrays, one for each step 1..11
        self.K = K
        self.r = r
        self.dt = dt
        self.df = np.exp(-r * dt)
        
    def exercise(self, current_state, time):
        current_state = np.asarray(current_state)
        if isinstance(time, (int, np.integer)):
            k = int(time)
        else:
            k = int(round(time * 12))
            
        k = max(1, min(12, k))
        
        P = np.maximum(self.K - current_state, 0.0)
        
        if k == 12:
            return P > 0.0
        
        idx = k - 1
        c_grid = self.cont_values[idx]
        
        # Interpolate continuation value from grid
        # np.interp expects xp to be increasing
        C = np.interp(current_state, self.grid, c_grid)
        
        return P >= C

def fit_exercise_policy(training_paths, contract):
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    T = contract.get('T', 1.0)
    n_steps = contract.get('n_steps', 12)
    dt = T / n_steps
    df = np.exp(-r * dt)
    
    paths = np.asarray(training_paths)
    if paths.ndim != 2:
        raise ValueError("training_paths must be 2D")
        
    N, D = paths.shape
    
    if D == 13:
        S = paths[:, 1:13]
    elif D == 12:
        S = paths[:, 0:12]
    else:
        S = paths[:, -12:]
        
    P = np.maximum(K - S, 0.0)
    
    # Define grid for kernel smoothing
    # Range based on typical GBM behavior
    S_min = np.min(S) * 0.9
    S_max = np.max(S) * 1.1
    if S_min <= 0: S_min = 1.0
    grid = np.linspace(S_min, S_max, 200)
    
    # Bandwidth for Gaussian kernel
    # Estimate std of S
    std_S = np.std(S)
    h = 1.06 * std_S * (N ** (-1.0 / 5.0))
    if h < 0.5: h = 0.5 # Minimum bandwidth
    
    cont_values = []
    
    # Backward induction
    # V_next is the value at step k+1 (already optimal)
    V_next = P[:, 11].copy()
    
    for k in range(11, 0, -1):
        idx_curr = k - 1
        S_curr = S[:, idx_curr]
        P_curr = P[:, idx_curr]
        
        # 1. Compute Continuation Value on the Grid
        # Weights: (M x N)
        # G[:, None] - S_curr[None, :]
        diff = grid[:, None] - S_curr[None, :]
        weights = np.exp(-0.5 * (diff / h) ** 2)
        
        # Numerator: sum(weights * V_next)
        num = weights @ V_next
        # Denominator: sum(weights)
        den = weights.sum(axis=1)
        
        # Avoid division by zero
        den[den < 1e-8] = 1e-8
        
        C_grid = num / den
        
        # Discounted continuation value on grid
        C_grid_disc = C_grid * df
        
        # Store for exercise decision
        cont_values.append(C_grid_disc)
        
        # 2. Compute Optimal Value for Training Paths (for next step)
        # We need C(S_curr) to compute V_curr for the training paths
        # We can use the grid interpolation or recompute. 
        # Interpolation is faster and consistent.
        C_curr = np.interp(S_curr, grid, C_grid_disc)
        
        V_curr = np.maximum(P_curr, C_curr)
        
        V_next = V_curr
        
    # Reverse cont_values because we built them from step 11 down to 1
    # cont_values[0] is for step 11, cont_values[1] for step 10, ...
    # We need index 0 to correspond to step 1.
    cont_values = cont_values[::-1]
    
    return ExercisePolicy(grid, cont_values, K, r, dt)
