import numpy as np
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    delta = 1.0 / N
    start_time = time.time()
    deadline = start_time + 580  # safety margin
    
    # Start with the linear star construction
    x0 = 0.0
    y0 = 0.0
    a = np.array([-i / (2.0 * N) for i in range(N)])  # x-positions for each i
    b = np.array([-j / (2.0 * N) for j in range(N)])  # y-positions for each j
    
    def compute_volume_separable(a, b):
        """Compute exact volume for separable construction."""
        # W_x(c) = max_i(a_i + c*i/8 + 1/8) - min_i(a_i + c*i/8)
        # Use numerical quadrature with many points
        c_vals = np.linspace(0, 1, 1001)
        # x_i + c*i/8 for all i, c
        # a[i] + c * i/8, shape (N, len(c_vals))
        x_left = a[:, None] + c_vals[None, :] * np.arange(N)[:, None] / N
        x_right = x_left + 1.0 / N
        W_x = x_right.max(axis=0) - x_left.min(axis=0)
        y_left = b[:, None] + c_vals[None, :] * np.arange(N)[:, None] / N
        y_right = y_left + 1.0 / N
        W_y = y_right.max(axis=0) - y_left.min(axis=0)
        area = W_x * W_y
        # Box dimensions
        X0 = a.min()
        X1 = (a + (np.arange(N) + 1) / N).max()
        Y0 = b.min()
        Y1 = (b + (np.arange(N) + 1) / N).max()
        box_area = (X1 - X0) * (Y1 - Y0)
        if box_area <= 1e-9:
            return float('inf')
        V = np.trapz(area, c_vals) * box_area
        return V
    
    def build_positions(a, b):
        positions = np.empty((64, 2), dtype=np.float64)
        for k in range(64):
            i = k // N
            j = k % N
            positions[k, 0] = a[i]
            positions[k, 1] = b[j]
        return positions
    
    best_a = a.copy()
    best_b = b.copy()
    best_V = compute_volume_separable(a, b)
    
    # Try small perturbations
    rng = np.random.default_rng(42)
    n_iter = 0
    step_size = 0.01
    
    while time.time() < deadline:
        n_iter += 1
        # Random perturbation
        da = rng.normal(0, step_size, N)
        db = rng.normal(0, step_size, N)
        trial_a = a + da
        trial_b = b + db
        trial_V = compute_volume_separable(trial_a, trial_b)
        
        if trial_V < best_V:
            best_V = trial_V
            best_a = trial_a
            best_b = trial_b
            a = trial_a
            b = trial_b
            step_size = max(0.001, step_size * 1.1)
        else:
            step_size = max(0.001, step_size * 0.9)
        
        # Also try coordinate-wise moves
        for dim in range(N):
            for delta_val in [0.005, -0.005, 0.002, -0.002]:
                trial_a2 = a.copy()
                trial_a2[dim] += delta_val
                trial_V2 = compute_volume_separable(trial_a2, b)
                if trial_V2 < best_V:
                    best_V = trial_V2
                    best_a = trial_a2
                    best_b = b.copy()
                    a = trial_a2
                    break
            for delta_val in [0.005, -0.005, 0.002, -0.002]:
                trial_b2 = b.copy()
                trial_b2[dim] += delta_val
                trial_V3 = compute_volume_separable(a, trial_b2)
                if trial_V3 < best_V:
                    best_V = trial_V3
                    best_a = a.copy()
                    best_b = trial_b2
                    b = trial_b2
                    break
    
    return build_positions(best_a, best_b)
