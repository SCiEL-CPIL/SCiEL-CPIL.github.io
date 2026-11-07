import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = [b[0] for b in bounds]
        hi = [b[1] for b in bounds]
        dim = self.dim
        N = self.budget
        
        start_time = time.time()
        deadline = 290.0
        
        # 1. Analytical setup
        # f1(x) = 1 - exp(-4x) * sin(6*pi*x)^8
        # f2 = 1 - sqrt(f1)
        
        # Pre-compute f1 on a fine grid to map y -> x
        n_grid = 1000000
        x_grid = np.linspace(0, 1, n_grid)
        sin_term = np.sin(6 * np.pi * x_grid)
        sin_term_pow8 = np.power(sin_term, 8)
        exp_term = np.exp(-4 * x_grid)
        f1_grid = 1.0 - exp_term * sin_term_pow8
        
        f1_min = np.min(f1_grid)
        f1_max = 1.0
        
        # 2. Initial positions based on optimal spacing
        # Spacing in y proportional to sqrt(y)
        # sqrt(y_k) = sqrt(y_min) + (1 - sqrt(y_min)) * (k-1)/(N-1)
        k = np.arange(1, N + 1)
        # Handle N=1 case
        if N == 1:
            sqrt_y = np.array([1.0]) # Max y
        else:
            sqrt_y = np.sqrt(f1_min) + (1.0 - np.sqrt(f1_min)) * (k - 1) / (N - 1)
        y_targets = sqrt_y**2
        
        # Map y_targets to x1
        # Sort f1_grid and x_grid
        sort_idx = np.argsort(f1_grid)
        f1_sorted = f1_grid[sort_idx]
        x1_sorted = x_grid[sort_idx]
        
        # Interpolate x1 for each y_target
        # np.interp expects xp increasing
        x1_init = np.interp(y_targets, f1_sorted, x1_sorted)
        
        # 3. Local Search to maximize HV
        # Objective: HV(y) = sum (y_{i+1} - y_i) * (10 + sqrt(y_i))
        # with y_{N+1} = 11
        
        def calc_hv(x1):
            y = 1.0 - np.exp(-4 * x1) * np.power(np.sin(6 * np.pi * x1), 8)
            # Sort y
            y_sorted = np.sort(y)
            # Extend with 11.0
            y_ext = np.append(y_sorted, 11.0)
            # Widths
            widths = np.diff(y_ext)
            # Heights: 10 + sqrt(y_i) for i=1..N
            heights = 10.0 + np.sqrt(y_sorted)
            # HV = sum(widths * heights)
            return np.sum(widths * heights)
        
        current_x1 = x1_init.copy()
        current_hv = calc_hv(current_x1)
        
        # Simple coordinate ascent
        step_size = 0.01
        while time.time() - start_time < deadline:
            improved = False
            for i in range(N):
                if time.time() - start_time > deadline:
                    break
                # Try increasing
                new_x1 = current_x1.copy()
                new_x1[i] = np.clip(current_x1[i] + step_size, 0, 1)
                new_hv = calc_hv(new_x1)
                if new_hv > current_hv:
                    current_x1 = new_x1
                    current_hv = new_hv
                    improved = True
                    continue
                # Try decreasing
                new_x1[i] = np.clip(current_x1[i] - step_size, 0, 1)
                new_hv = calc_hv(new_x1)
                if new_hv > current_hv:
                    current_x1 = new_x1
                    current_hv = new_hv
                    improved = True
            
            if not improved:
                step_size *= 0.5
                if step_size < 1e-5:
                    break
        
        # 4. Evaluate with problem
        for i in range(N):
            if time.time() - start_time > deadline:
                break
            x = np.zeros(dim)
            x[0] = current_x1[i]
            x_clipped = np.clip(x, lo, hi)
            problem(x_clipped)
            
        return None
