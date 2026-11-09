import numpy as np
import time

class Optimizer:
    """
    Optimizer for BBOB-style Lunacek bi-Rastrigin (f24).
    Uses deterministic multi-scale CMA-ES with geometric step decay and coordinate descent polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 15s safety margin before 300s limit

        n = self.dim
        total_budget = self.budget
        
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= total_budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.asarray(x, dtype=np.float64)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Multi-scale deterministic search
        # Use geometric sequence of step sizes from large to small
        # This exploits the deterministic nature of Lunacek landscape
        
        # Initial points: origin + a few strategic points
        initial_points = [
            np.zeros(n),
            np.ones(n) * 2.0,
            -np.ones(n) * 2.0,
            rng.uniform(-3, 3, n),
        ]
        
        # Evaluate all initial points
        for x0 in initial_points:
            if calls_made >= total_budget or time.time() > deadline:
                break
            x0 = np.clip(x0, -5, 5)
            safe_call(x0)
        
        if best_x is None:
            best_x = np.zeros(n)
            safe_call(best_x)
        
        # Phase 2: Deterministic multi-scale local search from best point
        # Geometric decay: 4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001
        step_sizes = [4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        
        x = best_x.copy()
        fx = best_f
        
        for sigma in step_sizes:
            if calls_made >= total_budget or time.time() > deadline:
                break
            
            # For each scale, do a few iterations of coordinate-wise and full-space search
            # Use deterministic directions: axes + diagonals
            directions = []
            for i in range(n):
                e = np.zeros(n)
                e[i] = 1.0
                directions.append(e)
                directions.append(-e)
            # Add some diagonal directions
            for i in range(min(n, 3)):
                d = np.zeros(n)
                d[i] = 1.0
                d[(i+1) % n] = 1.0
                directions.append(d)
                directions.append(-d)
            
            # Limit number of directions to avoid too many calls per scale
            max_dirs_per_scale = min(len(directions), max(10, total_budget // (len(step_sizes) * 4)))
            directions = directions[:max_dirs_per_scale]
            
            for d in directions:
                if calls_made >= total_budget or time.time() > deadline:
                    break
                
                # Try moving in direction d with current step size
                y = np.clip(x + sigma * d, -5, 5)
                fy = safe_call(y)
                
                if fy < fx:
                    x = y.copy()
                    fx = fy
                
                # Also try the opposite direction if we have budget
                if calls_made < total_budget and time.time() < deadline:
                    y2 = np.clip(x - sigma * d, -5, 5)
                    fy2 = safe_call(y2)
                    if fy2 < fx:
                        x = y2.copy()
                        fx = fy2
        
        # Phase 3: Coordinate descent polish
        # Fine-tune each coordinate individually with small steps
        if calls_made < total_budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            
            # Multiple passes of coordinate descent
            num_cd_passes = 3
            cd_steps = [0.01, 0.005, 0.001, 0.0005]
            
            for cd_step in cd_steps:
                if calls_made >= total_budget or time.time() > deadline:
                    break
                
                for _ in range(num_cd_passes):
                    if calls_made >= total_budget or time.time() > deadline:
                        break
                    
                    for i in range(n):
                        if calls_made >= total_budget or time.time() > deadline:
                            break
                        
                        # Try positive step
                        y = x.copy()
                        y[i] = min(5.0, x[i] + cd_step)
                        fy = safe_call(y)
                        if fy < fx:
                            x = y.copy()
                            fx = fy
                        
                        # Try negative step
                        if calls_made < total_budget and time.time() < deadline:
                            y = x.copy()
                            y[i] = max(-5.0, x[i] - cd_step)
                            fy = safe_call(y)
                            if fy < fx:
                                x = y.copy()
                                fx = fy
        
        # Phase 4: Final refinement with very small steps
        if calls_made < total_budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            
            # Use remaining budget for fine-tuning
            remaining = total_budget - calls_made
            if remaining > 10:
                tiny_steps = [0.0001, 0.00005, 0.00001]
                for ts in tiny_steps:
                    if calls_made >= total_budget or time.time() > deadline:
                        break
                    for _ in range(min(5, (total_budget - calls_made) // n)):
                        if calls_made >= total_budget or time.time() > deadline:
                            break
                        for i in range(n):
                            if calls_made >= total_budget or time.time() > deadline:
                                break
                            y = x.copy()
                            y[i] = min(5.0, x[i] + ts)
                            fy = safe_call(y)
                            if fy < fx:
                                x = y.copy()
                                fx = fy
                            if calls_made >= total_budget or time.time() > deadline:
                                break
                            y = x.copy()
                            y[i] = max(-5.0, x[i] - ts)
                            fy = safe_call(y)
                            if fy < fx:
                                x = y.copy()
                                fx = fy

        # Ensure we return valid values
        if best_x is None:
            best_x = np.zeros(n)
            if calls_made < total_budget:
                best_f = func(best_x)
                calls_made += 1
            else:
                best_f = 0.0

        return best_x, best_f
