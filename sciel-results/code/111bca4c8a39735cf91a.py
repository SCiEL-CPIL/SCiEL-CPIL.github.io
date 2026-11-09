"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with coordinate descent polish and adaptive step-size."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls = 0

        def time_ok():
            return time.time() < deadline and calls < budget

        # Phase 1: Initial random sampling to find a good starting point
        init_budget = min(30, budget // 10)
        for _ in range(init_budget):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            x = np.zeros(dim)
            fx = func(x)
            calls = 1
            best_f = fx
            best_x = x.copy()

        # Phase 2: Coordinate descent warm start to align with ridge structure
        # This helps early progress on f13's sharp ridge
        cd_budget = min(80, budget // 5)
        x = best_x.copy()
        fx = best_f
        
        for _ in range(cd_budget):
            if not time_ok():
                break
            improved = False
            for d in range(dim):
                if not time_ok():
                    break
                # Try moving in positive direction
                step = 0.5
                for _ in range(10):
                    if not time_ok():
                        break
                    trial = x.copy()
                    trial[d] = np.clip(trial[d] + step, -5, 5)
                    tf = func(trial)
                    calls += 1
                    if tf < fx:
                        x = trial
                        fx = tf
                        improved = True
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        step *= 0.5
                    else:
                        break
                # Try moving in negative direction
                step = 0.5
                for _ in range(10):
                    if not time_ok():
                        break
                    trial = x.copy()
                    trial[d] = np.clip(trial[d] - step, -5, 5)
                    tf = func(trial)
                    calls += 1
                    if tf < fx:
                        x = trial
                        fx = tf
                        improved = True
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        step *= 0.5
                    else:
                        break
            if not improved:
                break

        # Phase 3: (1+1)-ES with adaptive sigma and restarts
        max_restarts = 8
        stall_threshold = 20
        
        for restart in range(max_restarts):
            if not time_ok():
                break
            
            if restart == 0:
                x = best_x.copy()
                sigma = 0.5
            else:
                # Perturb best point for restart
                x = best_x + rng.standard_normal(dim) * 0.3
                x = np.clip(x, -5, 5)
                sigma = 0.5
            
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            stall_counter = 0
            
            while time_ok():
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.15
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                        stall_counter = 0
                    else:
                        stall_counter += 1
                else:
                    sigma *= 0.85
                    stall_counter += 1
                
                if stall_counter >= stall_threshold:
                    break

        # Phase 4: Final coordinate descent polish for high precision
        polish_budget = min(100, budget - calls)
        if polish_budget > 0 and time_ok():
            x = best_x.copy()
            fx = best_f
            
            for _ in range(polish_budget):
                if not time_ok():
                    break
                improved = False
                for d in range(dim):
                    if not time_ok():
                        break
                    # Fine-tune with smaller steps
                    for direction in [1, -1]:
                        step = 0.1
                        for _ in range(8):
                            if not time_ok():
                                break
                            trial = x.copy()
                            trial[d] = np.clip(trial[d] + direction * step, -5, 5)
                            tf = func(trial)
                            calls += 1
                            if tf < fx:
                                x = trial
                                fx = tf
                                improved = True
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()
                                step *= 0.5
                            else:
                                break
                if not improved:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
