"""Optimizer for BBOB-style f15 (Rastrigin rotated) with adaptive (1+1)-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive (1+1)-ES with 1/5 success rule and periodic exploration for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        # Smart initialization: try lattice points to exploit Rastrigin's periodicity
        # Rastrigin has minima at integer coordinates (after rotation/shift)
        # Try origin and a few structured points
        init_candidates = [
            np.zeros(dim),
            np.array([2.0] * dim),
            np.array([-2.0] * dim),
            np.array([3.0, -3.0, 2.0, -2.0, 1.0]),
            np.array([-1.0, 1.0, -1.0, 1.0, -1.0]),
        ]
        
        # Evaluate all init candidates to find the best starting point
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        for xc in init_candidates:
            if time.time() > deadline or calls_made >= budget:
                break
            xc_clipped = np.clip(xc, -5, 5)
            fc = func(xc_clipped)
            calls_made += 1
            if fc < best_f:
                best_f = fc
                best_x = xc_clipped.copy()
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1
        
        # Single sustained (1+1)-ES lineage with 1/5 success rule
        x = best_x.copy()
        fx = best_f
        sigma = 2.0  # Start with larger step for exploration
        
        # 1/5 success rule parameters
        sigma_min = 1e-4
        sigma_max = 5.0
        success_window = []
        window_size = 15
        
        # Exploration: periodically jump to escape local minima
        last_jump_iter = 0
        jump_interval = 50  # Jump every 50 iterations if stuck
        stuck_threshold = 0.1  # If improvement < this over window, consider stuck
        
        for i in range(budget - calls_made):
            if time.time() > deadline:
                break
            
            # Check if we should explore (jump to new region)
            if i - last_jump_iter >= jump_interval and i > 50:
                # Check if we're stuck: look at recent improvements
                if len(success_window) > 0:
                    recent_improvements = [1 for s in success_window[-window_size:] if s]
                    if len(recent_improvements) / min(len(success_window), window_size) < 0.2:
                        # Stuck - try a jump
                        jump_size = max(sigma * 3, 1.0)
                        jump_dir = rng.standard_normal(dim)
                        jump_dir /= (np.linalg.norm(jump_dir) + 1e-10)
                        y = x + jump_size * jump_dir
                        y = np.clip(y, -5, 5)
                        fy = func(y)
                        calls_made += 1
                        if fy < fx:
                            x = y
                            fx = fy
                            sigma = jump_size * 0.5
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                        last_jump_iter = i
                        continue
            
            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls_made += 1
            
            # Track success for 1/5 rule
            if fy < fx:
                success_window.append(1)
                x = y
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                success_window.append(0)
            
            # 1/5 success rule: adjust sigma based on success rate
            if len(success_window) >= window_size:
                success_rate = sum(success_window[-window_size:]) / window_size
                if success_rate > 0.5:
                    sigma *= 1.5
                elif success_rate < 0.2:
                    sigma *= 0.7
                # Otherwise keep sigma
                # Trim window
                if len(success_window) > window_size * 2:
                    success_window = success_window[-window_size:]
            
            # Clamp sigma
            sigma = np.clip(sigma, sigma_min, sigma_max)

        return best_x, best_f
# EVOLVE-BLOCK-END
