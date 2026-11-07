import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.start_time = time.time()
        # Deadline guard: 300s limit, keep 10s margin for safety
        self.deadline = self.start_time + 290.0

    def _check_time(self):
        return time.time() < self.deadline

    def __call__(self, func):
        # Wrap func to track calls and handle safety
        def safe_func(x):
            if self.calls >= self.budget:
                return self.best_f
            if not self._check_time():
                return self.best_f
            
            x = np.asarray(x, dtype=np.float64)
            # Clip to box to ensure valid input
            x_clipped = np.clip(x, -5, 5)
            
            try:
                f = float(func(x_clipped))
            except Exception:
                return 1e100
                
            self.calls += 1
            if f < self.best_f:
                self.best_f = f
                self.best_x = x_clipped.copy()
                
            return f

        # Strategy: Multi-Resolution Grid Scan
        # The Lunacek function is bi-modal. A simple grid scan can identify
        # the two main basins. We will perform a coarse scan, then refine
        # around the best points found.
        
        # Phase 1: Coarse Grid Scan
        # In 5D, a full grid of 5^5 = 3125 points is too many for 500 budget.
        # We need a sparse grid or a randomized grid.
        # Let's use a stratified random sampling approach or a specific pattern.
        
        # Given budget 500, let's split:
        # 100 calls: Coarse exploration (random points + center)
        # 300 calls: Medium refinement around best coarse points
        # 100 calls: Fine local search around global best
        
        # 1. Initial Exploration
        # Evaluate center
        x0 = np.zeros(self.dim)
        f0 = safe_func(x0)
        
        # Evaluate some random points to map the landscape
        rng = np.random.RandomState(42)
        coarse_points = []
        # Add center
        coarse_points.append(x0)
        
        # Add random points
        for _ in range(99):
            if self.calls >= self.budget or not self._check_time():
                break
            x_r = rng.uniform(-5, 5, self.dim)
            coarse_points.append(x_r)
            
        # Evaluate all coarse points
        for x in coarse_points:
            if self.calls >= self.budget or not self._check_time():
                break
            safe_func(x)
            
        # Sort coarse points by fitness
        # We need to re-evaluate or store fitness. safe_func updates self.best_f.
        # To find local bests, we need to know which x gave which f.
        # Let's keep a list of (x, f) for the coarse phase.
        # Since safe_func doesn't return the f for specific x easily in a loop without extra bookkeeping,
        # let's modify the approach: store results manually.
        
        # Actually, the current safe_func is fine, but we lost the association.
        # Let's re-do the coarse phase with explicit storage.
        
        # Reset best for this phase? No, keep global best.
        # Let's collect candidates.
        candidates = []
        
        # We already called safe_func for x0 and random points.
        # We can just use self.best_x as the anchor for refinement?
        # That might miss the other basin.
        
        # Better: Perform a structured coarse scan that stores (x, f).
        # But we already spent ~100 calls.
        # Let's use the remaining budget for refinement around the current best and a few other promising areas.
        
        # To handle bi-modality, we should also check the "other" side.
        # Lunacek has two minima. One is usually near the center/origin, one is shifted.
        # A common heuristic is to try the origin and a point shifted in a random direction.
        
        # Let's identify the current best.
        if self.best_x is None:
            self.best_x = x0
            self.best_f = f0
            
        # Phase 2: Local Refinement around Best
        # Use a local grid around self.best_x
        # Range: +/- 2.0
        # Resolution: 5x5x5x5x5 = 3125 points is too many.
        # Let's do a random local search or a smaller grid.
        
        # Let's try a local Nelder-Mead if available? No, stick to grid/random.
        # Random local search is robust.
        
        # Let's do a "Hill Climbing" with random restarts within a local box.
        
        # Define local box around best
        center = self.best_x.copy()
        box_size = 2.0
        
        # Generate random points in this box
        local_calls = 0
        max_local_calls = 200
        while local_calls < max_local_calls and self.calls < self.budget and self._check_time():
            x_new = center + rng.uniform(-box_size, box_size, self.dim)
            x_new = np.clip(x_new, -5, 5)
            f_new = safe_func(x_new)
            local_calls += 1
            
            if f_new < self.best_f:
                center = x_new.copy()
                # Shrink box slightly on success
                box_size *= 0.8
            else:
                # Keep box size or expand slightly if stuck? 
                # For now, just keep it.
                pass
        
        # Phase 3: Global Exploration for the Second Basin
        # The Lunacek function often has a second minimum far from the first.
        # If we only refined around the first, we might miss the global min if the first was local.
        # Let's try a few random restarts in the global space to see if we find something better.
        
        # If we have budget left, try 5 random restarts with local refinement.
        restarts = 5
        for r in range(restarts):
            if self.calls >= self.budget or not self._check_time():
                break
            
            # Start from a random point
            x_start = rng.uniform(-5, 5, self.dim)
            f_start = safe_func(x_start)
            
            if f_start < self.best_f:
                self.best_x = x_start.copy()
                self.best_f = f_start
                center = x_start.copy()
                box_size = 2.0
            else:
                # Even if not better, it might be in a different basin.
                # Let's do a small local search from here.
                center = x_start.copy()
                box_size = 1.0
            
            # Local refinement from this restart
            local_calls = 0
            max_local_calls = 30
            while local_calls < max_local_calls and self.calls < self.budget and self._check_time():
                x_new = center + rng.uniform(-box_size, box_size, self.dim)
                x_new = np.clip(x_new, -5, 5)
                f_new = safe_func(x_new)
                local_calls += 1
                
                if f_new < self.best_f:
                    self.best_x = x_new.copy()
                    self.best_f = f_new
                    center = x_new.copy()
                    box_size *= 0.8
                else:
                    # If we find a new local best in this basin, update center
                    # But we only update global best if it's better.
                    # For local search, we should track local best.
                    # This is getting complex. Let's simplify:
                    # Just keep sampling in the box.
                    pass
        
        # Final Polish: Very fine random walk around global best
        if self.best_x is not None and self.calls < self.budget and self._check_time():
            x = self.best_x.copy()
            f = self.best_f
            sigma = 0.1
            while self.calls < self.budget and self._check_time():
                z = rng.standard_normal(self.dim)
                x_new = x + sigma * z
                x_new = np.clip(x_new, -5, 5)
                f_new = safe_func(x_new)
                
                if f_new < f:
                    x = x_new
                    f = f_new
                    sigma *= 1.1
                else:
                    sigma *= 0.95
                
                if sigma < 1e-4:
                    sigma = 0.05

        return self.best_x, self.best_f
