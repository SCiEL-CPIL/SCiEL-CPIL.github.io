import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound_lo = -5.0
        self.bound_hi = 5.0
        self.start_time = None
        self.deadline = 325.0 # 5s margin

    def __call__(self, func):
        self.start_time = time.time()
        d = self.dim
        budget = self.budget
        lo = self.bound_lo
        hi = self.bound_hi
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Coarse Global Search (Scout)
        # Use a significant portion of the budget to find a good basin.
        # For d=5, 20-30 random points is a reasonable start.
        scout_budget = min(30, max(10, budget // 10))
        
        best_x = None
        best_f = np.inf
        scout_history = []
        
        # Sample uniformly
        for _ in range(scout_budget):
            x = rng.uniform(lo, hi, d)
            fx = func(x)
            scout_history.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        if best_x is None:
            best_x = np.zeros(d)
            best_f = func(best_x)
            
        # Sort scout history to find distinct basins
        scout_history.sort(key=lambda t: t[0])
        
        # Phase 2: Multi-Start Local Search (1+1-ES with Plateau Detection)
        # Select top K distinct starting points for local refinement.
        # "Distinct" means far enough from each other to likely be in different basins.
        K = 5
        starts = []
        threshold = 1.0 # Distance threshold for considering points as distinct
        
        for fx, x in scout_history:
            is_distinct = True
            for sx in starts:
                if np.linalg.norm(x - sx) < threshold:
                    is_distinct = False
                    break
            if is_distinct:
                starts.append(x)
            if len(starts) >= K:
                break
                
        # If we don't have enough distinct starts, fill with random or just use best
        while len(starts) < K:
            starts.append(best_x + rng.normal(0, 0.5, d))
            
        # Allocate remaining budget among starts
        remaining_budget = budget - scout_budget
        budget_per_start = max(10, remaining_budget // K)
        
        # For each start, run a (1+1)-ES with adaptive step size
        # (1+1)-ES is better for step functions (f7) as it doesn't get stuck in flat regions
        # if the step size is large enough to jump over plateaus, or small enough to exploit.
        
        global_best_x = best_x.copy()
        global_best_f = best_f
        
        for start_idx, start_x in enumerate(starts):
            # Check time
            if time.time() - self.start_time > self.deadline:
                break
                
            m = start_x.copy()
            sigma = 1.0 # Initial step size
            
            # If we are already at global best, start with smaller sigma?
            # No, let it explore.
            
            # Plateau detection:
            # If no improvement for N steps, increase sigma significantly.
            # If improvement is slow, decrease sigma.
            
            no_improve_count = 0
            last_best_f = m_f = func(m)
            # Actually m_f is already evaluated in scout or we need to eval it if it's a new start
            # Wait, start_x is from scout, so we have its f value?
            # Yes, in scout_history. But if we added random starts, we need to eval.
            if start_x in [sx for _, sx in scout_history]:
                 # Find f
                 for fx, sx in scout_history:
                     if np.allclose(sx, start_x):
                         m_f = fx
                         break
            else:
                 m_f = func(m)
                 
            if m_f < global_best_f:
                global_best_f = m_f
                global_best_x = m.copy()
                
            current_budget = budget_per_start
            while current_budget > 0:
                if time.time() - self.start_time > self.deadline:
                    break
                    
                # Generate candidate
                y = rng.standard_normal(d)
                x_new = m + sigma * y
                x_new = np.clip(x_new, lo, hi)
                
                f_new = func(x_new)
                current_budget -= 1
                
                if f_new < m_f:
                    # Improvement
                    m = x_new.copy()
                    m_f = f_new
                    no_improve_count = 0
                    # Shrink sigma slightly on success? Standard (1+1) often keeps sigma or shrinks.
                    # For step functions, keeping sigma or shrinking slowly is good.
                    sigma *= 0.95
                else:
                    # No improvement
                    no_improve_count += 1
                    # If stuck for a while, jump sigma to escape plateau
                    if no_improve_count > 10:
                        sigma *= 2.0
                        no_improve_count = 0 # Reset counter after jump
                    
                if m_f < global_best_f:
                    global_best_f = m_f
                    global_best_x = m.copy()
                    
                # Optional: if sigma becomes too small or too large, reset?
                if sigma < 1e-4:
                    break
                if sigma > 10.0:
                    sigma = 10.0

        return global_best_x, global_best_f
