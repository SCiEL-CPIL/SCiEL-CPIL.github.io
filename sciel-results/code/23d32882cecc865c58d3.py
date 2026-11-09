"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-start local search with Sobol initialization and adaptive (1+1)-ES refinement.
    
    Strategy:
    1. Initialize a diverse set of points using a low-discrepancy Sobol-like sequence
       (using Halton sequence as a simple deterministic low-discrepancy alternative).
    2. Evaluate all initial points to get broad coverage early (good for AOCC).
    3. For each of the top few candidates, run a (1+1)-ES refinement with adaptive step size.
    4. Track the global best throughout and return it.
    5. Explicit deadline guard to stay under 300s.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton_sequence(self, n, dim, skip=1):
        """Generate n Halton sequence points in [0,1)^dim, skipping first 'skip' points."""
        points = []
        for i in range(skip, skip + n):
            point = []
            for j in range(1, dim + 1):
                base = j + 1  # bases: 2, 3, 5, 7, ...
                f = 1.0
                result = 0.0
                i_val = i
                while i_val > 0:
                    f /= base
                    result += f * (i_val % base)
                    i_val //= base
                point.append(result)
            points.append(point)
        return np.array(points)

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Diverse initialization using Halton sequence
        # Use ~40% of budget for initialization to get good early coverage
        n_init = max(10, min(50, budget // 5))
        
        # Generate Halton points in [0,1)^dim, then scale to [-5,5]^dim
        halton_points = self._halton_sequence(n_init, dim, skip=1)
        init_points = halton_points * 10.0 - 5.0  # map [0,1) to [-5,5)

        # Evaluate all initial points
        best_x = None
        best_f = np.inf
        all_points = []
        all_fvals = []

        for p in init_points:
            if time.time() > deadline:
                break
            fp = func(p)
            all_points.append(p)
            all_fvals.append(fp)
            if fp < best_f:
                best_f = fp
                best_x = p.copy()

        # Phase 2: Refine the top candidates
        # Sort by function value and take top K for refinement
        if all_fvals:
            sorted_idx = np.argsort(all_fvals)
            top_k = min(5, len(sorted_idx))
            
            # For each top candidate, run (1+1)-ES refinement
            # Allocate remaining budget among refinements
            remaining_budget = budget - len(all_points)
            if remaining_budget <= 0:
                return best_x, best_f
            
            # Allocate more budget to better candidates
            for rank in range(top_k):
                if time.time() > deadline:
                    break
                
                idx = sorted_idx[rank]
                x = all_points[idx].copy()
                fx = all_fvals[idx]
                
                # Budget for this refinement: more for better candidates
                # Use a geometric allocation
                alloc = max(1, int(remaining_budget * (0.6 ** rank) / sum(0.6 ** r for r in range(top_k))))
                
                sigma = 1.0  # initial step size
                # Adaptive (1+1)-ES with 1/5 rule
                successes = 0
                trials = 0
                
                for _ in range(alloc):
                    if time.time() > deadline:
                        break
                    
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    
                    if fy < fx:
                        x, fx = y, fy
                        successes += 1
                        sigma *= 1.25
                    else:
                        sigma *= 0.85
                    
                    trials += 1
                    
                    # 1/5 success rule: adjust sigma based on success rate
                    if trials >= 10:
                        success_rate = successes / trials
                        if success_rate > 0.3:
                            sigma *= 1.1
                        elif success_rate < 0.1:
                            sigma *= 0.9
                        successes = 0
                        trials = 0
                    
                    # Update global best
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
            
            return best_x, best_f
        else:
            return best_x, best_f

# EVOLVE-BLOCK-END
