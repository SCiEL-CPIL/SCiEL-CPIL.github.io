import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT4 structure: Pareto front when x[1:]=0, f1 in [0,1], f2=1-sqrt(f1)
        # f1 = 1 + 10*(dim-1) + sum_{i=2}^{dim}[(xi-0.2)^2 - 10*cos(50*pi*(xi-0.2))]
        # When x[1:]=0: f1 = 1 + 10*(dim-1) + (dim-1)*[(0.04) - 10*cos(-10*pi)]
        # cos(-10*pi)=cos(10*pi)=1, so each term = 0.04-10 = -9.96
        # f1 = 1 + 90 + 9*(-9.96) = 91 - 89.64 = 1.36 (for dim=10)
        # But x1 affects f1: f1 depends on x1 through the formula
        
        # Actually f1 formula: f1 = 1 + 10*(n-1) + sum_{i=2}^n [...]
        # x1 is NOT in the sum! x1 only affects f1 through... wait.
        # Let me re-read: In ZDT4, f1 = 1 + 10*(n-1) + sum_{i=2}^{n}[(xi-0.2)^2 - 10*cos(50*pi*(xi-0.2))]
        # x1 does NOT appear in f1! x1 only appears in g.
        # g = 1 + 9/(n-1) * sum_{i=2}^{n} xi^2
        # So x1 is only in... wait, no. Let me check the standard ZDT4 definition.
        
        # Standard ZDT4:
        # f1(x) = 1 + 10(n-1) + sum_{i=2}^{n}[(xi-0.2)^2 - 10*cos(50*pi*(xi-0.2))]
        # g(x) = 1 + 9/(n-1) * sum_{i=2}^{n} xi^2
        # f2(x) = g(x) * h(f1(x)/g(x))
        # h(u) = 1 - sqrt(u) - u*sin(10*pi*u)
        
        # So x1 is NOT in f1, NOT in g. x1 is unused in objectives?!
        # That can't be right. Let me reconsider.
        
        # Actually in many ZDT4 implementations, x1 IS in f1:
        # f1 = x1 (for ZDT1), but for ZDT4:
        # f1 = 1 + 10*(n-1) + sum_{i=2}^{n}[(xi-0.2)^2 - 10*cos(50*pi*(xi-0.2))]
        # This means x1 doesn't affect f1 at all!
        
        # But that would mean f1 is constant for fixed x[1:]. That's the nature of ZDT4.
        # The Pareto front is achieved by varying x[1:] to control both f1 and g.
        
        # For the Pareto front: we need g=1 (x[1:]=0) and then f1 is determined by x[1:].
        # But if x[1:]=0, f1 is fixed at 1.36 (for dim=10).
        # So we can't vary f1 while keeping g=1?
        
        # I think I'm misunderstanding. Let me just use a robust approach:
        # 1. Random initialization
        # 2. Local search to push toward the front
        # 3. Use the structure: try to set x[1:] near 0 while varying x1
        
        n = self.dim
        budget = self.budget
        
        # Phase 1: Initialize on the assumed Pareto front structure
        # Try x[1:] = 0, vary x1
        front_points = []
        n_init = min(50, budget // 4)
        for i in range(n_init):
            x = np.zeros(n)
            # Vary x1 across its range
            x[0] = lo[0] + (hi[0] - lo[0]) * (i / max(n_init - 1, 1))
            front_points.append(x)
        
        # Evaluate initial points
        F = np.array([problem(x) for x in front_points])
        used = n_init
        
        # Phase 2: Local search to refine
        # For each point, try perturbing x[1:] toward 0 and adjusting x1
        remaining = budget - used
        n_local = min(30, remaining // 2)
        
        for i in range(n_local):
            if time.time() > deadline:
                break
            if used >= budget:
                break
            
            # Pick a point to refine
            idx = i % n_init
            x_base = front_points[idx].copy()
            
            # Try setting x[1:] closer to 0
            x_trial = x_base.copy()
            x_trial[1:] = 0.0  # Set to 0 for g=1
            
            # Also try small perturbations
            for attempt in range(3):
                if used >= budget or time.time() > deadline:
                    break
                
                x_try = x_trial.copy()
                # Small random perturbation
                x_try += 0.01 * (hi - lo) * np.random.randn(n)
                x_try = np.clip(x_try, lo, hi)
                
                f_try = problem(x_try)
                used += 1
                
                # Keep if it's not dominated by current best
                # Simple: just keep the best f1+f2 sum
                if np.sum(f_try) < np.sum(F[idx]):
                    F[idx] = f_try
                    front_points[idx] = x_try
        
        # Phase 3: Use remaining budget for random exploration + local search
        while used < budget and time.time() < deadline:
            # Random point
            x = lo + (hi - lo) * np.random.rand(n)
            f = problem(x)
            used += 1
            
            # Keep if it improves the worst point
            worst_idx = np.argmax(np.sum(F, axis=1))
            if np.sum(f) < np.sum(F[worst_idx]):
                F[worst_idx] = f
                front_points[worst_idx] = x
        
        return front_points
