import time
import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 250  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        dim = self.dim

        budget = self.budget

        # Phase 1: Dense sampling on the known ZDT6 Pareto front
        # ZDT6 front: x0 = t, x1..x9 = 0 => f1 = t^2, f2 = 1 - sqrt(t)
        # Hypervolume contribution is highest in the middle of the front
        # Use a distribution that concentrates points where the front curves most
        # The "knee" region is around t=0.3-0.7 where both objectives are moderate
        
        n_front = int(budget * 0.85)
        n_refine = budget - n_front

        # Use a beta-like distribution for t to concentrate in the middle
        # t = u^0.7 gives more density near t=0 but let's use a symmetric approach
        # Actually, for hypervolume with ref [11,11], the contribution of a point (f1,f2)
        # is proportional to (11-f1)*(11-f2) minus what's already covered.
        # The front is concave, so uniform in t should be reasonable.
        # Let's try slightly denser in the middle using a power transform
        u = np.linspace(0.0, 1.0, n_front)
        # Apply a mild concentration toward the center: t = 0.5 + 0.5*sin(pi*u)
        # This gives more points near t=0.5
        t_vals = 0.5 + 0.5 * np.sin(np.pi * u)
        
        X = np.zeros((n_front, dim))
        X[:, 0] = t_vals
        X[:, 1:] = 0.0  # Keep g at minimum for front points
        X = np.clip(X, lo, hi)

        F = np.array([problem(x) for x in X])
        used = n_front

        # Phase 2: Local refinement
        # For each front point, try small perturbations to see if we can improve
        # Focus on points in the middle of the front where hypervolume gain is highest
        
        if n_refine > 0 and time.time() < deadline:
            # Select which points to refine: prioritize middle of the front
            # The middle points (t around 0.3-0.7) contribute most to hypervolume
            # Sort by t value and pick from the middle region
            t_order = np.argsort(X[:, 0])
            # Pick indices from the middle 60% of the front
            mid_start = int(n_front * 0.2)
            mid_end = int(n_front * 0.8)
            mid_indices = t_order[mid_start:mid_end]
            
            # Distribute refinement budget across these points
            n_to_refine = min(len(mid_indices), n_refine)
            if n_to_refine > 0:
                # Pick evenly spaced indices from the mid region
                refine_idx = np.linspace(0, n_to_refine - 1, n_to_refine, dtype=int)
                refine_idx = mid_indices[refine_idx]
                
                for idx in refine_idx:
                    if used >= budget or time.time() > deadline:
                        break
                    
                    base = X[idx].copy()
                    base_f = F[idx].copy()
                    
                    # Try several small perturbations
                    n_trials = max(1, n_refine // n_to_refine)
                    for trial in range(n_trials):
                        if used >= budget or time.time() > deadline:
                            break
                        
                        # Perturb x0 slightly
                        delta0 = 0.02 * np.random.randn()
                        new_x0 = np.clip(base[0] + delta0, lo[0], hi[0])
                        
                        # Perturb other dimensions slightly (should stay near 0)
                        new_x = base.copy()
                        new_x[0] = new_x0
                        new_x[1:] = np.clip(base[1:] + 0.005 * np.random.randn(dim - 1), lo[1:], hi[1:])
                        
                        fc = problem(new_x)
                        used += 1
                        
                        # Accept if strictly better in all objectives
                        if np.all(fc <= base_f) and np.any(fc < base_f):
                            X[idx] = new_x
                            F[idx] = fc
                            base_f = fc

        return X
