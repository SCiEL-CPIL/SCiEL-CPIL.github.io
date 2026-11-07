import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj
        
        # State
        evals_used = 0
        archive_X = []
        archive_F = []
        
        def evaluate(x):
            nonlocal evals_used
            if evals_used >= budget or time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=np.float64)
            evals_used += 1
            return f
            
        def dominates(f1, f2):
            # f1 dominates f2 if f1 <= f2 and f1 != f2
            return np.all(f1 <= f2) and np.any(f1 < f2)
            
        def update_archive(x, f):
            if f is None:
                return
            # Check if f is dominated by any in archive
            # If yes, discard
            for ax, af in zip(archive_X, archive_F):
                if dominates(af, f):
                    return
                    
            # Remove points from archive dominated by f
            to_remove = []
            for i, (ax, af) in enumerate(zip(archive_X, archive_F)):
                if dominates(f, af):
                    to_remove.append(i)
                    
            for idx in sorted(to_remove, reverse=True):
                archive_X.pop(idx)
                archive_F.pop(idx)
                
            archive_X.append(x)
            archive_F.append(f)

        # 1. Initialization with Latin Hypercube for better spread
        n_init = 40
        # LHS: divide each dimension into n_init equal intervals, pick one random point per interval
        lhs_points = np.zeros((n_init, dim))
        for d in range(dim):
            intervals = np.linspace(0, 1, n_init + 1)
            for i in range(n_init):
                lhs_points[i, d] = intervals[i] + np.random.random() * (intervals[i+1] - intervals[i])
        
        X_init = lo + span * lhs_points
        for x in X_init:
            f = evaluate(x)
            if f is not None:
                update_archive(x, f)
        
        # 2. NSGA-II like Evolution
        # We will run a loop until budget is nearly exhausted or time is up.
        # We maintain a population. Since we are tracking an archive, 
        # for the evolutionary step, we can just generate random perturbations of existing archive points 
        # or use a simple EA on the whole space.
        # Given the small budget (200), a full NSGA-II with population 40 is feasible.
        # Let's implement a simplified EA:
        # Pop size 40.
        # While evals < budget - 40 (reserve for local):
        #    Generate 40 children from Pop (crossover/mutation)
        #    Evaluate children
        #    Combine Pop + Children
        #    Select best 40 (Non-dominated + Crowding)
        
        # However, managing the population separately from the global archive is tricky.
        # Simpler: Just use the Archive as the population base for mutation, 
        # but ensure diversity by keeping non-dominated points.
        
        # Let's stick to a robust simple EA on the full space, 
        # but biased towards good regions.
        
        # Actually, let's just do a loop of "Generate 20 new points based on archive, Evaluate, Update Archive".
        # This is essentially a (1+1) EA or similar, but with multiple starts.
        
        # Better: 
        # While evals_used < budget - 50:
        #   Pick a random point from archive (or random if empty)
        #   Mutate it
        #   Evaluate
        #   Add to archive
        #   (This is what the parent did, but we can improve mutation strategy)
        
        # Let's try a specific "Diversification" strategy:
        # We want to cover the front.
        # We will generate points using a mix of:
        # 1. Random in box
        # 2. Mutation of archive points
        
        # Let's reserve 60 evals for local polish.
        # Global phase budget: budget - 60 - n_init
        n_global = budget - 60 - n_init
        
        for _ in range(n_global):
            if evals_used >= budget - 60 or time.time() > deadline:
                break
                
            # Select parent
            if len(archive_X) == 0:
                x = np.random.uniform(lo, hi)
            else:
                # Pick random parent
                idx = np.random.randint(len(archive_X))
                x = np.array(archive_X[idx])
                
                # Mutation: Simpler adaptive step
                # Start large, shrink linearly
                progress = evals_used / budget
                scale = 0.3 * (1.0 - progress) + 0.03
                
                # Use archive spread in decision space to inform mutation
                if len(archive_X) >= 3:
                    X_arr = np.array(archive_X)
                    # Estimate typical step size in decision space
                    x_spread = np.std(X_arr, axis=0)
                    # Scale mutation based on how spread out we are
                    spread_factor = 1.0 + 0.3 * np.mean(x_spread / (span + 1e-10))
                    scale *= spread_factor
                
                x = x + scale * span * np.random.standard_normal(dim)
                
            x = np.clip(x, lo, hi)
            f = evaluate(x)
            if f is not None:
                update_archive(x, f)
                
        # 3. Local Polish with Tchebycheff decomposition
        # Select diverse points from archive and use multiple weight vectors
        if len(archive_X) > 0:
            F_arr = np.array(archive_F)
            # Sort by first objective to ensure we pick across the front
            order = np.argsort(F_arr[:, 0])
            
            # Select up to 10 points, evenly spaced in index
            n_points_to_refine = min(10, len(archive_X))
            # Pick indices
            indices = np.linspace(0, len(order)-1, n_points_to_refine).astype(int)
            
            # Define diverse weight vectors for Tchebycheff decomposition
            # Evenly spaced in [0.02, 0.98] to cover the front well
            n_weights = 10
            weight_vectors = [np.array([w, 1-w]) for w in np.linspace(0.02, 0.98, n_weights)]
            
            # Nadir point estimate: max of current archive
            z_max = np.max(F_arr, axis=0) if len(F_arr) > 0 else np.ones(n_obj)
            z_max = z_max + 1e-6 * np.ones(n_obj)
            
            # Cycle through weight vectors for each refinement point
            for i, idx in enumerate(indices):
                if evals_used >= budget or time.time() > deadline:
                    break
                
                x0 = np.array(archive_X[order[idx]])
                f0 = np.array(archive_F[order[idx]])
                
                # Use cycling weight vectors to explore different parts of the front
                w = weight_vectors[i % len(weight_vectors)]
                
                # Track evaluations in the local polish to stay within budget
                local_evals = 0
                max_local_evals = 6
                
                def obj(x, w=w, z_max=z_max):
                    nonlocal local_evals
                    if local_evals >= max_local_evals or evals_used >= budget or time.time() > deadline:
                        return np.inf
                    x_clipped = np.clip(x, lo, hi)
                    f = np.array(problem(x_clipped), dtype=np.float64)
                    local_evals += 1
                    evals_used += 1
                    # Tchebycheff: max(w * (f - z))
                    return np.max(w * (f - z_max))
                    
                try:
                    res = minimize(obj, x0, method='Nelder-Mead', 
                                   options={'maxiter': 30, 'maxfev': max_local_evals, 'xatol': 1e-8, 'fatol': 1e-8})
                    # Evaluate res.x explicitly to get its true F and add to archive
                    if evals_used < budget:
                        x_best = np.clip(res.x, lo, hi)
                        f_best = evaluate(x_best)
                        if f_best is not None:
                            update_archive(x_best, f_best)
                except Exception:
                    pass
                    
        return archive_X
