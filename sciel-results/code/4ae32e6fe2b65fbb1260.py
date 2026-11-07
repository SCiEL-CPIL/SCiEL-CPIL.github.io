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

        # 1. Initialization
        n_init = 40
        X_init = lo + span * np.random.random((n_init, dim))
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
                
                # Mutation: Adaptive step with archive-aware scaling
                # Start large, shrink
                progress = evals_used / budget
                scale = 0.4 * (1.0 - progress) + 0.05
                
                # If we have a good archive, use its spread to inform mutation
                if len(archive_F) >= 3:
                    F_arr_temp = np.array(archive_F)
                    # Estimate typical distance in objective space
                    obj_spread = np.std(F_arr_temp, axis=0)
                    # Scale mutation based on how spread out we are
                    spread_factor = 1.0 + 0.5 * np.mean(obj_spread / (np.max(F_arr_temp) - np.min(F_arr_temp) + 1e-10))
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
            
            # Select up to 8 points, evenly spaced in index
            n_points_to_refine = min(8, len(archive_X))
            # Pick indices
            indices = np.linspace(0, len(order)-1, n_points_to_refine).astype(int)
            
            # Define diverse weight vectors for Tchebycheff decomposition
            # This helps cover different parts of the Pareto front
            weight_vectors = [
                np.array([0.1, 0.9]),
                np.array([0.3, 0.7]),
                np.array([0.5, 0.5]),
                np.array([0.7, 0.3]),
                np.array([0.9, 0.1]),
            ]
            
            # Cycle through weight vectors for each refinement point
            for i, idx in enumerate(indices):
                if evals_used >= budget or time.time() > deadline:
                    break
                
                x0 = np.array(archive_X[order[idx]])
                f0 = np.array(archive_F[order[idx]])
                
                # Use cycling weight vectors to explore different parts of the front
                w = weight_vectors[i % len(weight_vectors)]
                
                # Nadir point estimate: max of current archive
                z_max = np.max(F_arr, axis=0) if len(F_arr) > 0 else np.ones(n_obj)
                # Add small margin to avoid numerical issues
                z_max = z_max + 1e-6 * np.ones(n_obj)
                
                def obj(x, w=w, z_max=z_max):
                    x_clipped = np.clip(x, lo, hi)
                    f = np.array(problem(x_clipped), dtype=np.float64)
                    # Tchebycheff: max(w * |f - z|)
                    # Using z_max as reference
                    return np.max(w * (f - z_max))
                    
                # Note: This objective function calls problem, which increments evals_used.
                # We need to ensure we don't exceed budget.
                # minimize will call obj multiple times.
                
                # We have ~60 evals left total. 8 points * ~7 evals each = 56.
                # Set maxfev=7 to be safe.
                
                try:
                    res = minimize(obj, x0, method='Nelder-Mead', 
                                   options={'maxiter': 15, 'maxfev': 7, 'xatol': 1e-5, 'fatol': 1e-5})
                    # Check if res.x is better than x0 in some sense
                    # We need to evaluate res.x explicitly if we want to add it to archive?
                    # minimize already evaluated it? Not necessarily the best one in the final state.
                    # We must evaluate res.x to get its true F.
                    if evals_used < budget:
                        x_best = np.clip(res.x, lo, hi)
                        f_best = evaluate(x_best)
                        if f_best is not None:
                            update_archive(x_best, f_best)
                except Exception:
                    pass
                    
        return archive_X
