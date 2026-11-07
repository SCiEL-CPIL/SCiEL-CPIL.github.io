"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simple evolutionary multi-objective search with adaptive step size and time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
        
        # Start with a small population, grow it over time
        pop = 10
        X = lo + (hi - lo) * rng.random((pop, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop
        
        # Adaptive step size: start small, increase if stuck, decrease if improving
        step_size = 0.05
        best_hv = -1.0
        last_improve_time = start_time
        no_improve_count = 0
        
        def compute_hv(F, ref):
            """Compute hypervolume for 3 objectives using simple inclusion-exclusion."""
            # Filter points strictly better than reference
            mask = np.all(F < ref, axis=1)
            F = F[mask]
            if len(F) == 0:
                return 0.0
            # Sort by first objective
            idx = np.argsort(F[:, 0])
            F = F[idx]
            # For 3D, we can use a sweep line approach
            # But for small populations, simple O(n^3) is fine
            # Actually, let's use a simpler approximation for speed
            # Hypervolume contribution: sum of volumes of boxes
            # This is an approximation - good enough for ranking
            total_hv = 0.0
            for i in range(len(F)):
                # Box from F[i] to min(F[j] for j!=i or ref)
                # Simplified: just use the volume from F[i] to ref, but that overcounts
                # Better: use the standard 3D hypervolume algorithm
                pass
            
            # Use a proper 3D hypervolume calculation
            # Sort by first objective
            F_sorted = F[np.argsort(F[:, 0])]
            
            # For each point, compute the volume it uniquely contributes
            # This is complex; let's use a simpler metric for internal ranking
            # Sum of normalized distances to origin (lower is better)
            # Actually, let's just use the sum of objectives as a proxy for now
            # and focus on the search strategy
            return -np.sum(F)
        
        while used < self.budget:
            # Check time
            if time.time() > deadline - 5:
                break
            
            # Find non-dominated points
            dominated = np.array([np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) for f in F])
            non_dom_idx = np.where(~dominated)[0]
            
            if len(non_dom_idx) == 0:
                parents = X
            else:
                parents = X[non_dom_idx]
            
            # Select parent with some randomness
            parent_idx = rng.integers(len(parents))
            parent = parents[parent_idx]
            
            # Adaptive mutation
            # If no improvement recently, increase step size
            if start_time + (time.time() - start_time) - last_improve_time > 2.0:
                no_improve_count += 1
                step_size = min(0.3, step_size * 1.5)
            else:
                no_improve_count = 0
                step_size = max(0.01, step_size * 0.95)
            
            child = parent + step_size * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1
            
            # Replace worst point (based on sum of objectives)
            worst = int(np.argmax(np.sum(F, axis=1)))
            
            # Track improvement
            if np.sum(fc) < np.sum(F[worst]):
                last_improve_time = time.time()
            
            X[worst], F[worst] = child, fc
            
            # Grow population if we have budget and it's small
            if pop < 30 and used < self.budget * 0.8 and used % 5 == 0:
                # Add a new random point
                new_point = lo + (hi - lo) * rng.random(self.dim)
                new_f = problem(new_point)
                used += 1
                if used < self.budget:
                    X = np.vstack([X, new_point])
                    F = np.vstack([F, new_f])
                    pop += 1
        
        return X
# EVOLVE-BLOCK-END
