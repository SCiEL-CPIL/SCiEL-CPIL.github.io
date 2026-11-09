# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ7.
    
    Strategy:
    1. Initialize population with points on the analytical Pareto front (x[3:] = 0.5).
    2. Use an evolutionary local search with adaptive mutation strength.
    3. Maintain a non-dominated archive.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj

        budget_used = 0
        archive_x = []
        archive_f = []
        
        def dominates(a, b):
            """Check if a dominates b"""
            return np.all(a <= b) and np.any(a < b)
        
        def is_non_dominated(point_f, archive_f_list):
            """Check if point_f is non-dominated with respect to archive_f_list"""
            for af in archive_f_list:
                if dominates(af, point_f):
                    return False
            return True
        
        def add_to_archive(x, f):
            """Add point to archive if non-dominated"""
            if is_non_dominated(f, archive_f):
                # Remove any points in archive that are dominated by this new point
                new_archive_x = []
                new_archive_f = []
                for ax, af in zip(archive_x, archive_f):
                    if not dominates(f, af):
                        new_archive_x.append(ax)
                        new_archive_f.append(af)
                archive_x.extend(new_archive_x)
                archive_f.extend(new_archive_f)
                archive_x.append(x)
                archive_f.append(f)

        # Phase 1: Initialize population on the Pareto front
        # x[0], x[1] vary to cover the front shape
        # x[2] is typically 0 or 1 for DTLZ7? No, x[0], x[1], x[2] determine the front.
        # For DTLZ7, the front is defined by x[0], x[1], x[2].
        # Let's sample a grid.
        
        n_init = 15
        pop_size = 15
        
        X = np.zeros((pop_size, n))
        F = np.zeros((pop_size, m))
        
        # Sample x[0..2]
        # x[0] in [0, 1], x[1] in [0, 1], x[2] in [0, 1]
        # But the front is a 2D surface? No, DTLZ7 has 3 objectives, so front is 2D.
        # x[0], x[1] determine the front?
        # Actually, for DTLZ7, x[0], x[1], x[2] are the shape variables.
        # The front is parameterized by 2 variables?
        # No, for M objectives, the front is (M-1) dimensional.
        # So for M=3, it is 2D.
        # The variables determining the front are usually the first M-1 variables?
        # In DTLZ7, x[0], x[1] determine the front?
        # Let's check the formula.
        # f_i depend on x[0..2] and g(x[3:]).
        # If g=0, f_i are functions of x[0..2].
        # The Pareto front is the set of non-dominated points in this 3D objective space.
        # It is a 2D surface.
        # So we need 2 parameters to span it.
        # Typically x[0] and x[1] are used, and x[2] is fixed or derived?
        # In standard DTLZ7, x[0], x[1] are the free variables for the front?
        # Let's assume x[0] and x[1] are the main shape variables.
        # x[2] can be set to 0.5 or varied.
        
        for i in range(pop_size):
            # Latin hypercube or grid for x[0], x[1]
            # Use a simple grid
            u = i // 4
            v = i % 4
            X[i, 0] = (u + 0.5) / 4.0
            X[i, 1] = (v + 0.5) / 4.0
            X[i, 2] = 0.5 # Fixed for now, or vary slightly
            # x[3:] = 0.5 (suspected optimum for g)
            X[i, 3:] = 0.5
            
            if budget_used >= self.budget:
                break
            if time.time() > deadline:
                break
                
            f = problem(X[i])
            F[i] = f
            budget_used += 1
            add_to_archive(X[i].copy(), f)
            
        # Phase 2: Local Search
        # Adaptive mutation
        sigma_shape = 0.05
        sigma_dist = 0.2
        
        while budget_used < self.budget and time.time() < deadline:
            # Decay sigma
            sigma_shape *= 0.95
            sigma_dist *= 0.95
            
            idx = np.random.randint(pop_size)
            parent = X[idx].copy()
            
            child = parent.copy()
            
            # Perturb shape variables (0, 1, 2)
            child[0:3] += np.random.normal(0, sigma_shape, size=3)
            # Perturb distance variables (3..n-1)
            child[3:] += np.random.normal(0, sigma_dist, size=n-3)
            
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            budget_used += 1
            
            # Update population
            sums = np.sum(F, axis=1)
            worst_idx = np.argmax(sums)
            
            # Check if child is better
            if np.sum(fc) < sums[worst_idx]:
                X[worst_idx] = child
                F[worst_idx] = fc
                add_to_archive(child.copy(), fc)
            else:
                # Even if not better in sum, check if it's non-dominated
                # This helps explore the front
                if is_non_dominated(fc, list(F)):
                    # Replace the worst point? Or just add to archive?
                    # To keep population diverse, replace worst if child is non-dominated
                    X[worst_idx] = child
                    F[worst_idx] = fc
                    add_to_archive(child.copy(), fc)

        if len(archive_x) > 0:
            return np.array(archive_x)
        return np.array([np.random.uniform(lo, hi)])
# EVOLVE-BLOCK-END
