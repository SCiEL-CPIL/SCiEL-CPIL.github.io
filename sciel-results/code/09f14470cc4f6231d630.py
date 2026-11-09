"""Multi-objective optimizer for DTLZ7 using analytical Pareto front initialization and local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ7.
    
    Strategy:
    1. Sample points on the analytical Pareto front of DTLZ7.
    2. For each point, optimize the distance variables (x[3:]) to minimize g.
    3. Use a small population-based local search with non-dominated sorting.
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

        # DTLZ7 structure:
        # x[0], x[1], x[2] determine the shape of the front
        # x[3:] are distance variables
        # The Pareto front is defined by g(x) = 0, which means x[3:] are at their optimal values.
        
        # For DTLZ7, the optimal distance variables are typically at the lower bound (0) 
        # or specific values that make g=0. Let's assume x[3:] = 0 is a good starting point.
        
        # Generate diverse points on the Pareto front
        # x[0], x[1], x[2] should be spread across [0,1]
        
        # We'll use a combination of:
        # 1. Grid points on the front
        # 2. Random points with local refinement
        
        budget_used = 0
        archive = []  # List of (x, f) pairs
        archive_x = []
        archive_f = []
        
        def dominates(a, b):
            """Check if a dominates b (a is better in all objectives, at least one strictly better)"""
            return np.all(a <= b) and np.any(a < b)
        
        def is_non_dominated(point_f, archive_f):
            """Check if point_f is non-dominated with respect to archive_f"""
            for af in archive_f:
                if dominates(af, point_f):
                    return False
            return True
        
        def add_to_archive(x, f):
            """Add point to archive if non-dominated"""
            non_local = is_non_dominated(f, archive_f)
            if non_local:
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
        
        # Phase 1: Sample points on the analytical Pareto front
        # For DTLZ7, the Pareto front is a 2D surface in objective space
        # x[0], x[1], x[2] parameterize the front
        # Let's sample a grid of points
        
        # Number of initial samples: use a fraction of the budget
        n_init = min(30, self.budget // 5)
        
        # Create diverse points on the front
        # x[0] in [0, 0.5], x[1] in [0, 0.5], x[2] in [0, 1] (typical for DTLZ7)
        # Actually, let's be more systematic
        
        # Sample x[0], x[1], x[2] from a Latin hypercube or grid
        xs0 = np.linspace(0.0, 0.5, n_init)
        xs1 = np.linspace(0.0, 0.5, n_init)
        
        # Create a set of initial points
        initial_points = []
        for i in range(n_init):
            x = np.zeros(n)
            x[0] = xs0[i % len(xs0)]
            x[1] = xs1[i % len(xs1)]
            x[2] = np.random.uniform(0, 1)
            # x[3:] = 0 (assumed optimal for distance)
            initial_points.append(x)
        
        # Evaluate initial points
        for x in initial_points:
            if budget_used >= self.budget:
                break
            if time.time() > deadline:
                break
            f = problem(x)
            add_to_archive(x.copy(), f)
            budget_used += 1
        
        # Phase 2: Local refinement using local search
        # For each point in the archive, try to improve it by perturbing x[3:]
        
        # We'll do a simple local search: perturb x[3:] and see if g decreases
        # But we don't have direct access to g, only to the objectives.
        
        # Alternative: use a population-based approach
        # Keep a population of the best non-dominated points
        # Mutate and evaluate
        
        # Let's do a simple evolutionary local search
        pop_size = 10
        X = np.array(archive_x[:pop_size]) if len(archive_x) >= pop_size else np.array(archive_x)
        F = np.array(archive_f[:len(X)])
        
        # If we don't have enough points, add random ones
        while len(X) < pop_size and budget_used < self.budget:
            x = np.random.uniform(lo, hi)
            f = problem(x)
            X = np.vstack([X, x]) if len(X) > 0 else x.reshape(1, -1)
            F = np.vstack([F, f]) if len(F) > 0 else f.reshape(1, -1)
            budget_used += 1
        
        # Local search loop
        while budget_used < self.budget and time.time() < deadline:
            # Select a random point from the population
            idx = np.random.randint(len(X))
            parent = X[idx].copy()
            
            # Mutate: perturb x[3:] with small steps
            # Also perturb x[0], x[1], x[2] slightly to explore the front
            mutation_strength = 0.05
            child = parent.copy()
            
            # Perturb all variables
            child += np.random.normal(0, mutation_strength, size=n)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            budget_used += 1
            
            # Replace the worst point in the population
            # Use a simple criterion: sum of objectives
            sums = np.sum(F, axis=1)
            worst_idx = np.argmax(sums)
            
            # Only replace if child is better or non-dominated
            if np.sum(fc) < sums[worst_idx] or not any(dominates(f, fc) for f in F):
                X[worst_idx] = child
                F[worst_idx] = fc
                add_to_archive(child.copy(), fc)
        
        # Return the archive (or a subset)
        # The return value is ignored, but we need to return something
        if len(archive_x) > 0:
            return np.array(archive_x)
        return np.array([np.random.uniform(lo, hi)])
# EVOLVE-BLOCK-END
