"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start local search using Nelder-Mead simplex restarts."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 310.0  # Safety margin before 330s kill
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # We want to maximize hypervolume, so we store all non-dominated points found
        # The evaluator records every returned objective vector.
        # We will return a list of points at the end, but __call__ return is ignored.
        # Wait, the contract says: "Every returned objective vector is recorded".
        # This implies that `problem(x)` returns the vector, and the evaluator records it.
        # The return value of __call__ is ignored.
        # So I just need to call `problem(x)` for valid x's.
        
        # Strategy:
        # 1. Generate a set of initial random points (say 10% of budget).
        # 2. For each remaining budget, perform a local search around the best non-dominated point found so far.
        #    Use a simple coordinate descent or Nelder-Mead variant.
        #    Given the tight budget (200) and dim (12), a full Nelder-Mead is expensive (dim+1 simplex).
        #    Let's use a "hill climbing" approach: pick a point, try small perturbations in random directions.
        #    If better (non-dominated), move there.
        
        # Better approach for MO: 
        # Keep an archive of non-dominated points.
        # Use the budget to explore around the archive.
        
        # Let's implement a simple MOEA/D-like or NSGA-II-lite? No, too complex for 200 evals.
        # Let's stick to a robust multi-start local search.
        
        # 1. Initialize with a spread of points.
        n_init = 20
        X_init = lo + (hi - lo) * rng.random((n_init, self.dim))
        
        # Evaluate initial points
        # We must record every evaluation.
        # The evaluator handles recording. We just call problem(x).
        
        # We need to track which points are non-dominated to guide search.
        # F_list stores (x, f) for non-dominated points.
        nd_points = []
        
        for x in X_init:
            f = problem(x)
            self._add_to_archive(nd_points, x, f)
            
        used = n_init
        
        # 2. Local search phase
        # We will pick a point from the archive and perturb it.
        # To be efficient, we'll use a simple random walk with step size decay.
        
        # Since budget is small, we can afford to do slightly larger steps initially.
        # Let's use a strategy: 
        # - Pick a random non-dominated point.
        # - Generate a candidate by adding Gaussian noise.
        # - If the candidate is non-dominated w.r.t the archive, add it.
        # - Even if dominated, if it's "close" to a boundary, maybe keep it? 
        #   No, for hypervolume, only non-dominated points on the front matter directly, 
        #   but intermediate points can help find the front.
        
        # Actually, the evaluator records EVERY point. 
        # The hypervolume is calculated on ALL recorded points.
        # So dominated points don't hurt the score (they just don't add volume), 
        # but they consume budget.
        # We want to maximize the number of non-dominated points found.
        
        # Let's use a "greedy" hill climbing on a scalarized objective?
        # No, let's use a simple evolutionary step but with local exploitation.
        
        # Revised Strategy:
        # 1. Init 10 points.
        # 2. Loop:
        #    - Select a parent from the current non-dominated set (uniformly).
        #    - Mutate with a step size that adapts (starts large, shrinks).
        #    - Evaluate.
        #    - If non-dominated, add to set.
        #    - This is essentially a (1+1) EA on the non-dominated set.
        
        # To improve on the seed (which does this with fixed 0.1 step), 
        # I will:
        # 1. Use a larger initial population (e.g., 30).
        # 2. Use adaptive step size: start at 0.5*(hi-lo), shrink by 0.95 each step.
        # 3. Use a "crowding" distance heuristic? No, too slow.
        # 4. Just pure exploitation of the best non-dominated points.
        
        # Let's try a different angle: 
        # The WFG4 problem is known to have a specific Pareto front.
        # With 12 dims and 200 budget, random search is weak.
        # Local search is strong.
        
        # Implementation:
        # 1. Init 15 random points.
        # 2. For remaining budget:
        #    - Pick a point from the current non-dominated archive.
        #    - Create a child using Gaussian perturbation with sigma = current_step.
        #    - current_step *= 0.98 (slow decay)
        #    - Clip to bounds.
        #    - Evaluate.
        #    - Update archive.
        
        # To ensure we explore the whole front, we should occasionally pick from the "edges" of the archive.
        # But simple uniform selection from the archive is a good start.
        
        # Let's refine the selection: 
        # Sort archive by first objective. Pick from the first 1/3, middle 1/3, last 1/3 with equal probability?
        # This ensures coverage.
        
        pass # placeholder for logic below

    def _add_to_archive(self, archive, x, f):
        """
        Add (x, f) to archive if it is non-dominated.
        archive is a list of (x, f).
        Returns True if added.
        """
        # Check if f is dominated by any existing point
        for _, f_existing in archive:
            if np.all(f_existing <= f) and np.any(f_existing < f):
                return False # f is dominated
        
        # Remove any existing points dominated by f
        new_archive = []
        for x_existing, f_existing in archive:
            if not (np.all(f <= f_existing) and np.any(f < f_existing)):
                new_archive.append((x_existing, f_existing))
        
        new_archive.append((x, f))
        
        # Clear and update to save memory/time (list reassignment)
        # Since we can't easily clear a passed list without returning it, 
        # let's just modify in place or return the new list.
        # For efficiency in the loop, let's just use a list and filter.
        archive.clear()
        archive.extend(new_archive)
        return True

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 310.0
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # 1. Initialization
        n_init = 15
        X_init = lo + range_ * rng.random((n_init, self.dim))
        
        archive = [] # List of (x, f)
        
        for x in X_init:
            f = problem(x)
            self._add_to_archive(archive, x, f)
            
        used = n_init
        step_size = 0.5 # Relative to range
        
        while used < self.budget:
            if time.time() - start_time > deadline:
                break
            
            if not archive:
                # Should not happen, but safety
                x = lo + range_ * rng.random(self.dim)
                f = problem(x)
                self._add_to_archive(archive, x, f)
                used += 1
                continue
            
            # Select parent
            # To encourage diversity, we pick from different "sectors" of the objective space
            # Simple strategy: Uniform random from archive
            idx = rng.integers(len(archive))
            parent_x, parent_f = archive[idx]
            
            # Mutate
            # Gaussian noise scaled by step_size
            noise = rng.standard_normal(self.dim) * (step_size * range_)
            child_x = parent_x + noise
            child_x = np.clip(child_x, lo, hi)
            
            # Evaluate
            child_f = problem(child_x)
            used += 1
            
            # Update archive
            self._add_to_archive(archive, child_x, child_f)
            
            # Decay step size
            step_size *= 0.95
            
        return archive # Return value is ignored, but good practice
# EVOLVE-BLOCK-END
