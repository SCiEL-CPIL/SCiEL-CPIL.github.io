"""Optimized program for WFG9 using DE with Tchebycheff decomposition."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """DE-based Multi-Objective Optimizer with Tchebycheff decomposition."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Generate reference points on the simplex for n_obj objectives
        # For 3 objectives, we need points on a triangle.
        # Number of reference points: let's use 25 for good coverage.
        n_ref = 25
        
        # Generate uniform points on the simplex
        ref_points = self._generate_ref_points(n_ref, self.n_obj)
        
        # Initialize population: for each reference point, generate a random solution
        pop_size = n_ref
        X = np.zeros((pop_size, self.dim))
        F = np.zeros((pop_size, self.n_obj))
        
        for i in range(pop_size):
            X[i] = lo + range_ * np.random.rand(self.dim)
            F[i] = problem(X[i])
        
        evaluations = pop_size
        
        # DE parameters
        CR = 0.9
        F_DE = 0.8
        
        # Main loop
        while evaluations < self.budget:
            if time.time() > deadline:
                break
            
            # For each subproblem (reference point), perform DE
            # We iterate through reference points. To respect budget, we do one DE step per eval.
            # We select a random reference point index to update.
            # Actually, standard DE-NSGA-II or similar iterates through the whole population or uses a specific schedule.
            # With budget 200 and pop 25, we have ~175 steps. We can cycle through the 25 individuals.
            
            # Let's just pick a random individual to evolve each step.
            # This ensures all subproblems get explored over time.
            idx = np.random.randint(pop_size)
            
            # Select parents: two other random individuals
            # Ensure they are different from idx
            candidates = [j for j in range(pop_size) if j != idx]
            p1_idx, p2_idx = np.random.choice(candidates, 2, replace=False)
            
            p1 = X[p1_idx]
            p2 = X[p2_idx]
            current = X[idx]
            
            # DE mutation: v = x_r1 + F * (x_r2 - x_r3) ? No, standard is x_r1 + F*(x_r2 - x_r3)
            # Let's pick 3 distinct parents for 'DE/rand/1'
            # We already have p1, p2. Let's pick p3.
            candidates3 = [j for j in range(pop_size) if j != idx and j != p1_idx and j != p2_idx]
            if len(candidates3) > 0:
                p3_idx = np.random.choice(candidates3)
            else:
                p3_idx = p2_idx # Fallback, though unlikely with pop 25
            
            p3 = X[p3_idx]
            
            v = p1 + F_DE * (p2 - p3)
            
            # Crossover
            child = np.zeros_like(v)
            j_rand = np.random.randint(self.dim)
            for i in range(self.dim):
                if np.random.rand() < CR or i == j_rand:
                    child[i] = v[i]
                else:
                    child[i] = current[i]
            
            # Clip to bounds
            child = np.clip(child, lo, hi)
            
            # Evaluate
            fc = problem(child)
            evaluations += 1
            
            # Update: Tchebycheff scalarization
            # Check if child is better than current for this reference point
            # Scalarized fitness: max_i (lambda_i * |f_i - z_i|)
            # We need a weight vector for this subproblem.
            w = ref_points[idx]
            
            # Get ideal point approximation (min of F)
            # For speed, we can use the current population's mins or a running min.
            # Let's use a running ideal point to avoid recalculating mins of F every time?
            # Recalculating mins of F (25x3) is fast.
            z = np.min(F, axis=0)
            
            def tcheb(f, w, z):
                # f is vector, w is weight, z is ideal
                # max_i (w_i * |f_i - z_i|)
                # Note: w should be normalized or scaled appropriately.
                # Standard: max_i (lambda_i * |f_i - z_i|)
                # Since f is minimized, and z is min, f-z >= 0.
                # If f < z (impossible if z is min of current pop, but possible if we update z later?), 
                # actually z is min of current F, so f_i >= z_i for all i in F.
                # But child fc might be better than z.
                # If fc_i < z_i, then |fc_i - z_i| = z_i - fc_i.
                # The formula is max_i (lambda_i * |f_i - z_i|).
                # Let's compute it.
                return np.max(w * np.abs(f - z))
            
            curr_fit = tcheb(F[idx], w, z)
            child_fit = tcheb(fc, w, z)
            
            # Accept if child is better (lower scalarized fitness)
            # Also, to maintain diversity, we might accept if it's close?
            # Standard DE for MOO often uses strict improvement or probabilistic acceptance.
            # Let's use strict improvement on the scalarized objective.
            # If equal, maybe accept with small prob? No, keep simple.
            if child_fit < curr_fit:
                X[idx] = child
                F[idx] = fc
            else:
                # Even if not better in scalarized sense, if it's non-dominated by current population?
                # No, that breaks the decomposition structure.
                # However, if we reject too much, we stagnate.
                # Let's add a small random restart or mutation if rejected?
                # Or just accept if it improves any objective?
                # Let's stick to Tchebycheff improvement for now.
                pass
            
        return X

    def _generate_ref_points(self, n_points, n_obj):
        """Generate n_points uniformly distributed on the (n_obj-1) simplex."""
        # For 3 objectives, simplex is a triangle.
        # We can use a grid or random sampling with rejection?
        # Simple method: generate random points in [0,1]^n, normalize to sum to 1.
        # This gives uniform distribution on simplex.
        pts = np.random.rand(n_points, n_obj)
        sums = pts.sum(axis=1, keepdims=True)
        pts = pts / sums
        return pts
# EVOLVE-BLOCK-END
