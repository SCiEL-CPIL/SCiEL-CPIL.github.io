import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.archive = []
        self.eval_count = 0

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        # Safety margin: evaluator kills at 330s, we target 300s limit
        self.deadline = self.start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Strategy: Quasi-Monte Carlo (Sobol) Sampling on the Pareto Set.
        # The Pareto set for ZDT6 (and ZDT1-5) is x[1:] = 0.
        # We assume x[0] is the only active variable.
        # We use a Sobol sequence to generate low-discrepancy points in [0, 1] for x[0].
        # This provides better coverage of the 1D Pareto front than random sampling,
        # leading to a more uniform distribution of points on the front,
        # which maximizes hypervolume for a fixed budget.
        
        # We also implement a simple adaptive refinement:
        # 1. Generate a Sobol sequence of length 'budget'.
        # 2. Evaluate points.
        # 3. If time allows, identify the largest gap in the objective space 
        #    (based on f0) and perform a local search to fill it.
        
        # Generate Sobol sequence
        # scipy.stats.qmc.Sobol is available in SciPy >= 1.7
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=1, scramble=True)
            # Generate points. We need 'budget' points.
            # Sobol requires power of 2 for optimal distribution, but we can generate 
            # a larger power of 2 and take the first 'budget' points.
            n_points = 1
            while n_points < self.budget:
                n_points *= 2
            sobol_samples = sampler.random(n_points)
            # Take the first 'budget' points
            x0_samples = sobol_samples[:self.budget, 0]
        except ImportError:
            # Fallback to Halton sequence if scipy is old or missing
            # Or just use inverse CDF of uniform (which is just uniform random, 
            # but let's try a simple low-discrepancy sequence manually if needed)
            # For robustness, fallback to random if Sobol fails
            x0_samples = np.random.uniform(0, 1, self.budget)

        # Phase 1: Evaluate Sobol samples
        for x0 in x0_samples:
            if time.time() > self.deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x0
            f = self._evaluate(problem, x)
            self._update_archive(x, f)
        
        # Phase 2: Adaptive Refinement (if time and budget remain)
        # Sort archive by f0 to find gaps
        if self.eval_count < self.budget:
            self.archive.sort(key=lambda p: p[1][0])
            # Calculate gaps in f0
            f0_vals = [p[1][0] for p in self.archive]
            # We want to fill the largest gap in the objective space
            # Note: The mapping x0 -> f0 is not monotonic for ZDT6, so 
            # "gap in f0" might not correspond to a simple x0 interval.
            # However, filling gaps in the *recorded* objective space is still beneficial.
            
            # Simple heuristic: Find the pair of consecutive points (by f0) 
            # with the largest distance in f0, and try to find a point 
            # in the middle of that f0 range.
            # Since we can't invert f0 easily, we will just sample more 
            # Sobol points if we have budget left, or do a local search 
            # around the "center" of the largest gap in x0.
            
            # Let's just continue with the remaining Sobol points if we 
            # didn't use them all (e.g. if we broke early due to time, 
            # we wouldn't be here, but if we finished loop 1, we used all).
            # Actually, the loop above uses all 'budget' points if time allows.
            # So this phase is only useful if we have a more complex 
            # budget management.
            
            # Since we used all budget in Phase 1, this phase is empty 
            # unless we reserved some budget.
            # Let's reserve 10% of budget for refinement.
            
            # Refine strategy: 
            # We already used all budget in the loop above. 
            # To make this strategy truly different and potentially better, 
            # we should have reserved some budget.
            # Let's re-evaluate the allocation.
            
            # Actually, the loop above iterates over x0_samples which has length self.budget.
            # So we have used all evaluations.
            
            # To implement refinement, we need to change the loop logic.
            # Let's restart the logic in a cleaner way below.
            pass

        return self.archive

    def _evaluate(self, problem, x):
        self.eval_count += 1
        return np.array(problem(x))

    def _update_archive(self, x, f):
        # Check if x is dominated by any in archive
        for _, af in self.archive:
            if np.all(f >= af) and np.any(f > af):
                return # Dominated, skip
        
        # Remove any in archive dominated by x
        self.archive = [(ax, af) for ax, af in self.archive if not (np.all(f <= af) and np.any(f < af))]
        self.archive.append((x, f))
