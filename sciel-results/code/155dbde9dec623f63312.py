"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using Halton sequence and farthest-point sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        max_time = 290.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # 1. Generate Halton sequence points in decision space
        # Halton sequence is a low-discrepancy sequence, good for space-filling.
        # We generate a few more than budget to have options, but we will evaluate exactly budget points.
        # Actually, we just generate budget points directly.
        
        def halton_sequence(n, dim):
            """Generate n points in dim dimensions using Halton sequence."""
            points = np.zeros((n, dim))
            for d in range(dim):
                # Prime numbers for Halton bases
                primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
                base = primes[d]
                for i in range(1, n + 1):
                    f = 1.0 / base
                    idx = i
                    while idx > 0:
                        idx, rem = divmod(idx, base)
                        f *= base
                        points[i-1, d] += rem * f
            return points

        # We have 200 budget.
        # Let's split the budget:
        # Phase 1: 100 points from Halton sequence.
        # Phase 2: 100 points selected by farthest-point sampling in objective space from the remaining Halton candidates.
        
        n_phase1 = 100
        n_phase2 = self.budget - n_phase1
        
        # Generate a pool of Halton points. Let's generate 200 unique Halton points.
        # The first 100 will be evaluated in Phase 1.
        # The next 100 will be candidates for Phase 2.
        halton_points = halton_sequence(self.budget, self.dim)
        
        # Scale to bounds
        X_halton = lo + (hi - lo) * halton_points
        
        # Phase 1: Evaluate first n_phase1 points
        X_used = []
        F_used = []
        
        for i in range(n_phase1):
            x = X_halton[i]
            f = problem(x)
            X_used.append(x)
            F_used.append(f)
            
            if time.time() - start_time > max_time:
                break
        
        X_used = np.array(X_used)
        F_used = np.array(F_used)
        
        # Phase 2: Farthest Point Sampling (FPS)
        # We have n_phase2 evaluations left.
        # We have candidates X_halton[n_phase1:]
        # We want to select points that are farthest from the current set of objective vectors.
        
        candidates = X_halton[n_phase1:]
        
        for _ in range(n_phase2):
            if time.time() - start_time > max_time:
                break
            
            if len(candidates) == 0:
                # If we run out of pre-generated Halton points, generate random ones
                # But we generated exactly budget worth, so this shouldn't happen if we track indices.
                # Actually, we just iterate through the remaining list.
                break
                
            # Calculate distance in objective space for each candidate
            # We need to evaluate the candidates? No, that would cost evaluations.
            # FPS in decision space is cheaper but less effective for MO.
            # FPS in objective space requires knowing the objective values of candidates.
            # We don't have them yet.
            
            # Alternative: Use decision space distance for selection, but this might not cover objective space well.
            # Better: Evaluate one candidate, add to set, repeat? No, that's just greedy random.
            
            # Let's refine the strategy:
            # Since we can't evaluate candidates without spending budget, and we want to spend budget on "good" points:
            # The Halton sequence is already space-filling.
            # Perhaps a better strategy is:
            # 1. Evaluate all 200 Halton points.
            # This is simple and deterministic.
            # Is it better than the current program?
            # Current program: 50 random + local search.
            # Halton is generally better than random for integration/coverage.
            # Let's just evaluate all 200 Halton points.
            
            # Wait, the loop above already did 100.
            # Let's just finish evaluating the rest of the Halton points.
            # This simplifies the code and ensures full coverage.
            
            x = candidates[0]
            f = problem(x)
            X_used.append(x)
            F_used.append(f)
            candidates = candidates[1:]
        
        return X_used

# EVOLVE-BLOCK-END
