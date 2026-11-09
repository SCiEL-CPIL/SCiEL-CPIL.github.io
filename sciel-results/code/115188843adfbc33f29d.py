"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """MOEA/D-DE: decomposition-based multi-objective optimizer with differential evolution."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit with 20s safety margin
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size: balance between diversity and computation
        N = min(30, max(10, self.budget // 7))
        
        # Generate weight vectors using simplex-lattice method
        weights = self._generate_weights(N, self.n_obj)
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Compute ideal point for normalization
        ideal = F.min(axis=0)
        nadir = F.max(axis=0)
        span = nadir - ideal
        span[span < 1e-10] = 1e-10
        
        # DE parameters
        CR = 0.5  # crossover rate
        F_mut = 0.5  # mutation factor
        
        while used < self.budget:
            # Check time budget
            if time.time() > deadline:
                break
            
            # For each subproblem, generate a child via DE
            for i in range(N):
                if used >= self.budget or time.time() > deadline:
                    break
                
                # Select neighborhood (simple: use all other solutions for small N)
                # For efficiency, pick 2 random parents different from i
                p1, p2 = rng.choice(N, size=2, replace=False)
                while p1 == i:
                    p1 = rng.integers(N)
                while p2 == i:
                    p2 = rng.integers(N)
                
                # DE/rand/1 mutation
                v = X[p1] + F_mut * (X[p2] - X[i])
                
                # Binary crossover
                j_rand = rng.integers(self.dim)
                mask = rng.random(self.dim) < CR
                mask[j_rand] = True
                child = np.where(mask, v, X[i])
                child = np.clip(child, lo, hi)
                
                # Evaluate
                fc = problem(child)
                used += 1
                
                # Update if child is better in terms of Tchebycheff aggregation
                self._update_solution(F, X, fc, child, i, weights, ideal, nadir, span)
            
            # Update ideal point
            ideal = F.min(axis=0)
            nadir = F.max(axis=0)
            span = nadir - ideal
            span[span < 1e-10] = 1e-10
        
        return X

    def _generate_weights(self, N, n_obj):
        """Generate weight vectors using simplex-lattice method."""
        rng = np.random.default_rng(42)
        if n_obj == 1:
            return np.ones((N, 1)) / N
        
        # Use uniform sampling on the simplex for simplicity
        weights = []
        for _ in range(N):
            w = rng.random(n_obj)
            w = w / w.sum()
            weights.append(w)
        return np.array(weights)

    def _update_solution(self, F, X, fc, child, i, weights, ideal, nadir, span):
        """Update solution i if child is better in Tchebycheff aggregation."""
        w = weights[i]
        
        # Current Tchebycheff value
        F_norm = (F[i] - ideal) / span
        current_tch = max(w * F_norm)
        
        # Child Tchebycheff value
        fc_norm = (fc - ideal) / span
        child_tch = max(w * fc_norm)
        
        if child_tch < current_tch:
            F[i] = fc
            X[i] = child
# EVOLVE-BLOCK-END
