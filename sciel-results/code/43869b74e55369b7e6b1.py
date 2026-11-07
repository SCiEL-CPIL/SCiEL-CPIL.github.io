"""Multi-objective optimizer for ZDT2."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective optimizer for ZDT2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim

        # ZDT2 specific: f1 = x1, f2 = (1+10*(n-1))*(1-sqrt(f1/g1)*g2)
        # Pareto front: x2...xn = 0, f1 in [0,1], f2 = 11*(1-sqrt(f1))
        
        # We'll use a strategy that exploits ZDT2 structure:
        # 1. First, try to find points on/near the Pareto front by setting x2...xn close to 0
        # 2. Then refine with local search
        
        # Initial population: mix of random and structured points
        pop_size = min(50, self.budget // 4)
        
        # Create structured initial population
        X = np.zeros((pop_size, n))
        F = np.zeros((pop_size, self.n_obj))
        
        # First half: structured points on/near Pareto front
        for i in range(pop_size // 2):
            x1 = (i + 1) / (pop_size // 2 + 1)  # Spread x1 in (0,1)
            X[i, 0] = x1
            X[i, 1:] = 0.0  # Set other variables to 0 for Pareto front
            F[i] = problem(X[i])
        
        # Second half: random points
        for i in range(pop_size // 2, pop_size):
            X[i] = lo + (hi - lo) * np.random.random(n)
            F[i] = problem(X[i])
        
        used = pop_size
        all_F = [F.copy()]  # Keep track of all evaluated objective vectors
        
        # Evolutionary loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Find non-dominated points
            nd_mask = self._nondominated_mask(F)
            parents = X[nd_mask] if np.any(nd_mask) else X
            
            # Select parent
            p_idx = np.random.randint(len(parents))
            parent = parents[p_idx]
            
            # Mutation: adaptive step size
            step = 0.05 * (hi - lo)
            child = parent + step * np.random.randn(n)
            child = np.clip(child, lo, hi)
            
            # For ZDT2, encourage x2...xn to be small
            if np.random.random() < 0.3:
                child[1:] = np.clip(child[1:] * 0.5, 0, 1)
            
            fc = problem(child)
            used += 1
            all_F.append(fc.reshape(1, -1))
            
            # Replace worst point
            worst = np.argmax(np.sum(F, axis=1))
            X[worst] = child
            F[worst] = fc
        
        # Final polish: try to improve points near the Pareto front
        if time.time() < deadline - 10 and used < self.budget:
            # Take the best points and do local refinement
            best_indices = np.argsort(np.sum(F, axis=1))[:10]
            for idx in best_indices:
                if used >= self.budget or time.time() > deadline:
                    break
                for _ in range(5):
                    if used >= self.budget or time.time() > deadline:
                        break
                    x = X[idx].copy()
                    # Local search: small perturbations
                    x[1:] = np.clip(x[1:] * 0.8, 0, 1)  # Push x2...xn toward 0
                    x[0] = np.clip(x[0] + 0.01 * np.random.randn(), 0, 1)
                    fx = problem(x)
                    used += 1
                    all_F.append(fx.reshape(1, -1))
                    if np.sum(fx) < np.sum(F[idx]):
                        X[idx] = x
                        F[idx] = fx
        
        return X

    def _nondominated_mask(self, F):
        """Return boolean mask of non-dominated points."""
        n = len(F)
        nd = np.ones(n, dtype=bool)
        for i in range(n):
            if not nd[i]:
                continue
            for j in range(n):
                if i == j or not nd[j]:
                    continue
                # j dominates i if j <= i in all objectives and j < i in at least one
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd[i] = False
                    break
        return nd
# EVOLVE-BLOCK-END
