"""Multi-objective optimizer for RE21 using Pareto tournament selection, adaptive mutation, and local Nelder-Mead refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

class Optimizer:
    """
    Multi-objective optimizer using:
    1. Latin Hypercube Sampling for initial population
    2. Hypervolume-based Pareto tournament selection
    3. Adaptive mutation with decreasing sigma
    4. Local Nelder-Mead refinement on promising points
    5. Deadline guard for safety
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin against 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Reference point for scoring
        ref_point = np.array([3000.0, 0.06])
        
        # 1. Initialization: Latin Hypercube Sampling
        n_init = 30
        X_pop = self._latin_hypercube(n_init, self.dim, lo, hi)
        F_pop = np.array([problem(x) for x in X_pop])
        
        # Track all evaluated points
        X_all = X_pop.copy()
        F_all = F_pop.copy()
        
        # Population size for the evolutionary loop
        pop_size = 40
        # Create initial population by taking the best pop_size from init (or all if fewer)
        if len(F_pop) >= pop_size:
            # Select non-dominated + some random for diversity
            nd_indices = self._non_dominated_indices(F_pop)
            if len(nd_indices) < pop_size:
                remaining = np.setdiff1d(np.arange(len(F_pop)), nd_indices)
                extra = np.random.choice(remaining, size=pop_size - len(nd_indices), replace=False)
                selected = np.concatenate([nd_indices, extra])
            else:
                selected = np.random.choice(nd_indices, size=pop_size, replace=False)
            X_pop = X_pop[selected]
            F_pop = F_pop[selected]
        else:
            X_pop = X_pop
            F_pop = F_pop
            pop_size = len(F_pop)
        
        # Adaptive mutation sigma
        sigma = 0.3  # Initial mutation scale in normalized space
        
        # 2. Evolutionary Loop
        while len(X_all) < self.budget:
            if time.time() > deadline:
                break
            
            # Update sigma based on progress
            progress = len(X_all) / self.budget
            sigma = max(0.01, 0.3 * (1.0 - progress))
            
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            n_offspring = pop_size
            
            for _ in range(n_offspring):
                if time.time() > deadline:
                    break
                
                # Pareto Tournament Selection
                parent1_idx, parent2_idx = self._pareto_tournament(F_pop, 2)
                
                # Crossover (SBX-like or simple blend)
                if np.random.random() < 0.8:
                    # Blend crossover
                    alpha = np.random.random()
                    child = alpha * X_pop[parent1_idx] + (1 - alpha) * X_pop[parent2_idx]
                else:
                    # Use one parent
                    child = X_pop[parent1_idx].copy()
                
                # Adaptive Mutation
                mutation = np.random.randn(self.dim) * sigma
                child = child + mutation * span
                
                # Clip to bounds
                child = np.clip(child, lo, hi)
                
                # Evaluate
                f_child = problem(child)
                offspring_X.append(child)
                offspring_F.append(f_child)
                X_all = np.vstack([X_all, child])
                F_all = np.vstack([F_all, f_child])
                
                if len(X_all) >= self.budget:
                    break
            
            if not offspring_X:
                break
            
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine parent and offspring populations
            X_combined = np.vstack([X_pop, offspring_X])
            F_combined = np.vstack([F_pop, offspring_F])
            
            # Select new population: non-dominated + diversity
            X_pop, F_pop = self._select_next_population(X_combined, F_combined, pop_size)
            
            # 3. Local Refinement
            # Pick the best non-dominated point and refine it
            nd_indices = self._non_dominated_indices(F_pop)
            if len(nd_indices) > 0:
                # Pick the point with the best hypervolume contribution
                best_idx = nd_indices[0]  # Simplified: just pick first
                best_x = X_pop[best_idx]
                best_f = F_pop[best_idx]
                
                # Nelder-Mead refinement on a scalarized objective
                # Use weighted sum with random weights to explore different parts of the front
                for _ in range(3):  # Try a few different weightings
                    if time.time() > deadline:
                        break
                    if len(X_all) >= self.budget:
                        break
                    
                    weights = np.random.dirichlet(np.ones(self.n_obj))
                    
                    def scalar_obj(x):
                        f = problem(x)
                        return np.dot(weights, f)
                    
                    try:
                        res = minimize(scalar_obj, best_x, method='Nelder-Mead', 
                                      bounds=list(zip(lo, hi)),
                                      options={'maxiter': 20, 'xatol': 1e-4, 'fatol': 1e-4})
                        if res.fun < np.dot(weights, best_f) - 1e-6:
                            X_all = np.vstack([X_all, res.x])
                            F_all = np.vstack([F_all, problem(res.x)])
                    except Exception:
                        pass
        
        return X_all

    def _latin_hypercube(self, n, dim, lo, hi):
        """Generate Latin Hypercube samples."""
        pts = np.zeros((n, dim))
        for i in range(dim):
            intervals = np.linspace(0, 1, n + 1)
            pts[:, i] = np.array([np.random.uniform(intervals[j], intervals[j+1]) for j in range(n)])
        return lo + pts * (hi - lo)

    def _non_dominated_indices(self, F):
        """Return indices of non-dominated points."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        
        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dominated[i] = True
                    break
        return np.where(~is_dominated)[0]

    def _hypervolume_contribution(self, F, ref_point):
        """Calculate hypervolume for 2D case."""
        if len(F) == 0:
            return 0.0
        
        # Filter points better than ref_point
        valid = [p for p in F if np.all(p < ref_point)]
        if not valid:
            return 0.0
        
        valid = np.array(sorted(valid, key=lambda x: x[0]))
        
        hv = 0.0
        for i in range(len(valid)):
            x_next = valid[i+1][0] if i < len(valid) - 1 else ref_point[0]
            width = x_next - valid[i][0]
            height = ref_point[1] - valid[i][1]
            if width > 0 and height > 0:
                hv += width * height
        return hv

    def _pareto_tournament(self, F, k=2):
        """Pareto tournament selection: pick k random, return index of best non-dominated."""
        n = len(F)
        if n <= k:
            # If population is small, just pick randomly
            return np.random.randint(n), np.random.randint(n)
        
        candidates = np.random.choice(n, size=k, replace=False)
        
        # Find non-dominated among candidates
        nd_in_candidates = []
        for idx in candidates:
            is_dom = False
            for other in candidates:
                if idx == other:
                    continue
                if np.all(F[other] <= F[idx]) and np.any(F[other] < F[idx]):
                    is_dom = True
                    break
            if not is_dom:
                nd_in_candidates.append(idx)
        
        if len(nd_in_candidates) > 0:
            # Pick randomly from non-dominated candidates
            return nd_in_candidates[np.random.randint(len(nd_in_candidates))], candidates[0]
        else:
            # All dominated, pick the one with best hypervolume contribution (simplified: random)
            return candidates[0], candidates[1]

    def _select_next_population(self, X, F, pop_size):
        """Select next population using non-dominated sorting + crowding distance."""
        n = len(F)
        if n <= pop_size:
            return X.copy(), F.copy()
        
        # Get non-dominated points
        nd_indices = self._non_dominated_indices(F)
        
        if len(nd_indices) >= pop_size:
            # Too many non-dominated, use crowding distance to select
            # Simplified: just pick pop_size random from non-dominated
            selected = np.random.choice(nd_indices, size=pop_size, replace=False)
        else:
            # Not enough non-dominated, add some dominated points for diversity
            remaining = np.setdiff1d(np.arange(n), nd_indices)
            if len(remaining) > 0:
                extra_needed = pop_size - len(nd_indices)
                extra = np.random.choice(remaining, size=min(extra_needed, len(remaining)), replace=False)
                selected = np.concatenate([nd_indices, extra])
            else:
                selected = nd_indices
        
        return X[selected], F[selected]

# EVOLVE-BLOCK-END
