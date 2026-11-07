"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Differential Evolution with Nelder-Mead polish for multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(0)
        
        # Population size: balance between exploration and generations
        pop_size = 20
        budget = self.budget
        
        # Initialize population uniformly
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size
        
        # DE/rand/1 parameters
        F_factor = 0.8  # scale factor
        CR = 0.5  # crossover rate
        
        # Track best non-dominated set for polish
        best_front = list(zip(X, F))
        
        while used < budget:
            # Check time
            if time.time() > deadline:
                break
            
            for i in range(pop_size):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Select three distinct random indices different from i
                idx = list(range(pop_size))
                idx.remove(i)
                a, b, c = rng.choice(idx, size=3, replace=False)
                
                # Create trial vector using DE/rand/1
                v = X[a] + F_factor * (X[b] - X[c])
                
                # Binomial crossover
                r = rng.random(self.dim)
                jrand = rng.integers(self.dim)
                for j in range(self.dim):
                    if r[j] < CR or j == jrand:
                        v[j] = v[j]
                    else:
                        v[j] = X[i][j]
                
                # Clip to bounds
                v = np.clip(v, lo, hi)
                
                # Evaluate
                fv = problem(v)
                used += 1
                
                # Compare with target
                if self._dominates(fv, F[i]):
                    F[i] = fv
                    X[i] = v
                elif not self._dominates(F[i], fv):
                    # Both non-dominated: keep the one with better crowding or just replace
                    # Simple strategy: replace with some probability
                    if rng.random() < 0.5:
                        F[i] = fv
                        X[i] = v
        
        # Nelder-Mead polish on diverse points from the front
        # Select up to 5 diverse points from non-dominated set
        nd_points = self._get_non_dominated(X, F)
        
        # Select diverse points for polish
        polish_points = self._select_diverse(nd_points, max_count=5)
        
        remaining_budget = budget - used
        polish_per_point = max(1, remaining_budget // max(1, len(polish_points)))
        
        for x0, f0 in polish_points:
            if used >= budget:
                break
            if time.time() > deadline:
                break
            
            # Simple local refinement using small steps
            best_x = x0.copy()
            best_f = f0.copy()
            step_size = 0.05
            
            for _ in range(min(polish_per_point, 20)):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Try perturbing in random directions
                direction = rng.standard_normal(self.dim)
                direction /= (np.linalg.norm(direction) + 1e-10)
                
                new_x = np.clip(best_x + step_size * direction, lo, hi)
                new_f = problem(new_x)
                used += 1
                
                if self._dominates(new_f, best_f):
                    best_x = new_x
                    best_f = new_f
                    step_size *= 0.7  # reduce step on success
                else:
                    step_size *= 0.5  # reduce step on failure
                
                # Also try opposite direction
                if used < budget and time.time() < deadline:
                    new_x2 = np.clip(best_x - step_size * direction, lo, hi)
                    new_f2 = problem(new_x2)
                    used += 1
                    if self._dominates(new_f2, best_f):
                        best_x = new_x2
                        best_f = new_f2
                        step_size *= 0.7
        
        return X

    def _dominates(self, a, b):
        """Check if a dominates b (a <= b in all objectives, a < b in at least one)."""
        return np.all(a <= b) and np.any(a < b)

    def _get_non_dominated(self, X, F):
        """Get non-dominated points."""
        nd = []
        for i in range(len(X)):
            dominated = False
            for j in range(len(X)):
                if i != j and self._dominates(F[j], F[i]):
                    dominated = True
                    break
            if not dominated:
                nd.append((X[i], F[i]))
        return nd

    def _select_diverse(self, points, max_count=5):
        """Select diverse points from a list of (x, f) tuples."""
        if len(points) <= max_count:
            return points
        
        selected = [points[0]]
        remaining = list(points[1:])
        
        for _ in range(max_count - 1):
            best_idx = 0
            best_min_dist = -1
            
            for i, (x, f) in enumerate(remaining):
                min_dist = min(np.linalg.norm(x - sx) for sx, sf in selected)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        return selected
# EVOLVE-BLOCK-END
