"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Decomposition-based multi-objective optimizer using Tchebycheff scalarization."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        
        # Generate 30 uniformly distributed weight vectors in the simplex
        n_weights = 30
        weights = self._generate_weight_vectors(n_weights, self.n_obj, rng)
        
        # Initialize with diverse random points
        n_init = min(15, self.budget // 2)
        X = lo + (hi - lo) * rng.random((n_init, self.dim))
        F = np.array([problem(x) for x in X])
        used = n_init
        
        # Track best point per weight vector
        best_X = [None] * n_weights
        best_F = [None] * n_weights
        best_tcheb = [np.inf] * n_weights
        
        # Assign initial points to nearest weight vectors
        for i in range(n_init):
            idx = self._assign_weight(F[i], weights, lo, hi)
            tcheb_val = self._tchebycheff_value(F[i], weights[idx], lo, hi)
            if tcheb_val < best_tcheb[idx]:
                best_tcheb[idx] = tcheb_val
                best_X[idx] = X[i].copy()
                best_F[idx] = F[i].copy()
        
        # Main loop: local search on each subproblem
        while used < self.budget:
            # Check time deadline
            if time.time() > deadline:
                break
            
            # Select next subproblem to work on (cycle through)
            for wi in range(n_weights):
                if used >= self.budget or time.time() > deadline:
                    break
                
                # Get current best for this subproblem
                if best_X[wi] is None:
                    # Initialize this subproblem with a random point
                    x = lo + (hi - lo) * rng.random(self.dim)
                    f = problem(x)
                    used += 1
                    best_X[wi] = x.copy()
                    best_F[wi] = f.copy()
                    best_tcheb[wi] = self._tchebycheff_value(f, weights[wi], lo, hi)
                    continue
                
                # Perform a few steps of Nelder-Mead-like local search
                x_best = best_X[wi].copy()
                f_best = best_F[wi].copy()
                tcheb_best = best_tcheb[wi]
                
                # Small perturbation search
                step_size = 0.05 * (hi - lo)
                for step in range(3):
                    if used >= self.budget or time.time() > deadline:
                        break
                    
                    # Random direction perturbation
                    direction = rng.standard_normal(self.dim)
                    direction /= (np.linalg.norm(direction) + 1e-10)
                    
                    # Try moving in this direction
                    x_new = x_best + step_size * direction
                    x_new = np.clip(x_new, lo, hi)
                    
                    f_new = problem(x_new)
                    used += 1
                    
                    tcheb_new = self._tchebycheff_value(f_new, weights[wi], lo, hi)
                    
                    if tcheb_new < tcheb_best:
                        x_best = x_new
                        f_best = f_new
                        tcheb_best = tcheb_new
                        best_X[wi] = x_best.copy()
                        best_F[wi] = f_best.copy()
                        best_tcheb[wi] = tcheb_best
                    else:
                        # Try opposite direction
                        x_new2 = x_best - step_size * direction
                        x_new2 = np.clip(x_new2, lo, hi)
                        f_new2 = problem(x_new2)
                        used += 1
                        
                        tcheb_new2 = self._tchebycheff_value(f_new2, weights[wi], lo, hi)
                        if tcheb_new2 < tcheb_best:
                            x_best = x_new2
                            f_best = f_new2
                            tcheb_best = tcheb_new2
                            best_X[wi] = x_best.copy()
                            best_F[wi] = f_best.copy()
                            best_tcheb[wi] = tcheb_best
                
                # Decrease step size slightly for next iteration
                step_size *= 0.9
        
        # Collect all non-dominated points
        all_X = [x for x in best_X if x is not None]
        all_F = [f for f in best_F if f is not None]
        
        if len(all_X) > 0:
            all_F = np.array(all_F)
            # Keep only non-dominated points
            nondom_mask = np.array([not self._is_dominated(all_F[i], all_F) for i in range(len(all_F))])
            all_X = np.array([all_X[i] for i in range(len(all_X)) if nondom_mask[i]])
        
        return all_X if len(all_X) > 0 else np.array([lo + (hi - lo) * 0.5])
    
    def _generate_weight_vectors(self, n_weights, n_obj, rng):
        """Generate uniformly distributed weight vectors in the (n_obj-1)-simplex."""
        # Use Dirichlet distribution for uniform distribution on simplex
        weights = rng.dirichlet(np.ones(n_obj), size=n_weights)
        # Add a few extreme points for better coverage
        extreme_weights = []
        for i in range(n_obj):
            w = np.zeros(n_obj)
            w[i] = 1.0
            extreme_weights.append(w)
        # Combine and deduplicate
        all_weights = list(weights) + extreme_weights
        # Select n_weights distinct ones
        selected = []
        for w in all_weights:
            if len(selected) >= n_weights:
                break
            if not any(np.allclose(w, s) for s in selected):
                selected.append(w)
        # Fill remaining with random
        while len(selected) < n_weights:
            selected.append(rng.dirichlet(np.ones(n_obj)))
        return np.array(selected[:n_weights])
    
    def _tchebycheff_value(self, f, weight, lo, hi):
        """Compute Tchebycheff scalarization value."""
        # Normalize objectives
        f_min = np.min(f)
        f_max = np.max(f)
        f_range = f_max - f_min + 1e-10
        f_norm = (f - f_min) / f_range
        return np.max(weight * f_norm)
    
    def _assign_weight(self, f, weights, lo, hi):
        """Assign objective vector to nearest weight vector."""
        f_min = np.min(f)
        f_max = np.max(f)
        f_range = f_max - f_min + 1e-10
        f_norm = (f - f_min) / f_range
        # Find weight vector that minimizes Tchebycheff distance
        best_idx = 0
        best_val = np.inf
        for i, w in enumerate(weights):
            val = np.max(w * f_norm)
            if val < best_val:
                best_val = val
                best_idx = i
        return best_idx
    
    def _is_dominated(self, f, all_f):
        """Check if f is dominated by any point in all_f."""
        for other_f in all_f:
            if np.all(other_f <= f) and np.any(other_f < f):
                return True
        return False
# EVOLVE-BLOCK-END
