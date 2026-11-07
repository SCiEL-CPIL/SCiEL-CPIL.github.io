"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer using Tchebycheff scalarization with multi-start Nelder-Mead.
    Strategy:
    1. Generate a set of Tchebycheff weight vectors that spread across the objective space.
    2. For each weight vector, run a local Nelder-Mead search to find a point on the Pareto front.
    3. Use the budget efficiently by allocating evaluations across multiple weight vectors.
    4. This approach is different from evolutionary strategies as it uses local search guided by
       scalarization, which can be more efficient for smooth, low-dimensional problems like RE34.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0
        range_vec = hi - lo
        
        # Generate Tchebycheff weight vectors
        # For 3 objectives, we want to spread weights across the simplex
        n_weights = 12  # Number of Tchebycheff directions
        weights = self._generate_tchebycheff_weights(self.n_obj, n_weights)
        
        # Allocate budget: reserve some for initialization, rest for local searches
        init_budget = 10
        remaining_budget = self.budget - init_budget
        evals_per_search = max(5, remaining_budget // len(weights))
        
        # Initialize with some random points to establish initial bounds
        X_init = np.random.rand(init_budget, self.dim) * range_vec + lo
        F_init = np.array([problem(x) for x in X_init])
        used = init_budget
        
        # Find initial min/max for each objective
        f_min = np.min(F_init, axis=0)
        f_max = np.max(F_init, axis=0)
        
        # For each weight vector, run a local search
        all_F = [f for f in F_init]
        
        for i, w in enumerate(weights):
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            
            # Initial point for this search: random or from initialization
            x0 = np.random.rand(self.dim) * range_vec + lo
            f0 = problem(x0)
            used += 1
            all_F.append(f0.copy())
            
            # Run Nelder-Mead with Tchebycheff scalarization
            x_best, f_best, evals_used = self._nelder_mead_tchebycheff(
                problem, x0, w, lo, hi, f_min, f_max, 
                evals_per_search, used, deadline
            )
            used += evals_used
            if f_best is not None:
                all_F.append(f_best.copy())
        
        return all_F

    def _generate_tchebycheff_weights(self, n_obj, n_weights):
        """
        Generate Tchebycheff weight vectors spread across the simplex.
        For n_obj objectives, weights sum to 1 and are non-negative.
        """
        weights = []
        
        # For 2 objectives: line from (1,0) to (0,1)
        # For 3 objectives: triangle vertices and edges
        if n_obj == 2:
            for i in range(n_weights):
                t = i / (n_weights - 1) if n_weights > 1 else 0.5
                w = np.array([1 - t, t])
                weights.append(w)
        elif n_obj == 3:
            # Generate points on the 2D simplex (triangle)
            # Use a grid-like approach
            n_per_edge = max(2, int(np.ceil(np.sqrt(n_weights))))
            for i in range(n_per_edge):
                for j in range(n_per_edge - i):
                    k = n_per_edge - i - j
                    w = np.array([i, j, k], dtype=float)
                    if np.sum(w) > 0:
                        w = w / np.sum(w)
                        weights.append(w)
            # Ensure we have at least n_weights points
            while len(weights) < n_weights:
                w = np.random.dirichlet(np.ones(n_obj))
                weights.append(w)
            # Trim if too many
            if len(weights) > n_weights:
                # Keep evenly spaced subset
                indices = np.linspace(0, len(weights) - 1, n_weights, dtype=int)
                weights = [weights[i] for i in indices]
        else:
            # General case: random Dirichlet samples
            for _ in range(n_weights):
                w = np.random.dirichlet(np.ones(n_obj))
                weights.append(w)
        
        return weights

    def _nelder_mead_tchebycheff(self, problem, x0, w, lo, hi, f_min, f_max, 
                                  max_evals, used, deadline):
        """
        Run Nelder-Mead local search minimizing the Tchebycheff scalarization:
        T(x, w) = max_j [ w_j * (f_j(x) - f_min_j) / (f_max_j - f_min_j + eps) ]
        """
        eps = 1e-8
        dim = self.dim
        
        # Initialize simplex
        simplex = [x0.copy()]
        f0 = problem(x0)
        t0 = self._tchebycheff_value(f0, w, f_min, f_max, eps)
        simplex_F = [f0.copy()]
        simplex_T = [t0]
        evals = 1
        
        # Create initial simplex by perturbing each dimension
        for d in range(dim):
            x_new = x0.copy()
            delta = 0.1 * (hi[d] - lo[d])
            x_new[d] += delta
            x_new = np.clip(x_new, lo, hi)
            
            if evals >= max_evals:
                break
            if time.time() > deadline:
                break
            
            f_new = problem(x_new)
            t_new = self._tchebycheff_value(f_new, w, f_min, f_max, eps)
            simplex.append(x_new.copy())
            simplex_F.append(f_new.copy())
            simplex_T.append(t_new)
            evals += 1
        
        # Nelder-Mead iterations
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink
        
        best_idx = 0
        for _ in range(max_evals - evals):
            if time.time() > deadline:
                break
            
            # Sort simplex by Tchebycheff value
            order = np.argsort(simplex_T)
            simplex = [simplex[i] for i in order]
            simplex_F = [simplex_F[i] for i in order]
            simplex_T = [simplex_T[i] for i in order]
            best_idx = 0
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Try reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, lo, hi)
            
            f_ref = problem(x_ref)
            t_ref = self._tchebycheff_value(f_ref, w, f_min, f_max, eps)
            evals += 1
            
            if t_ref < simplex_T[0]:
                # Try expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, lo, hi)
                f_exp = problem(x_exp)
                t_exp = self._tchebycheff_value(f_exp, w, f_min, f_max, eps)
                evals += 1
                
                if t_exp < t_ref:
                    simplex[-1] = x_exp
                    simplex_F[-1] = f_exp
                    simplex_T[-1] = t_exp
                else:
                    simplex[-1] = x_ref
                    simplex_F[-1] = f_ref
                    simplex_T[-1] = t_ref
            elif t_ref < simplex_T[-2]:
                # Accept reflection
                simplex[-1] = x_ref
                simplex_F[-1] = f_ref
                simplex_T[-1] = t_ref
            else:
                # Try contraction
                if t_ref < simplex_T[-1]:
                    # Outside contraction
                    x_con = centroid + rho * (x_ref - centroid)
                else:
                    # Inside contraction
                    x_con = centroid - rho * (centroid - simplex[-1])
                
                x_con = np.clip(x_con, lo, hi)
                f_con = problem(x_con)
                t_con = self._tchebycheff_value(f_con, w, f_min, f_max, eps)
                evals += 1
                
                if t_con < min(t_ref, simplex_T[-1]):
                    simplex[-1] = x_con
                    simplex_F[-1] = f_con
                    simplex_T[-1] = t_con
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lo, hi)
                        f_shrink = problem(simplex[i])
                        simplex_F[i] = f_shrink
                        simplex_T[i] = self._tchebycheff_value(f_shrink, w, f_min, f_max, eps)
                        evals += 1
                        if evals >= max_evals:
                            break
                    if evals >= max_evals:
                        break
        
        # Return best point found
        if simplex_T:
            best_local = np.argmin(simplex_T)
            return simplex[best_local], simplex_F[best_local], evals
        else:
            return None, None, evals

    def _tchebycheff_value(self, f, w, f_min, f_max, eps):
        """
        Compute Tchebycheff scalarization value.
        T(x, w) = max_j [ w_j * (f_j(x) - f_min_j) / (f_max_j - f_min_j + eps) ]
        """
        f_range = f_max - f_min + eps
        normalized = (f - f_min) / f_range
        weighted = w * normalized
        return np.max(weighted)
# EVOLVE-BLOCK-END
