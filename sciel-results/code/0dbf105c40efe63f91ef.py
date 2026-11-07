# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective Optimizer using grid-based decomposition and Tchebycheff local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0
        self.start_time = time.time()
        # Safety margin: 330s limit, use 300s deadline
        self.deadline = self.start_time + 300

    def _check_time(self):
        return time.time() > self.deadline

    def _evaluate(self, x, problem):
        if self.eval_count >= self.budget:
            return None
        self.eval_count += 1
        try:
            return problem(x)
        except Exception:
            return None

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        X_list = []
        F_list = []
        
        def add_point(x, f):
            if f is not None and len(f) == self.n_obj and np.all(np.isfinite(f)):
                x_clipped = np.clip(x, lo, hi)
                X_list.append(x_clipped)
                F_list.append(f)

        # Phase 1: Grid-based initialization
        # Use a 3x3x3x3 grid = 81 points for systematic coverage
        # Plus corners and center for additional coverage
        n_per_dim = 3
        grid_points = []
        for d in range(self.dim):
            grid_points.append(np.linspace(lo[d], hi[d], n_per_dim))
        
        # Generate all grid vertices
        grid_vertices = np.array(np.meshgrid(*grid_points, indexing='ij')).reshape(self.dim, -1).T
        
        # Add corners (already in grid) and center
        center = (lo + hi) / 2
        all_init_points = np.vstack([grid_vertices, center.reshape(1, -1)])
        
        # Evaluate grid points
        for x in all_init_points:
            if self.eval_count >= self.budget or self._check_time():
                break
            f = self._evaluate(x, problem)
            if f is not None:
                add_point(x, f)
                
        if not F_list:
            x0 = (lo + hi) / 2
            f0 = self._evaluate(x0, problem)
            if f0 is not None:
                add_point(x0, f0)
            return X_list if X_list else [x0]

        def get_nondominated_indices(F):
            n = len(F)
            if n == 0:
                return []
            is_dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if is_dominated[i]:
                    continue
                for j in range(i + 1, n):
                    if is_dominated[j]:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        is_dominated[j] = True
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dominated[i] = True
                        break
            return np.where(~is_dominated)[0]

        F = np.array(F_list)
        X = np.array(X_list)
        
        # Phase 2: Identify promising regions and refine
        # Use Tchebycheff decomposition with more weight vectors
        # Generate weight vectors on the simplex
        weight_vectors = []
        
        # Add extreme points
        for i in range(self.n_obj):
            w = np.zeros(self.n_obj)
            w[i] = 1.0
            weight_vectors.append(w)
        
        # Add edge midpoints
        for i in range(self.n_obj):
            for j in range(i + 1, self.n_obj):
                w = np.zeros(self.n_obj)
                w[i] = 0.5
                w[j] = 0.5
                weight_vectors.append(w)
        
        # Add more diverse points
        n_extra = 20
        for _ in range(n_extra):
            w = np.random.dirichlet(np.ones(self.n_obj))
            weight_vectors.append(w)
        
        # Remove duplicates
        unique_weights = []
        for w in weight_vectors:
            is_dup = False
            for uw in unique_weights:
                if np.allclose(w, uw, atol=1e-6):
                    is_dup = True
                    break
            if not is_dup:
                unique_weights.append(w)
        weight_vectors = unique_weights
        
        # Budget allocation: remaining budget for refinement
        remaining_budget = self.budget - self.eval_count
        n_searches = min(len(weight_vectors), max(1, remaining_budget // 8))
        maxfev_per_search = max(5, remaining_budget // max(1, n_searches))
        
        # Sort weight vectors by diversity (spread)
        # Prioritize weight vectors that point in different directions
        def weight_spread(w):
            return np.std(w)
        
        # Keep all weight vectors, but limit the number of searches
        for i, w in enumerate(weight_vectors):
            if i >= n_searches:
                break
            if self.eval_count >= self.budget or self._check_time():
                break
            
            # Find the best starting point for this weight direction
            F_nd = F[get_nondominated_indices(F)] if len(get_nondominated_indices(F)) > 0 else F
            z = np.min(F_nd, axis=0)
            
            scores = np.max(w * (F - z), axis=1)
            best_idx = np.argmin(scores)
            x_start = X[best_idx].copy()
            
            def objective(x, w=w, z=z):
                f = self._evaluate(x, problem)
                if f is None:
                    return 1e6
                return np.max(w * (f - z))
            
            try:
                res = minimize(
                    objective, 
                    x_start, 
                    method='Nelder-Mead', 
                    bounds=list(zip(lo, hi)),
                    options={
                        'maxfev': maxfev_per_search, 
                        'xatol': 1e-4, 
                        'fatol': 1e-4,
                        'adaptive': True
                    }
                )
                
                if res.x is not None:
                    f_final = self._evaluate(res.x, problem)
                    if f_final is not None:
                        add_point(res.x, f_final)
                        F = np.array(F_list)
                        X = np.array(X_list)
                        
            except Exception:
                pass

        # Phase 3: Additional refinement on the best non-dominated points
        nd_indices = get_nondominated_indices(F)
        if len(nd_indices) > 0:
            # Refine the top few non-dominated points
            F_nd = F[nd_indices]
            # Sort by sum of objectives (lower is better)
            sum_obj = np.sum(F_nd, axis=1)
            sorted_idx = np.argsort(sum_obj)
            
            n_refine = min(5, len(sorted_idx))
            for k in range(n_refine):
                if self.eval_count >= self.budget or self._check_time():
                    break
                
                idx = nd_indices[sorted_idx[k]]
                x_start = X[idx].copy()
                
                # Use a small perturbation-based local search
                for attempt in range(3):
                    if self.eval_count >= self.budget or self._check_time():
                        break
                    
                    # Random perturbation
                    step_size = 0.05 * (1 - attempt * 0.3)
                    x_perturbed = x_start + np.random.uniform(-step_size, step_size, self.dim)
                    x_perturbed = np.clip(x_perturbed, lo, hi)
                    
                    f_perturbed = self._evaluate(x_perturbed, problem)
                    if f_perturbed is not None:
                        # Check if this improves any objective
                        f_current = F[idx]
                        if np.any(f_perturbed < f_current - 1e-6):
                            add_point(x_perturbed, f_perturbed)
                            F = np.array(F_list)
                            X = np.array(X_list)
                            nd_indices = get_nondominated_indices(F)
                            break

        return X_list
# EVOLVE-BLOCK-END
