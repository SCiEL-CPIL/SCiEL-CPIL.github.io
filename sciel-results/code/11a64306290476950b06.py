import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj
        
        # State
        evals_used = 0
        archive_X = []
        archive_F = []
        
        def evaluate(x):
            nonlocal evals_used
            if evals_used >= budget or time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=np.float64)
            evals_used += 1
            return f
            
        def dominates(f1, f2):
            # f1 dominates f2 if f1 <= f2 and f1 != f2
            return np.all(f1 <= f2) and np.any(f1 < f2)
            
        def update_archive(x, f):
            if f is None:
                return
            # Check if f is dominated by any in archive
            for ax, af in zip(archive_X, archive_F):
                if dominates(af, f):
                    return
                    
            # Remove points from archive dominated by f
            to_remove = []
            for i, (ax, af) in enumerate(zip(archive_X, archive_F)):
                if dominates(f, af):
                    to_remove.append(i)
                    
            for idx in sorted(to_remove, reverse=True):
                archive_X.pop(idx)
                archive_F.pop(idx)
                
            archive_X.append(x)
            archive_F.append(f)

        def compute_hypervolume(points, ref_point):
            """Compute hypervolume for a set of points in 2D."""
            if len(points) == 0:
                return 0.0
            
            # Filter points that are better than reference in all objectives
            valid = []
            for p in points:
                if all(p[i] < ref_point[i] for i in range(len(p))):
                    valid.append(p)
            
            if len(valid) == 0:
                return 0.0
            
            # Sort by first objective (ascending)
            valid.sort(key=lambda x: x[0])
            
            hv = 0.0
            prev_y = ref_point[1]
            for i, p in enumerate(valid):
                # Remove dominated points
                if i > 0 and p[1] >= valid[i-1][1]:
                    continue
                # Hypervolume contribution
                width = ref_point[0] - p[0]
                height = prev_y - p[1]
                if width > 0 and height > 0:
                    hv += width * height
                prev_y = min(prev_y, p[1])
            
            return hv

        # 1. Initialization with Latin Hypercube Sampling
        n_init = 30
        lhs_points = np.zeros((n_init, dim))
        for d in range(dim):
            intervals = np.linspace(0, 1, n_init + 1)
            for i in range(n_init):
                lhs_points[i, d] = intervals[i] + np.random.random() * (intervals[i+1] - intervals[i])
        
        X_init = lo + span * lhs_points
        for x in X_init:
            f = evaluate(x)
            if f is not None:
                update_archive(x, f)
        
        # 2. Differential Evolution with Hypervolume-based selection
        # Population size for DE
        pop_size = 20
        
        # Initialize population from archive + random points
        population_X = []
        population_F = []
        
        # Start with archive points
        for x, f in zip(archive_X, archive_F):
            population_X.append(x)
            population_F.append(f)
        
        # Fill with random points
        while len(population_X) < pop_size and evals_used < budget:
            x = np.random.uniform(lo, hi)
            f = evaluate(x)
            if f is not None:
                population_X.append(x)
                population_F.append(f)
        
        # DE parameters
        CR = 0.5  # Crossover rate
        F = 0.8   # Scale factor
        
        # Main DE loop
        while evals_used < budget - 20 and time.time() < deadline:
            # Generate offspring using differential evolution
            new_X = []
            new_F = []
            
            n_new = min(pop_size, budget - evals_used - 20)
            
            for _ in range(n_new):
                if evals_used >= budget - 20 or time.time() > deadline:
                    break
                
                # Select three distinct random individuals
                if len(population_X) < 3:
                    # Fallback to random if population too small
                    x = np.random.uniform(lo, hi)
                else:
                    # Pick three distinct indices
                    idx_a = np.random.randint(len(population_X))
                    idx_b = np.random.randint(len(population_X))
                    while idx_b == idx_a:
                        idx_b = np.random.randint(len(population_X))
                    idx_c = np.random.randint(len(population_X))
                    while idx_c in (idx_a, idx_b):
                        idx_c = np.random.randint(len(population_X))
                    
                    # Differential mutation
                    x = population_X[idx_a] + F * (population_X[idx_b] - population_X[idx_c])
                
                # Binomial crossover
                x_new = np.copy(x)
                for d in range(dim):
                    if np.random.random() < CR:
                        x_new[d] = x[d]
                    else:
                        x_new[d] = population_X[idx_a][d]
                
                x_new = np.clip(x_new, lo, hi)
                f_new = evaluate(x_new)
                
                if f_new is not None:
                    new_X.append(x_new)
                    new_F.append(f_new)
            
            # Combine population with offspring
            combined_X = population_X + new_X
            combined_F = population_F + new_F
            
            # Select best pop_size based on hypervolume contribution
            # Use greedy hypervolume selection
            selected_indices = []
            selected_F = []
            
            remaining = list(range(len(combined_F)))
            
            # Sort by first objective for efficiency
            remaining.sort(key=lambda i: combined_F[i][0])
            
            while len(selected_indices) < pop_size and remaining:
                # Find point that adds most hypervolume
                best_idx = None
                best_hv = -1
                
                for idx in remaining:
                    # Compute HV with current selected + this point
                    test_selected = selected_F + [combined_F[idx]]
                    hv = compute_hypervolume(test_selected, np.array([3000.0, 0.06]))
                    if hv > best_hv:
                        best_hv = hv
                        best_idx = idx
                
                if best_idx is None:
                    break
                
                selected_indices.append(best_idx)
                selected_F.append(combined_F[best_idx])
                remaining.remove(best_idx)
            
            # Update population
            population_X = [combined_X[i] for i in selected_indices]
            population_F = [combined_F[i] for i in selected_indices]
            
            # Update archive with new population points
            for x, f in zip(population_X, population_F):
                update_archive(x, f)
        
        # 3. Local polish with Tchebycheff decomposition
        if len(archive_X) > 0:
            F_arr = np.array(archive_F)
            order = np.argsort(F_arr[:, 0])
            
            n_points_to_refine = min(6, len(archive_X))
            indices = np.linspace(0, len(order)-1, n_points_to_refine).astype(int)
            
            weight_vectors = [
                np.array([0.1, 0.9]),
                np.array([0.3, 0.7]),
                np.array([0.5, 0.5]),
                np.array([0.7, 0.3]),
                np.array([0.9, 0.1]),
            ]
            
            for i, idx in enumerate(indices):
                if evals_used >= budget or time.time() > deadline:
                    break
                
                x0 = np.array(archive_X[order[idx]])
                
                w = weight_vectors[i % len(weight_vectors)]
                z_max = np.max(F_arr, axis=0) if len(F_arr) > 0 else np.ones(n_obj)
                z_max = z_max + 1e-6 * np.ones(n_obj)
                
                def obj(x, w=w, z_max=z_max):
                    x_clipped = np.clip(x, lo, hi)
                    f = np.array(problem(x_clipped), dtype=np.float64)
                    return np.max(w * (f - z_max))
                
                try:
                    res = minimize(obj, x0, method='Nelder-Mead', 
                                   options={'maxiter': 10, 'maxfev': 5, 'xatol': 1e-5, 'fatol': 1e-5})
                    if evals_used < budget:
                        x_best = np.clip(res.x, lo, hi)
                        f_best = evaluate(x_best)
                        if f_best is not None:
                            update_archive(x_best, f_best)
                except Exception:
                    pass
                    
        return archive_X
