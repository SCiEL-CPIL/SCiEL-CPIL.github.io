"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

class Optimizer:
    """Combined approach: Space-filling initial design + Weighted Local Search refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0
        self.start_time = None

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        self.eval_count = 0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Time limit with safety margin (330s kill, 295s safety)
        time_limit = 295.0
        
        def check_time():
            return time.time() - self.start_time < time_limit
        
        def check_budget():
            return self.eval_count < self.budget
        
        def eval_and_count(x):
            if not check_budget() or not check_time():
                raise Exception("Budget or time exceeded")
            x = np.clip(x, lo, hi)
            res = problem(x)
            self.eval_count += 1
            return res

        # 1. Deterministic Space-Filling Initial Design (Halton Sequence)
        # This provides uniform coverage in the decision space, which often maps
        # to better coverage in objective space than random sampling for smooth problems.
        rng = np.random.default_rng(42)
        
        # Use Halton sequence for the first portion of the budget
        # Halton requires prime bases. For dim=4, use 2, 3, 5, 7
        primes = [2, 3, 5, 7]
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        # Reserve 30% of budget for initial space-filling sampling
        # This is more than the 5-10% used in previous versions to ensure good initial coverage
        initial_budget = int(self.budget * 0.3)
        
        # Generate Halton points
        # We need to map [0, 1) to [lo, hi]
        halton_points = []
        for i in range(1, initial_budget + 1):
            point = np.array([halton(i, primes[d]) for d in range(self.dim)])
            x = lo + (hi - lo) * point
            halton_points.append(x)
        
        # Evaluate initial points
        for x in halton_points:
            if not check_budget() or not check_time():
                break
            eval_and_count(x)
        
        # 2. Weighted Local Search with diverse weights
        # Define weight vectors on the simplex
        n_weights = 12
        weights = []
        for _ in range(n_weights):
            w = rng.dirichlet([1, 1, 1])
            weights.append(w)
            
        # Ensure vertices and edge midpoints
        weights.append([1, 0, 0])
        weights.append([0, 1, 0])
        weights.append([0, 0, 1])
        weights.append([0.5, 0.5, 0])
        weights.append([0.5, 0, 0.5])
        weights.append([0, 0.5, 0.5])
        
        # Deduplicate
        unique_weights = []
        for w in weights:
            w = np.array(w)
            w = w / w.sum()
            is_dup = False
            for uw in unique_weights:
                if np.allclose(w, uw, atol=1e-3):
                    is_dup = True
                    break
            if not is_dup:
                unique_weights.append(w)
        
        final_weights = unique_weights
        
        remaining_budget = self.budget - self.eval_count
        if remaining_budget <= 0:
            return None
            
        # Allocate remaining budget to weighted local search
        # Use multiple restarts per weight to escape local minima
        n_restarts_per_weight = 2
        total_slots = len(final_weights) * n_restarts_per_weight
        budget_per_slot = max(5, remaining_budget // total_slots)
        
        bounds_opt = list(zip(lo, hi))
        best_solutions = []
        
        for w in final_weights:
            for restart in range(n_restarts_per_weight):
                if not check_budget() or not check_time():
                    break
                
                # Start from a random point or one of the initial Halton points
                if rng.random() < 0.5 and len(halton_points) > 0:
                    x0 = halton_points[rng.integers(len(halton_points))]
                else:
                    x0 = lo + (hi - lo) * rng.random(self.dim)
                
                def obj(x, w=w):
                    if not check_budget() or not check_time():
                        raise Exception("Budget or time exceeded")
                    f = problem(x)
                    self.eval_count += 1
                    return np.dot(w, f)
                
                try:
                    res = opt.minimize(obj, x0, method='L-BFGS-B', bounds=bounds_opt, 
                                        options={'maxiter': 30, 'ftol': 1e-8, 'gtol': 1e-6})
                    if res.x is not None and res.fun is not None:
                        best_solutions.append(np.clip(res.x, lo, hi))
                except Exception:
                    pass
        
        # 3. Polish pass: refine the best solutions found
        if best_solutions and check_time():
            if len(best_solutions) > 8:
                # Sort by sum of objectives
                def solution_score(x):
                    try:
                        f = eval_and_count(x)
                        return np.sum(f)
                    except Exception:
                        return np.inf
                
                scored = [(solution_score(x), x) for x in best_solutions]
                scored.sort(key=lambda t: t[0])
                best_solutions = [x for _, x in scored[:8]]
            
            for x_best in best_solutions:
                if not check_budget() or not check_time():
                    break
                
                for obj_idx in range(self.n_obj):
                    if not check_budget() or not check_time():
                        break
                    
                    w_polish = np.zeros(self.n_obj)
                    w_polish[obj_idx] = 1.0
                    
                    def obj_polish(x, w=w_polish):
                        if not check_budget() or not check_time():
                            raise Exception("Budget or time exceeded")
                        f = problem(x)
                        self.eval_count += 1
                        return np.dot(w, f)
                    
                    try:
                        res = opt.minimize(obj_polish, x_best, method='L-BFGS-B', 
                                            bounds=bounds_opt,
                                            options={'maxiter': 20, 'ftol': 1e-10, 'gtol': 1e-8})
                    except Exception:
                        pass

        return None
# EVOLVE-BLOCK-END
