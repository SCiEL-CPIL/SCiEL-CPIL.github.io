"""Multi-Objective Evolutionary Strategy with Adaptive Local Refinement for RE21."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

class Optimizer:
    """
    Multi-objective optimizer using an Evolutionary Strategy (ES) 
    with adaptive weighted-sum local refinement.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin against 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # State tracking
        population = []  # List of (x, f) tuples
        eval_count = 0
        max_evals = self.budget
        
        def evaluate(x):
            nonlocal eval_count
            if eval_count >= max_evals:
                return None
            if time.time() > deadline:
                return None
            fx = problem(x)
            eval_count += 1
            return fx

        # 1. Initialization: Random sampling
        # We reserve some budget for evolution and local search.
        # Let's start with 20 random points to get a baseline.
        init_size = 20
        for _ in range(init_size):
            x = np.random.uniform(lo, hi)
            fx = evaluate(x)
            if fx is None:
                break
            population.append((x, fx))
        
        if not population:
            return np.array([])

        # Helper: Non-dominated sort (simplified for small population)
        def get_nondominated(points):
            if not points:
                return []
            nd_indices = []
            for i, (xi, fi) in enumerate(points):
                dominated = False
                for j, (xj, fj) in enumerate(points):
                    if i == j:
                        continue
                    if np.all(fj <= fi) and np.any(fj < fi):
                        dominated = True
                        break
                if not dominated:
                    nd_indices.append(i)
            return nd_indices

        # Helper: Local refinement using weighted sum
        def local_refine(x0, f0, weights, max_iters=50):
            """
            Refines x0 by minimizing a weighted sum of objectives.
            Returns the best point found (x_best, f_best).
            """
            if time.time() > deadline:
                return x0, f0
            
            # Define scalar objective
            def scalar_obj(x):
                nonlocal eval_count
                if eval_count >= max_evals or time.time() > deadline:
                    return 1e10
                fx = problem(x)
                eval_count += 1
                return np.dot(weights, fx)
            
            # Use L-BFGS-B with strict bounds
            try:
                # Limit function evaluations strictly
                options = {'maxiter': max_iters, 'maxfun': 10}
                res = minimize(scalar_obj, x0, method='L-BFGS-B', bounds=bounds, options=options)
                
                # The minimizer may have made several calls. 
                # We need the final point and its true objective values.
                x_best = res.x
                # We must evaluate x_best to get the true vector f_best for the archive
                # However, if we just return res.x, we don't have f_best recorded in our population list yet.
                # The evaluator records all calls to problem().
                # So we need to add (x_best, f_best) to our population.
                # We need f_best. We can get it from the last call? No, safer to evaluate once more?
                # No, that wastes budget. 
                # Actually, the 'problem' call inside scalar_obj is recorded.
                # But we don't know which one was the best.
                # To be safe and simple, we will assume the last point in the minimization 
                # or the res.x is good. 
                # Let's evaluate res.x explicitly to ensure we have the vector in our internal list.
                # This costs 1 extra eval.
                
                fx_best = evaluate(x_best)
                if fx_best is None:
                    return x0, f0
                return x_best, fx_best
            except Exception:
                return x0, f0

        # 2. Evolutionary Loop
        # We will iterate until budget is nearly exhausted.
        # Each generation:
        #   a. Select parents from non-dominated set (or all if small)
        #   b. Crossover and Mutation to create offspring
        #   c. Evaluate offspring
        #   d. Merge population, sort, truncate
        #   e. If budget allows, pick best non-dominated points for local refinement
        
        pop_size = 30
        max_generations = 10
        
        for gen in range(max_generations):
            if time.time() > deadline:
                break
            if eval_count > max_evals - 20: # Reserve 20 for final local refinements
                break
                
            # Get current non-dominated front
            nd_indices = get_nondominated(population)
            if not nd_indices:
                # Fallback: pick best by sum
                sums = [np.sum(f) for _, f in population]
                nd_indices = [np.argmin(sums)]
            
            nd_points = [population[i] for i in nd_indices]
            
            # 2a. Selection: Binary tournament on non-dominated points
            def select_parent():
                i = np.random.randint(len(nd_points))
                j = np.random.randint(len(nd_points))
                # If same, pick i. Otherwise, we need a metric.
                # For simplicity, just pick randomly from ND set.
                # This is a "random" selection from the front, which is standard for diversity.
                return nd_points[i][0]
            
            # 2b. Generate Offspring
            new_offspring = []
            n_offspring = pop_size // 2
            for _ in range(n_offspring):
                p1 = select_parent()
                p2 = select_parent()
                
                # SBX (Simulated Binary Crossover)
                eta_c = 20
                x_c = np.zeros(self.dim)
                for k in range(self.dim):
                    if np.random.rand() < 0.5:
                        # Standard SBX
                        if abs(p1[k] - p2[k]) < 1e-14:
                            x_c[k] = p1[k]
                        else:
                            u = np.random.rand()
                            if u <= 0.5:
                                beta = (2 * u) ** (1 / (eta_c + 1))
                            else:
                                beta = (1 / (2 * (1 - u))) ** (1 / (eta_c + 1))
                            x_c[k] = 0.5 * ((1 + beta) * p1[k] + (1 - beta) * p2[k])
                    else:
                        x_c[k] = p1[k]
                
                # Clip to bounds
                x_c = np.clip(x_c, lo, hi)
                
                # Polynomial Mutation
                eta_m = 20
                for k in range(self.dim):
                    if np.random.rand() < 0.8: # Mutation rate 0.8
                        delta1 = (x_c[k] - lo[k]) / span[k]
                        delta2 = (hi[k] - x_c[k]) / span[k]
                        u = np.random.rand()
                        if u < 0.5:
                            xy = 1 - delta1
                            mut_pow = 2 * (xy ** (eta_m + 1))
                            rand_pow = (1 + mut_pow) ** (-1 / (eta_m + 1))
                            deltaq = (rand_pow - 1) * delta1
                        else:
                            xy = 1 - delta2
                            mut_pow = 2 * (xy ** (eta_m + 1))
                            rand_pow = (1 + mut_pow) ** (-1 / (eta_m + 1))
                            deltaq = (rand_pow - 1) * delta2
                        x_c[k] += deltaq * span[k]
                        
                x_c = np.clip(x_c, lo, hi)
                new_offspring.append(x_c)
            
            # 2c. Evaluate Offspring
            for x in new_offspring:
                fx = evaluate(x)
                if fx is not None:
                    population.append((x, fx))
            
            # 2d. Truncate Population
            # Keep best 'pop_size' points based on non-domination + crowding (or just ND count)
            # Simple approach: Keep all non-dominated, then fill with best scalarized if too many
            nd_indices = get_nondominated(population)
            if len(nd_indices) > pop_size:
                # Too many non-dominated. Use sum of objectives to pick best pop_size
                nd_points = [population[i] for i in nd_indices]
                sums = [np.sum(f) for _, f in nd_points]
                sorted_nd = [nd_points[i] for i in np.argsort(sums)]
                population = sorted_nd[:pop_size]
            else:
                # If ND set is small, keep it all, and fill rest with best dominated
                nd_set = set(nd_indices)
                dominated = [i for i in range(len(population)) if i not in nd_set]
                # Sort dominated by sum
                dominated_sorted = sorted(dominated, key=lambda i: np.sum(population[i][1]))
                population = [population[i] for i in nd_indices] + [population[i] for i in dominated_sorted[:pop_size - len(nd_indices)]]
                
                # Ensure we don't exceed pop_size
                population = population[:pop_size]

        # 3. Final Local Refinement Phase
        # Use remaining budget to refine the best non-dominated points.
        # This is critical for hypervolume as it polishes the front.
        
        nd_indices = get_nondominated(population)
        if not nd_indices:
            sums = [np.sum(f) for _, f in population]
            nd_indices = [np.argmin(sums)]
        
        # Sort ND points by "promisingness" (e.g., distance to ideal point or sum)
        nd_points = [population[i] for i in nd_indices]
        # We want to refine points that are likely to improve HV.
        # Let's just take the top 5-10 from the ND front.
        # Since we don't know the ideal point, we use the sum of objectives as a heuristic.
        nd_points_sorted = sorted(nd_points, key=lambda p: np.sum(p[1]))
        
        # Determine how many local searches we can afford
        # Each local search takes ~10-15 evals.
        remaining_budget = max_evals - eval_count
        n_refines = min(10, remaining_budget // 15)
        
        if n_refines > 0:
            # Select diverse points from the ND front to refine
            # We'll pick the first n_refines from the sorted list
            # But to ensure diversity, we might spread them out.
            # For simplicity, take the best ones.
            targets = nd_points_sorted[:n_refines]
            
            for x0, f0 in targets:
                if eval_count >= max_evals - 5:
                    break
                if time.time() > deadline:
                    break
                
                # Generate a random weight vector for local search
                # This helps explore different parts of the front
                w = np.random.dirichlet(np.ones(self.n_obj))
                # Scale weights to be reasonable
                # If objectives are on different scales, this might be biased.
                # But L-BFGS-B will handle the gradient.
                
                x_new, f_new = local_refine(x0, f0, w)
                
                # Add to population
                population.append((x_new, f_new))
                
                # Optional: Immediate truncation to keep population manageable
                if len(population) > 100:
                    nd_indices = get_nondominated(population)
                    if len(nd_indices) > 50:
                        nd_points = [population[i] for i in nd_indices]
                        sums = [np.sum(f) for _, f in nd_points]
                        sorted_nd = [nd_points[i] for i in np.argsort(sums)]
                        population = sorted_nd[:50]
                    else:
                        population = population[:50] # Rough cut

        return np.array([x for x, f in population])
# EVOLVE-BLOCK-END
