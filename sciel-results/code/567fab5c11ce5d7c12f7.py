# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution


class Optimizer:
    """Multi-objective optimizer using a scalarized differential evolution approach."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        # WFG7 has 3 objectives. We'll use a weighted sum approach with multiple weights.
        # Since we need to minimize all 3 objectives, we'll sample weight vectors on the simplex.
        
        # Generate weight vectors on the 2-simplex (for 3 objectives)
        # We'll use a grid of weight vectors
        n_weights = 10
        weights = []
        for i in range(n_weights):
            for j in range(n_weights - i):
                k = n_weights - i - j
                w = np.array([i / n_weights, j / n_weights, k / n_weights])
                weights.append(w)
        weights = np.array(weights)

        # We have a budget of 200 evaluations. Let's split it among the weight vectors.
        # But first, let's do some random sampling to get an initial idea of the landscape.
        
        archive = []  # list of (x, f) tuples
        used = 0

        # Initial random sampling
        n_init = min(30, self.budget)
        X_init = lo + (hi - lo) * rng.random((n_init, self.dim))
        for x in X_init:
            f = problem(x)
            archive.append((x.copy(), f.copy()))
            used += 1
            if time.time() > deadline:
                break

        # Now use differential evolution on scalarized objectives
        # For each weight vector, run a short DE optimization
        
        # We'll allocate remaining budget among weight vectors
        remaining_budget = self.budget - used
        if remaining_budget <= 0:
            return archive

        n_weights_to_use = min(len(weights), max(1, remaining_budget // 15))
        # Select evenly spaced weight vectors
        indices = np.linspace(0, len(weights) - 1, n_weights_to_use, dtype=int)
        selected_weights = weights[indices]

        # For each weight, run a short DE
        for w in selected_weights:
            if used >= self.budget or time.time() > deadline:
                break
            
            # Scalarized objective
            def scalarized(x):
                f = problem(x)
                return float(np.dot(w, f))
            
            # Run differential evolution with limited evaluations
            # maxiter controls the number of generations, popsize controls population
            # We want to use about 15-20 evaluations per weight vector
            max_evals = min(15, self.budget - used)
            if max_evals < 5:
                break
            
            # Set up bounds for DE
            bounds_de = list(zip(lo, hi))
            
            try:
                # Use a small population and few iterations
                popsize = max(4, min(8, max_evals // 2))
                maxiter = max(1, max_evals // popsize)
                
                result = differential_evolution(
                    scalarized,
                    bounds_de,
                    maxiter=maxiter,
                    popsize=popsize,
                    tol=1e-6,
                    seed=0,
                    init='latinhypercube'
                )
                
                # The result.x is the best point found, but we need to count evaluations
                # DE with popsize p and maxiter m uses approximately p*(1+m) evaluations
                # But we don't have exact count, so let's just evaluate the result
                if result.success or result.fun < float('inf'):
                    x_best = np.clip(result.x, lo, hi)
                    f_best = problem(x_best)
                    archive.append((x_best.copy(), f_best.copy()))
                    used += 1
            except Exception:
                pass

        # If we still have budget, do more random sampling or local search
        if used < self.budget:
            # Do some local perturbation around the best points
            # Sort archive by sum of objectives
            archive.sort(key=lambda xf: np.sum(xf[1]))
            
            # Take top few points and perturb them
            n_local = min(10, len(archive), self.budget - used)
            for i in range(n_local):
                if used >= self.budget or time.time() > deadline:
                    break
                x_base, f_base = archive[i]
                # Perturb
                n_perturb = min(3, self.budget - used)
                for _ in range(n_perturb):
                    if used >= self.budget or time.time() > deadline:
                        break
                    x_pert = x_base + 0.05 * (hi - lo) * rng.standard_normal(self.dim)
                    x_pert = np.clip(x_pert, lo, hi)
                    f_pert = problem(x_pert)
                    archive.append((x_pert.copy(), f_pert.copy()))
                    used += 1

        return archive
# EVOLVE-BLOCK-END
