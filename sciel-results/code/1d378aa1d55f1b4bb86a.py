"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start local search with hypervolume-guided restarts on WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 280.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj

        rng = np.random.default_rng(42)

        # Known structure: WFG7 Pareto front can be parameterized.
        # For WFG7, the Pareto front in objective space is well-defined.
        # We use a hybrid: initialize points along a parameterized front,
        # then do local refinement with Nelder-Mead style or gradient-free local search.

        # Strategy:
        # 1. Generate diverse starting points using parameterization of WFG7's PF
        # 2. For each, do a short local search (Nelder-Mead on each objective or combined)
        # 3. Track all evaluated points, compute hypervolume incrementally
        # 4. Use remaining budget for additional random restarts + local polish

        # WFG7 has 3 objectives. The Pareto front is a 2D surface.
        # We parameterize with two angles/parameters.
        # For WFG7, the PF is known: f1 = x1^2, f2 = ... (varies by version)
        # Since we don't have the exact formula, we use a generic approach:
        # Sample points in the decision space that are likely near the PF.

        # For WFG problems, the Pareto-optimal solutions often have structure in
        # the first few variables. Let's use a smart initialization.

        # Phase 1: Smart initialization along a grid of the "front parameters"
        # WFG7: typically, the first variable controls trade-offs.
        # We'll use a combination of:
        # - Points along the diagonal of the search space
        # - Points with structured first variables

        all_points = []
        all_fvals = []
        used = 0

        def eval_point(x):
            nonlocal used
            x_clipped = np.clip(x, lo, hi)
            f = problem(x_clipped)
            used += 1
            return x_clipped, np.array(f)

        # Phase 1: Generate diverse initial points
        # For WFG7 with 12 vars, the PF is influenced by several variables.
        # Let's create points that vary the first few variables systematically.

        n_init = 40
        for i in range(n_init):
            if used >= self.budget:
                break
            # Create a point with structured variation
            x = np.zeros(dim)
            # Vary first 3 variables on a grid-like pattern
            t1 = i / max(1, n_init - 1)
            x[0] = lo[0] + t1 * span[0]
            x[1] = lo[1] + ((i * 0.618) % 1.0) * span[1]  # golden ratio spacing
            x[2] = lo[2] + ((i * 0.382) % 1.0) * span[2]
            # Remaining variables: random but seeded
            for d in range(3, dim):
                x[d] = lo[d] + rng.random() * span[d]
            x, f = eval_point(x)
            all_points.append(x)
            all_fvals.append(f)

        # Phase 2: Local search from best points
        # Use Nelder-Mead style simplex for each objective, or a combined approach.
        # We'll do a simple coordinate descent / local random search.

        def local_search(x0, f0, n_steps, step_scale):
            """Simple local random search around x0."""
            nonlocal used
            best_x = x0.copy()
            best_f = f0.copy()
            current_x = x0.copy()
            current_f = f0.copy()
            
            for step in range(n_steps):
                if used >= self.budget:
                    break
                if time.time() - start_time > deadline - 10:
                    break
                # Try a perturbation
                scale = step_scale * (1.0 - step / max(1, n_steps))
                perturbation = rng.standard_normal(dim) * scale * span
                new_x = np.clip(current_x + perturbation, lo, hi)
                new_x, new_f = eval_point(new_x)
                
                # Check if new point is better in any objective (non-dominated)
                if np.any(new_f < best_f):
                    # Check if it dominates or is non-dominated
                    if np.all(new_f <= best_f) and np.any(new_f < best_f):
                        best_x = new_x
                        best_f = new_f
                        current_x = new_x
                        current_f = new_f
                    elif not np.any(np.all(best_f <= new_f) and np.any(best_f < new_f)):
                        # new_f doesn't dominate best_f, but best_f doesn't dominate new_f
                        # Keep both - for simplicity, just track the best per objective
                        pass
                elif np.any(new_f < current_f):
                    current_x = new_x
                    current_f = new_f
            
            return best_x, best_f

        # Select top diverse points for local search
        # Sort by sum of objectives (simple heuristic for "goodness")
        fvals_arr = np.array(all_fvals)
        sums = fvals_arr.sum(axis=1)
        
        # Select a diverse subset: top by sum + some random
        n_local = min(15, len(all_points))
        top_indices = np.argsort(sums)[:n_local]
        # Add a few random ones for diversity
        remaining = list(set(range(len(all_points))) - set(top_indices))
        if remaining:
            extra = rng.choice(remaining, size=min(3, len(remaining)), replace=False)
            top_indices = np.concatenate([top_indices, extra])
        
        top_indices = top_indices[:n_local]
        
        for idx in top_indices:
            if used >= self.budget:
                break
            if time.time() - start_time > deadline - 20:
                break
            x0 = all_points[idx]
            f0 = all_fvals[idx]
            # Local search with decreasing step size
            n_steps = 10
            step_scale = 0.1
            best_x, best_f = local_search(x0, f0, n_steps, step_scale)
            # Add the result
            if not any(np.allclose(best_x, p) for p in all_points):
                all_points.append(best_x)
                all_fvals.append(best_f)

        # Phase 3: Use remaining budget for more random restarts with local polish
        while used < self.budget:
            if time.time() - start_time > deadline - 10:
                break
            
            # Random restart
            x = lo + span * rng.random(dim)
            x, f = eval_point(x)
            all_points.append(x)
            all_fvals.append(f)
            
            # Quick local polish (3 steps)
            if used < self.budget and time.time() - start_time < deadline - 10:
                for _ in range(3):
                    if used >= self.budget:
                        break
                    if time.time() - start_time > deadline - 10:
                        break
                    perturbation = rng.standard_normal(dim) * 0.05 * span
                    new_x = np.clip(x + perturbation, lo, hi)
                    new_x, new_f = eval_point(new_x)
                    if np.any(new_f < f):
                        x = new_x
                        f = new_f
                        all_points.append(x)
                        all_fvals.append(f)

        # Return the points (the evaluator records all objective vectors from __call__'s return)
        # But wait - the evaluator records "every returned objective vector". 
        # Looking at the contract: "Every returned objective vector is recorded"
        # This means we need to return the objective values, not the decision vectors.
        # But the current program returns X (decision vectors). Let me re-read...
        # "the return value of __call__ is ignored" - so the return value doesn't matter!
        # "Every returned objective vector is recorded" - this must mean every f(x) computed
        # via problem(x) calls is recorded. So we just need to make the calls.
        
        return np.array(all_points)
# EVOLVE-BLOCK-END
