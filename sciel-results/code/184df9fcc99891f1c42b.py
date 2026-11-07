"""Nelder-Mead based optimizer for BBOB f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Two-phase optimizer for BBOB f14:
    Phase 1: Coarse global search using stratified sampling to find promising regions.
    Phase 2: Intensive local refinement using Nelder-Mead with adaptive simplex.
    - Handles hidden shifts/rotations via broad initial exploration.
    - Deadline guard ensures safe termination.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 295
        rng = np.random.default_rng(42)
        d = self.dim
        max_calls = self.budget

        best_x = None
        best_f = float('inf')
        calls_used = 0
        eval_log = []  # Store (x, f) pairs for potential model fitting

        def try_eval(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= max_calls:
                return None
            if time.time() > deadline - 2:
                return None
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls_used += 1
            eval_log.append((x_clipped.copy(), fx))
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        def nelder_mead(start_x, initial_step=1.0, max_iters=200):
            """
            Nelder-Mead simplex optimization with adaptive step size.
            Returns (best_x_local, best_f_local)
            """
            nonlocal calls_used, best_x, best_f

            # Initialize simplex with larger initial step for better exploration
            simplex = [start_x.copy()]
            for i in range(d):
                v = start_x.copy()
                v[i] += initial_step
                simplex.append(v)

            fvals = []
            for v in simplex:
                fx = try_eval(v)
                if fx is None:
                    return start_x, best_f
                fvals.append(fx)

            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            prev_best_f = min(fvals)
            stagnant_iters = 0

            for it in range(max_iters):
                if calls_used >= max_calls or time.time() > deadline - 3:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Check convergence
                f_range = fvals[-1] - fvals[0]
                if f_range < 1e-8:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = try_eval(x_r)
                if f_r is None:
                    break

                accepted = False
                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = try_eval(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                    accepted = True
                elif f_r < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                    accepted = True
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = try_eval(x_c)
                    if f_c is None:
                        break

                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                        accepted = True
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_i = try_eval(simplex[i])
                            if f_i is None:
                                return simplex[0], fvals[0]
                            fvals[i] = f_i
                        accepted = True

                # Track stagnation
                current_best_f = min(fvals)
                if current_best_f < prev_best_f - 1e-10:
                    prev_best_f = current_best_f
                    stagnant_iters = 0
                else:
                    stagnant_iters += 1

                # If stagnant for too long, try a larger step
                if stagnant_iters >= 15:
                    # Restart simplex with larger step around current best
                    best_idx = np.argmin(fvals)
                    new_start = simplex[best_idx].copy()
                    simplex = [new_start.copy()]
                    for i in range(d):
                        v = new_start.copy()
                        v[i] += 2.0  # Larger step
                        v = np.clip(v, -5, 5)
                        simplex.append(v)
                    fvals = []
                    for v in simplex:
                        fx = try_eval(v)
                        if fx is None:
                            return new_start, min(fvals)
                        fvals.append(fx)
                    stagnant_iters = 0
                    prev_best_f = min(fvals)

                # Update global best
                local_best_f = min(fvals)
                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = simplex[np.argmin(fvals)].copy()

            # Return best from simplex
            local_best_idx = np.argmin(fvals)
            return simplex[local_best_idx].copy(), fvals[local_best_idx]

        # Phase 1: Coarse global search - stratified sampling
        # Use a grid-like stratified sampling to cover the space efficiently
        n_coarse = min(30, max(10, max_calls // 15))
        
        # Create stratified points: divide [-5,5]^d into cells
        # For d=5, use a structured approach
        coarse_points = []
        
        # Always include origin and axis-aligned points
        coarse_points.append(np.zeros(d))
        for i in range(d):
            for sign in [-1, 1]:
                p = np.zeros(d)
                p[i] = sign * 3.0
                coarse_points.append(p)
        
        # Add stratified random points
        # Divide each dimension into 4 strata
        strata_per_dim = 4
        cell_size = 10.0 / strata_per_dim
        
        # Generate stratified points
        for _ in range(n_coarse - len(coarse_points)):
            point = np.zeros(d)
            for i in range(d):
                stratum = rng.integers(strata_per_dim)
                point[i] = -5.0 + (stratum + rng.random()) * cell_size
            coarse_points.append(point)
        
        # Evaluate all coarse points
        for point in coarse_points:
            if calls_used >= max_calls or time.time() > deadline - 10:
                break
            fx = try_eval(point)
            if fx is None:
                break
        
        # Phase 2: Local refinement around the best points found
        # Sort by function value and refine top candidates
        if eval_log:
            # Get top 3 candidates
            sorted_evals = sorted(eval_log, key=lambda x: x[1])
            top_candidates = [x for x, f in sorted_evals[:3]]
            
            for candidate in top_candidates:
                if calls_used >= max_calls - 50 or time.time() > deadline - 15:
                    break
                
                # Refine with Nelder-Mead starting from this candidate
                remaining = max_calls - calls_used
                iters = min(100, max(30, remaining // 4))
                
                x_local, f_local = nelder_mead(candidate, initial_step=0.5, max_iters=iters)
                
                if f_local < best_f:
                    best_f = f_local
                    best_x = x_local.copy()
        
        # Phase 3: Final intensive refinement around the global best
        if best_x is not None and calls_used < max_calls - 30 and time.time() < deadline - 5:
            x_refined, f_refined = nelder_mead(best_x, initial_step=0.2, max_iters=80)
            if f_refined < best_f:
                best_f = f_refined
                best_x = x_refined.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
