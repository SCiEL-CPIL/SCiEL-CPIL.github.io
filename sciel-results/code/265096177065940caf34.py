"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic hierarchical search: coarse scan then adaptive local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)
        dim = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def check_deadline():
            return time.time() >= deadline or calls_made >= total_budget

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def run_local_refinement(x_init, sigma_init, max_calls):
            """Adaptive Nelder-Mead style local search with restarts on stagnation."""
            nonlocal calls_made, best_x, best_f

            # Build initial simplex
            simplex = [x_init.copy()]
            for i in range(dim):
                perturbation = x_init.copy()
                perturbation[i] += sigma_init
                simplex.append(np.clip(perturbation, -5, 5))

            # Evaluate initial simplex
            simplex_f = [evaluate(s) for s in simplex]

            local_best_x = x_init.copy()
            local_best_f = best_f
            no_improve = 0
            max_no_improve = 30

            while not check_deadline() and calls_made < max_calls:
                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                if simplex_f[0] < local_best_f:
                    local_best_f = simplex_f[0]
                    local_best_x = simplex[0].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= max_no_improve:
                    break

                # Compute centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                x_reflect = centroid + alpha * (centroid - simplex[-1])
                x_reflect = np.clip(x_reflect, -5, 5)
                f_reflect = evaluate(x_reflect)

                if f_reflect < simplex_f[0]:
                    # Expansion
                    gamma = 2.0
                    x_expand = centroid + gamma * (centroid - simplex[-1])
                    x_expand = np.clip(x_expand, -5, 5)
                    f_expand = evaluate(x_expand)
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        simplex_f[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        simplex_f[-1] = f_reflect
                elif f_reflect < simplex_f[-2]:
                    simplex[-1] = x_reflect
                    simplex_f[-1] = f_reflect
                else:
                    # Contraction
                    rho = 0.5
                    x_contract = centroid + rho * (simplex[-1] - centroid)
                    x_contract = np.clip(x_contract, -5, 5)
                    f_contract = evaluate(x_contract)
                    if f_contract < simplex_f[-1]:
                        simplex[-1] = x_contract
                        simplex_f[-1] = f_contract
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex_f[i] = evaluate(simplex[i])

                # Check convergence
                spread = np.max(np.std(simplex, axis=0))
                if spread < 1e-8:
                    break

            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

            return local_best_f, local_best_x

        # Phase 1: Structured coarse scan to locate promising basins
        # Use a deterministic grid-like sampling with some randomization for robustness
        scan_budget = min(300, total_budget - calls_made)

        # Generate structured scan points: combine grid points with random jitter
        # Use 3 levels per dimension for structured part, then fill with random
        grid_levels = [-4.0, -2.0, 0.0, 2.0, 4.0]

        # First, evaluate key structured points
        structured_points = []
        for v1 in grid_levels:
            for v2 in grid_levels:
                for v3 in grid_levels:
                    for v4 in grid_levels:
                        for v5 in grid_levels:
                            structured_points.append(np.array([v1, v2, v3, v4, v5]))

        # Shuffle and take first scan_budget points
        rng.shuffle(structured_points)
        for pt in structured_points[:scan_budget]:
            if check_deadline():
                break
            # Add small random jitter to avoid exact grid alignment issues
            jitter = rng.uniform(-0.3, 0.3, dim)
            pt_jittered = np.clip(pt + jitter, -5, 5)
            evaluate(pt_jittered)

        # Phase 2: Multi-start local refinement from top candidates
        if not check_deadline() and best_x is not None:
            remaining_budget = total_budget - calls_made

            # Use top 3 diverse starting points for local refinement
            # Re-evaluate to get sorted list of all evaluated points
            # For simplicity, just use the global best and a few random perturbations
            start_points = [best_x.copy()]

            # Add 2 diverse starts: perturb best_x in random directions
            for _ in range(2):
                direction = rng.standard_normal(dim)
                direction /= np.linalg.norm(direction) + 1e-10
                perturbed = best_x + 1.5 * direction
                perturbed = np.clip(perturbed, -5, 5)
                start_points.append(perturbed)

            # Allocate remaining budget among starts
            per_start_budget = max(20, remaining_budget // len(start_points))

            for i, sp in enumerate(start_points):
                if check_deadline() or calls_made >= total_budget:
                    break
                sigma_init = 1.0 if i == 0 else 0.5
                max_calls = total_budget if i == len(start_points) - 1 else calls_made + per_start_budget
                _, _ = run_local_refinement(sp, sigma_init, max_calls)

        return best_x, best_f
# EVOLVE-BLOCK-END
