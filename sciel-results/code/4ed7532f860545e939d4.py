"""Parallel Nelder-Mead simplex ensemble for BBOB f3 (Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Parallel asynchronous Nelder-Mead simplex ensemble.
    Multiple simplexes explore different basins simultaneously to escape
    Rastrigin's local minima. Uses adaptive step sizes and periodic
    re-seeding for diversity.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin before 300s

        dim = self.dim
        budget = self.budget

        calls_made = 0
        best_x = np.zeros(dim)
        best_fx = np.inf

        def safe_call(x):
            nonlocal calls_made, best_x, best_fx
            if calls_made >= budget or time.time() > deadline:
                return None
            x_clip = np.clip(x, -5.0, 5.0)
            f_val = func(x_clip)
            calls_made += 1
            if f_val < best_fx:
                best_fx = f_val
                best_x = x_clip.copy()
            return f_val

        # --- Initialize simplex ensemble ---
        # Start simplexes at corners and center of the search space
        # This ensures broad coverage of the domain
        rng = np.random.default_rng(42)

        # Define starting points: center + 4 corners of a smaller box
        # Use a box of [-3, 3] to stay within [-5, 5] with margin
        center = np.zeros(dim)
        corner_offset = 3.0

        # Generate 5 starting points
        start_points = [center.copy()]

        # Add 4 diverse points
        for i in range(4):
            # Use different patterns for diversity
            if i == 0:
                # Random point in [-3, 3]
                sp = rng.uniform(-corner_offset, corner_offset, dim)
            elif i == 1:
                # Point along first axis
                sp = np.zeros(dim)
                sp[0] = corner_offset
            elif i == 2:
                # Point along second axis
                sp = np.zeros(dim)
                sp[1] = -corner_offset
            else:
                # Another random point
                sp = rng.uniform(-corner_offset, corner_offset, dim)
            sp = np.clip(sp, -5, 5)
            start_points.append(sp)

        # Initialize simplexes around each starting point
        # Each simplex has dim+1 vertices
        n_simplexes = len(start_points)
        simplexes = []
        simplex_fvals = []
        simplex_best = []
        simplex_stagnation = []

        for sp in start_points:
            # Initialize simplex vertices
            simplex = [sp.copy()]
            fvals = []

            # Evaluate center
            f0 = safe_call(sp)
            if f0 is None:
                break
            fvals.append(f0)

            # Add dim more vertices with small perturbations
            for i in range(dim):
                xi = sp.copy()
                xi[i] += 0.5
                xi = np.clip(xi, -5, 5)
                fi = safe_call(xi)
                if fi is None:
                    break
                simplex.append(xi)
                fvals.append(fi)

            if len(fvals) == dim + 1:
                simplexes.append(simplex)
                simplex_fvals.append(fvals)
                simplex_best.append(min(fvals))
                simplex_stagnation.append(0)

        if not simplexes:
            return best_x, best_fx

        # --- Nelder-Mead parameters ---
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5

        # Track progress for adaptive re-seeding
        global_progress_count = 0
        last_best_fx = best_fx
        progress_window = 30  # Check for progress every 30 calls

        # Main loop: run all simplexes in round-robin fashion
        simplex_idx = 0
        while calls_made < budget and time.time() < deadline:
            # Check for global stagnation
            if calls_made - global_progress_count >= progress_window:
                if best_fx >= last_best_fx - 1e-6:
                    # No progress, re-seed the worst simplex
                    # Find the simplex with the worst best value
                    worst_idx = np.argmax(simplex_best)

                    # Re-initialize this simplex at a new random location
                    new_sp = rng.uniform(-4, 4, dim)
                    new_sp = np.clip(new_sp, -5, 5)

                    # Reset the simplex
                    simplex = [new_sp.copy()]
                    fvals = []

                    f0 = safe_call(new_sp)
                    if f0 is None:
                        break
                    fvals.append(f0)

                    for i in range(dim):
                        xi = new_sp.copy()
                        xi[i] += 0.5
                        xi = np.clip(xi, -5, 5)
                        fi = safe_call(xi)
                        if fi is None:
                            break
                        simplex.append(xi)
                        fvals.append(fi)

                    if len(fvals) == dim + 1:
                        simplexes[worst_idx] = simplex
                        simplex_fvals[worst_idx] = fvals
                        simplex_best[worst_idx] = min(fvals)
                        simplex_stagnation[worst_idx] = 0

                    last_best_fx = best_fx
                    global_progress_count = calls_made
                else:
                    # Progress made, reset counter
                    last_best_fx = best_fx
                    global_progress_count = calls_made

            # Run one iteration of the selected simplex
            idx = simplex_idx % n_simplexes
            simplex_idx += 1

            if idx >= len(simplexes):
                continue

            simplex = simplexes[idx]
            fvals = simplex_fvals[idx]

            # Sort simplex by fitness
            order_s = np.argsort(fvals)
            simplex = [simplex[i] for i in order_s]
            fvals = [fvals[i] for i in order_s]

            # Check if simplex is degenerate
            if fvals[0] >= fvals[-1] - 1e-8:
                # Reset this simplex
                new_sp = best_x + rng.normal(0, 0.5, dim)
                new_sp = np.clip(new_sp, -5, 5)

                simplex = [new_sp.copy()]
                fvals = []

                f0 = safe_call(new_sp)
                if f0 is None:
                    break
                fvals.append(f0)

                for i in range(dim):
                    xi = new_sp.copy()
                    xi[i] += 0.3
                    xi = np.clip(xi, -5, 5)
                    fi = safe_call(xi)
                    if fi is None:
                        break
                    simplex.append(xi)
                    fvals.append(fi)

                if len(fvals) == dim + 1:
                    simplexes[idx] = simplex
                    simplex_fvals[idx] = fvals
                    simplex_best[idx] = min(fvals)
                    simplex_stagnation[idx] = 0
                continue

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = safe_call(xr)
            if fr is None:
                break

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = safe_call(xe)
                if fe is None:
                    break
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                xc = centroid + rho * (simplex[-1] - centroid)
                xc = np.clip(xc, -5, 5)
                fc = safe_call(xc)
                if fc is None:
                    break
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fi = safe_call(simplex[i])
                        if fi is None:
                            break
                        fvals[i] = fi
                    if calls_made >= budget or time.time() > deadline:
                        break

            # Update stored simplex
            simplexes[idx] = simplex
            simplex_fvals[idx] = fvals
            simplex_best[idx] = min(fvals)

            # Track stagnation for this simplex
            if fvals[0] < simplex_best[idx] - 1e-6:
                simplex_stagnation[idx] = 0
            else:
                simplex_stagnation[idx] += 1

            # If a simplex has been stagnant for too long, reset it
            if simplex_stagnation[idx] > 20:
                new_sp = best_x + rng.normal(0, 1.0, dim)
                new_sp = np.clip(new_sp, -5, 5)

                simplex = [new_sp.copy()]
                fvals = []

                f0 = safe_call(new_sp)
                if f0 is None:
                    break
                fvals.append(f0)

                for i in range(dim):
                    xi = new_sp.copy()
                    xi[i] += 0.3
                    xi = np.clip(xi, -5, 5)
                    fi = safe_call(xi)
                    if fi is None:
                        break
                    simplex.append(xi)
                    fvals.append(fi)

                if len(fvals) == dim + 1:
                    simplexes[idx] = simplex
                    simplex_fvals[idx] = fvals
                    simplex_best[idx] = min(fvals)
                    simplex_stagnation[idx] = 0

        return best_x, best_fx
# EVOLVE-BLOCK-END
