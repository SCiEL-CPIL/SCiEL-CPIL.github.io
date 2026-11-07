"""Multi-objective optimizer on WFG4 using Constrained Chebyshev Scalarization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Constrained Chebyshev Scalarization for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 315.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        # Reference point for hypervolume
        ref = np.array([3.0, 5.0, 7.0])

        # Internal RNG
        rng = np.random.default_rng(12345)

        # Archive: list of (x, f) tuples
        archive = []

        def add_to_archive(x, f):
            """Add point to archive if not dominated, remove dominated points."""
            f = np.asarray(f, dtype=np.float64)
            # Check if any archived point dominates f
            for _, f_arch in archive:
                if np.all(f_arch <= f) and np.any(f_arch < f):
                    return  # f is dominated, don't add
            # Remove points dominated by f
            new_archive = []
            for x_arch, f_arch in archive:
                if not (np.all(f <= f_arch) and np.any(f < f_arch)):
                    new_archive.append((x_arch, f_arch))
            new_archive.append((x, f))
            archive.clear()
            archive.extend(new_archive)

        def chebyshev_scalarization(x, w, zmin, zmax, epsilon=1e-4):
            """
            Constrained Chebyshev scalarization.
            Minimize: max_i w_i * (f_i(x) - zmin_i) / (zmax_i - zmin_i + eps)
            Subject to: f_i(x) <= zmax_i + epsilon for all i
            """
            f = np.asarray(problem(x), dtype=np.float64)
            # Normalize objectives
            range_ = zmax - zmin + 1e-10
            normalized = (f - zmin) / range_
            # Chebyshev distance
            cheb = np.max(w * normalized)
            # Constraint violation: how much f exceeds zmax + epsilon
            violation = np.max(f - (zmax + epsilon))
            # Penalty
            penalty = 1000.0 * max(0.0, violation)
            return cheb + penalty, f, violation

        # Step 1: Get initial estimates of zmin and zmax via random sampling
        n_initial = min(50, self.budget // 4)
        initial_X = lo + (hi - lo) * rng.random((n_initial, self.dim))
        initial_F = np.array([problem(x) for x in initial_X])

        # Track evaluations
        evaluations = n_initial

        # Add initial points to archive
        for i in range(n_initial):
            add_to_archive(initial_X[i], initial_F[i])

        # Get zmin and zmax from initial samples
        zmin = initial_F.min(axis=0)
        zmax = initial_F.max(axis=0)
        # Ensure zmin < zmax
        for i in range(self.n_obj):
            if zmax[i] - zmin[i] < 1e-10:
                zmax[i] = zmin[i] + 1.0

        # Step 2: Systematic Chebyshev weight vectors
        # Generate weight vectors that span the Pareto front
        # For 3 objectives, use a grid on the simplex
        weight_vectors = []
        # Fine grid on the probability simplex for 3 objectives
        n_weights_per_axis = 8
        for i in range(n_weights_per_axis + 1):
            for j in range(n_weights_per_axis + 1 - i):
                k = n_weights_per_axis - i - j
                w = np.array([i, j, k], dtype=np.float64)
                w = w / (w.sum() + 1e-10)
                weight_vectors.append(w)

        # Also add some random weight vectors for diversity
        for _ in range(20):
            w = rng.random(self.n_obj)
            w = w / w.sum()
            weight_vectors.append(w)

        # Shuffle weight vectors
        rng.shuffle(weight_vectors)

        # Step 3: For each weight vector, solve the scalar subproblem
        # Use multiple restarts with local polishing
        max_restarts_per_w = 3
        max_local_iters = 50
        local_step = 0.05

        # Budget allocation
        # Reserve some budget for final refinement
        budget_for_scalar = int(self.budget * 0.85)
        budget_for_refine = self.budget - budget_for_scalar

        # Process weight vectors within budget
        for w in weight_vectors:
            if evaluations >= budget_for_scalar:
                break
            if time.time() > deadline - 30:
                break

            # Multiple restarts for this weight vector
            for restart in range(max_restarts_per_w):
                if evaluations >= budget_for_scalar:
                    break
                if time.time() > deadline - 30:
                    break

                # Initialize: random point or best from archive
                if archive and rng.random() < 0.5:
                    # Start from a random archive point with perturbation
                    idx = rng.integers(len(archive))
                    x0 = archive[idx][0].copy()
                    x0 += (hi - lo) * 0.1 * rng.standard_normal(self.dim)
                    x0 = np.clip(x0, lo, hi)
                else:
                    x0 = lo + (hi - lo) * rng.random(self.dim)

                # Local optimization using gradient-free approach
                best_x = x0.copy()
                best_val, best_f, _ = chebyshev_scalarization(x0, w, zmin, zmax)
                evaluations += 1

                # Simplex-like local search
                for it in range(max_local_iters):
                    if evaluations >= budget_for_scalar:
                        break
                    if time.time() > deadline - 30:
                        break

                    # Generate perturbation
                    step_size = local_step * (0.5 ** (it // 10))  # Decrease step size
                    delta = step_size * (hi - lo) * rng.standard_normal(self.dim)
                    x_new = np.clip(best_x + delta, lo, hi)

                    val_new, f_new, viol_new = chebyshev_scalarization(x_new, w, zmin, zmax)
                    evaluations += 1

                    if val_new < best_val:
                        best_val = val_new
                        best_x = x_new.copy()
                        best_f = f_new.copy()
                    else:
                        # Try opposite direction
                        x_opp = np.clip(best_x - delta, lo, hi)
                        val_opp, f_opp, _ = chebyshev_scalarization(x_opp, w, zmin, zmax)
                        evaluations += 1
                        if val_opp < best_val:
                            best_val = val_opp
                            best_x = x_opp.copy()
                            best_f = f_opp.copy()

                # Add best solution to archive
                if best_f is not None:
                    add_to_archive(best_x, best_f)

                # Update zmin and zmax
                zmin = np.minimum(zmin, best_f)
                zmax = np.maximum(zmax, best_f)
                for i in range(self.n_obj):
                    if zmax[i] - zmin[i] < 1e-10:
                        zmax[i] = zmin[i] + 1.0

        # Step 4: Final refinement - polish the best points in the archive
        if archive and evaluations < self.budget:
            # Sort archive by Chebyshev distance to ideal point
            ideal = zmin
            def cheb_dist(f):
                return np.max((f - ideal) / (zmax - zmin + 1e-10))

            # Take top points for refinement
            n_refine = min(len(archive), budget_for_refine // 10)
            if n_refine > 0:
                # Sort by objective sum (simple proxy)
                sorted_archive = sorted(archive, key=lambda xf: np.sum(xf[1]))
                refine_candidates = sorted_archive[:n_refine]

                for x_init, f_init in refine_candidates:
                    if evaluations >= self.budget:
                        break
                    if time.time() > deadline - 10:
                        break

                    # Fine local search around this point
                    best_x = x_init.copy()
                    best_f = f_init.copy()

                    # Multi-directional fine search
                    for _ in range(5):
                        if evaluations >= self.budget:
                            break
                        if time.time() > deadline - 10:
                            break

                        # Small random perturbation
                        delta = 0.01 * (hi - lo) * rng.standard_normal(self.dim)
                        x_new = np.clip(best_x + delta, lo, hi)
                        f_new = problem(x_new)
                        evaluations += 1

                        # Check if new point is non-dominated relative to best_f
                        if not (np.all(best_f <= f_new) and np.any(best_f < f_new)):
                            # Not dominated by current best, check if it dominates
                            if np.all(f_new <= best_f) and np.any(f_new < best_f):
                                best_x = x_new.copy()
                                best_f = f_new.copy()
                            else:
                                # Non-dominated, add to archive
                                add_to_archive(x_new, f_new)

        # Return the archive (list of x vectors)
        return [x for x, f in archive]
# EVOLVE-BLOCK-END
