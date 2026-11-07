"""Decomposition-based multi-objective optimizer for WFG7 using Tchebycheff scalarization."""
import time
import numpy as np


class Optimizer:
    """Decomposition-based optimizer using Tchebycheff scalarization with local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        rng = np.random.default_rng(42)

        # Generate uniformly distributed weight vectors on the 2D simplex (3 objectives)
        n_weights = 30
        weight_vectors = []
        # Use a grid approach on the simplex
        # For 3 objectives, we can use a triangular grid
        n_per_side = 6  # gives about 21 points, adjust to get ~30
        # Actually, let's use a more systematic approach
        # Generate points on simplex using stratified sampling
        for i in range(n_weights):
            # Use quasi-random (Halton-like) or just uniform random on simplex
            # Simple approach: generate 3 random numbers, normalize
            r = rng.random(3)
            r = r / r.sum()
            weight_vectors.append(r)
        weight_vectors = np.array(weight_vectors)

        # Ideal point estimate (will be updated)
        ideal_point = np.array([0.0, 0.0, 0.0])
        nadir_point = np.array([np.inf, np.inf, np.inf])

        def _tchebycheff_scalarization(f, w, ideal):
            """Tchebycheff scalarization: min max_i w_i * (f_i - ideal_i)"""
            return np.max(w * (f - ideal))

        def _dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def _is_non_dominated(F, f):
            for ef in F:
                if _dominates(ef, f):
                    return False
            return True

        def _remove_dominated(F, f):
            return [ef for ef in F if not _dominates(f, ef)]

        # Initialize with one solution per weight vector
        # Start from center of search space
        center = (lo + hi) * 0.5
        archive_F = []
        archive_X = []
        used = 0

        # Evaluate center first
        f_center = np.array(problem(center), dtype=np.float64)
        used += 1
        ideal_point = np.minimum(ideal_point, f_center)
        nadir_point = np.minimum(nadir_point, f_center)

        # For each weight vector, run a local search (Nelder-Mead style)
        # Budget allocation: ~6 evaluations per weight vector for local search
        evals_per_subproblem = max(3, (budget - used) // n_weights)

        for wi, w in enumerate(weight_vectors):
            if used >= budget or time.time() > deadline - 5:
                break

            # Initialize subproblem at center
            x = center.copy()
            f = f_center.copy()

            # Simple local search: try perturbations in different directions
            # Use a simple hill-climbing with adaptive step size
            step_size = 0.1 * (hi - lo)
            best_f = f.copy()
            best_x = x.copy()

            # Nelder-Mead simplex initialization
            # Create simplex around x
            simplex = [x.copy()]
            simplex_f = [f.copy()]

            # Initialize other simplex points
            for i in range(dim):
                perturbed = x.copy()
                perturbed[i] += step_size[i] if i < len(step_size) else step_size[0]
                perturbed = np.clip(perturbed, lo, hi)
                fp = np.array(problem(perturbed), dtype=np.float64)
                used += 1
                if used >= budget:
                    break
                simplex.append(perturbed)
                simplex_f.append(fp)

            # Update ideal/nadir
            for sf in simplex_f:
                ideal_point = np.minimum(ideal_point, sf)
                nadir_point = np.minimum(nadir_point, sf)

            # Run Nelder-Mead for a few iterations
            max_nm_iters = 5
            for nm_iter in range(max_nm_iters):
                if used >= budget or time.time() > deadline - 5:
                    break

                # Sort simplex by Tchebycheff scalarization value
                scalar_vals = [_tchebycheff_scalarization(sf, w, ideal_point) for sf in simplex_f]
                order = np.argsort(scalar_vals)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                scalar_vals = [scalar_vals[i] for i in order]

                # Compute centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + (centroid - simplex[-1])
                xr = np.clip(xr, lo, hi)
                fr = np.array(problem(xr), dtype=np.float64)
                used += 1
                if used >= budget:
                    break

                sr = _tchebycheff_scalarization(fr, w, ideal_point)

                if sr < scalar_vals[0]:
                    # Expansion
                    xe = centroid + 2 * (centroid - simplex[-1])
                    xe = np.clip(xe, lo, hi)
                    fe = np.array(problem(xe), dtype=np.float64)
                    used += 1
                    if used >= budget:
                        break
                    se = _tchebycheff_scalarization(fe, w, ideal_point)
                    if se < sr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif sr < scalar_vals[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Contraction
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, lo, hi)
                    fc = np.array(problem(xc), dtype=np.float64)
                    used += 1
                    if used >= budget:
                        break
                    sc = _tchebycheff_scalarization(fc, w, ideal_point)
                    if sc < scalar_vals[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            fi = np.array(problem(simplex[i]), dtype=np.float64)
                            used += 1
                            if used >= budget:
                                break
                            simplex_f[i] = fi
                        if used >= budget:
                            break

                # Update ideal/nadir
                for sf in simplex_f:
                    ideal_point = np.minimum(ideal_point, sf)
                    nadir_point = np.minimum(nadir_point, sf)

            # Add best point from this subproblem to archive
            best_idx = np.argmin([_tchebycheff_scalarization(sf, w, ideal_point) for sf in simplex_f])
            best_x = simplex[best_idx]
            best_f = simplex_f[best_idx]

            if _is_non_dominated(archive_F, best_f):
                archive_F = _remove_dominated(archive_F, best_f)
                archive_F.append(best_f)
                archive_X.append(best_x)

        # Final refinement: local search on archive points with remaining budget
        if used < budget and time.time() < deadline - 5:
            remaining = budget - used
            n_refine = min(remaining, len(archive_X) * 2)

            for _ in range(n_refine):
                if used >= budget or time.time() > deadline - 2:
                    break
                # Pick a random point from archive to refine
                idx = rng.integers(0, len(archive_X))
                base_x = archive_X[idx].copy()
                base_f = archive_F[idx].copy()

                # Try small perturbations
                for _ in range(2):
                    if used >= budget:
                        break
                    pert = base_x + 0.01 * (hi - lo) * rng.standard_normal(dim)
                    pert = np.clip(pert, lo, hi)
                    fp = np.array(problem(pert), dtype=np.float64)
                    used += 1
                    if _dominates(fp, base_f):
                        base_x = pert
                        base_f = fp

                # Update archive
                if _is_non_dominated(archive_F, base_f):
                    archive_F = _remove_dominated(archive_F, base_f)
                    archive_F.append(base_f)
                    archive_X.append(base_x)

        return archive_X if archive_X else [lo + (hi - lo) * 0.5]
