"""Multi-objective optimizer on WFG4 using hypervolume-indicator-driven sequential search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-indicator-driven sequential multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        budget = self.budget
        ref_point = np.array([3.0, 5.0, 7.0], dtype=np.float64)

        rng = np.random

        archive = []
        archive_X = []

        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def update_archive(f, x):
            for af in archive:
                if np.all(af <= f) and np.any(af < f):
                    return False
            new_archive = []
            new_archive_X = []
            for i, af in enumerate(archive):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive.append(af)
                    new_archive_X.append(archive_X[i])
            new_archive.append(f.copy())
            new_archive_X.append(x.copy())
            archive[:] = new_archive
            archive_X[:] = new_archive_X
            return True

        def hypervolume_2d(points, ref):
            """Compute exact 2D hypervolume."""
            valid = [p for p in points if np.all(p < ref)]
            n = len(valid)
            if n == 0:
                return 0.0
            if n == 1:
                p = valid[0]
                return float((ref[0] - p[0]) * (ref[1] - p[1]))

            valid_sorted = sorted(valid, key=lambda p: p[0])
            filtered = []
            min_y = float('inf')
            for p in valid_sorted:
                if p[1] < min_y:
                    filtered.append(p)
                    min_y = p[1]

            hv = 0.0
            for i in range(len(filtered)):
                p = filtered[i]
                if i == 0:
                    x_width = ref[0] - p[0]
                else:
                    x_width = filtered[i - 1][0] - p[0]
                y_height = ref[1] - p[1]
                hv += x_width * y_height
            return hv

        def hypervolume_3d(points, ref):
            """Compute exact hypervolume for 3D points with reference point."""
            valid = [p for p in points if np.all(p < ref)]
            n = len(valid)
            if n == 0:
                return 0.0
            if n == 1:
                p = valid[0]
                return float(np.prod(ref - p))

            valid_sorted = sorted(valid, key=lambda p: p[0])
            hv = 0.0

            for i in range(n):
                p = valid_sorted[i]
                if i == 0:
                    x_width = ref[0] - p[0]
                else:
                    x_width = valid_sorted[i - 1][0] - p[0]

                if x_width <= 0:
                    continue

                yz_points = [valid_sorted[j][1:] for j in range(i + 1)]
                hv_2d_with = hypervolume_2d(yz_points, np.array([ref[1], ref[2]]))

                if i > 0:
                    yz_points_without = [valid_sorted[j][1:] for j in range(i)]
                    hv_2d_without = hypervolume_2d(yz_points_without, np.array([ref[1], ref[2]]))
                    marginal_yz = hv_2d_with - hv_2d_without
                else:
                    marginal_yz = hv_2d_with

                hv += x_width * marginal_yz

            return hv

        # Phase 1: Front-biased initialization
        n_init = min(60, budget)
        used = 0

        for i in range(n_init):
            if used >= budget:
                break
            if time.time() > deadline - 5.0:
                break

            t = rng.random()
            x = np.zeros(dim)
            x[0] = lo[0] + span[0] * t

            for d in range(1, dim):
                u = rng.random()
                if rng.random() < 0.5:
                    x[d] = lo[d] + span[d] * u
                else:
                    x[d] = (lo[d] + hi[d]) / 2.0 + span[d] * 0.3 * rng.standard_normal()
                    x[d] = np.clip(x[d], lo[d], hi[d])

            f = problem(x)
            used += 1
            update_archive(f, x)

        # Phase 2: Hypervolume-driven sequential search
        best_hv = hypervolume_3d(archive, ref_point)
        stagnation_counter = 0
        stagnation_threshold = 20

        while used < budget:
            if time.time() > deadline - 5.0:
                break

            if len(archive) == 0:
                x = lo + span * rng.random(dim)
                f = problem(x)
                used += 1
                update_archive(f, x)
                continue

            # Adaptive exploration/exploitation
            explore_prob = 0.15 + 0.2 * min(stagnation_counter / stagnation_threshold, 1.0)

            if rng.random() < explore_prob:
                # Global exploration
                x = lo + span * rng.random(dim)
            else:
                # Local refinement around a random archive point
                idx = rng.randint(len(archive))
                base_x = archive_X[idx]

                # Adaptive step size
                progress = used / budget
                base_step = 0.3 * (1.0 - 0.7 * progress)

                # Try multiple Gaussian perturbations and pick the one with best marginal HV
                best_marginal = -1.0
                best_x = None

                n_tries = 5
                for _ in range(n_tries):
                    if time.time() > deadline - 10.0:
                        break

                    step = base_step * rng.standard_normal(dim)
                    x_try = base_x + step * span
                    x_try = np.clip(x_try, lo, hi)
                    f_try = problem(x_try)
                    used += 1

                    if used >= budget:
                        break

                    # Compute marginal HV
                    is_dominated = False
                    for af in archive:
                        if np.all(af <= f_try) and np.any(af < f_try):
                            is_dominated = True
                            break

                    if not is_dominated:
                        # Quick marginal HV estimate
                        hv_with = hypervolume_3d(archive + [f_try], ref_point)
                        hv_without = best_hv
                        marginal = hv_with - hv_without

                        if marginal > best_marginal:
                            best_marginal = marginal
                            best_x = x_try
                            best_f = f_try

                if best_x is not None:
                    update_archive(best_f, best_x)
                    current_hv = hypervolume_3d(archive, ref_point)
                    if current_hv > best_hv:
                        best_hv = current_hv
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                else:
                    stagnation_counter += 1

                if used >= budget:
                    break

            if used < budget:
                f = problem(x)
                used += 1
                if update_archive(f, x):
                    current_hv = hypervolume_3d(archive, ref_point)
                    if current_hv > best_hv:
                        best_hv = current_hv
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                else:
                    stagnation_counter += 1

        return archive_X if archive_X else [np.zeros(dim)]

# EVOLVE-BLOCK-END
