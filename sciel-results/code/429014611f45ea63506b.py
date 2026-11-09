"""Multi-objective optimizer for RE34 using iterative DE search with Pareto archive."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Iterative differential-evolution search with Pareto archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 310.0  # Safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        rng = np.random.default_rng(0)

        # ---------- Initialization: structured diversity ----------
        # 1) All 2^5 = 32 corners
        corners = np.array(
            [[hi[j] if (i >> j) & 1 else lo[j] for j in range(dim)] for i in range(32)]
        )

        # 2) Midpoint and face centers
        mid = (lo + hi) / 2.0
        face_centers = []
        for d in range(dim):
            for val in [lo[d], hi[d]]:
                fc = mid.copy()
                fc[d] = val
                face_centers.append(fc.copy())

        pop_size = 30
        init_points = [corners[0].copy()]
        for i in [1, 2, 4, 8, 16]:
            init_points.append(corners[i].copy())
        init_points.append(mid.copy())
        # Add some face centers for better coverage
        for fc in face_centers[:4]:
            init_points.append(fc.copy())

        # Fill remaining with stratified random sampling
        n_remaining = pop_size - len(init_points)
        if n_remaining > 0:
            for _ in range(n_remaining):
                x = lo + (hi - lo) * rng.random(dim)
                init_points.append(x)

        X = np.array(init_points[:pop_size])
        F = np.array([problem(x) for x in X])
        used = pop_size

        # ---------- Pareto archive ----------
        archive_X = []
        archive_F = []

        def dominates(a, b):
            """Return True if a dominates b (a <= b in all, a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def is_dominated(f, arch_F):
            for af in arch_F:
                if dominates(af, f):
                    return True
            return False

        def crowding_distance(arch_F):
            """Compute crowding distance for each member of the archive."""
            n = len(arch_F)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(n_obj):
                obj = np.array([af[m] for af in arch_F])
                order = np.argsort(obj)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                span = obj[order[-1]] - obj[order[0]]
                if span > 0:
                    for i in range(1, n - 1):
                        cd[order[i]] += (obj[order[i + 1]] - obj[order[i - 1]]) / span
            return cd

        def update_archive(x, f, arch_X, arch_F):
            """Add x/f to archive if non-dominated, removing dominated members."""
            if is_dominated(f, arch_F):
                return arch_X, arch_F
            for j, af in enumerate(arch_F):
                if np.allclose(f, af, atol=1e-12):
                    return arch_X, arch_F
            new_X = []
            new_F = []
            for ax, af in zip(arch_X, arch_F):
                if not dominates(f, af):
                    new_X.append(ax)
                    new_F.append(af)
            new_X.append(x.copy())
            new_F.append(f.copy())
            return new_X, new_F

        # Add initial population to archive
        for i in range(len(X)):
            archive_X, archive_F = update_archive(X[i], F[i], archive_X, archive_F)

        # ---------- Main search loop ----------
        F_factor = 0.7
        CR = 0.85

        while used < budget:
            if time.time() - start_time > deadline:
                break

            # Select parent from archive using crowding distance
            if len(archive_X) > 1:
                cd = crowding_distance(archive_F)
                i1 = rng.integers(len(archive_X))
                i2 = rng.integers(len(archive_X))
                parent_idx = i1 if cd[i1] >= cd[i2] else i2
                parent = archive_X[parent_idx]
            elif len(archive_X) == 1:
                parent = archive_X[0]
            else:
                parent = X[rng.integers(pop_size)]

            # DE mutation: select 3 distinct random indices from population
            idx1, idx2, idx3 = rng.choice(pop_size, size=3, replace=False)

            # Create trial vector
            trial = X[idx1] + F_factor * (X[idx2] - X[idx3])
            # Binary crossover with parent
            mask = rng.random(dim) < CR
            trial = np.where(mask, trial, parent)
            trial = np.clip(trial, lo, hi)

            ft = problem(trial)
            used += 1

            # Update archive
            archive_X, archive_F = update_archive(trial, ft, archive_X, archive_F)

            # Update population: find best candidate to replace
            # Score each population member by whether it's dominated and by objective sum
            best_replace_idx = -1
            best_replace_score = np.inf
            for i in range(pop_size):
                if dominates(ft, F[i]):
                    # Trial dominates this member: strong candidate for replacement
                    score = -1000.0 - np.sum(F[i])
                else:
                    # Not dominated by trial; use sum of objectives as proxy
                    score = np.sum(F[i])
                if score < best_replace_score:
                    best_replace_score = score
                    best_replace_idx = i

            if best_replace_idx >= 0:
                X[best_replace_idx] = trial
                F[best_replace_idx] = ft

            # Periodically refresh population with new random points to maintain diversity
            if used % 40 == 0 and used < budget - 5:
                idx = rng.integers(pop_size)
                new_x = lo + (hi - lo) * rng.random(dim)
                new_f = problem(new_x)
                used += 1
                X[idx] = new_x
                F[idx] = new_f
                archive_X, archive_F = update_archive(new_x, new_f, archive_X, archive_F)

        # ---------- Final polish: local optimization on top archive members ----------
        if len(archive_X) > 0 and time.time() - start_time < deadline - 20:
            # Sort archive by crowding distance, take top few for polishing
            cd = crowding_distance(archive_F)
            order = np.argsort(-cd)  # descending crowding distance
            n_polish = min(5, len(archive_X))
            polish_indices = order[:n_polish]

            for pi in polish_indices:
                if used >= budget or time.time() - start_time > deadline - 10:
                    break

                x0 = archive_X[pi].copy()
                # Define a scalar objective: weighted sum with weights based on normalized objectives
                # Use the normalized objectives to create a fair scalarization
                arch_F_arr = np.array(archive_F)
                obj_min = arch_F_arr.min(axis=0)
                obj_max = arch_F_arr.max(axis=0)
                obj_range = obj_max - obj_min
                obj_range[obj_range == 0] = 1.0
                weights = 1.0 / obj_range

                def scalar_obj(x):
                    fx = problem(x)
                    return np.sum(weights * fx)

                try:
                    res = minimize(
                        scalar_obj, x0,
                        method='Nelder-Mead',
                        options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4}
                    )
                    if used < budget:
                        fx = res.fun
                        # We need to get the actual 3 objectives
                        # Nelder-Mead doesn't track evaluations easily, so just evaluate
                        # Actually, let's just evaluate the result
                        if used + 1 <= budget:
                            f_res = problem(res.x)
                            used += 1
                            archive_X, archive_F = update_archive(res.x, f_res, archive_X, archive_F)
                except Exception:
                    pass

        return X
# EVOLVE-BLOCK-END
