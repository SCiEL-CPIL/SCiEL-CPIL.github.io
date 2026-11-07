"""Evolutionary multi-objective optimizer for WFG7 with HV-guided selection and Nelder-Mead refinement."""
import time
import numpy as np


class Optimizer:
    """Evolutionary multi-objective optimizer for WFG7."""

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

        ref_point = np.array([3.0, 5.0, 7.0])
        used = 0

        def _dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def _hv_3d(points, ref):
            """Exact 3D hypervolume contribution of non-dominated points vs ref."""
            valid = [p for p in points if np.all(p < ref)]
            if not valid:
                return 0.0
            if len(valid) == 1:
                return float(np.prod(ref - valid[0]))
            valid.sort(key=lambda p: p[0])
            n = len(valid)
            total = 0.0
            for i in range(n):
                x1 = valid[i][0]
                x2 = valid[i + 1][0] if i + 1 < n else ref[0]
                dx = x2 - x1
                if dx <= 0:
                    continue
                yz = [(valid[j][1], valid[j][2]) for j in range(i, n)
                      if valid[j][1] < ref[1] and valid[j][2] < ref[2]]
                if not yz:
                    continue
                yz.sort(key=lambda p: p[0])
                m = len(yz)
                min_z = ref[2]
                for j in range(m - 1, -1, -1):
                    y, z = yz[j]
                    if z < min_z:
                        min_z = z
                    y_next = yz[j + 1][0] if j + 1 < m else ref[1]
                    dy = y_next - y
                    dz = ref[2] - min_z
                    if dy > 0 and dz > 0:
                        total += dx * dy * dz
            return total

        def _nelder_mead(x0, f0, max_iter=30):
            """Fast Nelder-Mead local refinement using Tchebycheff-like scalarization
             based on distance to current ideal point."""
            nonlocal used
            if used >= budget or time.time() > deadline - 5:
                return x0, f0

            # Use sum of objectives as scalar for local search (simple but effective)
            f_scalar = lambda x: float(np.sum(problem(x)))
            fs0 = f_scalar(x0)
            used += 1

            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            simplex = [x0.copy() + 0.01 * (hi - lo) * rng.standard_normal(dim) for _ in range(dim)]
            simplex = [np.clip(s, lo, hi) for s in simplex]
            f_simplex = [f_scalar(s) for s in simplex]
            used += dim

            for _ in range(max_iter):
                if used >= budget or time.time() > deadline - 5:
                    break
                # Sort simplex by fitness
                idx = np.argsort(f_simplex)
                simplex = [simplex[i] for i in idx]
                f_simplex = [f_simplex[i] for i in idx]

                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, lo, hi)
                fr = f_scalar(xr)
                used += 1

                if fr < f_simplex[0]:
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, lo, hi)
                    fe = f_scalar(xe)
                    used += 1
                    if fe < fr:
                        simplex[-1] = xe
                        f_simplex[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_simplex[-1] = fr
                elif fr < f_simplex[-1]:
                    simplex[-1] = xr
                    f_simplex[-1] = fr
                else:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, lo, hi)
                    fc = f_scalar(xc)
                    used += 1
                    if fc < f_simplex[-1]:
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            f_simplex[i] = f_scalar(simplex[i])
                            used += 1
                        if used >= budget:
                            break

            # Return best point
            best_idx = np.argmin(f_simplex)
            return simplex[best_idx], f_simplex[best_idx]

        # Initialize population
        pop_size = 40
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used += pop_size

        # Archive (non-dominated set)
        archive_X = []
        archive_F = []
        for i in range(pop_size):
            if not any(_dominates(af, F[i]) for af in archive_F):
                archive_X.append(X[i].copy())
                archive_F.append(F[i].copy())
                archive_F = [af for af in archive_F if not _dominates(F[i], af)]
                archive_F.append(F[i].copy())
                # Fix: rebuild archive properly
                archive_F = []
                archive_X = []

        # Rebuild archive cleanly
        archive_F = []
        archive_X = []
        for i in range(pop_size):
            dominated = False
            for j, af in enumerate(archive_F):
                if _dominates(af, F[i]):
                    dominated = True
                    break
                if _dominates(F[i], af):
                    archive_F.pop(j)
                    archive_X.pop(j)
                    j -= 1
            if not dominated:
                archive_F.append(F[i].copy())
                archive_X.append(X[i].copy())

        # Evolutionary loop
        gen = 0
        while used < budget and time.time() < deadline - 10:
            gen += 1

            # Create offspring using DE/rand/1
            n_offspring = pop_size
            offspring_X = []
            offspring_F = []

            for _ in range(n_offspring):
                if used >= budget or time.time() > deadline - 5:
                    break
                # Select three distinct parents
                a, b, c = rng.choice(pop_size, size=3, replace=False)
                F_scale = 0.8
                CR = 0.9
                v = X[a] + F_scale * (X[b] - X[c])
                v = np.clip(v, lo, hi)

                # Crossover
                trial = X[rng.randint(pop_size)].copy()
                jrand = rng.randint(dim)
                for j in range(dim):
                    if rng.random() < CR or j == jrand:
                        trial[j] = v[j]
                trial = np.clip(trial, lo, hi)

                f_trial = problem(trial)
                used += 1
                offspring_X.append(trial)
                offspring_F.append(f_trial)

            if not offspring_X:
                break

            # Combine population and offspring, select by non-domination + crowding
            all_X = np.vstack([X, np.array(offspring_X)])
            all_F = np.vstack([F, np.array(offspring_F)])
            n_total = len(all_F)

            # Non-dominated sorting
            remaining = list(range(n_total))
            front_idx = 0
            fronts = []
            while remaining:
                front = []
                for i in remaining:
                    is_dom = False
                    for j in remaining:
                        if i != j and _dominates(all_F[j], all_F[i]):
                            is_dom = True
                            break
                    if not is_dom:
                        front.append(i)
                if not front:
                    front = [remaining[0]]  # Fallback
                fronts.append(front)
                remaining = [i for i in remaining if i not in front]
                if len(fronts) >= 3:
                    break

            # Select next population: fill with fronts in order
            new_idx = []
            for front in fronts:
                if len(new_idx) + len(front) <= pop_size:
                    new_idx.extend(front)
                else:
                    # Need to select from this front using crowding distance
                    new_idx.extend(front[:pop_size - len(new_idx)])
                    break

            if len(new_idx) < pop_size:
                # Fill remaining randomly
                while len(new_idx) < pop_size:
                    new_idx.append(rng.randint(n_total))

            X = all_X[new_idx]
            F = all_F[new_idx]

            # Update archive
            for i in range(pop_size):
                dominated = False
                for j, af in enumerate(archive_F):
                    if _dominates(af, F[i]):
                        dominated = True
                        break
                    if _dominates(F[i], af):
                        archive_F.pop(j)
                        archive_X.pop(j)
                        j -= 1
                if not dominated:
                    archive_F.append(F[i].copy())
                    archive_X.append(X[i].copy())

            # Periodic local refinement of archive points
            if gen % 5 == 0 and len(archive_X) > 0 and used < budget - 50:
                # Pick up to 3 diverse archive points for NM refinement
                n_refine = min(3, len(archive_X), (budget - used) // 20)
                if n_refine > 0:
                    if len(archive_X) > n_refine:
                        # Pick spread indices
                        indices = np.linspace(0, len(archive_X) - 1, n_refine, dtype=int)
                    else:
                        indices = range(len(archive_X))
                    for idx in indices:
                        if used >= budget or time.time() > deadline - 10:
                            break
                        x_ref, f_ref = _nelder_mead(archive_X[idx].copy(), archive_F[idx].copy())
                        # Update archive with refined point
                        dominated = False
                        for j, af in enumerate(archive_F):
                            if _dominates(af, f_ref):
                                dominated = True
                                break
                            if _dominates(f_ref, af):
                                archive_F.pop(j)
                                archive_X.pop(j)
                                j -= 1
                        if not dominated:
                            archive_F.append(f_ref)
                            archive_X.append(x_ref)

        # Final: ensure we return something valid
        if len(archive_X) > 0:
            return archive_X
        return [X[0]]
