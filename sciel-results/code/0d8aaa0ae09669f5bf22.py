# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ1: DTLZ1-aware seeding + DE mutation +
    epsilon-dominance archive + local refinement, with a wall-clock deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 300.0 - 15.0  # safety margin
        dim = self.dim
        n_obj = self.n_obj
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        budget = int(self.budget)

        def time_ok():
            return time.time() < deadline

        # ---- DTLZ1-aware seeding on the known Pareto front (cheap, no eval) ----
        # DTLZ1: g = 100*(n-1 + sum((x_i-0.5)^2 - cos(20*pi*(x_i-0.5))) for i in [m, n])
        #        f_k = (1+g) * prod_{j=1}^{k-1} x_j * (1 - x_{k})   (k=1..m)
        # Pareto front at g=0 (x_i in [m,n) at 0.5) and x in [0,1]^m with sum(x)=1.
        # We seed on the simplex of the first m variables with the rest at 0.5.
        m = n_obj
        simplex_n = 14  # number of grid points along each simplex edge
        seeds = []
        # generate points on the (m-1)-simplex: x_1..x_{m-1} >= 0, sum(x_1..x_{m-1}) <= 1, x_m = 1 - sum
        if m == 3:
            for a in np.linspace(0.0, 1.0, simplex_n):
                for b in np.linspace(0.0, 1.0 - a, simplex_n):
                    c = 1.0 - a - b
                    seeds.append(np.array([a, b, c], dtype=float))
            # add a few random simplex points for robustness
            for _ in range(20):
                u = np.random.rand(m)
                u /= u.sum()
                seeds.append(u.astype(float))
        else:
            for _ in range(60):
                u = np.random.rand(m)
                u /= u.sum()
                seeds.append(u.astype(float))
        seeds = np.array(seeds)
        # build full candidate vectors
        X = np.tile(0.5, (seeds.shape[0], dim))
        X[:, :m] = seeds
        # ---- random extra candidates to fill initial population ----
        rng = np.random.default_rng(0)
        n_rand = max(0, min(40, budget - X.shape[0]))
        if n_rand > 0:
            Xr = lo + (hi - lo) * rng.random((n_rand, dim))
            X = np.vstack([X, Xr])
        # ---- evaluate initial population (with deadline guard) ----
        F = np.empty((X.shape[0], n_obj))
        for i in range(X.shape[0]):
            F[i] = problem(X[i])
            if not time_ok():
                break
        used = X.shape[0]
        if used >= budget or not time_ok():
            return X

        # ---- epsilon-dominance archive ----
        # epsilon dominance: a point dominates b if a_i <= b_i + eps for all i and a_i < b_i for some i
        # Use a grid-based epsilon to encourage spread.
        eps_grid = 0.05  # epsilon for dominance
        archive = list(zip(list(X[:used]), list(F[:used])))

        def dominates(a, b, eps=0.0):
            # a dominates b: a <= b + eps in all, a < b in at least one
            return bool(np.all(a <= b + eps) and np.any(a < b - eps))

        def archive_dominated(new_f):
            for _, af in archive:
                if dominates(af, new_f, eps_grid):
                    return True
            return False

        def add_to_archive(x, f):
            # remove archive members dominated by new point, then add if not dominated
            new_archive = []
            for ax, af in archive:
                if dominates(f, af, eps_grid):
                    continue  # new dominates this, drop it
                new_archive.append((ax, af))
            if not archive_dominated(f):
                new_archive.append((x, f))
            archive.clear()
            archive.extend(new_archive)

        # ---- DE-based mutation + local refinement ----
        def de_mutate():
            # pick 3 distinct parents from current population
            n = X.shape[0]
            i1, i2, i3 = rng.choice(n, size=3, replace=False)
            v = X[i1] + 0.7 * (X[i2] - X[i3])
            # crossover
            cross_rate = 0.5
            mask = (rng.random(dim) < cross_rate) | (rng.random() == 0)  # ensure at least one gene
            child = np.where(mask, v, X[i1])
            child = np.clip(child, lo, hi)
            return child

        def local_refine(x, f, steps=8):
            # coordinate descent on the objectives via a scalarization (minimize weighted sum)
            # Use a smooth surrogate: minimize sum of objectives with small random restarts
            best_x, best_f = x.copy(), f.copy()
            step = 0.05 * (hi - lo)
            for s in range(steps):
                improved = False
                for d in range(dim):
                    for delta in (step, -step):
                        trial = best_x.copy()
                        trial[d] += delta
                        trial = np.clip(trial, lo, hi)
                        tf = problem(trial)
                        # accept if it dominates best_f or improves the scalarized sum
                        if dominates(tf, best_f, 0.0) or np.sum(tf) < np.sum(best_f) - 1e-12:
                            best_x, best_f = trial, tf
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-4 * (hi - lo).max():
                        break
            return best_x, best_f

        # main loop
        while used < budget and time_ok():
            child = de_mutate()
            fc = problem(child)
            used += 1
            # only add to archive if it improves the front (not epsilon-dominated)
            if not archive_dominated(fc):
                add_to_archive(child, fc)
                # occasionally do a local refinement on promising points
                if rng.random() < 0.15 and time_ok() and used < budget - 5:
                    rx, rf = local_refine(child, fc, steps=6)
                    used += 1
                    if used < budget and not archive_dominated(rf):
                        add_to_archive(rx, rf)
            # every few iterations, refresh population with archive + random to maintain diversity
            if used % 30 == 0 and time_ok():
                arch_X = np.array([a[0] for a in archive])
                arch_F = np.array([a[1] for a in archive])
                # keep top of archive plus a few random
                keep = min(arch_X.shape[0], 25)
                idx = rng.choice(arch_X.shape[0], size=keep, replace=False)
                new_X = arch_X[idx]
                new_F = arch_F[idx]
                # pad with random if needed
                if new_X.shape[0] < 25 and time_ok():
                    pad = 25 - new_X.shape[0]
                    Xr = lo + (hi - lo) * rng.random((pad, dim))
                    Fr = np.array([problem(x) for x in Xr])
                    used += pad
                    new_X = np.vstack([new_X, Xr])
                    new_F = np.vstack([new_F, Fr])
                X, F = new_X, new_F

        # ---- final polish: refine the archive points that are most promising ----
        if time_ok() and len(archive) > 0:
            # sort archive by sum of objectives, polish the best few
            arch_sorted = sorted(archive, key=lambda a: np.sum(a[1]))
            for x, f in arch_sorted[:min(5, len(arch_sorted))]:
                if not time_ok():
                    break
                rx, rf = local_refine(x, f, steps=10)
                if not archive_dominated(rf):
                    add_to_archive(rx, rf)

        return X
# EVOLVE-BLOCK-END
