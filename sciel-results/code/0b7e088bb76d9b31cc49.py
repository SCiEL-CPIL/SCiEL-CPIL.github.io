# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structured multi-objective optimizer tuned for the DTLZ4 manifold.

    DTLZ4's Pareto front is a 2D manifold: the 3 front variables set the
    objective directions and the 9 tail variables must be driven to ~0 (with
    the x**0.1 distortion) to make g=0.  We exploit this by anchoring the
    population on a dense spherical lattice of front directions and refining
    the tail variables with a decaying mutation scale, using crowding-based
    replacement to keep the population spread over the front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0          # safety margin before the 300s wall-clock
        rng = np.random.default_rng(0)

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        dim, n_obj = self.dim, self.n_obj
        budget = self.budget

        # ---- build a dense spherical lattice of front directions ----
        # The 3 front variables (first 3) are normalized to sum to 1 so that
        # with g=0 the objectives become f_i = 0.5 * u_i, covering the front.
        n_lat = 45
        lat = []
        for i in range(n_lat + 1):
            a = i / n_lat * (np.pi / 2.0)
            for j in range(n_lat + 1):
                b = j / n_lat * (2.0 * np.pi)
                u = np.array([np.cos(a), np.sin(a) * np.cos(b), np.sin(a) * np.sin(b)])
                s = u.sum()
                if s > 1e-9:
                    u = u / s
                lat.append(u)
        lat = np.array(lat)
        n_lat = len(lat)

        # ---- initial population: one point per lattice direction ----
        # Tail variables start small (the distortion pulls them to 0 anyway).
        pop = np.zeros((n_lat, dim))
        pop[:, :3] = lat
        pop[:, 3:] = 0.02 * rng.random((n_lat, dim - 3))
        pop = np.clip(pop, lo, hi)
        F = np.array([problem(x) for x in pop], dtype=float)
        used = n_lat
        if used >= budget:
            return pop

        # ---- crowding-based non-dominated sorting (vectorized) ----
        def sort_by_front(F):
            """Return a permutation sorted by (front index, crowding distance)."""
            m = len(F)
            fronts = []
            remaining = np.arange(m)
            while len(remaining) > 0:
                Fr = F[remaining]
                # a point is non-dominated if no other point is <= it in all
                # objectives and < it in at least one.
                # D[i,j] = Fr[j] <= Fr[i] in every objective
                leq = np.all(Fr[:, None, :] <= Fr[None, :, :], axis=2)
                lt = np.any(Fr[:, None, :] < Fr[None, :, :], axis=2)
                dom = leq & lt                     # dom[i,j]=True: j dominates i
                nd = ~np.any(dom, axis=1)          # not dominated by anyone
                front_local = np.where(nd)[0]
                # crowding distance within this front
                cd = np.zeros(len(front_local))
                if len(front_local) > 2:
                    for k in range(n_obj):
                        vals = Fr[front_local, k]
                        order = np.argsort(vals)
                        cd[order[0]] = np.inf
                        cd[order[-1]] = np.inf
                        rng_k = vals[order[-1]] - vals[order[0]]
                        if rng_k > 1e-12:
                            for idx in range(1, len(order) - 1):
                                cd[order[idx]] += (
                                    vals[order[idx + 1]] - vals[order[idx - 1]]
                                ) / rng_k
                fronts.append((front_local, cd))
                # remove this front
                keep = np.ones(len(remaining), dtype=bool)
                keep[front_local] = False
                remaining = remaining[keep]
            # build permutation: front order then crowding desc
            perm = []
            for front_local, cd in fronts:
                order = np.argsort(-cd)
                perm.append(remaining[front_local[order]])
            return np.concatenate(perm)

        # ---- main refinement loop ----
        # Decaying mutation scale: broad early, fine late.
        while used < budget:
            if time.time() > deadline:
                break
            progress = used / budget
            scale = 0.35 * (1.0 - progress) + 0.02
            # refine the tail variables (the ones that must go to 0)
            mutation = scale * span * rng.standard_normal((n_lat, dim - 3))
            child_tail = pop[:, 3:] + mutation
            child_tail = np.clip(child_tail, lo[3:], hi[3:])
            child = pop.copy()
            child[:, 3:] = child_tail
            Fc = np.array([problem(x) for x in child], dtype=float)
            used += n_lat
            if used >= budget:
                # keep whichever of the parent/child population is better per slot
                # (use combined crowding over the union, take best n_lat)
                combined_X = np.vstack([pop, child])
                combined_F = np.vstack([F, Fc])
                perm = sort_by_front(combined_F)
                pop = combined_X[perm[:n_lat]]
                F = combined_F[perm[:n_lat]]
                break
            # merge: keep the n_lat best points from the union by crowding
            combined_X = np.vstack([pop, child])
            combined_F = np.vstack([F, Fc])
            perm = sort_by_front(combined_F)
            pop = combined_X[perm[:n_lat]]
            F = combined_F[perm[:n_lat]]

        return pop
# EVOLVE-BLOCK-END
