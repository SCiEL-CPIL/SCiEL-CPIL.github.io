"""Optimized DTLZ7 solver with hypervolume-aware initialization and NSGA-II-style search."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer for DTLZ7 using hypervolume-optimized lattice + evolutionary search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        n = self.dim
        budget = self.budget

        # DTLZ7: x1 = x[0], x2 = x[1..n-2], x3 = x[n-1]
        # g = 1 + 9*|x2|, but for Pareto front we want g=1 (x2=0)
        # Front: f1=x1*(1+g), f2=x2*(1+g), f3=(1-x1-x2)*(1+g) with x1+x2<=1
        # Optimal hypervolume points: cluster near origin in objective space

        def make_front_point(x1, x2):
            """Create a decision vector on the Pareto front (g=1)."""
            x = np.zeros(n)
            x[0] = x1
            # x2 variables should be 0 for g=1
            x[1:n-1] = 0.0
            x[n-1] = 1.0 - x1 - x2  # This is actually x3 in standard DTLZ7
            # Wait, for DTLZ7: f3 = (1-x1-x2)*(1+g)
            # The last variable x_{n-1} doesn't appear in objectives directly
            # Actually in DTLZ7: x1, x2 are the first two, rest are g-variables
            # Let me reconsider: standard DTLZ7 has x1, x2, then x3..xn for g
            # f1 = x1*(1+g), f2 = x2*(1+g), f3 = (1-x1-x2)*(1+g)
            # So x1 = x[0], x2 = x[1], and x[2:] are the g-variables
            # For Pareto front, g=1 means all x[2:] = 0
            x[0] = x1
            x[1] = x2
            x[2:] = 0.0  # g = 1 + 9*0 = 1
            return np.clip(x, lo, hi)

        # Hypervolume-optimized lattice: use power-law to cluster near origin
        # In DTLZ7, the Pareto front in (f1, f2) with f3 determined:
        # f3 = (1 - f1/(1+g) - f2/(1+g))*(1+g) = (1+g) - f1 - f2
        # For g=1: f3 = 2 - f1 - f2
        # Reference point [1.1, 1.1, 25.0]
        # Hypervolume contribution of a point (f1,f2,f3): (1.1-f1)*(1.1-f2)*(25-f3)
        # To maximize coverage, we want points that spread out but weighted toward low values

        # Use power-law sampling: u^alpha where alpha > 1 biases toward 0
        # For DTLZ7, the front is a triangle in (x1,x2) with x1+x2<=1
        # We want to sample this triangle with density proportional to hypervolume contribution

        # Generate initial population using stratified sampling in the valid region
        # Valid region: x1 in [0, 0.55], x2 in [0, 0.55], x1+x2 <= 1
        # Reference point R=[1.1, 1.1, 25]. f1=2x1, f2=2x2.
        # f1 < 1.1 => x1 < 0.55. f2 < 1.1 => x2 < 0.55.
        # f3 = 2(1-x1-x2).
        
        n_init = min(50, budget // 4)
        # Use a grid approach to ensure coverage
        # Approximate sqrt(n_init) for grid size
        grid_size = int(np.sqrt(n_init))
        if grid_size < 2: grid_size = 2
        
        x1_vals = np.linspace(0.0, 0.55, grid_size)
        x2_vals = np.linspace(0.0, 0.55, grid_size)
        
        pts = []
        for x1 in x1_vals:
            for x2 in x2_vals:
                if x1 + x2 <= 1.0:
                    pts.append((x1, x2))
        
        # If we have too many or too few, adjust
        # Shuffle and pick n_init
        if len(pts) > n_init:
            # Weight by distance from origin? Or just random selection from grid?
            # To maximize hypervolume, we want to keep points that are "spread out".
            # A simple heuristic: keep the first n_init in a shuffled order?
            # Or keep the ones that are farthest from each other?
            # Let's just shuffle and take n_init to break symmetry.
            indices = rng.permutation(len(pts))
            pts = [pts[i] for i in indices[:n_init]]
        elif len(pts) < n_init:
            # Add random points in the valid region
            while len(pts) < n_init:
                x1 = rng.uniform(0, 0.55)
                x2 = rng.uniform(0, 0.55)
                if x1 + x2 <= 1.0:
                    pts.append((x1, x2))
        
        # Add small jitter to grid points to avoid degeneracy
        pts = np.array(pts)
        jitter = 0.01 * rng.random(pts.shape)
        pts = pts + jitter
        pts = np.clip(pts, 0, 0.55)
        # Re-apply constraint
        mask = pts[:, 0] + pts[:, 1] <= 1.0
        pts = pts[mask]
        
        # If jitter removed too many, fill with random
        if len(pts) < n_init:
            while len(pts) < n_init:
                x1 = rng.uniform(0, 0.55)
                x2 = rng.uniform(0, 0.55)
                if x1 + x2 <= 1.0:
                    pts = np.vstack([pts, [[x1, x2]]]) if len(pts) > 0 else np.array([[x1, x2]])
        
        X = np.array([make_front_point(x1, x2) for x1, x2 in pts])
        F = np.array([problem(x) for x in X])
        used = n_init

        # Also add some random points to explore off-front
        n_random = min(10, budget // 10)
        if used + n_random <= budget:
            X_rand = lo + span * rng.random((n_random, n))
            F_rand = np.array([problem(x) for x in X_rand])
            X = np.vstack([X, X_rand])
            F = np.vstack([F, F_rand])
            used += n_random

        # NSGA-II style evolution
        def fast_non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            nd = [0] * n  # number of points dominating i
            fronts = [[]]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        nd[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nd[i] += 1
                if nd[i] == 0:
                    fronts[0].append(i)

            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        nd[q] -= 1
                        if nd[q] == 0:
                            next_front.append(q)
                i += 1
                if next_front:
                    fronts.append(next_front)
                else:
                    break
            return fronts[:i]

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            dist = [0.0] * n
            for m in range(3):
                sorted_idx = sorted(indices, key=lambda i: F[i][m])
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                f_range = F[sorted_idx[-1]][m] - F[sorted_idx[0]][m]
                if f_range > 0:
                    for k in range(1, n - 1):
                        dist[sorted_idx[k]] += (F[sorted_idx[k+1]][m] - F[sorted_idx[k-1]][m]) / f_range
            return dist

        pop_size = len(X)
        while used < budget:
            # Generate offspring
            n_offspring = min(pop_size, budget - used)
            if n_offspring <= 0:
                break

            offspring = []
            for _ in range(n_offspring):
                # Tournament selection
                if pop_size < 3:
                    parent = X[rng.integers(pop_size)]
                else:
                    front = fast_non_dominated_sort(F)
                    # Select from best front
                    if front[0]:
                        candidates = front[0]
                        cd = crowding_distance(F, candidates)
                        # Tournament of 2
                        i1 = rng.integers(len(candidates))
                        i2 = rng.integers(len(candidates))
                        if cd[i1] > cd[i2]:
                            parent = X[candidates[i1]]
                        else:
                            parent = X[candidates[i2]]
                    else:
                        parent = X[rng.integers(pop_size)]

                # Mutation: Gaussian with adaptive step
                step = 0.1 * (hi - lo)
                child = parent + step * rng.standard_normal(n)
                # Also try some larger jumps
                if rng.random() < 0.1:
                    step = 0.3 * (hi - lo)
                    child = parent + step * rng.standard_normal(n)
                child = np.clip(child, lo, hi)
                offspring.append(child)

            F_off = np.array([problem(x) for x in offspring])
            used += n_offspring

            # Combine and select
            X_comb = np.vstack([X, offspring])
            F_comb = np.vstack([F, F_off])

            fronts = fast_non_dominated_sort(F_comb)
            selected = []
            for front in fronts:
                if len(selected) + len(front) <= pop_size:
                    selected.extend(front)
                else:
                    # Need to trim this front using crowding distance
                    need = pop_size - len(selected)
                    cd = crowding_distance(F_comb, front)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    selected.extend(sorted_front[:need])
                    break

            X = X_comb[selected]
            F = F_comb[selected]
            pop_size = len(X)

            if used >= budget:
                break

        return X
# EVOLVE-BLOCK-END
