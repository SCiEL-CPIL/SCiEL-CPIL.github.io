# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using a hybrid approach:
    1. Structured initial sampling with cosine spacing
    2. Evolutionary search with crowding distance
    3. Local refinement using Nelder-Mead on promising solutions
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # DTLZ4 has a specific structure: first 12-2=10 variables affect g,
        # last 2 affect position on the front. But since we're optimizing
        # 12 variables for 3 objectives, we need to be careful.
        # For DTLZ4 with M=3, typically:
        # - First K variables determine g(x) where g is minimized
        # - Remaining M-1 variables determine position on the front
        # Here dim=12, n_obj=3, so we can assume standard DTLZ4 structure

        # Initial population with cosine spacing for better coverage
        pop_size = min(40, self.budget // 5)
        X = []
        F = []
        used = 0

        # Add structured points using cosine spacing
        n_structured = min(pop_size // 2, 15)
        for i in range(n_structured):
            # Cosine spacing in [0, 1]
            t = (i + 0.5) / n_structured
            # Create a point that spreads across the objective space
            x = np.zeros(self.dim)
            # For DTLZ4, the last M-1=2 variables control position on front
            # Use cosine spacing for these
            theta1 = np.arcsin(2 * t - 1)  # Map to [-pi/2, pi/2]
            # Actually, let's use a simpler approach: spread in the position variables
            x[self.dim - 2] = 0.5 + 0.5 * np.cos(np.pi * t)
            x[self.dim - 1] = 0.5 + 0.5 * np.sin(np.pi * t)
            # First dim-2 variables set to 0.5 (which gives g=0 for DTLZ4)
            x[:self.dim - 2] = 0.5
            x = np.clip(x, lo, hi)
            X.append(x)
            F.append(np.array(problem(x)))
            used += 1
            if used >= self.budget or time.time() > deadline:
                break

        # Add random points to fill up initial population
        while len(X) < pop_size and used < self.budget:
            x = lo + range_ * rng.random(self.dim)
            X.append(x)
            F.append(np.array(problem(x)))
            used += 1
            if time.time() > deadline:
                break

        X = np.array(X)
        F = np.array(F)

        # Crowding distance calculation
        def crowding_distance(F, n_obj):
            if len(F) <= 2:
                return np.full(len(F), np.inf)
            cd = np.zeros(len(F))
            for m in range(n_obj):
            # Sort by objective m
                order = np.argsort(F[:, m])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                f_min, f_max = F[order[0], m], F[order[-1], m]
                if f_max - f_min > 1e-12:
                    for i in range(1, len(order) - 1):
                        cd[order[i]] += (F[order[i + 1], m] - F[order[i - 1], m]) / (f_max - f_min)
            return cd

        # Non-dominated sorting
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = np.zeros(n, dtype=int)
            fronts = [[]]
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_count[p] += 1
                if n_count[p] == 0:
                    fronts[0].append(p)
            i = 0
            while len(fronts[i]) > 0:
                Q = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_count[q] -= 1
                        if n_count[q] == 0:
                            Q.append(q)
                i += 1
                if len(Q) > 0:
                    fronts.append(Q)
            return fronts[:-1]

        # Evolutionary loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Get non-dominated fronts
            fronts = non_dominated_sort(F)
            if not fronts:
                fronts = [[i for i in range(len(F))]]

            # Select parents from first front (or first few fronts)
            front0 = fronts[0]
            if len(front0) == 0:
                front0 = [0]

            # Compute crowding distance for selection
            cd = crowding_distance(F[front0], self.n_obj)

            # Tournament selection: pick 2 random from front0, select higher CD
            p1_idx = front0[rng.integers(len(front0))]
            p2_idx = front0[rng.integers(len(front0))]
            if cd[p1_idx] >= cd[p2_idx]:
                parent = X[p1_idx]
            else:
                parent = X[p2_idx]

            # Mutation: adaptive step size
            # Use smaller steps as we approach the deadline
            elapsed = time.time() - start_time
            remaining = max(1, self.budget - used)
            step_scale = 0.05 * (remaining / self.budget)
            step_scale = max(0.005, step_scale)

            # Gaussian mutation with occasional larger jumps
            if rng.random() < 0.1:
                # Larger jump for exploration
                child = parent + step_scale * 3 * (hi - lo) * rng.standard_normal(self.dim)
            else:
                child = parent + step_scale * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)

            fc = np.array(problem(child))
            used += 1

            # Replace worst point (highest crowding distance in last front, or random)
            if len(fronts) > 1:
                last_front = fronts[-1]
                if len(last_front) > 1:
                    cd_last = crowding_distance(F[last_front], self.n_obj)
                    worst_in_last = last_front[np.argmax(cd_last)]
                    X[worst_in_last] = child
                    F[worst_in_last] = fc
                else:
                    # Replace a random point
                    idx = rng.integers(len(F))
                    X[idx] = child
                    F[idx] = fc
            else:
                # Single front: replace point with highest crowding distance
                cd_all = crowding_distance(F, self.n_obj)
                worst = np.argmax(cd_all)
                X[worst] = child
                F[worst] = fc

        # Local refinement: use Nelder-Mead on top solutions
        if used < self.budget and time.time() < deadline:
            from scipy.optimize import minimize

            # Get non-dominated points
            fronts = non_dominated_sort(F)
            non_dom_indices = fronts[0] if fronts else []

            # Select top points by crowding distance for local optimization
            if len(non_dom_indices) > 0:
                cd_nd = crowding_distance(F[non_dom_indices], self.n_obj)
                # Sort by crowding distance, take top few
                n_refine = min(5, len(non_dom_indices), (self.budget - used) // 3)
                if n_refine > 0:
                    # Get indices sorted by CD
                    sorted_nd = [non_dom_indices[i] for i in np.argsort(cd_nd)[::-1][:n_refine]]

                    for idx in sorted_nd:
                        if used >= self.budget or time.time() > deadline:
                            break

                        x0 = X[idx].copy()
                        f0 = F[idx].copy()

                        # Define scalarized objective using weighted sum with weights from CD
                        # Use equal weights as a simple approach
                        weights = np.ones(self.n_obj) / self.n_obj

                        def scalarized(x):
                            return np.dot(weights, np.array(problem(x)))

                        try:
                            res = minimize(scalarized, x0, method='Nelder-Mead',
                                         bounds=list(zip(lo, hi)),
                                         options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-6})
                            if res.success and res.fun < np.dot(weights, f0):
                                x_new = np.clip(res.x, lo, hi)
                                f_new = np.array(problem(x_new))
                                # Check if it's non-dominated or improves
                                # Replace if it's better in at least one objective and not worse in all
                                if np.any(f_new < f0) and not np.all(f_new > f0):
                                    X[idx] = x_new
                                    F[idx] = f_new
                                    used += 1  # Count the final evaluation
                        except Exception:
                            pass

        return X
# EVOLVE-BLOCK-END
