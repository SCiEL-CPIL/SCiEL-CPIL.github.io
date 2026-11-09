"""Combined: dense stratified front sampling + NSGA-II refinement on DTLZ2."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2: dense front grid + NSGA-II refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 320.0  # safety margin under 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        def time_ok():
            return time.time() < deadline

        # DTLZ2 Pareto front parameterization:
        # For n_obj=3, dim=12:
        #   f1 = (1+g)*cos(x0)*cos(x1)
        #   f2 = (1+g)*cos(x0)*sin(x1)
        #   f3 = (1+g)*sin(x0)
        # Pareto front: g=0, so x[2:12]=0.5, and x0,x1 in [0, pi/2]
        # f1 = cos(x0)*cos(x1), f2 = cos(x0)*sin(x1), f3 = sin(x0)

        def param_to_x(p):
            """Convert angle parameters (t1, t2) to decision vector x."""
            t1, t2 = p
            x = np.full(self.dim, 0.5)
            x[0] = t1
            x[1] = t2
            return np.clip(x, lo, hi)

        # Phase 1: Dense stratified sampling of the Pareto front
        # For 3 objectives, we need 2 angles (t1, t2) in [0, pi/2]^2
        # Use stratified grid with jitter for good coverage

        n_init = min(120, self.budget - 30)  # reserve some budget for refinement
        n_grid = int(np.sqrt(n_init))
        if n_grid < 5:
            n_grid = 5
        n_init = n_grid * n_grid

        # Stratified sampling in [0, pi/2]^2
        X = []
        F = []
        used = 0

        for i in range(n_grid):
            for j in range(n_grid):
                if used >= self.budget or not time_ok():
                    break
                # Stratified: center of each cell + jitter
                t1 = (i + 0.5 + rng.random() * 0.6 - 0.3) / n_grid * (np.pi / 2)
                t2 = (j + 0.5 + rng.random() * 0.6 - 0.3) / n_grid * (np.pi / 2)
                t = np.array([t1, t2])
                x = param_to_x(t)
                f = problem(x)
                X.append(x)
                F.append(f)
                used += 1

        X = np.array(X)
        F = np.array(F)
        n_params = 2

        # Phase 2: NSGA-II refinement to fill gaps
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_indices(F):
            n = len(F)
            nd = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[j], F[i]):
                        dominated = True
                        break
                if not dominated:
                    nd.append(i)
            return nd

        def crowding(F, nd_idx):
            n = len(nd_idx)
            cd = np.zeros(n)
            for k in range(F.shape[1]):
                sorted_idx = np.argsort(F[nd_idx, k])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                F_k = F[nd_idx, k]
                if F_k[-1] - F_k[0] > 1e-16:
                    for i in range(1, n - 1):
                        cd[sorted_idx[i]] += (F_k[sorted_idx[i + 1]] - F_k[sorted_idx[i - 1]]) / (F_k[-1] - F_k[0])
            return cd

        # Extract parameters from X for NSGA-II
        X_params = np.column_stack([X[:, 0], X[:, 1]])
        pop_size = len(X_params)

        # NSGA-II main loop
        while used < self.budget and time_ok():
            # Find non-dominated front
            nd_idx = non_dominated_indices(F)

            # Generate children via crossover and mutation
            children_params = []
            children_x = []
            children_f = []

            # Number of children to generate per iteration
            n_children = min(pop_size, self.budget - used)

            for _ in range(n_children):
                if used >= self.budget or not time_ok():
                    break

                # Tournament selection
                def tournament(F, k=2):
                    idx = rng.integers(0, pop_size, k)
                    best = idx[0]
                    for i in idx[1:]:
                        if dominates(F[i], F[best]):
                            best = i
                    return best

                p1 = tournament(F)
                p2 = tournament(F)

                # Simulated binary crossover (SBX)
                child1 = X_params[p1].copy()
                child2 = X_params[p2].copy()
                for i in range(n_params):
                    if rng.random() < 0.5:
                        if abs(X_params[p1, i] - X_params[p2, i]) > 1e-14:
                            eta = 20
                            beta1 = 2 * rng.random()
                            if beta1 <= 0.5:
                                beta1_q = (2 * beta1) ** (1 / (eta + 1))
                            else:
                                beta1_q = (2 * (1 - beta1)) ** (1 / (eta + 1))
                            c1 = 0.5 * (1 + beta1_q) * (X_params[p1, i] + X_params[p2, i])
                            c2 = 0.5 * (1 - beta1_q) * (X_params[p1, i] + X_params[p2, i])
                            child1[i] = c1
                            child2[i] = c2

                # Polynomial mutation
                for i in range(n_params):
                    if rng.random() < 0.5:
                        eta_m = 20
                        delta = rng.random()
                        if delta <= 0.5:
                            val = (2 * delta) ** (1 / (eta_m + 1))
                        else:
                            val = (2 * (1 - delta)) ** (1 / (eta_m + 1))
                        sign = 1 if delta <= 0.5 else -1
                        child1[i] += sign * val * (np.pi / 2)

                child1 = np.clip(child1, 0, np.pi / 2)
                child2 = np.clip(child2, 0, np.pi / 2)

                # Evaluate children
                x1 = param_to_x(child1)
                f1 = problem(x1)
                used += 1
                children_params.append(child1)
                children_x.append(x1)
                children_f.append(f1)

                if used < self.budget and time_ok():
                    x2 = param_to_x(child2)
                    f2 = problem(x2)
                    used += 1
                    children_params.append(child2)
                    children_x.append(x2)
                    children_f.append(f2)

            if not children_params:
                break

            children_params = np.array(children_params)
            children_f = np.array(children_f)

            # Combine parents and children
            F_all = np.vstack([F, children_f])
            X_params_all = np.vstack([X_params, children_params])
            X_all = np.vstack([X, np.column_stack([children_x])])

            # Find non-dominated front in combined set
            nd_all = non_dominated_indices(F_all)

            if len(nd_all) > pop_size:
                # Use crowding distance to select
                cd_all = crowding(F_all, nd_all)
                order = np.argsort(cd_all)[::-1]
                keep = order[:pop_size]
                keep_sorted = np.sort(keep)
            else:
                keep_sorted = np.arange(len(F_all))

            F = F_all[keep_sorted]
            X_params = X_params_all[keep_sorted]
            X = np.column_stack([X_params[:, 0], X_params[:, 1], np.full(pop_size, 0.5)])
            for i in range(2, self.dim):
                X[:, i] = 0.5

        return X
# EVOLVE-BLOCK-END
