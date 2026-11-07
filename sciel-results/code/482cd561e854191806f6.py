import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj

        # Budget allocation
        init_size = min(20, self.budget // 5)
        remaining = self.budget - init_size

        # Initialize with random samples
        X_init = lo + (hi - lo) * np.random.random((init_size, n))
        F_init = np.array([problem(x) for x in X_init])
        evals_used = init_size

        # Archive: list of (x, f)
        archive = []
        for x, f in zip(X_init, F_init):
            dominated = False
            for _, f_j in archive:
                if np.all(f_j <= f) and np.any(f_j < f):
                    dominated = True
                    break
            if not dominated:
                # Remove points dominated by new point
                new_archive = []
                for x_j, f_j in archive:
                    if not (np.all(f <= f_j) and np.any(f < f_j)):
                        new_archive.append((x_j, f_j))
                new_archive.append((x, f))
                archive = new_archive

        def crowding_distance(points):
            if len(points) <= 2:
                return [float('inf')] * len(points)
            dist = [0.0] * len(points)
            for k in range(m):
                indices = sorted(range(len(points)), key=lambda i: points[i][k])
                dist[indices[0]] += float('inf')
                dist[indices[-1]] += float('inf')
                f_range = points[indices[-1]][k] - points[indices[0]][k]
                if f_range == 0:
                    continue
                for i in range(1, len(indices) - 1):
                    dist[indices[i]] += (points[indices[i+1]][k] - points[indices[i-1]][k]) / f_range
            return dist

        max_archive_size = 30

        # Main optimization loop
        while evals_used < self.budget:
            if time.time() > deadline:
                break

            # Selection strategy
            if len(archive) < 2:
                parent_idx = 0
                x_parent = archive[parent_idx][0]
                sigma = 0.15
                x_child = x_parent + sigma * (hi - lo) * np.random.standard_normal(n)
            else:
                strategy = np.random.random()
                if strategy < 0.6:
                    # Mutation from a random archive point
                    p_idx = np.random.randint(len(archive))
                    x_parent = archive[p_idx][0]
                    sigma = 0.08
                    x_child = x_parent + sigma * (hi - lo) * np.random.standard_normal(n)
                elif strategy < 0.85:
                    # Interpolation between two archive points
                    i1, i2 = np.random.choice(len(archive), 2, replace=False)
                    x1 = archive[i1][0]
                    x2 = archive[i2][0]
                    alpha = np.random.random()
                    x_child = x1 + alpha * (x2 - x1)
                    x_child += 0.02 * (hi - lo) * np.random.standard_normal(n)
                else:
                    # Crossover with Gaussian noise
                    i1, i2 = np.random.choice(len(archive), 2, replace=False)
                    x1 = archive[i1][0]
                    x2 = archive[i2][0]
                    x_child = 0.5 * (x1 + x2) + 0.05 * (hi - lo) * np.random.standard_normal(n)

            x_child = np.clip(x_child, lo, hi)
            f_child = problem(x_child)
            evals_used += 1

            # Check domination
            dominated = False
            for _, f_j in archive:
                if np.all(f_j <= f_child) and np.any(f_j < f_child):
                    dominated = True
                    break

            if not dominated:
                # Remove points dominated by child
                new_archive = []
                for x_j, f_j in archive:
                    if not (np.all(f_child <= f_j) and np.any(f_child < f_j)):
                        new_archive.append((x_j, f_j))
                new_archive.append((x_child, f_child))
                archive = new_archive

                # If too large, remove least crowded
                if len(archive) > max_archive_size:
                    f_list = [f for _, f in archive]
                    cd = crowding_distance(f_list)
                    min_idx = int(np.argmin(cd))
                    archive.pop(min_idx)

        return None
