import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([3000.0, 0.06])
        
        # Archive of points that contribute to hypervolume
        # We maintain a list of (x, f) pairs
        archive = []
        
        def hypervolume_contribution(f_new, archive_f):
            """
            Compute the hypervolume contribution of f_new given the existing archive.
            Only points strictly better than reference point count.
            Reference point is [3000.0, 0.06].
            We want to maximize the volume dominated by the union of all points.
            
            For 2 objectives, the hypervolume of a set of points can be computed by sorting.
            Contribution of a new point is the area it adds to the existing union.
            """
            # Filter out points that are not strictly better than ref
            valid_f = [f for f in archive_f if f[0] < ref[0] and f[1] < ref[1]]
            
            if f_new[0] >= ref[0] or f_new[1] >= ref[1]:
                return 0.0
            
            # If no valid points in archive, contribution is the rectangle from f_new to ref
            if not valid_f:
                return (ref[0] - f_new[0]) * (ref[1] - f_new[1])
            
            # Check if f_new is dominated by any point in archive
            for f in valid_f:
                if f[0] <= f_new[0] and f[1] <= f_new[1] and (f[0] < f_new[0] or f[1] < f_new[1]):
                    return 0.0
            
            # f_new is not dominated. Calculate the new area it adds.
            # Sort valid_f + [f_new] by f[0]
            points = sorted(valid_f + [f_new], key=lambda x: x[0])
            
            # Compute hypervolume of the new set
            def calc_hv(points, ref):
                if not points:
                    return 0.0
                # Sort by first objective
                points = sorted(points, key=lambda x: x[0])
                hv = 0.0
                prev_y = ref[1]
                for i, p in enumerate(points):
                    # The width is p[0] to next point's x (or ref[0])
                    # But we need to be careful: the "skyline" is formed by the Pareto front.
                    # Since we only keep non-dominated points in archive, all points are on the front.
                    # Hypervolume is sum of rectangles.
                    next_x = points[i+1][0] if i + 1 < len(points) else ref[0]
                    # The height is from p[1] to the "upper" envelope.
                    # Actually, for 2D, if points are sorted by x ascending, and are non-dominated,
                    # y must be ascending as well? No, non-dominated in minimization means
                    # if x1 < x2, then y1 > y2.
                    # So sorted by x ascending, y is descending.
                    # The area is sum over each point p_i of (x_{i+1} - x_i) * (ref_y - y_i)
                    # where x_{last+1} = ref_x.
                    pass
                
                # Let's do the standard 2D HV calculation.
                # Sort by f1 ascending.
                # For each point, the width is min(next_point_f1, ref_f1) - current_f1.
                # The height is ref_f2 - current_f2.
                # Wait, this is only correct if the points form a "staircase" towards the origin.
                # For non-dominated points sorted by f1 ascending, f2 is descending.
                # The union of rectangles [f1, ref_f1] x [f2, ref_f2] is the HV.
                # The area is sum_{i} (x_{i+1} - x_i) * (ref_y - y_i)
                # where x_{n+1} = ref_x.
                
                total_hv = 0.0
                for i in range(len(points)):
                    x_i = points[i][0]
                    y_i = points[i][1]
                    x_next = points[i+1][0] if i + 1 < len(points) else ref[0]
                    # Width
                    w = x_next - x_i
                    # Height
                    h = ref[1] - y_i
                    total_hv += w * h
                return total_hv

            hv_before = calc_hv(valid_f, ref)
            hv_after = calc_hv(points, ref)
            return hv_after - hv_before

        def add_to_archive(x, f):
            """Add point to archive if it improves hypervolume, else remove dominated points."""
            if f[0] >= ref[0] or f[1] >= ref[1]:
                return False
            
            # Check if dominated
            for i, (ax, af) in enumerate(archive):
                if af[0] <= f[0] and af[1] <= f[1] and (af[0] < f[0] or af[1] < f[1]):
                    return False
            
            # Check if it dominates any existing points
            new_archive = []
            for i, (ax, af) in enumerate(archive):
                if f[0] <= af[0] and f[1] <= af[1] and (f[0] < af[0] or f[1] < af[1]):
                    # f dominates af, do not add af
                    continue
                new_archive.append((ax, af))
            
            new_archive.append((x, f))
            
            # Prune: keep only non-dominated points
            # Since we just added a non-dominated point and removed dominated ones,
            # the archive might still have points that are now dominated by each other?
            # No, the archive was previously non-dominated. Adding one non-dominated point
            # and removing those it dominates keeps the set non-dominated.
            # However, we should ensure we don't have duplicates or near-duplicates that don't add HV.
            
            # Calculate HV contribution of the new point in the context of the *pruned* archive
            # Actually, simpler: just keep the archive small and non-dominated.
            # If the archive grows too large, prune by HV contribution.
            
            self.archive = new_archive
            return True

        # Initialize archive
        self.archive = []
        
        # 1. Initial Sampling: Latin Hypercube or Random
        N_init = 50
        X = lo + (hi - lo) * np.random.random((N_init, self.dim))
        F = np.array([problem(x) for x in X])
        used = N_init
        
        for i in range(N_init):
            add_to_archive(X[i], F[i])
        
        # 2. Local Search / Evolutionary Loop
        # Use a simple evolutionary strategy on the archive or a separate population.
        # Given the small budget, a direct search on the Pareto front is efficient.
        
        # We will maintain a population of size N_pop = 30.
        # Initialize population from archive if possible, else random.
        N_pop = 30
        if len(self.archive) >= N_pop:
            # Sample from archive
            indices = np.random.choice(len(self.archive), N_pop, replace=False)
            X_pop = np.array([self.archive[i][0] for i in indices])
            F_pop = np.array([self.archive[i][1] for i in indices])
        else:
            # Fill rest with random
            X_pop = np.zeros((N_pop, self.dim))
            F_pop = np.zeros((N_pop, self.n_obj))
            for i in range(len(self.archive)):
                X_pop[i] = self.archive[i][0]
                F_pop[i] = self.archive[i][1]
            for i in range(len(self.archive), N_pop):
                X_pop[i] = lo + (hi - lo) * np.random.random(self.dim)
                F_pop[i] = problem(X_pop[i])
                used += 1
        
        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Select two parents from population (random or tournament)
            # Simple random selection to maintain diversity
            i1 = np.random.randint(N_pop)
            i2 = np.random.randint(N_pop)
            
            # Crossover (SBX)
            x1 = X_pop[i1]
            x2 = X_pop[i2]
            
            child1 = x1.copy()
            child2 = x2.copy()
            eta = 20.0
            for j in range(self.dim):
                if np.random.random() < 0.5:
                    if abs(x1[j] - x2[j]) > 1e-14:
                        u = np.random.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (eta + 1))
                        else:
                            beta = (2 * (1 - u)) ** (-1 / (eta + 1))
                        
                        child1[j] = 0.5 * ((1 + beta) * x1[j] + (1 - beta) * x2[j])
                        child2[j] = 0.5 * ((1 - beta) * x1[j] + (1 + beta) * x2[j])
            
            # Mutation
            eta_m = 20.0
            for j in range(self.dim):
                if np.random.random() < 1.0 / self.dim:
                    u = np.random.random()
                    if u < 0.5:
                        delta_q = (2 * u) ** (1 / (eta_m + 1)) - 1
                    else:
                        delta_q = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                    child1[j] += delta_q * (hi[j] - lo[j])
            
            for j in range(self.dim):
                if np.random.random() < 1.0 / self.dim:
                    u = np.random.random()
                    if u < 0.5:
                        delta_q = (2 * u) ** (1 / (eta_m + 1)) - 1
                    else:
                        delta_q = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                    child2[j] += delta_q * (hi[j] - lo[j])
            
            child1 = np.clip(child1, lo, hi)
            child2 = np.clip(child2, lo, hi)
            
            # Evaluate child 1
            if used < self.budget:
                f1 = problem(child1)
                used += 1
                
                # Update archive
                add_to_archive(child1, f1)
                
                # Replace worst in population
                # Worst is the one with lowest HV contribution? Or just random?
                # Let's replace the one that is dominated or has lowest crowding.
                # For simplicity, replace random or the one that is dominated.
                # Find if child1 dominates any pop member
                replaced = False
                for k in range(N_pop):
                    if f1[0] <= F_pop[k][0] and f1[1] <= F_pop[k][1] and (f1[0] < F_pop[k][0] or f1[1] < F_pop[k][1]):
                        X_pop[k] = child1
                        F_pop[k] = f1
                        replaced = True
                        break
                if not replaced:
                    # Replace random
                    k = np.random.randint(N_pop)
                    X_pop[k] = child1
                    F_pop[k] = f1
            
            # Evaluate child 2
            if used < self.budget:
                f2 = problem(child2)
                used += 1
                
                add_to_archive(child2, f2)
                
                replaced = False
                for k in range(N_pop):
                    if f2[0] <= F_pop[k][0] and f2[1] <= F_pop[k][1] and (f2[0] < F_pop[k][0] or f2[1] < F_pop[k][1]):
                        X_pop[k] = child2
                        F_pop[k] = f2
                        replaced = True
                        break
                if not replaced:
                    k = np.random.randint(N_pop)
                    X_pop[k] = child2
                    F_pop[k] = f2

        return X_pop
