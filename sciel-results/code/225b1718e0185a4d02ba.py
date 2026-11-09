import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer using differential evolution with archive-based selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _is_dominated(self, a, b):
        """Check if a dominates b (a is better than b in all objectives, strictly in at least one)."""
        return np.all(a <= b) and np.any(a < b)

    def _get_non_dominated_indices(self, F):
        """Return indices of non-dominated solutions in F."""
        n = len(F)
        if n == 0:
            return []
        
        dominated = [False] * n
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if self._is_dominated(F[i], F[j]):
                    dominated[j] = True
        
        return [i for i in range(n) if not dominated[i]]

    def _compute_crowding_distance(self, F, indices):
        """Compute crowding distance for given indices in F."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        distances = np.zeros(n)
        F_sub = F[indices]
        n_obj = F_sub.shape[1]
        
        for m in range(n_obj):
            order = np.argsort(F_sub[:, m])
            distances[order[0]] = np.inf
            distances[order[-1]] = np.inf
            
            f_min = F_sub[order[0], m]
            f_max = F_sub[order[-1], m]
            if f_max - f_min > 1e-15:
                for i in range(1, n - 1):
                    distances[order[i]] += (F_sub[order[i + 1], m] - F_sub[order[i - 1], m]) / (f_max - f_min)
        
        return distances

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo

        # Archive size for storing non-dominated solutions
        archive_size = 40
        
        # Initialize with random points spread across the domain
        n_init = 20
        X_init = np.random.uniform(lo, hi, size=(n_init, self.dim))
        F_init = np.array([problem(x) for x in X_init])
        eval_count = n_init

        # Keep track of all evaluated points
        all_X = list(X_init)
        all_F = list(F_init)

        # Initialize archive with non-dominated solutions
        archive_X = []
        archive_F = []
        for i in range(n_init):
            f_i = F_init[i]
            dominated = False
            for j in range(len(archive_F)):
                if self._is_dominated(archive_F[j], f_i):
                    dominated = True
                    break
            if not dominated:
                # Remove any archive members dominated by f_i
                new_archive_X = []
                new_archive_F = []
                for j in range(len(archive_F)):
                    if not self._is_dominated(f_i, archive_F[j]):
                        new_archive_X.append(archive_X[j])
                        new_archive_F.append(archive_F[j])
                archive_X = new_archive_X
                archive_F = new_archive_F
                archive_X.append(X_init[i])
                archive_F.append(f_i)

        # Differential evolution parameters
        F_de = 0.8  # Differential weight
        CR = 0.9  # Crossover rate
        pop_size = 30  # Population size for DE

        # Initialize DE population
        pop_X = np.random.uniform(lo, hi, size=(pop_size, self.dim))
        pop_F = np.array([problem(x) for x in pop_X])
        eval_count += pop_size
        all_X.extend(pop_X)
        all_F.extend(pop_F)

        # Main loop
        generation = 0
        while eval_count < self.budget:
            if time.time() > deadline - 10:
                break
            
            generation += 1
            
            # Update archive with current population
            for i in range(pop_size):
                f_i = pop_F[i]
                dominated = False
                for j in range(len(archive_F)):
                    if self._is_dominated(archive_F[j], f_i):
                        dominated = True
                        break
                if not dominated:
                    # Remove any archive members dominated by f_i
                    new_archive_X = []
                    new_archive_F = []
                    for j in range(len(archive_F)):
                        if not self._is_dominated(f_i, archive_F[j]):
                            new_archive_X.append(archive_X[j])
                            new_archive_F.append(archive_F[j])
                    archive_X = new_archive_X
                    archive_F = new_archive_F
                    archive_X.append(pop_X[i])
                    archive_F.append(f_i)
            
            # Limit archive size using crowding distance
            if len(archive_F) > archive_size:
                archive_F_arr = np.array(archive_F)
                cd = self._compute_crowding_distance(archive_F_arr, list(range(len(archive_F_arr))))
                order = np.argsort(-cd)
                keep = order[:archive_size]
                archive_X = [archive_X[i] for i in keep]
                archive_F = [archive_F[i] for i in keep]

            # Create offspring using DE
            offspring_X = []
            offspring_F = []
            
            for i in range(pop_size):
                if eval_count >= self.budget:
                    break
                if time.time() > deadline - 5:
                    break
                
                # Select three distinct parents
                idxs = [i]
                while len(idxs) < 3:
                    j = np.random.randint(pop_size)
                    if j not in idxs:
                        idxs.append(j)
                
                a, b, c = pop_X[idxs[0]], pop_X[idxs[1]], pop_X[idxs[2]]
                
                # DE mutation
                v = a + F_de * (b - c)
                v = np.clip(v, lo, hi)
                
                # Binary crossover
                trial = np.random.uniform(0, 1, self.dim) < CR
                trial[i] = True  # Ensure at least one gene from mutant
                trial_x = np.where(trial, v, pop_X[i])
                trial_x = np.clip(trial_x, lo, hi)
                
                # Evaluate
                trial_f = problem(trial_x)
                eval_count += 1
                all_X.append(trial_x)
                all_F.append(trial_f)
                
                # Selection: keep the better of trial and parent
                if self._is_dominated(trial_f, pop_F[i]):
                    # Trial dominates parent
                    offspring_X.append(trial_x)
                    offspring_F.append(trial_f)
                elif self._is_dominated(pop_F[i], trial_f):
                    # Parent dominates trial
                    offspring_X.append(pop_X[i])
                    offspring_F.append(pop_F[i])
                else:
                    # Neither dominates - keep both in archive, select based on crowding
                    # For simplicity, keep the one with better crowding in the current front
                    # But for DE, we just keep the trial to maintain diversity
                    offspring_X.append(trial_x)
                    offspring_F.append(trial_f)
            
            if not offspring_X:
                break
            
            pop_X = np.array(offspring_X)
            pop_F = np.array(offspring_F)

        # Final archive update
        for i in range(len(all_F)):
            f_i = all_F[i]
            dominated = False
            for j in range(len(archive_F)):
                if self._is_dominated(archive_F[j], f_i):
                    dominated = True
                    break
            if not dominated:
                # Remove any archive members dominated by f_i
                new_archive_X = []
                new_archive_F = []
                for j in range(len(archive_F)):
                    if not self._is_dominated(f_i, archive_F[j]):
                        new_archive_X.append(archive_X[j])
                        new_archive_F.append(archive_F[j])
                archive_X = new_archive_X
                archive_F = new_archive_F
                archive_X.append(all_X[i])
                archive_F.append(f_i)
        
        # Limit final archive size
        if len(archive_F) > archive_size:
            archive_F_arr = np.array(archive_F)
            cd = self._compute_crowding_distance(archive_F_arr, list(range(len(archive_F_arr))))
            order = np.argsort(-cd)
            keep = order[:archive_size]
            archive_X = [archive_X[i] for i in keep]
            archive_F = [archive_F[i] for i in keep]

        return np.array(all_X)
