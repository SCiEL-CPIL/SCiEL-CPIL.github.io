import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        timeout = 295.0  # Safety margin before 300s hard limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Archive of non-dominated points
        archive_X = []
        archive_F = []
        evals = 0
        
        def non_dominated_sort(F):
            """
            Returns a list of fronts. Each front is a list of indices.
            """
            n = len(F)
            if n == 0:
                return []
            
            S = [[] for _ in range(n)]  # S[i] = set of indices dominated by i
            n_d = [0] * n  # n_d[i] = number of points dominating i
            rank = [0] * n
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    # Check if j dominates i
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_d[i] += 1
            
            fronts = []
            current_front = []
            for i in range(n):
                if n_d[i] == 0:
                    rank[i] = 0
                    current_front.append(i)
            
            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                current_front = next_front
                k += 1
            
            return fronts

        def crowding_distance(F, indices):
            """
            Calculate crowding distance for points with given indices.
            Returns a dictionary {index: distance}
            """
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}
            
            m = len(indices)
            cd = {i: 0.0 for i in indices}
            
            for k in range(self.n_obj):
                # Sort indices by k-th objective
                sorted_indices = sorted(indices, key=lambda i: F[i][k])
                
                # Assign infinite distance to boundary points
                cd[sorted_indices[0]] = float('inf')
                cd[sorted_indices[-1]] = float('inf')
                
                f_min = F[sorted_indices[0]][k]
                f_max = F[sorted_indices[-1]][k]
                
                if f_max - f_min == 0:
                    continue
                
                for i in range(1, m - 1):
                    cd[sorted_indices[i]] += (F[sorted_indices[i+1]][k] - F[sorted_indices[i-1]][k]) / (f_max - f_min)
            
            return cd

        def update_archive_simple(x, f):
            """Simpler archive update"""
            # Check if f is dominated by any archive point
            for f_arch in archive_F:
                if np.all(f_arch <= f) and np.any(f_arch < f):
                    return  # f is dominated, don't add
            
            # Remove archive points dominated by f
            new_archive_X = []
            new_archive_F = []
            for i in range(len(archive_F)):
                if not (np.all(f <= archive_F[i]) and np.any(f < archive_F[i])):
                    new_archive_X.append(archive_X[i])
                    new_archive_F.append(archive_F[i])
            
            # Add new point
            new_archive_X.append(x)
            new_archive_F.append(f)
            
            archive_X.clear()
            archive_F.clear()
            archive_X.extend(new_archive_X)
            archive_F.extend(new_archive_F)
            
        def tournament_select():
            """
            Select a parent from the archive using tournament selection
            based on non-dominated sorting and crowding distance.
            """
            if not archive_X:
                return None
            
            n = len(archive_X)
            if n == 1:
                return archive_X[0]
            
            # Get non-dominated fronts
            F_arr = np.array(archive_F)
            fronts = non_dominated_sort(F_arr)
            
            # Assign ranks
            rank = [0] * n
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i
            
            # Calculate crowding distance for the first front (usually the most populated or important)
            # Actually, we should calculate for all points in the selection pool?
            # Standard NSGA-II calculates CD for the front being filled.
            # Here, we are selecting one parent. 
            # We can calculate CD for the entire archive or just the best front.
            # Let's calculate CD for the best front (rank 0) and assign 0 to others?
            # Or just use rank as the primary key and CD as secondary.
            
            # For simplicity and speed, let's just pick randomly from the best front found in a tournament?
            # No, let's do a proper tournament.
            
            # Pick 2 random candidates
            i1 = np.random.randint(n)
            i2 = np.random.randint(n)
            
            if rank[i1] != rank[i2]:
                return archive_X[i1] if rank[i1] < rank[i2] else archive_X[i2]
            
            # Same rank, use crowding distance
            # Calculate CD for the front containing i1
            front_indices = [idx for idx in range(n) if rank[idx] == rank[i1]]
            cd = crowding_distance(F_arr, front_indices)
            
            if cd[i1] > cd[i2]:
                return archive_X[i1]
            else:
                return archive_X[i2]

        # Initialization: sample points spread across the space
        # Use a stratified approach for better initial coverage
        N_init = min(20, budget // 10)
        for i in range(N_init):
            if time.time() - start_time > timeout:
                break
            x = lo + (hi - lo) * np.random.random(dim)
            f = problem(x)
            evals += 1
            update_archive_simple(x, f)
        
        # Main loop: iterate through archive and perturb
        while evals < budget:
            if time.time() - start_time > timeout:
                break
            
            if not archive_X:
                # Fallback: random point
                x = lo + (hi - lo) * np.random.random(dim)
            else:
                # Pick a point from archive using tournament selection
                x_parent = tournament_select()
                
                # Adaptive step size: start large, shrink over time
                progress = evals / budget
                sigma = (1.0 - 0.5 * progress) * (hi - lo) * 0.3
                x = x_parent + sigma * np.random.standard_normal(dim)
                x = np.clip(x, lo, hi)
            
            f = problem(x)
            evals += 1
            update_archive_simple(x, f)
        
        return archive_X if archive_X else [np.zeros(dim)]
