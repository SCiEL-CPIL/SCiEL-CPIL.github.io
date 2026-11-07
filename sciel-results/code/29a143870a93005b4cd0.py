"""Multi-objective optimizer for WFG4 using NSGA-II with global archive."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II based multi-objective optimizer for WFG4 with global archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 285.0  # Safety margin from 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        span = hi - lo

        ref = np.array([3.0, 5.0, 7.0])

        # Parameters
        pop_size = 30
        cx_prob = 0.9
        mut_prob = 1.0 / dim
        eta_c = 15.0
        eta_m = 20.0

        used = 0
        # Global archive: list of (x, f) tuples
        archive_X = []
        archive_F = []

        def time_left():
            return deadline - (time.time() - start_time)

        def evaluate(x):
            nonlocal used
            f = problem(x)
            used += 1
            return f

        def is_dominated(f_new, F_archive):
            """Check if f_new is dominated by any point in F_archive."""
            if len(F_archive) == 0:
                return False
            F_arr = np.array(F_archive)
            # f_new is dominated if exists f_a such that f_a <= f_new and f_a != f_new
            dominated = np.any(np.all(F_arr <= f_new, axis=1) & np.any(F_arr < f_new, axis=1))
            return dominated

        def update_archive(x, f):
            """Add (x, f) to archive if non-dominated, removing dominated points."""
            nonlocal archive_X, archive_F
            # Check if f is dominated by existing archive
            if is_dominated(f, archive_F):
                return False
            # Remove any archive points dominated by f
            new_archive_X = []
            new_archive_F = []
            for i in range(len(archive_X)):
                f_old = archive_F[i]
                # Check if old is dominated by new
                if np.all(f <= f_old) and np.any(f < f_old):
                    continue  # old is dominated, skip
                new_archive_X.append(archive_X[i])
                new_archive_F.append(archive_F[i])
            new_archive_X.append(x)
            new_archive_F.append(f)
            archive_X = new_archive_X
            archive_F = new_archive_F
            return True

        def non_dominated_sort(F):
            """
            Fast non-dominated sorting (vectorized).
            Returns a list of lists of indices.
            """
            n = len(F)
            if n == 0:
                return [], np.zeros(0, dtype=int)
            
            # Vectorize dominance check
            # F[i] dominates F[j] if all(F[i] <= F[j]) and any(F[i] < F[j])
            S = [[] for _ in range(n)]
            n_d = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_d[j] += 1
            
            fronts = []
            current_front = [i for i in range(n) if n_d[i] == 0]
            
            if current_front:
                for i in current_front:
                    rank[i] = 0
                fronts.append(current_front)
            
            k = 1
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            rank[j] = k
                            next_front.append(j)
                if not next_front:
                    break
                fronts.append(next_front)
                current_front = next_front
                k += 1
                
            return fronts, rank

        def crowding_distance(F, front_indices):
            """
            Calculate crowding distance for solutions in a front.
            Returns an array of distances indexed by position in front_indices.
            """
            m = len(front_indices)
            if m <= 2:
                return np.full(m, float('inf'))
            
            dist = np.zeros(m)
            F_sub = F[front_indices]
            
            for k in range(n_obj):
                sorted_pos = np.argsort(F_sub[:, k])
                sorted_F_k = F_sub[sorted_pos, k]
                
                # First and last have infinite distance
                dist[sorted_pos[0]] = float('inf')
                dist[sorted_pos[-1]] = float('inf')
                
                range_k = sorted_F_k[-1] - sorted_F_k[0]
                if range_k == 0:
                    continue
                    
                for i in range(1, m - 1):
                    dist[sorted_pos[i]] += (sorted_F_k[i + 1] - sorted_F_k[i - 1]) / range_k
                    
            return dist

        def select_parents(pop_rank, pop_dist):
            """
            Binary tournament selection.
            Returns indices of two parents.
            """
            n = len(pop_rank)
            
            def tournament():
                i1 = np.random.randint(0, n)
                i2 = np.random.randint(0, n)
                if pop_rank[i1] < pop_rank[i2]:
                    return i1
                elif pop_rank[i2] < pop_rank[i1]:
                    return i2
                else:
                    if pop_dist[i1] > pop_dist[i2]:
                        return i1
                    else:
                        return i2
            
            return tournament(), tournament()

        def sbx_crossover(p1, p2):
            """
            Simulated Binary Crossover.
            """
            if np.random.random() > cx_prob:
                return p1.copy(), p2.copy()
            
            c1 = p1.copy()
            c2 = p2.copy()
            
            for i in range(dim):
                if np.random.random() < 0.5:
                    if p1[i] != p2[i]:
                        if p1[i] < p2[i]:
                            x1, x2 = p1[i], p2[i]
                        else:
                            x1, x2 = p2[i], p1[i]
                        
                        u = np.random.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (eta_c + 1))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / (eta_c + 1))
                        
                        c1[i] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                        c2[i] = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
                        
                        # Clip to bounds
                        c1[i] = np.clip(c1[i], lo[i], hi[i])
                        c2[i] = np.clip(c2[i], lo[i], hi[i])
                        
            return c1, c2

        def polynomial_mutation(x):
            """
            Polynomial Mutation.
            """
            x_mut = x.copy()
            for i in range(dim):
                if np.random.random() < mut_prob:
                    u = np.random.random()
                    if u < 0.5:
                        delta_1 = (2 * u) ** (1 / (eta_m + 1)) - 1
                    else:
                        delta_1 = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                    
                    x_mut[i] = x[i] + delta_1 * (hi[i] - lo[i])
                    x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
            return x_mut

        # Phase 1: Initialization with LHS
        init_X = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = np.random.permutation(pop_size)
            init_X[:, d] = (perm + np.random.random(pop_size)) / pop_size
        init_X = lo + init_X * span
        
        init_F = []
        for i in range(pop_size):
            if used >= budget or time_left() < 5.0:
                break
            f = evaluate(init_X[i])
            init_F.append(f)
            update_archive(init_X[i].copy(), f.copy())
            
        # If we ran out of budget during init, return what we have
        if used >= budget or time_left() < 5.0:
            if archive_X:
                return np.array(archive_X)
            return np.array(init_X[:used]) if used > 0 else np.zeros((1, dim))

        pop_X = init_X[:used]
        pop_F = np.array(init_F)

        # Main Loop
        while used < budget and time_left() > 15.0:
            # Compute ranks and crowding distances
            fronts, pop_rank = non_dominated_sort(pop_F)
            pop_dist = np.zeros(len(pop_X))
            for front in fronts:
                if len(front) > 0:
                    dists = crowding_distance(pop_F, front)
                    for i, idx in enumerate(front):
                        pop_dist[idx] = dists[i]
            
            # Generate Offspring
            off_X = []
            off_F = []
            
            while len(off_X) < pop_size and used < budget and time_left() > 5.0:
                p1_idx, p2_idx = select_parents(pop_rank, pop_dist)
                c1, c2 = sbx_crossover(pop_X[p1_idx], pop_X[p2_idx])
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                f1 = evaluate(c1)
                off_X.append(c1)
                off_F.append(f1)
                update_archive(c1.copy(), f1.copy())
                
                if used < budget and time_left() > 5.0:
                    f2 = evaluate(c2)
                    off_X.append(c2)
                    off_F.append(f2)
                    update_archive(c2.copy(), f2.copy())
                    
            if not off_X:
                break
                
            # Combine Parents and Offspring
            comb_X = np.vstack([pop_X, np.array(off_X)])
            comb_F = np.vstack([pop_F, np.array(off_F)])
            
            # Sort and Select Next Generation
            fronts, comb_rank = non_dominated_sort(comb_F)
            comb_dist = np.zeros(len(comb_X))
            for front in fronts:
                if len(front) > 0:
                    dists = crowding_distance(comb_F, front)
                    for i, idx in enumerate(front):
                        comb_dist[idx] = dists[i]
            
            next_X = []
            next_F = []
            
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    next_X.extend([comb_X[i] for i in front])
                    next_F.extend([comb_F[i] for i in front])
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(next_X)
                    # Sort front indices by crowding distance (descending)
                    front_sorted = sorted(front, key=lambda i: comb_dist[i], reverse=True)
                    selected = front_sorted[:remaining]
                    next_X.extend([comb_X[i] for i in selected])
                    next_F.extend([comb_F[i] for i in selected])
                    break
            
            if not next_X:
                break
                
            pop_X = np.array(next_X)
            pop_F = np.array(next_F)

        # Final local polish: use remaining budget to refine archive points
        if used < budget and time_left() > 5.0 and len(archive_X) > 0:
            # Pick a few archive points and do local search around them
            n_polish = min(10, len(archive_X))
            # Select diverse points from archive
            if n_polish < len(archive_X):
                # Simple selection: take evenly spaced indices
                indices = np.linspace(0, len(archive_X) - 1, n_polish, dtype=int)
            else:
                indices = np.arange(len(archive_X))
            
            for idx in indices:
                if used >= budget or time_left() < 3.0:
                    break
                base_x = archive_X[idx].copy()
                # Try small perturbations
                for attempt in range(5):
                    if used >= budget or time_left() < 3.0:
                        break
                    # Random direction with small step
                    direction = np.random.randn(dim)
                    direction /= np.linalg.norm(direction) + 1e-10
                    step = 0.1 * span
                    new_x = base_x + direction * step * np.random.random(dim)
                    new_x = np.clip(new_x, lo, hi)
                    new_f = evaluate(new_x)
                    # Check if this improves any objective
                    base_f = archive_F[idx]
                    if np.any(new_f < base_f) and not np.all(new_f >= base_f):
                        # Potentially better, add to archive
                        if not is_dominated(new_f, archive_F):
                            update_archive(new_x.copy(), new_f.copy())
                            base_x = new_x.copy()
                            base_f = new_f.copy()

        # Return the global archive
        if archive_X:
            return np.array(archive_X)
        elif len(pop_X) > 0:
            return np.array(pop_X)
        else:
            return np.zeros((1, dim))

# EVOLVE-BLOCK-END
