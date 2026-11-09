"""Multi-objective optimizer using NSGA-II style approach with tournament selection."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer using NSGA-II with non-dominated sorting and crowding distance."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        budget = self.budget
        
        # NSGA-II Parameters
        pop_size = min(40, budget // 5)
        if pop_size < 20:
            pop_size = 20
            
        # Ensure we have enough budget
        if budget < pop_size + 10:
            pop_size = max(5, budget - 10)

        # Initialize Population
        population_x = lo + rng.random((pop_size, dim)) * span
        population_f = np.array([problem(x) for x in population_x])
        used = pop_size
        
        # Archive of all non-dominated solutions found
        archive_x = []
        archive_f = []
        
        def dominates(f1, f2):
            """Check if f1 dominates f2: f1 <= f2 and f1 != f2"""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def is_dominated_by_any(f, f_list):
            """Check if f is dominated by any point in f_list"""
            for ef in f_list:
                if dominates(ef, f):
                    return True
            return False

        def add_to_archive(x, f):
            """Add to archive if non-dominated, remove dominated points"""
            # Check if dominated by any existing
            for ef in archive_f:
                if dominates(ef, f):
                    return False
            
            # Remove points dominated by new point
            keep = []
            for i, ef in enumerate(archive_f):
                if not dominates(f, ef):
                    keep.append(i)
            
            if len(keep) < len(archive_f):
                archive_x[:] = [archive_x[i] for i in keep]
                archive_f[:] = [archive_f[i] for i in keep]
            
            archive_x.append(x)
            archive_f.append(f)
            return True

        def crowding_distance(f_arr):
            """Calculate crowding distances for a set of objective vectors"""
            n = len(f_arr)
            if n <= 2:
                return np.full(n, np.inf)
            
            crowd = np.zeros(n)
            f_min = f_arr.min(axis=0)
            f_max = f_arr.max(axis=0)
            f_range = f_max - f_min
            f_range[f_range == 0] = 1.0
            
            for m in range(self.n_obj):
                order = np.argsort(f_arr[:, m])
                crowd[order[0]] = np.inf
                crowd[order[-1]] = np.inf
                if n > 2:
                    for i in range(1, n - 1):
                        dist = f_arr[order[i+1], m] - f_arr[order[i-1], m]
                        crowd[order[i]] += dist / f_range[m]
            
            return crowd

        def non_dominated_sort(f_arr):
            """Return list of fronts, each a list of indices"""
            n = len(f_arr)
            S = [[] for _ in range(n)]  # S[i] = set of solutions dominated by i
            n_count = np.zeros(n, dtype=int)  # number of solutions that dominate i
            rank = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(f_arr[i], f_arr[j]):
                        S[i].append(j)
                    elif dominates(f_arr[j], f_arr[i]):
                        n_count[i] += 1
            
            # First front
            front = []
            for i in range(n):
                if n_count[i] == 0:
                    rank[i] = 0
                    front.append(i)
            
            fronts = []
            k = 0
            while front:
                fronts.append(front)
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                front = next_front
                k += 1
            
            # Add remaining (should not happen with proper implementation)
            remaining = [i for i in range(n) if rank[i] == 0 and i not in [idx for f in fronts for idx in f]]
            if remaining:
                fronts.append(remaining)
            
            return fronts

        # Initialize archive with initial population
        for i in range(pop_size):
            add_to_archive(population_x[i].copy(), population_f[i].copy())

        def create_child(pop_x, pop_f):
            """Create a child using SBX crossover and polynomial mutation"""
            # Binary tournament selection
            def tournament_select(pop_f, k=2):
                indices = rng.choice(len(pop_f), size=k, replace=False)
                best = indices[0]
                for idx in indices[1:]:
                    # Compare using rank + crowding distance
                    # For simplicity, use dominance check
                    if dominates(pop_f[idx], pop_f[best]):
                        best = idx
                    elif not dominates(pop_f[best], pop_f[idx]):
                        # Non-dominated, pick randomly
                        if rng.random() < 0.5:
                            best = idx
                return best
            
            p1_idx = tournament_select(pop_f)
            p2_idx = tournament_select(pop_f)
            p1 = pop_x[p1_idx]
            p2 = pop_x[p2_idx]
            
            # Simulated Binary Crossover
            eta_c = 20.0
            child1 = np.zeros(dim)
            child2 = np.zeros(dim)
            
            for j in range(dim):
                if rng.random() < 0.5:
                    # No crossover
                    child1[j] = p1[j]
                    child2[j] = p2[j]
                else:
                    if abs(p1[j] - p2[j]) < 1e-14:
                        child1[j] = p1[j]
                        child2[j] = p2[j]
                    else:
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1.0 / (eta_c + 1))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
                        
                        x1 = 0.5 * ((1 + beta) * p1[j] + (1 - beta) * p2[j])
                        x2 = 0.5 * ((1 - beta) * p1[j] + (1 + beta) * p2[j])
                        
                        # Clip to bounds
                        x1 = np.clip(x1, lo[j], hi[j])
                        x2 = np.clip(x2, lo[j], hi[j])
                        
                        child1[j] = x1
                        child2[j] = x2
            
            # Polynomial Mutation
            eta_m = 20.0
            for j in range(dim):
                if rng.random() < 1.0 / dim:
                    u = rng.random()
                    if u < 0.5:
                        delta_q = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                    else:
                        delta_q = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                    child1[j] += delta_q * (hi[j] - lo[j])
                    child1[j] = np.clip(child1[j], lo[j], hi[j])
                
                if rng.random() < 1.0 / dim:
                    u = rng.random()
                    if u < 0.5:
                        delta_q = (2 * u) ** (1.0 / (eta_m + 1)) - 1
                    else:
                        delta_q = 1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1))
                    child2[j] += delta_q * (hi[j] - lo[j])
                    child2[j] = np.clip(child2[j], lo[j], hi[j])
            
            return child1, child2

        # Main NSGA-II loop
        while used < budget - 1:
            # Generate offspring
            offspring_x = []
            offspring_f = []
            
            while len(offspring_x) < pop_size and used < budget:
                c1, c2 = create_child(population_x, population_f)
                f1 = problem(c1)
                f2 = problem(c2)
                used += 2
                
                offspring_x.append(c1)
                offspring_f.append(f1)
                
                if used < budget:
                    offspring_x.append(c2)
                    offspring_f.append(f2)
            
            offspring_x = np.array(offspring_x[:pop_size])
            offspring_f = np.array(offspring_f[:pop_size])
            
            # Combine parent and offspring
            combined_x = np.vstack([population_x, offspring_x])
            combined_f = np.vstack([population_f, offspring_f])
            
            # Non-dominated sorting
            fronts = non_dominated_sort(combined_f)
            
            # Select next generation
            next_x = []
            next_f = []
            for front in fronts:
                if len(next_x) + len(front) <= pop_size:
                    for idx in front:
                        next_x.append(combined_x[idx])
                        next_f.append(combined_f[idx])
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - len(next_x)
                    front_f = np.array([combined_f[idx] for idx in front])
                    cd = crowding_distance(front_f)
                    # Sort by crowding distance descending
                    order = np.argsort(-cd)
                    for i in range(remaining):
                        idx = front[order[i]]
                        next_x.append(combined_x[idx])
                        next_f.append(combined_f[idx])
                    break
            
            population_x = np.array(next_x)
            population_f = np.array(next_f)
            
            # Update archive
            for i in range(pop_size):
                add_to_archive(population_x[i].copy(), population_f[i].copy())

        # Final archive pruning if needed
        if len(archive_f) > 100:
            # Keep only top 100 by crowding distance
            f_arr = np.array(archive_f)
            cd = crowding_distance(f_arr)
            order = np.argsort(-cd)
            top_indices = order[:100]
            archive_x = [archive_x[i] for i in top_indices]
            archive_f = [archive_f[i] for i in top_indices]

        return population_x

# EVOLVE-BLOCK-END
