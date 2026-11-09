"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


class Optimizer:
    """Differential evolution based multi-objective optimizer for DTLZ2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref_point = np.array([2.5] * self.n_obj)
        
        # We will collect all evaluated points
        all_points = []
        all_objectives = []
        
        # Use differential evolution with a custom objective that returns a scalar
        # But since we need multi-objective, we'll use a different approach:
        # We'll run DE on a surrogate that tries to balance objectives, but more importantly,
        # we'll collect all intermediate evaluations.
        
        # Actually, let's use a simpler approach: use scipy's differential_evolution
        # but we need to track all evaluations. Since DE doesn't easily expose intermediate
        # evaluations in a clean way, let's implement our own DE-like optimizer.
        
        # Better approach: Implement NSGA-II or a simple multi-objective DE
        # Given the budget of 200 and 12 dimensions, let's use a population-based approach
        
        # Let's use a simple multi-objective differential evolution
        pop_size = 30
        max_iter = (self.budget - pop_size) // pop_size if self.budget > pop_size else 1
        
        # Initialize population
        rng = np.random.default_rng(42)
        population = np.random.uniform(lo, hi, (pop_size, self.dim))
        objectives = np.array([problem(x) for x in population])
        
        all_points.extend(list(population))
        all_objectives.extend(list(objectives))
        eval_count = pop_size
        
        def dominates(a, b):
            """Check if a dominates b (a is better in all objectives and at least one strictly better)"""
            return np.all(a <= b) and np.any(a < b)
        
        def non_dominated_sort(objs):
            """Return indices of non-dominated points"""
            n = len(objs)
            dominated = [False] * n
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if dominates(objs[j], objs[i]):
                        dominated[i] = True
                        break
            return [i for i in range(n) if not dominated[i]]
        
        def crowding_distance(objs, indices):
            """Calculate crowding distance for the given indices"""
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}
            
            dist = {i: 0.0 for i in indices}
            for m in range(self.n_obj):
                sorted_indices = sorted(indices, key=lambda i: objs[i][m])
                dist[sorted_indices[0]] = float('inf')
                dist[sorted_indices[-1]] = float('inf')
                obj_range = objs[sorted_indices[-1]][m] - objs[sorted_indices[0]][m]
                if obj_range == 0:
                    continue
                for i in range(1, len(sorted_indices) - 1):
                    idx = sorted_indices[i]
                    dist[idx] += (objs[sorted_indices[i+1]][m] - objs[sorted_indices[i-1]][m]) / obj_range
            return dist
        
        while eval_count < self.budget:
            # Find non-dominated front
            nondom_idx = non_dominated_sort(objectives)
            
            # Select parents from non-dominated front
            if len(nondom_idx) < 3:
                # If too few non-dominated, select from all
                parent_indices = list(range(len(population)))
            else:
                parent_indices = nondom_idx
            
            # Create child using differential evolution mutation
            p1, p2, p3 = rng.choice(parent_indices, size=3, replace=False)
            mutation_factor = 0.5
            crossover_rate = 0.9
            
            # DE/rand/1 mutation
            mutant = population[p1] + mutation_factor * (population[p2] - population[p3])
            
            # Binomial crossover
            crossover_point = rng.integers(self.dim)
            child = population[p1].copy()
            for i in range(self.dim):
                if rng.random() < crossover_rate or i == crossover_point:
                    child[i] = mutant[i]
            
            child = np.clip(child, lo, hi)
            child_obj = np.array(problem(child))
            eval_count += 1
            
            all_points.append(child)
            all_objectives.append(child_obj)
            
            # Replace worst individual in population
            # Use sum of objectives as a simple fitness
            fitness = np.sum(objectives, axis=1)
            worst_idx = np.argmax(fitness)
            population[worst_idx] = child
            objectives[worst_idx] = child_obj
        
        # Now compute hypervolume indicator
        # Filter points that are strictly better than reference point in all objectives
        valid_mask = np.all(np.array(all_objectives) < ref_point, axis=1)
        valid_objectives = np.array(all_objectives)[valid_mask]
        
        if len(valid_objectives) == 0:
            return np.array([])
        
        return valid_objectives
# EVOLVE-BLOCK-END
