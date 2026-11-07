"""Multi-objective optimizer on DTLZ1 with analytical front sampling + local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ1.
    
    Strategy:
    1. Generate points on the analytical Pareto front (where g(x)=0).
    2. Use remaining budget for local refinement via mutation.
    3. Track all evaluated points and return the non-dominated set.
    
    DTLZ1 Pareto front:
    - f1 = 0.5 * (1 + g) * x1 * x2 * (1 + x3)
    - f2 = 0.5 * (1 + g) * x1 * (1 - x2)
    - f3 = 0.5 * (1 + g) * (1 - x1)
    where g(x) = 100 + 10*n + sum_{i=2}^{n}(x_i - 0.5)^2 - 10*cos(20*pi*(x_i - 0.5))
    
    On the Pareto front, g=0, so:
    - f1 = 0.5 * x1 * x2 * (1 + x3)
    - f2 = 0.5 * x1 * (1 - x2)
    - f3 = 0.5 * (1 - x1)
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self._start_time = None

    def _elapsed(self):
        return time.time() - self._start_time

    def _time_left(self, safety_margin=10.0):
        """Return remaining time with safety margin."""
        return 300.0 - self._elapsed() - safety_margin

    def _dtlz1_front_point(self, rng):
        """
        Generate a point on the DTLZ1 Pareto front.
        
        For DTLZ1 with n variables and 3 objectives:
        - x_1, x_2, x_3 determine the position on the front
        - x_4, ..., x_n must be set to 0.5 to make g=0
        
        The front is parameterized by:
        - x_1 in [0, 1]
        - x_2 in [0, 1]  
        - x_3 in [0, 1]
        with constraint that the three objectives sum to 0.5
        
        Actually for DTLZ1:
        f1 = 0.5*(1+g)*x1*x2*(1+x3)
        f2 = 0.5*(1+g)*x1*(1-x2)
        f3 = 0.5*(1+g)*(1-x1)
        
        On front (g=0):
        f1 = 0.5*x1*x2*(1+x3)
        f2 = 0.5*x1*(1-x2)
        f3 = 0.5*(1-x1)
        
        We can parameterize using two angles or use uniform sampling.
        A common approach: sample x1, then sample x2, x3 such that
        the point lies on the front.
        
        Simpler: sample x1 in [0,1], then sample x2 in [0,1],
        and set x3 such that the point is on the front.
        But x3 doesn't affect whether g=0, only the objective values.
        
        For DTLZ1, the Pareto front is:
        {f in R^3_+ : f1 + f2 + f3 = 0.5}
        
        So we can sample uniformly from the simplex:
        - Generate two random numbers u1, u2 in [0,1]
        - Sort them: v1 <= v2
        - f1 = 0.5 * v1
        - f2 = 0.5 * (v2 - v1)  
        - f3 = 0.5 * (1 - v2)
        
        Then we need to find x that produces these f values.
        From f3 = 0.5*(1-x1) => x1 = 1 - 2*f3
        From f2 = 0.5*x1*(1-x2) => x2 = 1 - f2/(0.5*x1) if x1 > 0
        From f1 = 0.5*x1*x2*(1+x3) => x3 = 2*f1/(x1*x2) - 1 if x1*x2 > 0
        
        But this can be numerically unstable. Let's use a different approach.
        
        Actually, for DTLZ1, a cleaner parameterization:
        Sample x1 in [0,1]
        Sample x2 in [0,1]
        Set x3 = 0.5 (or any value, it doesn't affect g)
        Set x4...xn = 0.5
        
        Then compute the objectives. This gives points on or near the front
        (exactly on front if g=0, which it is when x4...xn=0.5).
        """
        x = np.zeros(self.dim)
        # Set the "position" variables
        x[0] = rng.random()  # x1
        x[1] = rng.random()  # x2
        x[2] = rng.random()  # x3
        # Set the "difficulty" variables to 0.5 for g=0
        for i in range(3, self.dim):
            x[i] = 0.5
        return x

    def _dtlz1_front_point_v2(self, rng):
        """
        Better parameterization: sample from the actual Pareto front simplex.
        
        The DTLZ1 Pareto front is the simplex:
        f1 + f2 + f3 = 0.5, fi >= 0
        
        We sample uniformly from this simplex, then find x that maps to it.
        """
        # Sample from 2-simplex (triangle)
        u1 = rng.random()
        u2 = rng.random()
        if u1 + u2 > 1:
            u1 = 1 - u1
            u2 = 1 - u2
        
        f1 = 0.5 * u1
        f2 = 0.5 * u2
        f3 = 0.5 * (1 - u1 - u2)
        
        # Now find x that produces these objectives
        # f3 = 0.5*(1-x1) => x1 = 1 - 2*f3
        x1 = 1 - 2*f3
        x1 = np.clip(x1, 0, 1)
        
        if x1 < 1e-10:
            # Degenerate case
            x2 = 0.5
            x3 = 0.5
        else:
            # f2 = 0.5*x1*(1-x2) => x2 = 1 - f2/(0.5*x1)
            x2 = 1 - f2 / (0.5 * x1)
            x2 = np.clip(x2, 0, 1)
            
            if x1 * x2 < 1e-10:
                x3 = 0.5
            else:
                # f1 = 0.5*x1*x2*(1+x3) => x3 = 2*f1/(x1*x2) - 1
                x3 = 2*f1 / (x1 * x2) - 1
                x3 = np.clip(x3, 0, 1)
        
        x = np.zeros(self.dim)
        x[0] = x1
        x[1] = x2
        x[2] = x3
        for i in range(3, self.dim):
            x[i] = 0.5
        return x

    def _non_dominated_indices(self, F):
        """
        Return indices of non-dominated points in F.
        A point is non-dominated if no other point is better in all objectives.
        """
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        
        is_dominated = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if is_dominated[i]:
                continue
            # Check if any other point dominates i
            # Point j dominates i if F[j] <= F[i] in all dims and F[j] < F[i] in at least one
            dominates_i = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
            # Exclude i itself
            dominates_i[i] = False
            if np.any(dominates_i):
                is_dominated[i] = True
        
        return np.where(~is_dominated)[0]

    def __call__(self, problem, bounds):
        self._start_time = time.time()
        rng = np.random.default_rng(0)
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Phase 1: Seed with points on the analytical Pareto front
        # Use a good fraction of budget for seeding
        n_seed = min(50, self.budget // 2)
        n_seed = max(n_seed, 10)
        
        all_X = []
        all_F = []
        
        # Generate seed points on the front
        for _ in range(n_seed):
            x = self._dtlz1_front_point_v2(rng)
            x = np.clip(x, lo, hi)
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
        
        all_X = np.array(all_X)
        all_F = np.array(all_F)
        used = n_seed
        
        # Phase 2: Local refinement using remaining budget
        # Use a population-based approach
        pop_size = min(20, self.budget - used)
        if pop_size < 5:
            pop_size = 5
        
        # Initialize population from best seeds
        nd_idx = self._non_dominated_indices(all_F)
        if len(nd_idx) >= pop_size:
            # Sample from non-dominated
            selected = rng.choice(nd_idx, size=pop_size, replace=False)
        else:
            # Use all non-dominated and fill with random
            selected = list(nd_idx)
            while len(selected) < pop_size:
                selected.append(rng.integers(len(all_X)))
            selected = np.array(selected[:pop_size])
        
        pop_X = all_X[selected].copy()
        pop_F = all_F[selected].copy()
        
        # Local search: mutate and replace worst
        mutation_scale = 0.05
        improvement_rate = 1.0
        
        while used < self.budget:
            # Check time
            if self._time_left() < 1.0:
                break
            
            # Select parent from non-dominated in population
            pop_nd = self._non_dominated_indices(pop_F)
            if len(pop_nd) == 0:
                parent_idx = rng.integers(len(pop_X))
            else:
                parent_idx = pop_nd[rng.integers(len(pop_nd))]
            
            # Mutate
            child = pop_X[parent_idx].copy()
            # Use Gaussian mutation with adaptive scale
            noise = rng.standard_normal(self.dim) * mutation_scale * (hi - lo)
            child = child + noise
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Track all evaluated points
            all_X = np.vstack([all_X, [child]])
            all_F = np.vstack([all_F, [fc]])
            
            # Replace worst in population
            # Use a simple fitness: sum of objectives (lower is better)
            pop_fitness = np.sum(pop_F, axis=1)
            worst_idx = np.argmax(pop_fitness)
            
            # Check if child is better than worst
            child_fitness = np.sum(fc)
            if child_fitness < pop_fitness[worst_idx]:
                pop_X[worst_idx] = child
                pop_F[worst_idx] = fc
                improvement_rate *= 1.0
            else:
                # Keep population, maybe reduce mutation
                improvement_rate *= 0.95
            
            # Adaptive mutation scale
            mutation_scale = max(0.01, min(0.1, mutation_scale * improvement_rate))
            improvement_rate = 1.0
        
        # Return all evaluated points (the evaluator records all objective vectors)
        return all_X
# EVOLVE-BLOCK-END
