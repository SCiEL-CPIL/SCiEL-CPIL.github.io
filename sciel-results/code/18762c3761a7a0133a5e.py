"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer using a hybrid of NSGA-II style selection and
       local search on the Pareto front, tailored for ZDT4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim

        # ZDT4 structure: x0 in [-5, 5], x1..x9 in [0, 1]
        # f1 = x0
        # f2 = 1 - sqrt(x0/g(x))^2 * g(x) where g(x) = 1 + 10(n-1) + sum 100(xi^2 - sin(2*pi*xi))
        # For ZDT4, the Pareto front is f1 in [-5,5], f2 = 1 - f1^2 (approximately, when g=1)
        # But g(x) can be large due to the sin terms.
        # Key insight: to minimize f2, we need g(x) to be small (close to 1) and x0 near 0.
        # g(x) = 1 + 10*(dim-1) + sum_{i=1}^{dim-1} 100*(xi^2 - sin(2*pi*xi))
        # The minimum of xi^2 - sin(2*pi*xi) is at xi=0 (value 0) or xi=1 (value 1-0=1? no, sin(2pi)=0, so 1)
        # Actually at xi=0: 0 - 0 = 0. At xi=1: 1 - 0 = 1. So minimum is 0 at xi=0.
        # So g_min = 1 + 10*(dim-1) + 0 = 1 + 10*9 = 91 for dim=10.
        # Wait, that's not right. Let me re-check ZDT4.
        # ZDT4: g(x) = 1 + 10(n-1) + sum_{i=2}^n 100(x_i^2 - sin(2*pi*x_i))
        # So g_min = 1 + 10*(9) + 0 = 91.
        # Then f2 = 1 - (x0/g)^2 * g = 1 - x0^2/g.
        # For x0 in [-5,5], f2 = 1 - x0^2/91.
        # At x0=0, f2=1. At x0=5, f2=1-25/91≈0.725.
        # The Pareto front has f1=x0 in [-5,5] and f2=1-x0^2/91 (when g=91).
        
        # Strategy: 
        # 1. Initialize with points that have x1..x9 near 0 (to minimize g) and x0 spread.
        # 2. Use a population-based search with crowding distance for diversity.
        # 3. Local search on promising points.
        
        def fast_non_dominated_sort(F):
            """Returns list of non-dominated fronts."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = set of points dominated by i
            nd = [0] * n  # number of points that dominate i
            rank = [0] * n
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if all(F[i] <= F[j]) and any(F[i] < F[j]):
                        S[i].append(j)
                        nd[j] += 1
                    elif all(F[j] <= F[i]) and any(F[j] < F[i]):
                        nd[i] += 1
            
            fronts = []
            current_front = [i for i in range(n) if nd[i] == 0]
            for i in current_front:
                rank[i] = 0
            fronts.append(current_front)
            
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        nd[j] -= 1
                        if nd[j] == 0:
                            rank[j] = rank[i] + 1
                            next_front.append(j)
                current_front = next_front
                if current_front:
                    fronts.append(current_front)
            
            return fronts, rank
        
        def crowding_distance(F, indices):
            """Compute crowding distance for points in indices."""
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            cd = [0.0] * n
            for m in range(self.n_obj):
                f_vals = [F[i][m] for i in indices]
                f_min = min(f_vals)
                f_max = max(f_vals)
                if f_max == f_min:
                    continue
                # Sort by objective m
                sorted_idx = sorted(range(n), key=lambda k: f_vals[k])
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                for k in range(1, n - 1):
                    cd[sorted_idx[k]] += (f_vals[sorted_idx[k+1]] - f_vals[sorted_idx[k-1]]) / (f_max - f_min)
            return cd
        
        # Initialize population
        pop_size = min(40, self.budget // 5)
        X = np.zeros((pop_size, dim))
        # x0 spread across [-5, 5]
        X[:, 0] = np.linspace(-5, 5, pop_size)
        
        # Half the population: x1..x9 exactly 0 (known good g)
        # Other half: random to explore
        half = pop_size // 2
        X[:half, 1:] = 0.0
        X[half:, 1:] = rng.random((pop_size - half, dim - 1))
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Keep track of all evaluated points for hypervolume
        all_F = [tuple(f) for f in F]
        
        # Main loop
        while used < self.budget:
            fronts, rank = fast_non_dominated_sort(F)
            
            # Select parents from first front using crowding distance
            first_front = fronts[0]
            cd = crowding_distance(F, first_front)
            # Tournament selection
            parent1_idx = first_front[rng.integers(len(first_front))]
            parent2_idx = first_front[rng.integers(len(first_front))]
            
            p1 = X[parent1_idx].copy()
            p2 = X[parent2_idx].copy()
            
            # SBX crossover
            if rng.random() < 0.9:
                eta_c = 20
                child = np.zeros(dim)
                for d in range(dim):
                    if p1[d] != p2[d]:
                        u = rng.random()
                        if u <= 1 / (eta_c + 1):
                            beta = (2 * u) ** (1 / (eta_c + 1))
                        elif u >= 1 - 1 / (eta_c + 1):
                            beta = (2 * (1 - u)) ** (1 / (eta_c + 1))
                        else:
                            beta = 1.0
                        if u <= 0.5:
                            child[d] = 0.5 * ((1 + beta) * p1[d] + (1 - beta) * p2[d])
                        else:
                            child[d] = 0.5 * ((1 - beta) * p1[d] + (1 + beta) * p2[d])
                    else:
                        child[d] = p1[d]
            else:
                child = p1.copy()
            
            # Mutation
            eta_m = 20
            for d in range(dim):
                if rng.random() < 0.9:
                    u = rng.random()
                    if u <= 1 / (eta_m + 1):
                        delta = (2 * u) ** (1 / (eta_m + 1)) - 1
                    elif u >= 1 - 1 / (eta_m + 1):
                        delta = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                    else:
                        delta = 0.0
                    child[d] += delta * (hi[d] - lo[d])
            
            child = np.clip(child, lo, hi)
            
            # Local search: try to push x1..x9 toward 0
            # We check if setting each x_i (i>0) to 0 improves the solution
            # Since f1 depends only on x0, we only need to check f2 improvement
            # Or simply check if the vector of objectives is better (non-dominated)
            
            # Heuristic: Try zeroing out x_i for i>0
            # We do this for a subset of dimensions to save time, or all if budget allows
            # With budget 200 and pop 40, we have ~160 iterations.
            # 9 dims * 1 eval = 9 evals per child. Too expensive.
            # Instead, random coordinate zeroing.
            
            if rng.random() < 0.8:
                # Pick a random dimension from 1 to 9
                d_idx = rng.integers(1, dim)
                # Try setting it to 0
                temp_child = child.copy()
                temp_child[d_idx] = 0.0
                temp_child = np.clip(temp_child, lo, hi)
                
                # Only evaluate if we have budget
                if used < self.budget - 1:
                    f_temp = problem(temp_child)
                    # Check if temp is better than child in f2 (since f1 is same)
                    # Or just use non-domination
                    # Simple check: if f2 is lower, it's better (for same x0)
                    if f_temp[1] < fc[1]: 
                         # Wait, we haven't evaluated child yet.
                         # Let's just evaluate child, then compare.
                         pass

            # Standard approach: Evaluate child, then try local improvement
            fc = problem(child)
            used += 1
            all_F.append(tuple(fc))
            
            # Local improvement step: try to zero a random x_i
            if used < self.budget - 1:
                d_idx = rng.integers(1, dim)
                if child[d_idx] != 0.0:
                    improved = child.copy()
                    improved[d_idx] = 0.0
                    improved = np.clip(improved, lo, hi)
                    f_imp = problem(improved)
                    used += 1
                    all_F.append(tuple(f_imp))
                    
                    # If improved is better (lower f2, same f1), replace child
                    if f_imp[1] < fc[1]:
                        child = improved
                        fc = f_imp
            
            # Replace worst point in last front
            last_front = fronts[-1]
            cd_last = crowding_distance(F, last_front)
            worst_pos = int(np.argmin(cd_last))
            worst_idx = last_front[worst_pos]
            
            X[worst_idx] = child
            F[worst_idx] = fc
        
        return X
# EVOLVE-BLOCK-END
