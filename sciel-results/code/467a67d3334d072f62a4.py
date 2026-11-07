import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4 using structured search + NSGA-II."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT4 structure: x1 in [0,10], x2..x10 in [-5,5]
        # f1 = 1 + 8*sin^2(pi*x1), f2 = g(x)*h(f1,g(x))
        # g(x) = 1 + 10*(n-1) + sum((xi-0.5)^2 - 10*cos(20*pi*(xi-0.5))) for i=2..n
        # h(u,v) = 1 - sqrt(u/v) if v>0 else 0

        # Strategy: 
        # 1. Sample x1 near integers (0,1,2,...,10) to get low f1
        # 2. For each good x1, optimize x2..x10 to minimize g
        # 3. Use evolutionary search on the full space as backup

        # Phase 1: Structured initialization exploiting ZDT4 structure
        # x1 candidates: near integers 0..10
        x1_candidates = np.arange(0, 11, 0.5)  # 22 candidates
        # Add some random x1 values
        rng = np.random.default_rng(42)
        x1_random = rng.uniform(0, 10, 10)
        x1_pool = np.concatenate([x1_candidates, x1_random])

        # For x2..x10, try values near 0.5 (where cos term is maximized, minimizing g)
        # g is minimized when (xi-0.5)^2 is small and cos(20*pi*(xi-0.5)) is max (=1)
        # cos(20*pi*(xi-0.5)) = 1 when 20*pi*(xi-0.5) = 2*k*pi => xi-0.5 = k/10
        # So xi near 0.5, 1.5, -0.5, etc. But bounds are [-5,5]
        # Best is xi=0.5 for the cos term, and (xi-0.5)^2=0

        # Generate initial population
        pop_size = min(40, budget // 5)
        X = np.zeros((pop_size, dim))
        
        # First half: structured x1, optimized x2..x10 near 0.5
        n_struct = pop_size // 2
        for i in range(n_struct):
            X[i, 0] = x1_pool[i % len(x1_pool)]
            # x2..x10 near 0.5 with small perturbation
            X[i, 1:] = 0.5 + rng.uniform(-0.1, 0.1, dim - 1)
        
        # Second half: random
        for i in range(n_struct, pop_size):
            X[i, 0] = rng.uniform(0, 10)
            X[i, 1:] = rng.uniform(-5, 5, dim - 1)

        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        used = pop_size

        # NSGA-II style evolution
        def fast_non_dominated_sort(F):
            """Return list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = set of points dominated by i
            n_dom = np.zeros(n, dtype=int)  # number of points dominating i
            fronts = [[]]
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    fronts[0].append(i)
            
            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            next_front.append(q)
                i += 1
                fronts.append(next_front)
            fronts.pop()  # remove empty last
            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for given indices."""
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            n = len(indices)
            dist = np.zeros(n)
            F_sub = F[indices]
            
            for m in range(F_sub.shape[1]):
                sorted_idx = np.argsort(F_sub[:, m])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                f_range = F_sub[sorted_idx[-1], m] - F_sub[sorted_idx[0], m]
                if f_range > 0:
                    for k in range(1, n - 1):
                        dist[sorted_idx[k]] += (F_sub[sorted_idx[k + 1], m] - F_sub[sorted_idx[k - 1], m]) / f_range
            
            return dist

        def crossover(x1, x2):
            """SBX crossover."""
            if rng.random() < 0.9:
                beta = rng.random()
                if beta <= 0.5:
                    alpha = (2 * beta) ** (1 / 21)
                else:
                    alpha = (1 / (2 * (1 - beta))) ** (1 / 21)
                c1 = 0.5 * ((1 + alpha) * x1 + (1 - alpha) * x2)
                c2 = 0.5 * ((1 - alpha) * x1 + (1 + alpha) * x2)
                return c1, c2
            return x1.copy(), x2.copy()

        def mutate(x, rate=0.1, sigma=0.2):
            """Gaussian mutation."""
            mask = rng.random(dim) < rate
            x_new = x.copy()
            x_new[mask] += sigma * rng.standard_normal(np.sum(mask))
            return np.clip(x_new, lo, hi)

        # Evolution loop
        while used < budget:
            if time.time() > deadline:
                break
            
            # Non-dominated sort
            fronts = fast_non_dominated_sort(F)
            
            # Select parents using tournament on fronts + crowding
            children = []
            while len(children) < pop_size:
                # Tournament selection
                idx1 = rng.integers(len(F))
                idx2 = rng.integers(len(F))
                
                # Find front ranks
                rank1 = rank2 = 0
                for fi, front in enumerate(fronts):
                    if idx1 in front:
                        rank1 = fi
                    if idx2 in front:
                        rank2 = fi
                    if rank1 > 0 or rank2 > 0:
                        break
                
                if rank1 < rank2:
                    p1 = idx1
                elif rank2 < rank1:
                    p1 = idx2
                else:
                    # Same front, use crowding
                    front_idx = [fi for fi, f in enumerate(fronts) if idx1 in f][0]
                    front_members = fronts[front_idx]
                    cd = crowding_distance(F, front_members)
                    pos1 = front_members.index(idx1)
                    pos2 = front_members.index(idx2)
                    p1 = idx1 if cd[pos1] > cd[pos2] else idx2
                
                # Second parent
                idx3 = rng.integers(len(F))
                idx4 = rng.integers(len(F))
                rank3 = rank4 = 0
                for fi, front in enumerate(fronts):
                    if idx3 in front:
                        rank3 = fi
                    if idx4 in front:
                        rank4 = fi
                    if rank3 > 0 or rank4 > 0:
                        break
                
                if rank3 < rank4:
                    p2 = idx3
                elif rank4 < rank3:
                    p2 = idx4
                else:
                    front_idx = [fi for fi, f in enumerate(fronts) if idx3 in f][0]
                    front_members = fronts[front_idx]
                    cd = crowding_distance(F, front_members)
                    pos3 = front_members.index(idx3)
                    pos4 = front_members.index(idx4)
                    p2 = idx3 if cd[pos3] > cd[pos4] else idx4
                
                # Crossover
                c1, c2 = crossover(X[p1], X[p2])
                # Mutation
                c1 = mutate(c1)
                c2 = mutate(c2)
                children.append(c1)
                children.append(c2)
                if len(children) >= pop_size:
                    break
            
            children = children[:pop_size]
            
            # Evaluate children
            for child in children:
                if used >= budget or time.time() > deadline:
                    break
                fc = problem(child)
                used += 1
                
                # Replace worst point in population
                # Simple: replace the point with highest sum of objectives
                worst = np.argmax(np.sum(F, axis=1))
                X[worst] = child
                F[worst] = fc

        return X
