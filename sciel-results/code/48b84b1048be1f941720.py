"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ7-specific optimizer exploiting known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ7 structure:
        # f1 = x1, f2 = x2
        # f3 = (1 + g(x_{3..22})) * h(x_{1..2}, g)
        # g = sum(x_{3..22}^2) (or similar, depends on exact definition)
        # Pareto front: g=0, so x_{3..22}=0, and h=1, so f3=1
        # Front is 2D: f1 in [0,1], f2 in [0,1], f3=1
        
        # Strategy: 
        # 1. Use a portion of budget to find points with g≈0
        # 2. Spread those points across f1-f2 plane
        # 3. Use remaining budget for refinement
        
        n_evals = 0
        all_X = []
        all_F = []
        
        def evaluate(x):
            nonlocal n_evals
            f = problem(x)
            n_evals += 1
            return f
        
        # Phase 1: Find points on/near Pareto front (g≈0)
        # g depends on x_{3..22}. For DTLZ7, typically g = sum(x_{M+1..n}^2) or similar.
        # To minimize g, set x_{3..22} close to 0.
        
        # Initialize with points biased toward g=0
        # x1, x2 random in [0,1]
        # x3..x22 close to 0
        
        n_init = 50
        X_init = np.zeros((n_init, self.dim))
        X_init[:, 0] = np.linspace(0, 1, n_init)  # spread f1
        X_init[:, 1] = np.linspace(0, 1, n_init)  # spread f2
        # Add some randomness
        X_init[:, 0] += np.random.uniform(-0.1, 0.1, n_init)
        X_init[:, 1] += np.random.uniform(-0.1, 0.1, n_init)
        X_init[:, 2:] = np.random.uniform(0, 0.05, (n_init, self.dim - 2))  # small values for g≈0
        X_init = np.clip(X_init, lo, hi)
        
        for i in range(n_init):
            if n_evals >= self.budget:
                break
            F_i = evaluate(X_init[i])
            all_X.append(X_init[i])
            all_F.append(F_i)
        
        # Phase 2: Evolutionary refinement
        # Keep a population of the best points
        pop_size = 30
        
        # Sort by hypervolume contribution (approximate: non-dominated + diversity)
        # For simplicity, keep non-dominated points and fill with best dominated
        
        def get_nondominated(F):
            """Return indices of non-dominated points."""
            F = np.array(F)
            n = len(F)
            nondom = []
            for i in range(n):
                is_dom = False
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dom = True
                        break
                if not is_dom:
                    nondom.append(i)
            return nondom
        
        # Build initial population from best points
        if len(all_F) > 0:
            F_arr = np.array(all_F)
            nondom_idx = get_nondominated(F_arr)
            if len(nondom_idx) > pop_size:
                # Sample pop_size from non-dominated
                pop_idx = np.random.choice(nondom_idx, size=pop_size, replace=False)
            else:
                pop_idx = nondom_idx
                if len(pop_idx) < pop_size:
                    remaining = [i for i in range(len(F_arr)) if i not in nondom_idx]
                    extra = np.random.choice(remaining, size=min(pop_size - len(pop_idx), len(remaining)), replace=False)
                    pop_idx = np.concatenate([pop_idx, extra])
            
            X_pop = np.array([all_X[i] for i in pop_idx])
            F_pop = np.array([all_F[i] for i in pop_idx])
        else:
            X_pop = X_init[:pop_size]
            F_pop = np.array([evaluate(x) for x in X_pop])
        
        # Evolutionary loop
        while n_evals < self.budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            n_offspring = min(pop_size, self.budget - n_evals)
            
            X_off = np.zeros((n_offspring, self.dim))
            F_off = np.zeros((n_offspring, self.n_obj))
            
            for i in range(n_offspring):
                if n_evals >= self.budget:
                    break
                
                # Select two parents
                p1 = np.random.randint(len(X_pop))
                p2 = np.random.randint(len(X_pop))
                
                # Crossover: blend
                alpha = np.random.uniform(0.2, 0.8)
                child = alpha * X_pop[p1] + (1 - alpha) * X_pop[p2]
                
                # Mutation: Gaussian, with larger sigma for g-variables (x_{3..22})
                # to explore g=0 region
                sigma = np.full(self.dim, 0.05)
                sigma[2:] = 0.1  # larger mutation for g-variables
                child += np.random.normal(0, sigma)
                
                # Bias x_{3..22} toward 0 (since g=0 is Pareto front)
                # Soft constraint: pull x_{3..22} toward 0
                child[2:] = child[2:] * 0.5  # shrink toward 0
                
                child = np.clip(child, lo, hi)
                
                X_off[i] = child
                F_off[i] = evaluate(child)
            
            # Combine and select
            X_comb = np.vstack([X_pop, X_off])
            F_comb = np.vstack([F_pop, F_off])
            
            # Select next population: non-dominated + diversity
            nondom_idx = get_nondominated(F_comb)
            if len(nondom_idx) >= pop_size:
                # Use crowding-like selection: spread in f1-f2 plane
                # For DTLZ7, front is 2D in f1-f2
                # Select points that maximize spread
                selected = []
                remaining = list(nondom_idx)
                
                # Always keep extremes
                f1_vals = F_comb[remaining, 0]
                f2_vals = F_comb[remaining, 1]
                
                # Keep min/max f1, min/max f2
                idx_min_f1 = remaining[np.argmin(f1_vals)]
                idx_max_f1 = remaining[np.argmax(f1_vals)]
                idx_min_f2 = remaining[np.argmin(f2_vals)]
                idx_max_f2 = remaining[np.argmax(f2_vals)]
                
                for idx in [idx_min_f1, idx_max_f1, idx_min_f2, idx_max_f2]:
                    if idx not in selected:
                        selected.append(idx)
                
                # Fill rest with random from non-dominated
                remaining = [i for i in nondom_idx if i not in selected]
                if len(remaining) > 0:
                    needed = pop_size - len(selected)
                    extra = np.random.choice(remaining, size=min(needed, len(remaining)), replace=False)
                    selected.extend(extra.tolist())
                
                pop_idx = np.array(selected[:pop_size])
            else:
                # Not enough non-dominated, fill with best dominated
                dominated = [i for i in range(len(F_comb)) if i not in nondom_idx]
                # Sort dominated by sum of objectives (lower is better)
                if dominated:
                    obj_sums = np.sum(F_comb[dominated], axis=1)
                    sorted_dom = [dominated[i] for i in np.argsort(obj_sums)]
                    pop_idx = np.array(nondom_idx + sorted_dom[:pop_size - len(nondom_idx)])
                else:
                    pop_idx = np.array(nondom_idx)
            
            X_pop = X_comb[pop_idx]
            F_pop = F_comb[pop_idx]
            
            # Update all_X and all_F
            for i in range(len(pop_idx)):
                all_X.append(X_comb[pop_idx[i]])
                all_F.append(F_comb[pop_idx[i]])
        
        return all_X
# EVOLVE-BLOCK-END
