# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dominates(self, a, b):
        return np.all(a <= b) and np.any(a < b)

    def _fast_non_dominated_sort(self, F):
        """Efficient non-dominated sorting. Returns list of fronts (lists of indices)."""
        n = len(F)
        if n == 0:
            return []
        
        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=np.int32)
        
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if self._dominates(F[p], F[q]):
                    S[p].append(q)
                elif self._dominates(F[q], F[p]):
                    nd[p] += 1
        
        front = [i for i in range(n) if nd[i] == 0]
        fronts = []
        if not front:
            front = [0]
        fronts.append(front)
        
        while front:
            next_front = []
            for p in front:
                for q in S[p]:
                    nd[q] -= 1
                    if nd[q] == 0:
                        next_front.append(q)
            if next_front:
                fronts.append(next_front)
            front = next_front
        return fronts

    def _generate_sobol(self, n, dim, lo, hi):
        """Generate n Sobol points mapped to [lo, hi]."""
        sobol = qmc.Sobol(d=dim, scramble=True)
        n_gen = 1
        while n_gen < n:
            n_gen *= 2
        X = sobol.random(n_gen)[:n]
        return lo + X * (hi - lo)

    def _normalize(self, F):
        """Normalize objectives to [0, 1] range for scalarization."""
        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        F_range = F_max - F_min
        F_range[F_range == 0] = 1.0
        return (F - F_min) / F_range

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        all_X = []
        all_F = []
        used = 0
        
        def eval_point(x):
            """Evaluate a single point and add to archive."""
            nonlocal used
            if time.time() > deadline - 10:
                return None
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
            used += 1
            return f
        
        # Phase 1: Sobol Initial Sampling (40% of budget for better coverage)
        n_init = max(10, int(0.4 * self.budget))
        X_sobol = self._generate_sobol(n_init, self.dim, lo, hi)
        
        for x in X_sobol:
            if time.time() > deadline - 10:
                break
            eval_point(x)
        
        if len(all_F) == 0:
            return all_X
        
        all_F = np.array(all_F)
        all_X = np.array(all_X)
        
        # Phase 2: DE with adaptive parameters and more aggressive exploration
        if used < self.budget:
            pop_size = max(15, min(30, int(0.25 * self.budget)))
            
            # Initialize population from non-dominated points
            fronts = self._fast_non_dominated_sort(all_F)
            if fronts and len(fronts[0]) > 0:
                nd_idx = fronts[0]
                if len(nd_idx) < pop_size:
                    others = [i for i in range(len(all_F)) if i not in nd_idx]
                    if others:
                        n_add = pop_size - len(nd_idx)
                        add_idx = np.random.choice(others, size=min(n_add, len(others)), replace=False)
                        nd_idx = np.concatenate([nd_idx, add_idx])
                np.random.shuffle(nd_idx)
                pop_X = all_X[nd_idx[:pop_size]].copy()
                pop_F = all_F[nd_idx[:pop_size]].copy()
            else:
                pop_X = all_X[:pop_size].copy()
                pop_F = all_F[:pop_size].copy()
            
            # Pad if needed
            if len(pop_X) < pop_size:
                n_pad = pop_size - len(pop_X)
                X_pad = lo + np.random.rand(n_pad, self.dim) * (hi - lo)
                for x in X_pad:
                    eval_point(x)
                pop_X = np.vstack([pop_X, X_pad])
                pop_F = np.vstack([pop_F, np.array(all_F[-n_pad:])])
            
            # Adaptive DE parameters
            F_DE_start = 0.9
            F_DE_end = 0.5
            CR = 0.8
            batch_size = max(2, min(8, (self.budget - used) // 4))
            
            gen_count = 0
            while used < self.budget and time.time() < deadline - 15:
                # Adapt F_DE over generations
                progress = min(1.0, gen_count / max(1, (self.budget - used) // batch_size))
                F_DE = F_DE_start + (F_DE_end - F_DE_start) * progress
                
                n_batch = min(batch_size, self.budget - used)
                if n_batch <= 0:
                    break
                
                # For each trial, select parents and generate
                trials_X = []
                targets = []
                for _ in range(n_batch):
                    r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                    v = pop_X[r1] + F_DE * (pop_X[r2] - pop_X[r3])
                    v = np.clip(v, lo, hi)
                    
                    j_rand = np.random.randint(self.dim)
                    trial = pop_X[r1].copy()
                    for j in range(self.dim):
                        if np.random.rand() < CR or j == j_rand:
                            trial[j] = v[j]
                    trials_X.append(trial)
                    targets.append(r1)
                
                trials_X = np.array(trials_X)
                
                # Evaluate batch
                for i, trial in enumerate(trials_X):
                    if time.time() > deadline - 10:
                        break
                    f_trial = problem(trial)
                    used += 1
                    r1 = targets[i]
                    
                    # Selection: replace if dominates target
                    if self._dominates(f_trial, pop_F[r1]):
                        pop_X[r1] = trial
                        pop_F[r1] = f_trial
                    
                    all_X = np.vstack([all_X, trial])
                    all_F = np.vstack([all_F, f_trial])
                
                gen_count += 1
                
                # Update population periodically with non-dominated points
                if gen_count % 3 == 0 and len(all_F) > pop_size:
                    fronts = self._fast_non_dominated_sort(all_F)
                    if fronts and len(fronts[0]) >= pop_size:
                        nd_idx = fronts[0]
                        indices = np.random.choice(nd_idx, pop_size, replace=False)
                        pop_X = all_X[indices].copy()
                        pop_F = all_F[indices].copy()
        
        # Phase 3: Local refinement using scalarization with weighted sum
        if used < self.budget and time.time() < deadline - 20:
            fronts = self._fast_non_dominated_sort(all_F)
            if fronts:
                nd_idx = fronts[0]
                # Take top non-dominated points for refinement
                n_refine = min(8, len(nd_idx), (self.budget - used) // 2)
                if n_refine > 0:
                    refine_idx = np.random.choice(nd_idx, n_refine, replace=False)
                    
                    for idx in refine_idx:
                        if used >= self.budget or time.time() > deadline - 15:
                            break
                        center = all_X[idx]
                        
                        # Use weighted sum scalarization to find local optima
                        # Generate diverse weight vectors
                        n_weights = min(4, self.budget - used)
                        for _ in range(n_weights):
                            if used >= self.budget or time.time() > deadline - 15:
                                break
                            
                            # Random weight vector
                            weights = np.random.dirichlet(np.ones(self.n_obj))
                            
                            def scalarized(x):
                                f = problem(x)
                                return np.dot(weights, f)
                            
                            # L-BFGS-B optimization from the center
                            try:
                                result = minimize(
                                    scalarized,
                                    center,
                                    method='L-BFGS-B',
                                    bounds=list(zip(lo, hi)),
                                    options={'maxiter': 50, 'ftol': 1e-10}
                                )
                                if result.success and used < self.budget:
                                    x_opt = result.x
                                    f_opt = problem(x_opt)
                                    used += 1
                                    all_X = np.vstack([all_X, x_opt])
                                    all_F = np.vstack([all_F, f_opt])
                            except:
                                pass
        
        # Phase 4: Final fine-grained local search around best points
        if used < self.budget and time.time() < deadline - 5:
            # Get the non-dominated set
            fronts = self._fast_non_dominated_sort(all_F)
            if fronts:
                nd_idx = fronts[0]
                n_final = min(5, len(nd_idx), self.budget - used)
                if n_final > 0:
                    final_idx = np.random.choice(nd_idx, n_final, replace=False)
                    sigma = 0.02  # very small perturbation
                    
                    for idx in final_idx:
                        if used >= self.budget or time.time() > deadline - 5:
                            break
                        center = all_X[idx]
                        n_local = min(2, self.budget - used)
                        for _ in range(n_local):
                            if used >= self.budget or time.time() > deadline - 5:
                                break
                            perturbation = np.random.randn(self.dim) * sigma
                            x_local = np.clip(center + perturbation, lo, hi)
                            f_local = problem(x_local)
                            used += 1
                            all_X = np.vstack([all_X, x_local])
                            all_F = np.vstack([all_F, f_local])
        
        return all_X
# EVOLVE-BLOCK-END
