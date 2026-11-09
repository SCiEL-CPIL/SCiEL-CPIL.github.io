"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Multi-objective optimizer using efficient NSGA-II with Sobol initialization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """Efficient vectorized non-dominated sort. Returns list of fronts (lists of indices)."""
        n = len(F)
        if n == 0:
            return []
        
        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=np.int32)
        
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    nd[p] += 1
        
        front = [i for i in range(n) if nd[i] == 0]
        if not front:
            front = [0]
        
        fronts = [front]
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

    def _crowding_distance(self, F, indices):
        """Calculate crowding distance for a front."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        dist = np.zeros(n)
        F_sub = F[indices]
        
        for m in range(self.n_obj):
            order = np.argsort(F_sub[:, m])
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            
            min_val = F_sub[order[0], m]
            max_val = F_sub[order[-1], m]
            if max_val == min_val:
                continue
                
            for k in range(1, n - 1):
                dist[order[k]] += (F_sub[order[k + 1], m] - F_sub[order[k - 1], m]) / (max_val - min_val)
        return dist

    def _sobol_sample(self, n, dim, lo, hi, seed):
        """Generate n Sobol quasi-random points in [lo, hi]^dim."""
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        samples = sampler.random(n)
        return lo + (hi - lo) * samples

    def _sbx_crossover(self, x1, x2, lo, hi, eta=20.0):
        """Simulated binary crossover with correct beta computation."""
        c1 = x1.copy()
        c2 = x2.copy()
        for j in range(self.dim):
            if np.random.random() < 0.5:
                if abs(x1[j] - x2[j]) > 1e-10:
                    if np.random.random() < 0.5:
                        if x1[j] < x2[j]:
                            beta1 = 1 + 2 * (x1[j] - lo[j]) / (x2[j] - x1[j])
                            beta2 = 1 + 2 * (hi[j] - x2[j]) / (x2[j] - x1[j])
                        else:
                            beta1 = 1 + 2 * (x2[j] - lo[j]) / (x1[j] - x2[j])
                            beta2 = 1 + 2 * (hi[j] - x1[j]) / (x1[j] - x2[j])
                    else:
                        if x1[j] < x2[j]:
                            beta1 = 1 + 2 * (hi[j] - x2[j]) / (x2[j] - x1[j])
                            beta2 = 1 + 2 * (x1[j] - lo[j]) / (x2[j] - x1[j])
                        else:
                            beta1 = 1 + 2 * (hi[j] - x1[j]) / (x1[j] - x2[j])
                            beta2 = 1 + 2 * (x2[j] - lo[j]) / (x1[j] - x2[j])
                    
                    alpha1 = 2 + 1 / (beta1 + 1)
                    alpha2 = 2 + 1 / (beta2 + 1)
                    
                    u1 = np.random.random()
                    u2 = np.random.random()
                    
                    if u1 <= 1 / alpha1:
                        c1[j] = (x1[j] + x2[j]) / 2 + (x1[j] - x2[j]) * (2 * u1) ** (1 / alpha1) / 2
                    elif u1 >= 1 - 1 / alpha1:
                        c1[j] = (x1[j] + x2[j]) / 2 - (x2[j] - x1[j]) * (2 * (1 - u1)) ** (1 / alpha1) / 2
                    else:
                        c1[j] = x1[j]
                    
                    if u2 <= 1 / alpha2:
                        c2[j] = (x1[j] + x2[j]) / 2 + (x2[j] - x1[j]) * (2 * u2) ** (1 / alpha2) / 2
                    elif u2 >= 1 - 1 / alpha2:
                        c2[j] = (x1[j] + x2[j]) / 2 - (x1[j] - x2[j]) * (2 * (1 - u2)) ** (1 / alpha2) / 2
                    else:
                        c2[j] = x2[j]
        
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Use Sobol quasi-random for initial population for better coverage
        pop_size = min(50, self.budget // 4)
        X = self._sobol_sample(pop_size, self.dim, lo, hi, seed=42)
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track all evaluated points for final hypervolume
        all_X = [X.copy()]
        all_F = [F.copy()]
        evaluated_set = set()
        for x in X:
            evaluated_set.add(tuple(np.round(x, 12)))
        
        gen = 0
        total_gens = max(1, (self.budget - pop_size) // pop_size)
        
        while used < self.budget:
            if time.time() > deadline - 5.0:
                break
            
            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(F)
            
            # Create offspring
            offspring = []
            
            # Adaptive mutation rate: start large, decrease over generations
            progress = min(1.0, gen / total_gens)
            mutation_scale = span * (0.25 * (1 - progress) + 0.03 * progress)
            
            while len(offspring) < pop_size and used + len(offspring) < self.budget:
                p1 = self._tournament_select(F, fronts)
                p2 = self._tournament_select(F, fronts)
                
                # Simulated binary crossover (SBX)
                if np.random.random() < 0.9:
                    c1, c2 = self._sbx_crossover(X[p1], X[p2], lo, hi, eta=20.0)
                else:
                    c1 = X[p1].copy()
                    c2 = X[p2].copy()
                
                # Gaussian mutation
                for c in [c1, c2]:
                    if np.random.random() < 0.9:
                        c = c + mutation_scale * np.random.standard_normal(self.dim)
                        c = np.clip(c, lo, hi)
                    
                    # Deduplication check
                    key = tuple(np.round(c, 12))
                    if key not in evaluated_set:
                        evaluated_set.add(key)
                        offspring.append(c)
            
            if len(offspring) == 0:
                break
            
            offspring = np.array(offspring[:min(pop_size, self.budget - used)])
            
            F_off = np.array([problem(x) for x in offspring])
            used += len(offspring)
            
            all_X.append(offspring.copy())
            all_F.append(F_off.copy())
            
            # Combine parent + offspring
            X_comb = np.vstack([X, offspring])
            F_comb = np.vstack([F, F_off])
            
            # Select next generation
            fronts_comb = self._fast_non_dominated_sort(F_comb)
            next_gen = []
            for front in fronts_comb:
                if len(next_gen) + len(front) <= pop_size:
                    next_gen.extend(front)
                else:
                    need = pop_size - len(next_gen)
                    dist = self._crowding_distance(F_comb, front)
                    order = np.argsort(dist)[::-1]
                    next_gen.extend([front[i] for i in order[:need]])
                    break
            
            X = X_comb[next_gen]
            F = F_comb[next_gen]
            gen += 1
        
        # Final local refinement around non-dominated points
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline - 3.0:
            # Get non-dominated points from all evaluations
            all_X_flat = np.vstack(all_X)
            all_F_flat = np.vstack(all_F)
            fronts_all = self._fast_non_dominated_sort(all_F_flat)
            if fronts_all:
                nd_indices = fronts_all[0]
                nd_X = all_X_flat[nd_indices]
                
                # Multi-scale local perturbations around best points
                scales = [0.1, 0.05, 0.02, 0.01]
                local_off = []
                per_scale = max(1, remaining // (len(nd_X) * len(scales)))
                
                for base in nd_X:
                    for scale in scales:
                        for _ in range(per_scale):
                            if time.time() > deadline - 3.0:
                                break
                            pert = base + scale * span * np.random.standard_normal(self.dim)
                            pert = np.clip(pert, lo, hi)
                            key = tuple(np.round(pert, 12))
                            if key not in evaluated_set:
                                evaluated_set.add(key)
                                local_off.append(pert)
                
                if local_off:
                    local_off = np.array(local_off[:remaining])
                    F_local = np.array([problem(x) for x in local_off])
                    used += len(local_off)
                    all_X.append(local_off.copy())
                    all_F.append(F_local.copy())
        
        return np.vstack(all_X)

    def _tournament_select(self, F, fronts):
        """Tournament selection based on front rank."""
        rank = np.zeros(len(F), dtype=int)
        for i, front in enumerate(fronts):
            for idx in front:
                rank[idx] = i
        
        t1 = np.random.randint(len(F))
        t2 = np.random.randint(len(F))
        if rank[t1] < rank[t2]:
            return t1
        elif rank[t2] < rank[t1]:
            return t2
        else:
            return np.random.choice([t1, t2])
# EVOLVE-BLOCK-END
