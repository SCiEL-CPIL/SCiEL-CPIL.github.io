import numpy as np
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        
        # Population size
        N = 20
        
        # Initialize population using Sobol sequence for better spread
        # Sobol sequence requires power of 2 for dimensionality, but we can just take the first N points
        # Or use random if Sobol is too complex for arbitrary N.
        # Let's use random for simplicity but with a good seed, or a simple Latin Hypercube.
        # Latin Hypercube is good for 1D per variable.
        points = []
        for i in range(N):
            # Simple random initialization is often sufficient if mutation is strong
            x = lo + (hi - lo) * rng.random(n)
            points.append(x)
        X = np.array(points)
        
        F = np.array([problem(x) for x in X])
        used = N
        
        # DE-style or EA-style update
        # Let's use a simple EA with SBX and PM
        while used < self.budget:
            # Generate offspring
            Y = np.empty_like(X)
            F_Y = np.empty_like(F)
            
            # We generate offspring one by one or in batches?
            # To respect budget exactly, we generate 1 offspring, evaluate, and update selection logic?
            # Standard EA generates N offspring. If budget < N, we stop.
            # Here budget is 200, N=20. So we can do full generations.
            
            batch_size = min(N, self.budget - used)
            if batch_size == 0: break
            
            for i in range(batch_size):
                # Crossover
                if rng.random() < 1.0:
                    p1_idx = rng.integers(N)
                    p2_idx = rng.integers(N)
                    while p2_idx == p1_idx:
                        p2_idx = rng.integers(N)
                    
                    p1 = X[p1_idx]
                    p2 = X[p2_idx]
                    
                    # SBX
                    x1 = p1.copy()
                    x2 = p2.copy()
                    
                    for j in range(n):
                        if rng.random() < 0.5:
                            # Apply SBX to this gene
                            if abs(x1[j] - x2[j]) < 1e-14:
                                y1 = x1[j]
                                y2 = x2[j]
                            else:
                                u = rng.random()
                                beta = 0.0
                                if u <= 1.0 / 21.0:
                                    beta = (2.0 * u) ** (1.0 / 21.0)
                                else:
                                    beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / 21.0)
                                
                                y1 = 0.5 * ((1.0 + beta) * x1[j] + (1.0 - beta) * x2[j])
                                y2 = 0.5 * ((1.0 + 1.0/beta) * x1[j] + (1.0 - 1.0/beta) * x2[j])
                                
                                if y1 > hi[j]:
                                    y1 = hi[j]
                                    beta = (hi[j] - x1[j]) / (x1[j] - x2[j]) if x1[j] != x2[j] else 1.0
                                    y2 = 0.5 * ((1.0 + 1.0/beta) * x1[j] + (1.0 - 1.0/beta) * x2[j])
                                elif y1 < lo[j]:
                                    y1 = lo[j]
                                    beta = (lo[j] - x1[j]) / (x1[j] - x2[j]) if x1[j] != x2[j] else 1.0
                                    y2 = 0.5 * ((1.0 + 1.0/beta) * x1[j] + (1.0 - 1.0/beta) * x2[j])
                                    
                                if y2 > hi[j]:
                                    y2 = hi[j]
                                    beta = (hi[j] - x2[j]) / (x2[j] - x1[j]) if x2[j] != x1[j] else 1.0
                                    y1 = 0.5 * ((1.0 + 1.0/beta) * x2[j] + (1.0 - 1.0/beta) * x1[j])
                                elif y2 < lo[j]:
                                    y2 = lo[j]
                                    beta = (lo[j] - x2[j]) / (x2[j] - x1[j]) if x2[j] != x1[j] else 1.0
                                    y1 = 0.5 * ((1.0 + 1.0/beta) * x2[j] + (1.0 - 1.0/beta) * x1[j])
                                
                                x1[j] = y1
                                x2[j] = y2
                        else:
                            x1[j] = x1[j]
                            x2[j] = x2[j]
                            
                    # We only need one child
                    child = x1
                else:
                    child = X[rng.integers(N)].copy()
                    
                # Mutation
                for j in range(n):
                    if rng.random() < 1.0 / n:
                        u = rng.random()
                        delta = 0.0
                        if u < 0.5:
                            delta = (2.0 * u) ** (1.0 / 21.0) - 1.0
                        else:
                            delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / 21.0)
                        
                        new_val = child[j] + delta * (hi[j] - lo[j])
                        child[j] = np.clip(new_val, lo[j], hi[j])
                        
                Y[i] = child
                F_Y[i] = problem(child)
                used += 1
                
            # Combine and Select
            # We have X (N) and Y (batch_size). 
            # If batch_size == N, we do standard NSGA-II selection.
            # If batch_size < N (last partial batch), we just add them and keep the best N from the combined set?
            # Actually, if we only generate partial, we can just append and truncate.
            
            if batch_size == N:
                # Standard NSGA-II Selection
                R = np.vstack([X, Y])
                F_R = np.vstack([F, F_Y])
                
                # Non-dominated sorting
                ranks = np.zeros(len(R), dtype=int)
                front_idx = 0
                remaining = np.arange(len(R))
                fronts = []
                
                while len(remaining) > 0:
                    front = []
                    for i in remaining:
                        dominated = False
                        for j in remaining:
                            if i == j: continue
                            # Check if i is dominated by j
                            if np.all(F_R[j] <= F_R[i]) and np.any(F_R[j] < F_R[i]):
                                dominated = True
                                break
                        if not dominated:
                            front.append(i)
                    fronts.append(front)
                    for i in front:
                        ranks[i] = front_idx
                    remaining = [i for i in remaining if i not in front]
                    front_idx += 1
                    
                # Crowding distance
                dist = np.zeros(len(R))
                for f in fronts:
                    if len(f) == 0: continue
                    if len(f) == 1:
                        for i in f: dist[i] = np.inf
                        continue
                    for k in range(m):
                        sorted_f = sorted(f, key=lambda i: F_R[i][k])
                        dist[sorted_f[0]] = np.inf
                        dist[sorted_f[-1]] = np.inf
                        min_val = F_R[sorted_f[0]][k]
                        max_val = F_R[sorted_f[-1]][k]
                        range_val = max_val - min_val
                        if range_val == 0: continue
                        for i in range(1, len(sorted_f)-1):
                            idx = sorted_f[i]
                            dist[idx] += (F_R[sorted_f[i+1]][k] - F_R[sorted_f[i-1]][k]) / range_val
                
                # Select N best
                # Sort by rank, then by distance (descending)
                order = np.lexsort((dist * -1, ranks))
                X = R[order[:N]]
                F = F_R[order[:N]]
            else:
                # Partial batch: just add and keep best N?
                # Simpler: just append and if size > N, do a quick non-dominated sort and keep best N.
                R = np.vstack([X, Y])
                F_R = np.vstack([F, F_Y])
                # Quick selection: keep non-dominated, if too many, random drop or hypervolume?
                # For simplicity and speed, let's just do a full NSGA-II selection logic here too, 
                # but only on the combined set, keeping N.
                # Re-use the logic above but ensure we keep exactly N.
                
                ranks = np.zeros(len(R), dtype=int)
                front_idx = 0
                remaining = np.arange(len(R))
                fronts = []
                
                while len(remaining) > 0:
                    front = []
                    for i in remaining:
                        dominated = False
                        for j in remaining:
                            if i == j: continue
                            if np.all(F_R[j] <= F_R[i]) and np.any(F_R[j] < F_R[i]):
                                dominated = True
                                break
                        if not dominated:
                            front.append(i)
                    fronts.append(front)
                    for i in front:
                        ranks[i] = front_idx
                    remaining = [i for i in remaining if i not in front]
                    front_idx += 1
                    
                dist = np.zeros(len(R))
                for f in fronts:
                    if len(f) == 0: continue
                    if len(f) == 1:
                        for i in f: dist[i] = np.inf
                        continue
                    for k in range(m):
                        sorted_f = sorted(f, key=lambda i: F_R[i][k])
                        dist[sorted_f[0]] = np.inf
                        dist[sorted_f[-1]] = np.inf
                        min_val = F_R[sorted_f[0]][k]
                        max_val = F_R[sorted_f[-1]][k]
                        range_val = max_val - min_val
                        if range_val == 0: continue
                        for i in range(1, len(sorted_f)-1):
                            idx = sorted_f[i]
                            dist[idx] += (F_R[sorted_f[i+1]][k] - F_R[sorted_f[i-1]][k]) / range_val
                            
                order = np.lexsort((dist * -1, ranks))
                X = R[order[:N]]
                F = F_R[order[:N]]
                
        return X
