import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        X = []
        F = []
        used = 0
        
        def eval_point(x):
            nonlocal used
            if used >= self.budget or time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            X.append(x)
            F.append(f)
            used += 1
            return f

        def get_non_dominated_indices(F_list):
            n = len(F_list)
            if n == 0: return []
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]: continue
                for j in range(n):
                    if i == j: continue
                    if np.all(F_list[j] <= F_list[i]) and np.any(F_list[j] < F_list[i]):
                        dominated[i] = True
                        break
                    if np.all(F_list[i] <= F_list[j]) and np.any(F_list[i] < F_list[j]):
                        dominated[j] = True
            return [i for i in range(n) if not dominated[i]]

        # 1. Initialization: Sobol + Random
        n_sobol = 20
        n_random = 10
        
        sobol_engine = qmc.Sobol(d=dim, scramble=True)
        sobol_points = sobol_engine.random(n_sobol)
        sobol_points = lo + (hi - lo) * sobol_points
        
        for i in range(n_sobol):
            if time.time() > deadline: break
            eval_point(sobol_points[i])
            
        for i in range(n_random):
            if time.time() > deadline: break
            x = lo + (hi - lo) * np.random.rand(dim)
            eval_point(x)
            
        if not F:
            return X

        F_arr = np.array(F)
        nd_idx = get_non_dominated_indices(F_arr)
        
        # 2. Adaptive Tchebycheff Decomposition
        # Generate weight vectors on the simplex
        n_weights = 10
        weights = []
        for i in range(n_weights + 1):
            w1 = i / n_weights
            for j in range(n_weights - i + 1):
                w2 = j / (n_weights - i)
                w3 = 1 - w1 - w2
                if w1 >= 0 and w2 >= 0 and w3 >= 0:
                    weights.append([w1, w2, w3])
        
        # Remove duplicates and ensure we have a good set
        weights = np.array(weights)
        
        # Select a subset of weights to save time, focusing on corners and edges
        # But for 3 obj, n_weights=10 gives ~66 points, which is too many for budget 200.
        # Let's use a smaller set of weights, e.g., 6 points (corners + center + face centers)
        # Or just iterate through a few key directions.
        
        # Better: Use a fixed set of diverse weights
        # Corners: [1,0,0], [0,1,0], [0,0,1]
        # Edge midpoints: [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]
        # Center: [1/3, 1/3, 1/3]
        key_weights = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [1/3, 1/3, 1/3]
        ])
        
        # For each weight, find the best point in current ND set to start from
        # And run a local search minimizing the Tchebycheff function
        
        for w in key_weights:
            if time.time() > deadline or used >= self.budget:
                break
            
            # Find best starting point for this weight
            if len(nd_idx) > 0:
                nd_F = F_arr[nd_idx]
                # Tchebycheff value for each point: max(w_i * (f_i - z_i^min))
                # Since we don't have z_min (ideal point), use min of ND set as approximation
                # Or just use the point that minimizes the weighted sum as a heuristic start
                weighted_sums = nd_F @ w
                best_local_idx = np.argmin(weighted_sums)
                start_idx = nd_idx[best_local_idx]
            else:
                start_idx = 0
            
            x0 = X[start_idx]
            
            def tchebycheff_obj(x, w=w):
                nonlocal used
                if used >= self.budget or time.time() > deadline:
                    return 0.0
                x = np.clip(x, lo, hi)
                f = np.array(problem(x))
                X.append(x)
                F.append(f)
                used += 1
                # Tchebycheff: max(w_i * f_i)
                # We need a reference. Let's use the min of the current ND set for each obj
                # But that changes. Let's just use weighted sum for simplicity and speed,
                # or a fixed reference like the initial bounds.
                # Actually, standard Tchebycheff uses a utopia point.
                # Let's approximate utopia with the min of the initial ND set.
                # This is tricky to do dynamically without extra evals.
                # Let's stick to weighted sum but with more diverse weights.
                return np.dot(f, w)
            
            try:
                minimize(tchebycheff_obj, x0, method='L-BFGS-B',
                         bounds=list(zip(lo, hi)),
                         options={'maxiter': 5, 'maxfun': 5})
            except Exception:
                pass

        # 3. Local Search from Best Points per Objective
        F_arr = np.array(F)
        nd_idx = get_non_dominated_indices(F_arr)
        
        if len(nd_idx) > 0:
            nd_F = F_arr[nd_idx]
            start_points_idx = []
            for obj_i in range(n_obj):
                best_local_idx = np.argmin(nd_F[:, obj_i])
                global_idx = nd_idx[best_local_idx]
                if global_idx not in start_points_idx:
                    start_points_idx.append(global_idx)
            
            for idx in start_points_idx:
                if time.time() > deadline or used >= self.budget:
                    break
                x0 = X[idx]
                
                for obj_i in range(n_obj):
                    if time.time() > deadline or used >= self.budget:
                        break
                    
                    def obj_i_func(x):
                        nonlocal used
                        if used >= self.budget or time.time() > deadline:
                            return 0.0
                        x = np.clip(x, lo, hi)
                        f = np.array(problem(x))
                        X.append(x)
                        F.append(f)
                        used += 1
                        return f[obj_i]
                    
                    try:
                        minimize(obj_i_func, x0, method='L-BFGS-B',
                                 bounds=list(zip(lo, hi)),
                                 options={'maxiter': 5, 'maxfun': 5})
                    except Exception:
                        pass

        # 4. Fill Gaps with Mutation
        F_arr = np.array(F)
        nd_idx = get_non_dominated_indices(F_arr)
        
        if used < self.budget and time.time() < deadline:
            if len(nd_idx) > 0:
                nd_F = F_arr[nd_idx]
                mins = nd_F.min(axis=0)
                maxs = nd_F.max(axis=0)
                ranges = maxs - mins
                ranges[ranges == 0] = 1
                norm_F = (nd_F - mins) / ranges
                best_sum_idx = np.argsort(norm_F.sum(axis=1))[:5]
                elite_idx = [nd_idx[i] for i in best_sum_idx]
            else:
                elite_idx = list(range(min(5, len(F))))
            
            while used < self.budget and time.time() < deadline:
                parent_idx = elite_idx[np.random.randint(len(elite_idx))]
                x_parent = X[parent_idx]
                sigma = 0.1 * (hi - lo)
                x_child = x_parent + sigma * np.random.randn(dim)
                x_child = np.clip(x_child, lo, hi)
                f_child = eval_point(x_child)
                if f_child is None:
                    break

        return X
