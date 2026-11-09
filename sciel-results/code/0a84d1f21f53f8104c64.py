import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
                start_time = time.time()
                deadline = start_time + 300 # Safety margin
                lo = np.array([b[0] for b in bounds])
                hi = np.array([b[1] for b in bounds])
                n_lhs = 80
                n_nm = 8
                nm_steps = 3
                
                # 1. LHS Sampling
                # Standard LHS: for each dimension, divide [0,1] into n_lhs intervals, pick one random point in each, permute for each dimension.
                indices = np.arange(n_lhs)
                lhs_points = np.zeros((n_lhs, self.dim))
                for d in range(self.dim):
                    # Divide into n_lhs strata
                    strata = np.linspace(0, 1, n_lhs + 1)
                    # Pick one random point in each stratum
                    points_in_strata = np.random.uniform(strata[:-1], strata[1:])
                    # Shuffle
                    np.random.shuffle(points_in_strata)
                    lhs_points[:, d] = points_in_strata
                
                # Scale to bounds
                lhs_points = lo + lhs_points * (hi - lo)
                
                # Evaluate LHS
                F_lhs = np.array([problem(x) for x in lhs_points])
                used = n_lhs
                
                # 2. Non-dominated selection
                # Simple non-dominated check
                nd_mask = np.zeros(n_lhs, dtype=bool)
                for i in range(n_lhs):
                    dominated = False
                    for j in range(n_lhs):
                        if i != j:
                            if np.all(F_lhs[j] <= F_lhs[i]) and np.any(F_lhs[j] < F_lhs[i]):
                                dominated = True
                                break
                    if not dominated:
                        nd_mask[i] = True
                
                nd_indices = np.where(nd_mask)[0]
                if len(nd_indices) < n_nm:
                    # If not enough non-dominated, take random from all
                    nd_indices = np.random.choice(n_lhs, n_nm, replace=False)
                else:
                    # Sample n_nm from non-dominated
                    nd_indices = np.random.choice(nd_indices, n_nm, replace=False)
                
                # 3. Nelder-Mead Refinement
                # Weights for scalarization
                weights = np.random.dirichlet(np.ones(self.n_obj), size=n_nm)
                
                for k in range(n_nm):
                    if used >= self.budget:
                        break
                    if time.time() > deadline - 10: # Safety margin
                        break
                    
                    x0 = lhs_points[nd_indices[k]]
                    w = weights[k]
                    
                    # Define scalar objective
                    def scalar_obj(x):
                        return np.dot(problem(x), w)
                    
                    # Nelder-Mead
                    # Initial simplex: x0 and x0 + delta * e_i
                    delta = 0.05 * (hi - lo)
                    simplex = [x0.copy()]
                    for i in range(self.dim):
                        xi = x0.copy()
                        xi[i] += delta[i]
                        simplex.append(xi)
                    
                    # Evaluate initial simplex
                    simplex_f = []
                    for x in simplex:
                        if used >= self.budget:
                            break
                        if time.time() > deadline - 5:
                            break
                        f = problem(x)
                        simplex_f.append(np.dot(f, w))
                        used += 1
                    
                    if len(simplex_f) < self.dim + 1:
                        continue # Not enough budget for full simplex
                    
                    # NM Steps
                    for step in range(nm_steps):
                        if used >= self.budget:
                            break
                        if time.time() > deadline - 5:
                            break
                        
                        # Sort simplex by objective value
                        order = np.argsort(simplex_f)
                        simplex = [simplex[i] for i in order]
                        simplex_f = [simplex_f[i] for i in order]
                        
                        x_worst = simplex[-1]
                        f_worst = simplex_f[-1]
                        
                        # Centroid of best n points
                        centroid = np.mean(simplex[:-1], axis=0)
                        
                        # Reflection
                        x_r = centroid + 1.0 * (centroid - x_worst)
                        x_r = np.clip(x_r, lo, hi)
                        f_r = np.dot(problem(x_r), w)
                        used += 1
                        
                        if f_r < simplex_f[0]: # Better than best
                            # Expansion
                            x_e = centroid + 2.0 * (centroid - x_worst)
                            x_e = np.clip(x_e, lo, hi)
                            f_e = np.dot(problem(x_e), w)
                            used += 1
                            if f_e < f_r:
                                simplex[-1] = x_e
                                simplex_f[-1] = f_e
                            else:
                                simplex[-1] = x_r
                                simplex_f[-1] = f_r
                        elif f_r < simplex_f[-2]: # Between best and second worst
                            simplex[-1] = x_r
                            simplex_f[-1] = f_r
                        else: # Contraction
                            # Inside contraction
                            x_c = centroid + 0.5 * (x_worst - centroid)
                            x_c = np.clip(x_c, lo, hi)
                            f_c = np.dot(problem(x_c), w)
                            used += 1
                            if f_c < f_worst:
                                simplex[-1] = x_c
                                simplex_f[-1] = f_c
                            else:
                                # Shrink
                                for i in range(1, len(simplex)):
                                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                    simplex_f[i] = np.dot(problem(simplex[i]), w)
                                    used += 1
                                    if used >= self.budget:
                                        break
                                if used >= self.budget:
                                    break
                        if used >= self.budget:
                            break
                return None
