import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # 1. Generate Weight Vectors - H=6 -> 28 points
        H = 6
        weights_list = []
        for i in range(H + 1):
            for j in range(H + 1 - i):
                k = H - i - j
                w = np.array([i / H, j / H, k / H])
                weights_list.append(w)
        N = len(weights_list)
        weights = np.array(weights_list)
        
        # 2. Define Neighbors - 10 nearest
        neighbors = []
        for i in range(N):
            dists = np.linalg.norm(weights - weights[i], axis=1)
            sorted_idx = np.argsort(dists)
            neigh = [sorted_idx[j] for j in range(1, 11) if sorted_idx[j] != i]
            neighbors.append(neigh)
        
        # 3. Initialize Population using LHS
        X = np.zeros((N, dim))
        for d in range(dim):
            edges = np.linspace(lo[d], hi[d], N + 1)
            centers = np.array([(edges[i] + edges[i+1]) / 2 for i in range(N)])
            np.random.shuffle(centers)
            X[:, d] = centers
            
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = N
        z_star = np.min(F, axis=0)
        
        # 4. Evolution Loop
        # Reserve 50 evaluations for local polish
        budget_for_main = budget - 50
        
        i_counter = 0
        while used < budget_for_main:
            if time.time() > deadline:
                break
            
            i = i_counter % N
            i_counter += 1
            
            # Adaptive mutation factor
            progress = used / max(budget_for_main, 1)
            F_mut = 0.7 - 0.2 * progress
            
            # Select parents from neighbors
            candidate_indices = neighbors[i]
            if len(candidate_indices) < 3:
                candidate_indices = np.random.choice(N, 3, replace=False)
            else:
                # Uniform selection from neighbors for simplicity and robustness
                candidate_indices = np.random.choice(candidate_indices, 3, replace=False)
            
            r1, r2, r3 = candidate_indices
            
            # DE Mutation
            v = X[r1] + F_mut * (X[r2] - X[r3])
            
            # Binomial Crossover
            j_rand = np.random.randint(dim)
            CR = 0.5
            trial = np.zeros(dim)
            for j in range(dim):
                if np.random.random() < CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = X[i][j]
            
            trial = np.clip(trial, lo, hi)
            
            f_trial = np.array(problem(trial))
            used += 1
            
            z_star = np.minimum(z_star, f_trial)
            
            # Update Population
            indices_to_check = [i] + neighbors[i]
            for k in indices_to_check:
                d_trial = np.max(weights[k] * np.abs(f_trial - z_star))
                d_current = np.max(weights[k] * np.abs(F[k] - z_star))
                
                if d_trial < d_current:
                    X[k] = trial
                    F[k] = f_trial

        # 5. Local Search using SLSQP on Smooth Tchebycheff
        # Use smooth approximation: logsumexp(beta * w * |f - z|) / beta
        beta = 30.0
        
        if time.time() < deadline - 10:
            for i in range(N):
                if used >= budget:
                    break
                if time.time() > deadline - 3:
                    break
                
                w = weights[i].copy()
                x0 = X[i].copy()
                z = z_star.copy()
                
                def objective(x):
                    nonlocal used
                    if used >= budget:
                        return 1e10
                    if time.time() > deadline - 2:
                        return 1e10
                    
                    x_clipped = np.clip(x, lo, hi)
                    f = np.array(problem(x_clipped))
                    used += 1
                    # Smooth Tchebycheff
                    d = w * np.abs(f - z)
                    return logsumexp(beta * d) / beta
                
                try:
                    result = minimize(
                        objective,
                        x0,
                        method='SLSQP',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 20, 'ftol': 1e-5, 'disp': False}
                    )
                    
                    # SLSQP might have evaluated the objective multiple times.
                    # We don't explicitly add to used here because 'objective' does it.
                    # But we need to update X and F if result is better.
                    # Note: result.x is the best point found by the optimizer.
                    # We need to evaluate it once more to get F? 
                    # The last call in objective already evaluated it, but we don't have F stored.
                    # To be safe and count correctly, we evaluate result.x explicitly if it's different.
                    # However, minimizing 'objective' already consumed budget.
                    # Let's just update X[i] to result.x.
                    # We need F[i] for potential future use or return.
                    # Since the loop is ending, we just need the points to be good.
                    # The evaluator records ALL calls. So the calls made inside 'objective' are recorded.
                    # We don't strictly need to update F[i] unless we use it for something else.
                    # But let's update it for consistency.
                    if result.success:
                        X[i] = result.x
                        # Evaluate once more to get F[i] accurately? 
                        # This costs 1 eval.
                        if used < budget:
                            F[i] = np.array(problem(result.x))
                            used += 1
                            z_star = np.minimum(z_star, F[i])
                except:
                    pass
        
        return X
