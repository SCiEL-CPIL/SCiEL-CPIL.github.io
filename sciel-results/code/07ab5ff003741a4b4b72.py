import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # 1. Generate Weight Vectors on Simplex - structured grid
        # For 3 objectives, use H=5 -> 21 points
        H = 5
        weights_list = []
        for i in range(H + 1):
            for j in range(H + 1 - i):
                k = H - i - j
                w = np.array([i / H, j / H, k / H])
                weights_list.append(w)
        N = len(weights_list)
        weights = np.array(weights_list)
        
        # 2. Define Neighbors - 6 nearest for better exploration
        neighbors = []
        for i in range(N):
            dists = np.linalg.norm(weights - weights[i], axis=1)
            sorted_idx = np.argsort(dists)
            neigh = [sorted_idx[j] for j in range(1, 7) if sorted_idx[j] != i]
            neighbors.append(neigh)
        
        # 3. Initialize Population using LHS with better spread
        X = np.zeros((N, dim))
        for d in range(dim):
            # Use stratified sampling with jitter for better coverage
            edges = np.linspace(lo[d], hi[d], N + 1)
            centers = np.array([(edges[i] + edges[i+1]) / 2 for i in range(N)])
            # Add small random jitter to avoid exact grid alignment
            jitter = (hi[d] - lo[d]) / N * 0.1 * np.random.randn(N)
            centers = np.clip(centers + jitter, lo[d], hi[d])
            np.random.shuffle(centers)
            X[:, d] = centers
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = N
        
        # Initialize Ideal Point
        z_star = np.min(F, axis=0)
        
        # 4. Evolution Loop
        budget_for_main = budget - 50  # Reserve 50 for Nelder-Mead polish
        
        i_counter = 0
        gen = 0
        
        while used < budget_for_main:
            if time.time() > deadline:
                break
            
            i = i_counter % N
            i_counter += 1
            gen += 1
            
            # Adaptive mutation factor: start moderate, decrease over time
            progress = used / max(budget_for_main, 1)
            F_mut = 0.7 - 0.2 * progress  # 0.7 -> 0.5
            
            # Select parents from neighbors
            candidate_indices = neighbors[i]
            if len(candidate_indices) < 3:
                candidate_indices = np.random.choice(N, 3, replace=False)
            else:
                dists_to_i = np.array([np.linalg.norm(weights[c] - weights[i]) for c in candidate_indices])
                probs = 1.0 / (dists_to_i + 1e-6)
                probs = probs / probs.sum()
                candidate_indices = np.random.choice(candidate_indices, 3, replace=False, p=probs)
            
            r1, r2, r3 = candidate_indices
            
            # DE Mutation
            v = X[r1] + F_mut * (X[r2] - X[r3])
            
            # Binomial Crossover
            j_rand = np.random.randint(dim)
            CR = 0.6  # Slightly higher crossover rate for more exploration
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
            
            # Update Population (MOEA/D with Tchebycheff)
            indices_to_check = [i] + neighbors[i]
            
            for k in indices_to_check:
                d_trial = np.max(weights[k] * np.abs(f_trial - z_star))
                d_current = np.max(weights[k] * np.abs(F[k] - z_star))
                
                if d_trial < d_current:
                    X[k] = trial
                    F[k] = f_trial
        
        # 5. Nelder-Mead Local Search for each weight vector
        # Use smooth weighted-sum scalarization for better convergence
        if time.time() < deadline - 15:
            for i in range(N):
                if used >= budget:
                    break
                if time.time() > deadline - 5:
                    break
                
                w = weights[i]
                x0 = X[i].copy()
                
                # Define smooth scalarized objective (weighted sum with ideal point normalization)
                def objective(x):
                    nonlocal used
                    if used >= budget:
                        return 1e10
                    if time.time() > deadline - 3:
                        return 1e10
                    
                    x_clipped = np.clip(x, lo, hi)
                    f = np.array(problem(x_clipped))
                    used += 1
                    z = z_star
                    # Weighted sum: sum(w_j * f_j) - smooth approximation
                    return np.dot(w, f)
                
                # Run Nelder-Mead with tighter tolerances
                try:
                    result = minimize(
                        objective,
                        x0,
                        method='Nelder-Mead',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 50, 'xatol': 1e-5, 'fatol': 1e-5, 'initial_simplex': np.array([x0 + 0.05*(hi-lo)*np.random.randn(dim) for _ in range(dim+1)])}
                    )
                    
                    # Only update if we actually found improvement
                    f_new = np.array(problem(result.x))
                    used += 1
                    if np.all(f_new <= F[i] + 1e-10) and np.any(f_new < F[i] - 1e-10):
                        X[i] = result.x
                        F[i] = f_new
                        z_star = np.minimum(z_star, F[i])
                except:
                    pass
        
        return X
