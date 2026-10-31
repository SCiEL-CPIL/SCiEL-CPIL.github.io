"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Strategy: Differential Evolution (DE) global search.
    # This is a derivative-free, population-based stochastic global optimizer.
    # It is robust against local minima and does not require gradient information.
    # It explores the space by creating trial vectors from the population, which
    # helps escape local optima that gradient-based methods like L-BFGS-B might get stuck in.
    
    def get_sq_dists(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        return sq_dists[iu]

    def score_raw(pts):
        d_sq = get_sq_dists(pts)
        d_min = np.sqrt(d_sq.min())
        d_max = np.sqrt(d_sq.max())
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def normalize(pts):
        pts = pts - np.mean(pts, axis=0)
        scale = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
        if scale > 1e-9:
            pts = pts / scale
        return pts

    # Initialize population
    # We use a mix of random points and grid points to seed the population.
    pop_size = 50
    dim = N * 2
    
    # Create a diverse initial population
    population = np.zeros((pop_size, dim))
    
    # 1. Random points in a box
    for i in range(pop_size // 2):
        population[i] = np.random.uniform(-2, 2, dim)
        population[i] = normalize(population[i].reshape(N, 2)).ravel()
        
    # 2. Perturbed 4x4 grid
    x = np.arange(4) - 1.5
    y = np.arange(4) - 1.5
    xx, yy = np.meshgrid(x, y)
    base_grid = np.column_stack([xx.ravel(), yy.ravel()])
    base_grid = normalize(base_grid)
    
    for i in range(pop_size // 2, pop_size):
        noise = np.random.normal(0, 0.1, dim)
        perturbed = base_grid.ravel() + noise
        population[i] = normalize(perturbed.reshape(N, 2)).ravel()
        
    # Evaluate initial population
    scores = np.array([score_raw(p.reshape(N, 2)) for p in population])
    
    # DE Parameters
    F = 0.8  # Scaling factor
    CR = 0.9  # Crossover rate
    max_iter = 2000
    best_idx = np.argmin(scores)
    best_x = population[best_idx].copy()
    best_score = scores[best_idx]
    
    # Differential Evolution Loop
    for gen in range(max_iter):
        if time.time() > deadline:
            break
            
        for i in range(pop_size):
            # Select three distinct random indices different from i
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            while r1 == i or r2 == i or r3 == i:
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            
            # Create trial vector
            v = population[r1] + F * (population[r2] - population[r3])
            
            # Normalize trial vector to maintain scale consistency
            v = normalize(v.reshape(N, 2)).ravel()
            
            # Crossover
            j_rand = np.random.randint(0, dim)
            u = np.where(np.random.rand(dim) < CR, v, population[i])
            u[j_rand] = v[j_rand]
            
            # Evaluate trial
            trial_pts = u.reshape(N, 2)
            trial_score = score_raw(trial_pts)
            
            # Selection
            if trial_score <= scores[i]:
                population[i] = u
                scores[i] = trial_score
                if trial_score < best_score:
                    best_score = trial_score
                    best_idx = i
                    best_x = u.copy()
                    
    # Final polish: Use the best found so far as a starting point for a local search
    # using Nelder-Mead (derivative-free local optimizer) to refine the global best.
    # This combines global exploration (DE) with local exploitation (NM).
    
    from scipy.optimize import minimize
    
    def objective(x):
        pts = x.reshape(N, 2)
        return score_raw(pts)
    
    # Run Nelder-Mead on the best point found by DE
    try:
        res = minimize(
            objective,
            best_x,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        if res.fun < best_score:
            best_x = res.x
            best_score = res.fun
    except Exception:
        pass # Keep DE result if NM fails

    final_pts = best_x.reshape(N, 2)
    return final_pts

# EVOLVE-BLOCK-END
