"""Heilbronn triangles: 11 points in equilateral triangle, maximize min triangle area."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0

# Precompute indices for all triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])

def _bary_to_cart(bary):
    """Convert barycentric coords (u, v) with u>=0, v>=0, u+v<=1 to cartesian."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * np.sqrt(3) / 2
    return np.column_stack((x, y))

def _cart_to_bary(points):
    """Convert cartesian points to barycentric (u, v) where point = u*V0 + v*V1 + w*V2, w=1-u-v."""
    # x = v + (1-u-v)*0.5 = v + 0.5 - 0.5u - 0.5v = 0.5 + 0.5v - 0.5u
    # y = (1-u-v)*sqrt(3)/2
    y = points[:, 1]
    w = 2.0 * y / np.sqrt(3)
    x = points[:, 0]
    # x = 0.5 + 0.5v - 0.5u => 2x - 1 = v - u
    # w = 1 - u - v => u + v = 1 - w
    # v - u = 2x - 1
    # v + u = 1 - w
    # 2v = 1 - w + 2x - 1 = 2x - w => v = x - w/2
    # 2u = 1 - w - (2x - 1) = 2 - 2x - w => u = 1 - x - w/2
    v = x - w / 2.0
    u = 1.0 - x - w / 2.0
    return np.column_stack((u, v))

def _clamp_bary(bary):
    """Clamp barycentric coordinates to the valid region u>=0, v>=0, u+v<=1."""
    u = np.maximum(0.0, bary[:, 0])
    v = np.maximum(0.0, bary[:, 1])
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack((u, v))

def _all_dets_from_bary(bary):
    """Compute all 165 determinants (twice the area ratio) in barycentric coords."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    P = np.column_stack((u, v, w))
    
    i_idx = _IDX_TRIPLES[:, 0]
    j_idx = _IDX_TRIPLES[:, 1]
    k_idx = _IDX_TRIPLES[:, 2]
    
    Pi = P[i_idx]
    Pj = P[j_idx]
    Pk = P[k_idx]
    
    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
           
    return det

def _min_area_ratio_from_bary(bary):
    """Returns the minimum absolute determinant (proportional to area ratio)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(w < -1e-9):
        return -1.0
        
    dets = _all_dets_from_bary(bary)
    return np.min(np.abs(dets))

def _objective_smooth(x, beta, penalty_lambda=100.0):
    """
    Smooth objective for L-BFGS-B.
    x: (2N,) array of u, v coords.
    """
    bary = x.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Penalty for invalid barycentric coords
    penalty = 0.0
    excess = np.maximum(0, u + v - 1.0)
    penalty += penalty_lambda * np.sum(excess**2)
    neg_u = np.maximum(0, -u)
    penalty += penalty_lambda * np.sum(neg_u**2)
    neg_v = np.maximum(0, -v)
    penalty += penalty_lambda * np.sum(neg_v**2)
    
    dets = _all_dets_from_bary(bary)
    areas = np.abs(dets)
    
    if len(areas) == 0:
        return penalty
        
    m = np.min(areas)
    shifted = areas - m
    log_sum_exp = np.log(np.sum(np.exp(-beta * shifted)))
    
    smooth_min = m - (1.0 / beta) * log_sum_exp
    
    return -smooth_min + penalty

def _objective_slsqp(x):
    """
    Objective for SLSQP: minimize -min_area.
    x: (2N,) array.
    """
    bary = x.reshape(N, 2)
    bary_clamped = _clamp_bary(bary)
    dets = _all_dets_from_bary(bary_clamped)
    min_area = np.min(np.abs(dets))
    return -min_area

def _structured_init_fps():
    """Farthest Point Sampling on a grid of barycentric coordinates."""
    n_grid = 20
    candidates = []
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            u = i / n_grid
            v = j / n_grid
            candidates.append([u, v])
    candidates = np.array(candidates)
    
    selected = [np.array([1/3, 1/3])]
    
    for _ in range(N - 1):
        dists = []
        for c in candidates:
            min_d = np.min(np.linalg.norm(c - np.array(selected), axis=1))
            dists.append(min_d)
        dists = np.array(dists)
        max_idx = np.argmax(dists)
        selected.append(candidates[max_idx])
        
    return np.array(selected)

def _project_to_bary_simplex(bary):
    """Project points to the valid barycentric simplex."""
    return _clamp_bary(bary)

def _local_optimize_lbfgs(x0, maxiter=300, beta=50.0):
    """Run L-BFGS-B local optimization from x0 with smooth objective."""
    try:
        bounds = [(0, 1)] * (2 * N)
        res = minimize(
            lambda x: _objective_smooth(x, beta),
            x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-9}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt.flatten()
    except Exception:
        return x0

def _local_optimize_nm(x0, maxiter=300):
    """Run Nelder-Mead local optimization from x0."""
    try:
        res = minimize(
            _objective_slsqp, x0, method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt.flatten()
    except Exception:
        return x0

def _local_optimize_slsqp(x0, maxiter=200):
    """Run SLSQP local optimization from x0."""
    try:
        bounds = [(0, 1)] * (2 * N)
        res = minimize(
            _objective_slsqp, x0, method='SLSQP',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-12}
        )
        bary_opt = _clamp_bary(res.x.reshape(N, 2))
        return bary_opt.flatten()
    except Exception:
        return x0

def _genetic_algorithm(rng, pop_size=30, generations=20, elite_size=5, mutation_rate=0.1, mutation_scale=0.05):
    """
    Genetic algorithm to evolve the set of 11 points.
    """
    # Initialize population
    population = []
    
    # Add FPS init
    population.append(_structured_init_fps().flatten())
    
    # Add structured inits
    magic_pts = np.array([
        [1.0, 0.0], [0.0, 1.0], [0.0, 0.0],
        [1/3, 1/3],
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
        [0.25, 0.0], [0.0, 0.25], [0.25, 0.25], [0.5, 0.25],
    ])
    population.append(_clamp_bary(magic_pts).flatten())
    
    # Add random inits
    for _ in range(pop_size - 2):
        pts = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            while u + v > 1.0:
                u = rng.random()
                v = rng.random()
            pts[i, 0] = u
            pts[i, 1] = v
        population.append(_clamp_bary(pts).flatten())
    
    # Evaluate initial population
    fitness = np.array([_min_area_ratio_from_bary(ind.reshape(N, 2)) for ind in population])
    
    for gen in range(generations):
        # Sort by fitness (descending)
        order = np.argsort(fitness)[::-1]
        population = [population[i] for i in order]
        fitness = fitness[order]
        
        # Elite selection
        new_population = population[:elite_size]
        
        # Crossover and mutation
        while len(new_population) < pop_size:
            # Tournament selection
            p1_idx = rng.choice(pop_size, size=3, replace=False)
            p2_idx = rng.choice(pop_size, size=3, replace=False)
            p1 = population[p1_idx[np.argmax(fitness[p1_idx])]]
            p2 = population[p2_idx[np.argmax(fitness[p2_idx])]]
            
            # Crossover (uniform)
            mask = rng.random(p1.shape) < 0.5
            child = np.where(mask, p1, p2)
            
            # Mutation
            if rng.random() < mutation_rate:
                child += rng.normal(0, mutation_scale, size=child.shape)
            
            child = _clamp_bary(child.reshape(N, 2)).flatten()
            new_population.append(child)
        
        population = new_population
        fitness = np.array([_min_area_ratio_from_bary(ind.reshape(N, 2)) for ind in population])
        
        # Local optimization on best individuals
        for i in range(min(3, len(population))):
            if fitness[i] > 0:
                opt = _local_optimize_lbfgs(population[i], maxiter=100, beta=30.0)
                opt_fitness = _min_area_ratio_from_bary(opt.reshape(N, 2))
                if opt_fitness > fitness[i]:
                    population[i] = opt
                    fitness[i] = opt_fitness
                order = np.argsort(fitness)[::-1]
                population = [population[j] for j in order]
                fitness = fitness[order]
        
        # Update mutation scale
        mutation_scale *= 0.95
    
    best_idx = np.argmax(fitness)
    return population[best_idx]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def time_left():
        return deadline - time.time()
        
    best_x = None
    best_score = -float('inf')
    
    rng = np.random.RandomState(42)
    
    # Phase 1: Genetic Algorithm
    if time_left() > 10.0:
        ga_result = _genetic_algorithm(rng, pop_size=25, generations=15, elite_size=4, mutation_rate=0.15, mutation_scale=0.08)
        ga_score = _min_area_ratio_from_bary(ga_result.reshape(N, 2))
        if ga_score > best_score:
            best_score = ga_score
            best_x = ga_result.copy()
    
    # Phase 2: Multi-start local optimization
    inits = []
    
    # FPS Init
    inits.append(_structured_init_fps().flatten())
    
    # Random Inits
    for _ in range(10):
        pts = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            while u + v > 1.0:
                u = rng.random()
                v = rng.random()
            pts[i, 0] = u
            pts[i, 1] = v
        inits.append(_clamp_bary(pts).flatten())
        
    # Magic Init
    magic_pts = np.array([
        [1.0, 0.0], [0.0, 1.0], [0.0, 0.0],
        [1/3, 1/3],
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
        [0.25, 0.0], [0.0, 0.25], [0.25, 0.25], [0.5, 0.25],
    ])
    inits.append(_clamp_bary(magic_pts).flatten())
    
    # Add GA result if found
    if best_x is not None:
        inits.append(best_x.copy())
    
    # Optimize each init
    for init in inits:
        if time_left() < 2.0:
            break
        
        # L-BFGS-B with smooth objective
        opt1 = _local_optimize_lbfgs(init, maxiter=200, beta=50.0)
        score1 = _min_area_ratio_from_bary(opt1.reshape(N, 2))
        if score1 > best_score:
            best_score = score1
            best_x = opt1.copy()
        
        # Nelder-Mead
        if time_left() > 2.0:
            opt2 = _local_optimize_nm(opt1, maxiter=200)
            score2 = _min_area_ratio_from_bary(opt2.reshape(N, 2))
            if score2 > best_score:
                best_score = score2
                best_x = opt2.copy()
        
        # SLSQP
        if time_left() > 2.0:
            opt3 = _local_optimize_slsqp(opt2 if score2 > score1 else opt1, maxiter=100)
            score3 = _min_area_ratio_from_bary(opt3.reshape(N, 2))
            if score3 > best_score:
                best_score = score3
                best_x = opt3.copy()
    
    # Phase 3: Perturbation-based refinement
    if best_x is not None and time_left() > 5.0:
        for scale in [0.05, 0.03, 0.01, 0.005]:
            if time_left() < 2.0:
                break
            for _ in range(5):
                if time_left() < 1.0:
                    break
                pert = rng.normal(0, scale, size=(N, 2))
                start = _clamp_bary(best_x.reshape(N, 2) + pert).flatten()
                opt = _local_optimize_lbfgs(start, maxiter=150, beta=80.0)
                score = _min_area_ratio_from_bary(opt.reshape(N, 2))
                if score > best_score:
                    best_score = score
                    best_x = opt.copy()
    
    # Phase 4: Final fine-tuning
    if best_x is not None and time_left() > 2.0:
        opt_final = _local_optimize_slsqp(best_x, maxiter=200)
        score_final = _min_area_ratio_from_bary(opt_final.reshape(N, 2))
        if score_final > best_score:
            best_score = score_final
            best_x = opt_final.copy()
    
    # Fallback
    if best_x is None:
        best_x = _structured_init_fps().flatten()
    
    # Convert to cartesian
    bary = _clamp_bary(best_x.reshape(N, 2))
    return _bary_to_cart(bary)

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
