import numpy as np
import scipy.spatial as spat
import time

N = 14
DEADLINE = 145  # safety margin before 150s kill

def _calc_min_tri_area(points, prev_indices=None):
    """
    Compute minimum triangle area. If prev_indices is provided, first check
    whether those triples still yield a value >= prev_min (caller must pass
    prev_min separately, but here we just recompute all for correctness).
    For speed we always recompute but use a tighter loop.
    """
    n = len(points)
    # Precompute all pairwise vectors? No, just iterate.
    min_area = np.inf
    # Use itertools for clarity, but for n=14, 364 triples is small.
    # Actually, vectorized is faster.
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _hull_area(points):
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def _score(points):
    hull_a = _hull_area(points)
    if hull_a < 1e-10:
        return 0.0
    min_tri = _calc_min_tri_area(points)
    return min_tri / hull_a

def _normalize(points):
    """Normalize to unit hull area."""
    hull_a = _hull_area(points)
    if hull_a < 1e-10:
        return points, 1e-10
    return points / np.sqrt(hull_a), hull_a

def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)
    return np.column_stack([xs, ys])

def _nelder_mead(points, max_iter=200, tol=1e-10):
    """
    Simple Nelder-Mead implementation for a function f(x) where x is flattened points.
    We minimize -score.
    """
    def f(x):
        pts = x.reshape(N, 2)
        s = _score(pts)
        return -s

    x0 = points.flatten()
    n_dim = len(x0)
    
    # Initial simplex
    simplex = [x0.copy() for _ in range(n_dim + 1)]
    for i in range(n_dim):
        simplex[i+1] = x0.copy()
        simplex[i+1][i] += 0.01
    
    # Evaluate initial simplex
    fvals = [f(x) for x in simplex]
    
    # Nelder-Mead parameters
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for _ in range(max_iter):
        # Sort by fvals
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if abs(fvals[0] - fvals[-1]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n_dim + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = f(simplex[i])
    
    # Return best point
    best_idx = np.argmin(fvals)
    return np.array(simplex[best_idx]).reshape(N, 2)

def _differential_evolution(population, scores, rng, F=0.8, CR=0.9, max_iter=500):
    """
    Basic Differential Evolution.
    population: list of arrays (N, 2)
    scores: list of floats
    """
    pop_size = len(population)
    current_best_idx = np.argmin(scores)  # We minimize -score, so min score is best? No, max score.
    # Let's stick to maximizing score.
    # DE usually minimizes. Let's define fitness = -score.
    
    for i in range(max_iter):
        # For each individual, create mutant
        new_population = []
        new_scores = []
        
        for j in range(pop_size):
            # Choose 3 distinct random individuals
            a, b, c = rng.choice(pop_size, size=3, replace=False)
            while a == b or a == c or b == c:
                a, b, c = rng.choice(pop_size, size=3, replace=False)
            
            # Mutant vector
            v = population[a] + F * (population[b] - population[c])
            
            # Normalize
            v, _ = _normalize(v)
            
            # Crossover
            trial = np.zeros_like(population[j])
            cross_mask = rng.random((N, 2)) < CR
            # Ensure at least one gene is crossed
            if not np.any(cross_mask):
                idx = rng.randint(N)
                trial[idx, 0] = v[idx, 0]
                trial[idx, 1] = v[idx, 1]
            else:
                trial = np.where(cross_mask, v, population[j])
            
            # Normalize trial
            trial, _ = _normalize(trial)
            
            # Evaluate
            trial_score = _score(trial)
            
            # Selection
            if trial_score >= scores[j]:
                new_population.append(trial)
                new_scores.append(trial_score)
            else:
                new_population.append(population[j])
                new_scores.append(scores[j])
        
        population = new_population
        scores = new_scores
        
        # Early stop if no improvement in last 50 iterations?
        # Just run for max_iter or time limit
        
        if time.time() > DEADLINE - 10:
            break

    return population, scores

def construct():
    start_time = time.time()
    rng = np.random.RandomState(42)
    
    # 1. Generate initial population
    population = []
    scores = []
    
    # Add regular polygon
    reg_poly = _regular_polygon(N)
    reg_poly, _ = _normalize(reg_poly)
    population.append(reg_poly)
    scores.append(_score(reg_poly))
    
    # Add random points in circle
    for _ in range(10):
        r = np.sqrt(rng.random(N))
        theta = rng.uniform(0, 2 * np.pi, N)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        pts, _ = _normalize(pts)
        population.append(pts)
        scores.append(_score(pts))
    
    # Add perturbed regular polygons
    for seed in range(5):
        noise_scale = 0.05 + 0.02 * seed
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        angle_noise = rng.normal(0, noise_scale * 0.1, N)
        angles += angle_noise
        radii = np.ones(N)
        radius_noise = rng.normal(0, noise_scale, N)
        radii += radius_noise
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        pts, _ = _normalize(pts)
        population.append(pts)
        scores.append(_score(pts))
    
    # 2. Differential Evolution
    population, scores = _differential_evolution(population, scores, rng, max_iter=300)
    
    # 3. Find best individual
    best_idx = np.argmax(scores)
    best_points = population[best_idx].copy()
    best_score = scores[best_idx]
    
    # 4. Nelder-Mead refinement on best point
    if time.time() < DEADLINE - 20:
        refined_points = _nelder_mead(best_points, max_iter=100)
        refined_score = _score(refined_points)
        if refined_score > best_score:
            best_points = refined_points
            best_score = refined_score
    
    # 5. Final check and return
    if time.time() < DEADLINE - 5:
        # One last try with a different seed for DE if time permits?
        # No, just return best.
        pass
        
    return np.array(best_points)
