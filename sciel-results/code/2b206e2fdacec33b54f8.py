"""Optimized program for Heilbronn convex regions: 13 points maximizing min triangle area / hull area.
Strategy: smooth LogSumExp surrogate + BFGS + targeted polish + differential evolution.
"""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13
TIME_LIMIT = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _all_triangle_areas(pts):
    """Compute areas of all C(n,3) triangles. Returns array of shape (C(n,3),)."""
    n = len(pts)
    areas = []
    for i in range(n - 2):
        pi = pts[i]
        rest = pts[i+1:]
        m = len(rest)
        v1 = rest[:, np.newaxis, :] - pi
        v2 = rest[np.newaxis, :, :] - pi
        cross = v1[:, :, 0] * v2[:, :, 1] - v1[:, :, 1] * v2[:, :, 0]
        idx = np.triu_indices(m, k=1)
        areas.append(np.abs(cross[idx]))
    areas = np.concatenate(areas)
    return 0.5 * areas

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Compute min triangle area / convex hull area."""
    try:
        hull_area = _hull_area(pts)
        if hull_area < 1e-12:
            return 0.0
        min_area = np.min(_all_triangle_areas(pts))
        return min_area / hull_area
    except Exception:
        return 0.0

def _soft_min_score(pts, beta=50.0):
    """Smooth surrogate: -logsumexp(-beta * areas) / hull_area.
    This approximates min(areas)/hull_area smoothly."""
    try:
        areas = _all_triangle_areas(pts)
        hull_area = _hull_area(pts)
        if hull_area < 1e-12:
            return -1e10
        # Soft min of areas
        neg_areas = -beta * areas
        # Numerically stable logsumexp
        max_neg = np.max(neg_areas)
        logsum = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
        soft_min = -logsum / beta
        return soft_min / hull_area
    except Exception:
        return -1e10

def _neg_soft_min_score(x, beta=50.0):
    """Negative soft min score for minimization. x is flat array of shape (2n,)."""
    pts = x.reshape(-1, 2)
    return -_soft_min_score(pts, beta)

def _neg_score(x):
    """Negative exact score for minimization."""
    pts = x.reshape(-1, 2)
    return -_score(pts)

def _halton_sequence(n, base):
    """Generate n Halton sequence values in [0,1)."""
    result = []
    f = 1.0
    for i in range(1, n + 1):
        x = 0.0
        i_temp = i
        while i_temp > 0:
            f /= base
            x += f * (i_temp % base)
            i_temp //= base
        result.append(x)
    return np.array(result)

def _generate_initializations():
    """Generate multiple initial configurations."""
    inits = []
    
    # 1. Regular 13-gon on unit circle
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts_circle = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(pts_circle)
    
    # 2. Halton sequence in unit square
    x = _halton_sequence(N, 2)
    y = _halton_sequence(N, 3)
    pts_halton = np.column_stack([x, y])
    inits.append(pts_halton)
    
    # 3. Halton on circle (angular Halton)
    angles_halton = 2 * np.pi * _halton_sequence(N, 2)
    radii = 0.5 + 0.5 * _halton_sequence(N, 3)
    pts_halton_circle = np.column_stack([radii * np.cos(angles_halton), radii * np.sin(angles_halton)])
    inits.append(pts_halton_circle)
    
    # 4. Grid-like: mix of boundary and interior
    angles_boundary = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    boundary_pts = np.column_stack([np.cos(angles_boundary), np.sin(angles_boundary)])
    interior_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    interior_radii = [0.3, 0.5, 0.3, 0.5]
    interior_pts = np.column_stack([
        [r * np.cos(a) for r, a in zip(interior_radii, interior_angles)],
        [r * np.sin(a) for r, a in zip(interior_radii, interior_angles)]
    ])
    pts_mixed = np.vstack([boundary_pts, interior_pts])
    inits.append(pts_mixed)
    
    # 5. Another mixed: 10 boundary, 3 interior
    angles_b = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    b_pts = np.column_stack([np.cos(angles_b), np.sin(angles_b)])
    i_angles = np.linspace(0, 2 * np.pi, 3, endpoint=False) + np.pi / 6
    i_radii = [0.4, 0.6, 0.4]
    i_pts = np.column_stack([
        [r * np.cos(a) for r, a in zip(i_radii, i_angles)],
        [r * np.sin(a) for r, a in zip(i_radii, i_angles)]
    ])
    pts_mixed2 = np.vstack([b_pts, i_pts])
    inits.append(pts_mixed2)
    
    # 6. Random with seed
    rng = np.random.RandomState(42)
    pts_rand = rng.random((N, 2))
    inits.append(pts_rand)
    
    # 7. Regular polygon with one center point and 12 on circle
    angles_12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    circle_12 = np.column_stack([np.cos(angles_12), np.sin(angles_12)])
    center = np.array([[0.0, 0.0]])
    pts_center = np.vstack([circle_12, center])
    inits.append(pts_center)
    
    # 8. Fibonacci lattice in unit square
    golden = (1 + np.sqrt(5)) / 2
    idx = np.arange(N)
    fx = (idx / golden) % 1.0
    fy = (idx / (golden * golden)) % 1.0
    pts_fib = np.column_stack([fx, fy])
    inits.append(pts_fib)
    
    # 9. Slightly perturbed regular polygon
    angles_pert = np.linspace(0, 2 * np.pi, N, endpoint=False) + 0.05 * np.sin(np.arange(N))
    radii_pert = 1.0 + 0.05 * np.cos(2 * np.arange(N))
    pts_pert = np.column_stack([radii_pert * np.cos(angles_pert), radii_pert * np.sin(angles_pert)])
    inits.append(pts_pert)
    
    return inits

def _bfgs_optimize(pts, beta, time_limit, start_time):
    """Run BFGS on the smooth surrogate objective."""
    x0 = pts.flatten()
    remaining = time_limit - (time.time() - start_time)
    if remaining < 5:
        return pts, _score(pts)
    
    max_iter = min(500, int(remaining * 50))
    
    result = minimize(
        _neg_soft_min_score,
        x0,
        args=(beta,),
        method='BFGS',
        options={'maxiter': max_iter, 'gtol': 1e-8}
    )
    
    best_pts = result.x.reshape(-1, 2)
    best_score = _score(best_pts)
    
    # Also try the initial point
    s0 = _score(pts)
    if s0 > best_score:
        best_pts = pts
        best_score = s0
    
    return best_pts, best_score

def _targeted_polish(pts, time_limit, start_time):
    """Polish by targeting the worst triangles: move points that form small triangles."""
    best_pts = pts.copy()
    best_score = _score(pts)
    
    for iteration in range(20):
        remaining = time_limit - (time.time() - start_time)
        if remaining < 2:
            break
        
        areas = _all_triangle_areas(best_pts)
        min_area = np.min(areas)
        
        # Find which triangles have the smallest areas
        # We need to identify the indices of the worst triangles
        # Reconstruct triangle indices
        n = len(best_pts)
        tri_indices = []
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    tri_indices.append((i, j, k))
        tri_indices = np.array(tri_indices)
        
        # Find worst 5 triangles
        worst_idx = np.argsort(areas)[:5]
        
        # For each worst triangle, try to move one of its vertices
        for widx in worst_idx:
            i, j, k = tri_indices[widx]
            # Try small perturbations of each vertex in the triangle
            for vi in [i, j, k]:
                for dx in [-0.02, 0.02]:
                    for dy in [-0.02, 0.02]:
                        candidate = best_pts.copy()
                        candidate[vi, 0] += dx
                        candidate[vi, 1] += dy
                        s = _score(candidate)
                        if s > best_score:
                            best_score = s
                            best_pts = candidate.copy()
        
        # Also try moving points not in worst triangles but close to others
        remaining = time_limit - (time.time() - start_time)
        if remaining < 2:
            break
    
    return best_pts, best_score

def _differential_evolution_search(pts, time_limit, start_time, pop_size=30, max_gen=50):
    """Differential evolution on the smooth surrogate."""
    remaining = time_limit - (time.time() - start_time)
    if remaining < 10:
        return pts, _score(pts)
    
    x0 = pts.flatten()
    n_params = len(x0)
    
    # Initialize population
    rng = np.random.RandomState(123)
    pop = np.zeros((pop_size, n_params))
    pop[0] = x0
    for i in range(1, pop_size):
        pop[i] = x0 + rng.normal(0, 0.05, n_params)
    
    # Evaluate population
    scores = np.array([_soft_min_score(pop[i].reshape(-1, 2), beta=50.0) for i in range(pop_size)])
    
    best_idx = np.argmax(scores)
    best_x = pop[best_idx].copy()
    best_s = scores[best_idx]
    
    F = 0.8  # scale factor
    CR = 0.9  # crossover rate
    
    gen = 0
    while gen < max_gen:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 5:
            break
        
        for i in range(pop_size):
            remaining = time_limit - (time.time() - start_time)
            if remaining < 5:
                break
            
            # Select three distinct random individuals
            while True:
                r1, r2, r3 = rng.choice(pop_size, 3, replace=False)
                if len(set([i, r1, r2, r3])) == 4:
                    break
            
            # Mutation
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            
            # Crossover
            trial = pop[i].copy()
            j_rand = rng.randint(n_params)
            for j in range(n_params):
                if rng.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
            
            # Evaluation
            trial_pts = trial.reshape(-1, 2)
            trial_score = _soft_min_score(trial_pts, beta=50.0)
            
            if trial_score >= scores[i]:
                pop[i] = trial
                scores[i] = trial_score
                if trial_score > best_s:
                    best_s = trial_score
                    best_x = trial.copy()
        
        gen += 1
    
    best_pts = best_x.reshape(-1, 2)
    final_score = _score(best_pts)
    
    # Compare with input
    s_input = _score(pts)
    if s_input > final_score:
        best_pts = pts
        final_score = s_input
    
    return best_pts, final_score

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    inits = _generate_initializations()
    
    best_pts = None
    best_score = -1
    
    # Evaluate all initializations
    for i, init in enumerate(inits):
        if time.time() > deadline - 40:
            break
        s = _score(init)
        if s > best_score:
            best_score = s
            best_pts = init.copy()
    
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = _score(best_pts)
    
    # Phase 1: BFGS on smooth surrogate from each initialization
    for i, init in enumerate(inits):
        if time.time() > deadline - 20:
            break
        s_init = _score(init)
        if s_init < best_score * 0.3:
            continue
        
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        pts_opt, score_opt = _bfgs_optimize(init, beta=50.0, time_limit=deadline, start_time=start_time)
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt
    
    # Phase 2: Differential evolution from best point
    if time.time() < deadline - 30:
        pts_de, score_de = _differential_evolution_search(
            best_pts, time_limit=deadline, start_time=start_time,
            pop_size=25, max_gen=40
        )
        if score_de > best_score:
            best_score = score_de
            best_pts = pts_de
    
    # Phase 3: BFGS polish on best
    if time.time() < deadline - 10:
        pts_pol, score_pol = _bfgs_optimize(best_pts, beta=100.0, time_limit=deadline, start_time=start_time)
        if score_pol > best_score:
            best_score = score_pol
            best_pts = pts_pol
    
    # Phase 4: Targeted polish
    if time.time() < deadline - 3:
        pts_tp, score_tp = _targeted_polish(best_pts, time_limit=deadline, start_time=start_time)
        if score_tp > best_score:
            best_score = score_tp
            best_pts = pts_tp
    
    # Final: one more BFGS with high beta
    if time.time() < deadline - 2:
        pts_final, score_final = _bfgs_optimize(best_pts, beta=200.0, time_limit=deadline, start_time=start_time)
        if score_final > best_score:
            best_score = score_final
            best_pts = pts_final
    
    return best_pts

# EVOLVE-BLOCK-END
