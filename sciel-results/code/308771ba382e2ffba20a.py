# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 560.0  # 40s safety margin before 600s kill

def _min_triangle_area(points):
    """
    Compute the minimum area among all triangles formed by triples of points.
    Vectorized for speed.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Generate all combinations of 3 indices
    indices = np.array(list(itertools.combinations(range(n), 3)))
    
    # Extract points
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    # Calculate areas: 0.5 * |cross(p2-p1, p3-p1)|
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    return float(np.min(areas))

def _hull_area(points):
    """
    Compute the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the ratio A_min / A_hull.
    """
    a_min = _min_triangle_area(points)
    a_hull = _hull_area(points)
    
    if a_hull < 1e-10:
        return 0.0
        
    return a_min / a_hull

def _soft_min_objective(params, beta):
    """
    Objective function using a soft-min approximation for the minimum triangle area.
    We minimize the negative of the score: -A_min / A_hull.
    """
    points = params.reshape(13, 2)
    
    # Compute all triangle areas
    n = 13
    indices = np.array(list(itertools.combinations(range(n), 3)))
    p1 = points[indices[:, 0]]
    p2 = points[indices[:, 1]]
    p3 = points[indices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = np.abs(cross) * 0.5
    
    # Soft min approximation: -log(sum(exp(-beta * areas))) / beta
    # This is differentiable and approximates min(areas) as beta -> inf
    # To avoid overflow/underflow, we shift by the minimum
    min_area = np.min(areas)
    exp_term = np.exp(-beta * (areas - min_area))
    soft_min_val = min_area - np.log(np.sum(exp_term)) / beta
    
    a_hull = _hull_area(points)
    if a_hull < 1e-10:
        return 1e10
    
    # Maximize soft_min_val / a_hull -> Minimize -soft_min_val / a_hull
    return -soft_min_val / a_hull

def _exact_objective(params):
    """
    Exact objective function for final polish.
    """
    points = params.reshape(13, 2)
    s = _score(points)
    return -s

def _regular_polygon(n, radius=1.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(angles) * radius
    y = np.sin(angles) * radius
    return np.column_stack((x, y))

def _fibonacci_spiral(n, radius=1.0):
    """
    Generate points inside a circle using a Fibonacci-like spiral for even distribution.
    """
    golden_angle = np.pi * (3 - np.sqrt(5))
    i = np.arange(n)
    r = radius * np.sqrt((i + 0.5) / n)
    theta = i * golden_angle
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def _concentric_rings(n_total, num_rings, radius=1.0):
    """
    Generate points on concentric rings.
    For n=13: ring 0 has 1 point (center), ring 1 has 6 points, ring 2 has 6 points
    """
    points = []
    
    # Center point
    if num_rings >= 1:
        points.append([0.0, 0.0])
        n_total -= 1
    
    # Distribute remaining points on rings
    if num_rings == 1:
        # All on one ring
        angles = np.linspace(0, 2 * np.pi, n_total, endpoint=False)
        x = np.cos(angles) * radius
        y = np.sin(angles) * radius
        points.extend(zip(x, y))
    elif num_rings >= 2:
        # Ring 1: 6 points
        ring1_count = min(6, n_total)
        angles1 = np.linspace(0, 2 * np.pi, ring1_count, endpoint=False)
        r1 = radius * 0.5
        x1 = np.cos(angles1) * r1
        y1 = np.sin(angles1) * r1
        points.extend(zip(x1, y1))
        n_total -= ring1_count
        
        # Ring 2: remaining points
        if n_total > 0:
            ring2_count = n_total
            angles2 = np.linspace(0, 2 * np.pi, ring2_count, endpoint=False) + np.pi / ring2_count
            r2 = radius * 1.0
            x2 = np.cos(angles2) * r2
            y2 = np.sin(angles2) * r2
            points.extend(zip(x2, y2))
    
    return np.array(points)

def _hexagonal_lattice(n, spacing=1.0):
    """
    Generate points on a hexagonal lattice pattern.
    """
    points = []
    # Center
    points.append([0.0, 0.0])
    
    # First ring: 6 points
    for i in range(6):
        angle = i * np.pi / 3
        points.append([spacing * np.cos(angle), spacing * np.sin(angle)])
    
    # Second ring: 12 points (we only need 1 more to reach 13)
    for i in range(12):
        angle = i * np.pi / 6
        r = spacing * np.sqrt(3)
        points.append([r * np.cos(angle), r * np.sin(angle)])
    
    # Take first n points
    return np.array(points[:n])

def _sort_by_radius_angle(points):
    """
    Sort points by (radius, angle) to maintain topology.
    """
    radii = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    angles = np.arctan2(points[:, 1], points[:, 0])
    
    # Sort by radius first, then angle
    order = np.lexsort((angles, radii))
    return points[order]

def heilbronn_convex13() -> np.ndarray:
    n = 13
    
    # 1. Generate diverse initializations
    inits = []
    inits.append(_regular_polygon(n))
    inits.append(_fibonacci_spiral(n))
    inits.append(_concentric_rings(n, 2))
    inits.append(_hexagonal_lattice(n))
    
    # Add a few random inits
    rng = np.random.default_rng(42)
    for _ in range(5):
        inits.append(rng.random((n, 2)) * 2 - 1)
        
    # Also add the regular polygon with a slight perturbation
    inits.append(_regular_polygon(n) + rng.normal(0, 0.1, (n, 2)))
    inits.append(_concentric_rings(n, 2) + rng.normal(0, 0.05, (n, 2)))

    best_params = inits[0].flatten()
    best_score = _score(inits[0])
    
    # Evaluate all inits and keep the best
    for init in inits:
        s = _score(init)
        if s > best_score:
            best_score = s
            best_params = init.flatten()

    # 2. Optimization Strategy:
    # Phase 1: L-BFGS-B with soft-min surrogate (increasing beta)
    # Phase 2: Powell with exact objective
    # Phase 3: Nelder-Mead for fine-tuning
    # Phase 4: Simulated annealing to escape local optima
    
    current_params = best_params.copy()
    
    # Annealing schedule for beta
    betas = [10, 50, 200, 1000, 5000]
    
    for beta in betas:
        if time.time() > _DEADLINE - 120:
            break
            
        try:
            res = minimize(
                _soft_min_objective,
                current_params,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 30}
            )
            if res.fun < -best_score: # res.fun is -score
                current_params = res.x
                best_score = -res.fun
            # Even if not better, use the result as starting point for next beta
            # as it might be in a better basin for the sharper landscape
            if res.success:
                 current_params = res.x
        except Exception:
            pass

    # 3. Final Polish with Powell on the exact objective
    if time.time() < _DEADLINE - 60:
        try:
            res = minimize(
                _exact_objective,
                current_params,
                method='Powell',
                options={'maxiter': 1000, 'maxfev': 20000, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            if -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass
            
    # 4. Nelder-Mead for fine-tuning
    if time.time() < _DEADLINE - 45:
        try:
            res = minimize(
                _exact_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass
            
    # 5. Simulated annealing to escape local optima
    if time.time() < _DEADLINE - 30:
        sa_start = best_params.copy()
        sa_temp = 0.1
        sa_cooling = 0.95
        sa_best = best_params.copy()
        sa_best_score = best_score
        
        while time.time() < _DEADLINE - 10 and sa_temp > 0.001:
            # Generate perturbation
            noise_scale = sa_temp * 0.1
            perturbed = sa_best + rng.normal(0, noise_scale, (n, 2))
            
            s_pert = _score(perturbed)
            s_cur = sa_best_score
            
            if s_pert > s_cur:
                sa_best = perturbed
                sa_best_score = s_pert
                if sa_best_score > best_score:
                    best_score = sa_best_score
                    best_params = sa_best.copy()
            else:
                # Accept with probability exp((s_pert - s_cur) / sa_temp)
                delta = s_pert - s_cur
                if delta < 0:
                    prob = np.exp(delta / sa_temp)
                    if rng.random() < prob:
                        sa_best = perturbed
                        sa_best_score = s_pert
            
            sa_temp *= sa_cooling
        
        # Final polish after SA
        if time.time() < _DEADLINE - 5:
            try:
                res = minimize(
                    _exact_objective,
                    sa_best,
                    method='Powell',
                    options={'maxiter': 200, 'maxfev': 5000, 'xtol': 1e-12, 'ftol': 1e-12}
                )
                if -res.fun > best_score:
                    best_score = -res.fun
                    best_params = res.x
            except Exception:
                pass
            
    # 6. If time remains, try a few random restarts with the same pipeline
    # but shorter budgets
    while time.time() < _DEADLINE - 20:
        init = rng.random((n, 2)) * 2 - 1
        s_init = _score(init)
        if s_init < best_score * 0.8: # Heuristic: skip very bad inits
            continue
            
        p = init.flatten()
        
        # Shorter annealing
        for beta in [100, 1000]:
             if time.time() > _DEADLINE - 10:
                 break
             try:
                 res = minimize(
                     _soft_min_objective,
                     p,
                     args=(beta,),
                     method='L-BFGS-B',
                     options={'maxiter': 300, 'ftol': 1e-12}
                 )
                 if res.success:
                     p = res.x
             except:
                 break
                 
        # Quick polish
        try:
            res = minimize(
                _exact_objective,
                p,
                method='Powell',
                options={'maxiter': 200, 'maxfev': 5000}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_params = res.x
        except:
            pass

    return best_params.reshape(13, 2)

# EVOLVE-BLOCK-END
