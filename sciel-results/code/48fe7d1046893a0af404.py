"""Optimized program for Heilbronn convex regions: 13 points maximizing min triangle area / hull area."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
TIME_LIMIT = 140.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START

def _all_triangle_areas(pts):
    """Compute areas of all C(n,3) triangles. Returns array of shape (C(n,3),)."""
    n = len(pts)
    # pts shape (n, 2)
    # Use cross product: area = 0.5 * |cross(pj-pi, pk-pi)|
    # Vectorized: for each i, compute cross products with all j>i, k>j
    areas = []
    for i in range(n - 2):
        pi = pts[i]
        rest = pts[i+1:]
        # For each pair (j, k) in rest, compute cross product
        # rest shape (m, 2) where m = n-1-i
        m = len(rest)
        # v1 = rest[j] - pi, v2 = rest[k] - pi
        # cross = v1[:,0]*v2[:,1] - v1[:,1]*v2[:,0]
        # We need all pairs j < k
        # Use broadcasting
        v1 = rest[:, np.newaxis, :] - pi  # (m, 1, 2)
        v2 = rest[np.newaxis, :, :] - pi  # (1, m, 2)
        cross = v1[:, :, 0] * v2[:, :, 1] - v1[:, :, 1] * v2[:, :, 0]  # (m, m)
        # Take upper triangle (j < k)
        idx = np.triu_indices(m, k=1)
        areas.append(np.abs(cross[idx]))
    areas = np.concatenate(areas)
    return 0.5 * areas

def _score(pts):
    """Compute min triangle area / convex hull area."""
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # in 2D, volume is area
        if hull_area < 1e-12:
            return 0.0
        min_area = np.min(_all_triangle_areas(pts))
        return min_area / hull_area
    except Exception:
        return 0.0

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
    # 9 points on circle, 4 inside
    angles_boundary = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    boundary_pts = np.column_stack([np.cos(angles_boundary), np.sin(angles_boundary)])
    # 4 interior points at various radii
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
    
    return inits

def _nelder_mead_maximize(pts, score_func, max_iter=5000, tol=1e-10, time_limit=None, start_time=None):
    """Simple Nelder-Mead to maximize score."""
    n = len(pts) * 2  # number of parameters
    # Flatten points
    x0 = pts.flatten()
    
    # Initial simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = 0.01 if abs(x0[i]) < 0.1 else 0.01 * abs(x0[i])
        xi[i] += step
        simplex.append(xi)
    
    # Evaluate initial simplex
    scores = []
    for i in range(n + 1):
        pts_i = simplex[i].reshape(-1, 2)
        scores.append(score_func(pts_i))
    
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    best_x = simplex[best_idx].copy()
    
    iterations = 0
    while iterations < max_iter:
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Sort simplex by score (descending - we want to maximize)
        order = np.argsort(scores)[::-1]
        simplex = [simplex[i].copy() for i in order]
        scores = [scores[i] for i in order]
        
        # Worst point
        worst = simplex[-1]
        worst_score = scores[-1]
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        x_ref = centroid + alpha * (centroid - worst)
        pts_ref = x_ref.reshape(-1, 2)
        score_ref = score_func(pts_ref)
        
        if scores[0] >= score_ref > worst_score:
            simplex[-1] = x_ref
            scores[-1] = score_ref
            iterations += 1
            # Check if best improved
            cur_best = max(scores)
            if cur_best > best_score:
                best_score = cur_best
                best_x = np.argmax(scores)  # placeholder
                best_x = simplex[np.argmax(scores)].copy()
            continue
        
        # Expansion
        gamma = 2.0
        if score_ref >= scores[0]:
            x_exp = centroid + gamma * (centroid - worst)
            pts_exp = x_exp.reshape(-1, 2)
            score_exp = score_func(pts_exp)
            if score_exp > score_ref:
                simplex[-1] = x_exp
                scores[-1] = score_exp
            else:
                simplex[-1] = x_ref
                scores[-1] = score_ref
            iterations += 1
            cur_best = max(scores)
            if cur_best > best_score:
                best_score = cur_best
                best_x = simplex[np.argmax(scores)].copy()
            continue
        
        # Contraction
        rho = 0.5
        x_con = centroid + rho * (worst - centroid)
        pts_con = x_con.reshape(-1, 2)
        score_con = score_func(pts_con)
        
        if score_con > worst_score:
            simplex[-1] = x_con
            scores[-1] = score_con
            iterations += 1
            cur_best = max(scores)
            if cur_best > best_score:
                best_score = cur_best
                best_x = simplex[np.argmax(scores)].copy()
            continue
        
        # Shrink
        for i in range(1, n + 1):
            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
            pts_i = simplex[i].reshape(-1, 2)
            scores[i] = score_func(pts_i)
        iterations += 1
        cur_best = max(scores)
        if cur_best > best_score:
            best_score = cur_best
            best_x = simplex[np.argmax(scores)].copy()
        
        # Check convergence
        if abs(scores[0] - worst_score) < tol:
            break
    
    return best_x.reshape(-1, 2), best_score

def _local_search(pts, score_func, time_limit, start_time):
    """Local search with Nelder-Mead, with time guard."""
    remaining = time_limit - (time.time() - start_time)
    if remaining < 5:
        return pts, score_func(pts)
    
    max_iter = min(3000, int(remaining * 200))  # rough estimate
    pts_opt, score_opt = _nelder_mead_maximize(
        pts, score_func, max_iter=max_iter, 
        time_limit=remaining - 2, start_time=start_time
    )
    return pts_opt, score_opt

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    inits = _generate_initializations()
    
    best_pts = None
    best_score = -1
    
    # Evaluate all initializations
    for i, init in enumerate(inits):
        if time.time() > deadline - 30:
            break
        s = _score(init)
        if s > best_score:
            best_score = s
            best_pts = init.copy()
    
    if best_pts is None:
        # Fallback: regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = _score(best_pts)
    
    # Local search on each initialization (with time guard)
    for i, init in enumerate(inits):
        if time.time() > deadline - 10:
            break
        # Only do local search if this init is promising
        s_init = _score(init)
        if s_init < best_score * 0.5:
            continue
        
        remaining = deadline - 10 - time.time()
        if remaining < 5:
            break
        
        pts_opt, score_opt = _local_search(init, _score, remaining, start_time)
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt
    
    # Final check
    if time.time() > deadline - 2:
        pass
    
    return best_pts

# EVOLVE-BLOCK-END
