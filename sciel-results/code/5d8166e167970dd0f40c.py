import numpy as np
import time
import scipy.spatial as spat
from itertools import combinations

N = 13

# Pre-compute triangle indices for N=13
_triple_idx = list(combinations(range(N), 3))
_i_arr = np.array([t[0] for t in _triple_idx])
_j_arr = np.array([t[1] for t in _triple_idx])
_k_arr = np.array([t[2] for t in _triple_idx])
_n_triples = len(_triple_idx)

def _triangle_areas(points):
    """Compute areas of all C(N,3) triangles."""
    pts = points
    v1 = pts[_j_arr] - pts[_i_arr]
    v2 = pts[_k_arr] - pts[_i_arr]
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(det)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def calculate_score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(points) < 3:
        return 0.0
    
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return 0.0

    areas = _triangle_areas(points)
    min_area = np.min(areas)
    
    return min_area / hull_area

def _generate_concentric(radii, angles):
    """
    Generate points from concentric regular polygons.
    radii: list of radii for each ring
    angles: list of angular offsets for each ring
    The number of points in each ring is fixed by the configuration.
    """
    # Configuration: 1 center, 4 inner, 8 outer
    # Total 13 points
    pts = []
    
    # Ring 0: Center (radius 0)
    if radii[0] > 1e-6:
        # If radius is not 0, treat as a ring with 1 point? No, standard is center is 0.
        # But to allow optimization, we might allow the "center" to move? 
        # Actually, a 1-point ring is just a point.
        # Let's stick to:
        # Ring 0: 1 point at (0,0) + offset? No, affine transform handles global shift/rotation.
        # We fix global center at (0,0).
        # So Ring 0 is fixed at (0,0).
        pts.append([0.0, 0.0])
    else:
        pts.append([0.0, 0.0])
        
    # Ring 1: 4 points
    r1 = radii[1]
    a1 = angles[1]
    for i in range(4):
        theta = a1 + 2 * np.pi * i / 4
        pts.append([r1 * np.cos(theta), r1 * np.sin(theta)])
        
    # Ring 2: 8 points
    r2 = radii[2]
    a2 = angles[2]
    for i in range(8):
        theta = a2 + 2 * np.pi * i / 8
        pts.append([r2 * np.cos(theta), r2 * np.sin(theta)])
        
    return np.array(pts)

def _optimize_concentric(radii, angles, time_budget):
    """
    Local optimization of the concentric ring configuration.
    Uses coordinate descent on radii and angles.
    """
    current_r = radii.copy()
    current_a = angles.copy()
    current_pts = _generate_concentric(current_r, current_a)
    current_score = calculate_score(current_pts)
    
    best_pts = current_pts.copy()
    best_score = current_score
    
    # Parameters to optimize:
    # radii[1], radii[2]
    # angles[1], angles[2]
    # (radii[0] and angles[0] are fixed/irrelevant for a single center point)
    
    # Initial step sizes
    step_r = 0.05
    step_a = 0.05
    
    start_time = time.time()
    
    while time.time() - start_time < time_budget:
        improved = False
        
        # Try adjusting radii[1]
        for delta in [step_r, -step_r]:
            trial_r = current_r.copy()
            trial_r[1] += delta
            if trial_r[1] < 0: continue
            trial_pts = _generate_concentric(trial_r, current_a)
            trial_score = calculate_score(trial_pts)
            if trial_score > current_score:
                current_r = trial_r
                current_pts = trial_pts
                current_score = trial_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
        
        # Try adjusting radii[2]
        for delta in [step_r, -step_r]:
            trial_r = current_r.copy()
            trial_r[2] += delta
            if trial_r[2] < 0: continue
            trial_pts = _generate_concentric(trial_r, current_a)
            trial_score = calculate_score(trial_pts)
            if trial_score > current_score:
                current_r = trial_r
                current_pts = trial_pts
                current_score = trial_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
        
        # Try adjusting angles[1]
        for delta in [step_a, -step_a]:
            trial_a = current_a.copy()
            trial_a[1] += delta
            trial_pts = _generate_concentric(current_r, trial_a)
            trial_score = calculate_score(trial_pts)
            if trial_score > current_score:
                current_a = trial_a
                current_pts = trial_pts
                current_score = trial_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
        
        # Try adjusting angles[2]
        for delta in [step_a, -step_a]:
            trial_a = current_a.copy()
            trial_a[2] += delta
            trial_pts = _generate_concentric(current_r, trial_a)
            trial_score = calculate_score(trial_pts)
            if trial_score > current_score:
                current_a = trial_a
                current_pts = trial_pts
                current_score = trial_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_pts = current_pts.copy()
        
        if not improved:
            # Reduce step sizes
            step_r *= 0.8
            step_a *= 0.8
            if step_r < 1e-5:
                break

    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = -1.0
    best_points = None
    
    # Define a few initial concentric configurations
    # Radii: [center, inner_4, outer_8]
    # Angles: [center, inner_4, outer_8]
    
    initial_configs = [
        # 1. Standard symmetric: Inner at 0, Outer at 0 (aligned)
        {
            'radii': [0.0, 0.5, 1.0],
            'angles': [0.0, 0.0, 0.0]
        },
        # 2. Staggered: Inner at 0, Outer at pi/8 (half step)
        {
            'radii': [0.0, 0.5, 1.0],
            'angles': [0.0, 0.0, np.pi / 8]
        },
        # 3. Different radii ratio
        {
            'radii': [0.0, 0.6, 1.0],
            'angles': [0.0, 0.0, np.pi / 8]
        },
        # 4. Another ratio
        {
            'radii': [0.0, 0.4, 1.0],
            'angles': [0.0, 0.0, 0.0]
        },
        # 5. Perturbed standard
        {
            'radii': [0.0, 0.55, 0.95],
            'angles': [0.0, 0.1, 0.1]
        }
    ]
    
    # Score and optimize each
    n_configs = len(initial_configs)
    time_per_config = (deadline - start_time - 5.0) / n_configs
    time_per_config = max(time_per_config, 5.0)
    
    for i, config in enumerate(initial_configs):
        if time.time() > deadline - 5.0:
            break
            
        radii = np.array(config['radii'])
        angles = np.array(config['angles'])
        
        # Initial score
        pts = _generate_concentric(radii, angles)
        s = calculate_score(pts)
        
        if s > best_score:
            best_score = s
            best_points = pts.copy()
            
        # Optimize
        opt_pts, opt_score = _optimize_concentric(radii, angles, time_per_config)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts.copy()
            
    # Fallback: If the symmetric approach fails or yields poor results, 
    # try a generic SA on the full space as a backup, using remaining time
    if best_score < 0.1: # Threshold for "failure"
         if time.time() < deadline - 10.0:
             # Quick SA fallback
             rng = np.random.default_rng(42)
             # Start from best found so far or random
             if best_points is None:
                 angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
                 best_points = np.column_stack((np.cos(angles), np.sin(angles)))
             
             current_pts = best_points.copy()
             current_score = best_score
             
             T = 0.05
             for _ in range(1000):
                 if time.time() > deadline - 2.0:
                     break
                 trial_pts = current_pts + rng.normal(0, 0.05, (N, 2))
                 trial_score = calculate_score(trial_pts)
                 if trial_score > current_score or rng.random() < np.exp((trial_score - current_score) / T):
                     current_pts = trial_pts
                     current_score = trial_score
                     if current_score > best_score:
                         best_score = current_score
                         best_points = current_pts.copy()
                 T *= 0.99

    if best_points is None:
        # Absolute fallback
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
