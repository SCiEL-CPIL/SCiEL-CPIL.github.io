"""Heilbronn convex regions: maximize min triangle area over convex hull area for 13 points.
Strategy: Symmetry-constrained grid search.
We assume the optimal configuration has a regular k-gon on the boundary and 13-k points 
lying on the symmetry axes (radii) of that k-gon. We search over k, the radii of the 
interior points, and a global angular offset.
"""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from itertools import combinations

_DEADLINE = 145.0  # Safety margin from 150s limit
_START = time.time()

# Pre-compute triangle indices for vectorized area calculation
TRI_INDICES = np.array(list(combinations(range(N), 3)))
NUM_TRI = len(TRI_INDICES)

def _min_triangle_area_exact(pts):
    """Minimum area of any triangle formed by 3 of the points. Vectorized."""
    p_i = pts[TRI_INDICES[:, 0]]
    p_j = pts[TRI_INDICES[:, 1]]
    p_k = pts[TRI_INDICES[:, 2]]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)

def _convex_hull_area(pts):
    """Area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    return _min_triangle_area_exact(pts) / hull_area

def _generate_symmetric_config(k, radii, offset):
    """
    Generate points from a symmetric structure.
    k points on a circle of radius 1 (regular k-gon).
    13-k points on symmetry axes.
    
    For a regular k-gon, symmetry axes are:
    - If k is even: k/2 axes through opposite vertices, k/2 axes through midpoints of opposite edges.
    - If k is odd: k axes through vertices and midpoints of opposite edges.
    
    We place the interior points on a subset of these axes.
    To maximize symmetry and likely optimality, we distribute them evenly among the available axes.
    """
    pts = []
    
    # 1. Boundary points (regular k-gon)
    # Centered at origin, radius 1
    angles_boundary = 2 * np.pi * np.arange(k) / k + offset
    x_b = np.cos(angles_boundary)
    y_b = np.sin(angles_boundary)
    pts.extend(zip(x_b, y_b))
    
    # 2. Interior points
    n_int = N - k
    if n_int > 0:
        # Determine axes of symmetry
        # For simplicity and robustness, we use the axes passing through the vertices 
        # and the midpoints of edges.
        # Actually, for a regular k-gon, the set of symmetry axes is well defined.
        # Let's define the axes angles.
        
        if k % 2 == 0:
            # Axes through vertices: 0, pi/k, 2pi/k ... (k/2 axes)
            # Axes through edge midpoints: pi/(2k), 3pi/(2k) ... (k/2 axes)
            # Total k axes.
            # We can just use the k axes defined by the vertices and edge midpoints.
            # Angle of vertex i: i * 2pi/k
            # Angle of edge midpoint i: (i + 0.5) * 2pi/k
            axis_angles = []
            for i in range(k // 2):
                axis_angles.append(i * 2 * np.pi / k) # Vertex axis
                axis_angles.append((i + 0.5) * 2 * np.pi / k) # Edge axis
            # Sort to be consistent
            axis_angles.sort()
        else:
            # k axes, all passing through a vertex and the midpoint of the opposite edge.
            # Angle of vertex i: i * 2pi/k
            # The axis goes through the vertex and the center.
            axis_angles = [i * 2 * np.pi / k for i in range(k)]
            
        # We have n_int points to place.
        # Strategy: Assign points to axes.
        # To maintain high symmetry, we should place points on axes such that the configuration 
        # retains as much symmetry as possible.
        # A simple heuristic: distribute the n_int points among the first n_int axes? 
        # Or distribute evenly?
        # Given the small number, let's try distributing them evenly among the available axes.
        # If n_int < len(axis_angles), we pick a subset.
        # To break rotational symmetry slightly if needed, we could use different radii, 
        # but the prompt says "radial distances ... of these symmetric layers".
        # This implies we might have multiple "layers" of interior points.
        
        # Let's assume the interior points are placed on the axes closest to the "top" (angle 0+offset) 
        # or distributed evenly.
        # Actually, the most symmetric placement for n_int points on k axes is to pick 
        # n_int axes evenly spaced.
        
        if n_int <= len(axis_angles):
            # Pick n_int axes evenly spaced from the available axes
            indices = np.linspace(0, len(axis_angles) - 1, n_int, dtype=int)
            chosen_axes = [axis_angles[i] for i in indices]
        else:
            # If n_int > k, we must put multiple points on some axes.
            # This is less likely for N=13 and typical k (e.g., k=12, n_int=1).
            # For k=6, n_int=7. 7 points on 6 axes. One axis gets 2, others 1.
            # This breaks symmetry. 
            # For the grid search, we will assume n_int <= k or handle the specific case.
            # For now, just cycle through axes.
            chosen_axes = [axis_angles[i % len(axis_angles)] for i in range(n_int)]

        # Assign radii to these axes
        # radii is a list of length n_int
        for i in range(n_int):
            r = radii[i]
            angle = chosen_axes[i] + offset
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            pts.append((x, y))
            
    return np.array(pts)

def construct():
    global _START, _DEADLINE
    
    best_score = -np.inf
    best_pts = None
    
    # We will search over k (number of boundary points)
    # k must be at least 3.
    # If k is too small, the convex hull area is small relative to the spread of interior points?
    # Actually, the convex hull IS the boundary points if they form a convex polygon containing others.
    # We assume the boundary points form the convex hull.
    
    # Candidate k values:
    # 13 points. 
    # k=13: Regular 13-gon.
    # k=12: 12-gon + 1 center.
    # k=11: 11-gon + 2 interior.
    # k=10: 10-gon + 3 interior.
    # ...
    # k=6: 6-gon + 7 interior.
    
    # Let's prioritize k=12, 13, 11, 10 as they are common in Heilbronn problems.
    # Also k=7, 8, 9.
    
    k_candidates = [13, 12, 11, 10, 9, 8, 7, 6]
    
    for k in k_candidates:
        if time.time() > _DEADLINE - 10:
            break
            
        n_int = N - k
        
        # Grid search parameters:
        # 1. offset: global rotation. Due to symmetry, we only need to search [0, 2pi/k) 
        #    if the interior points break the symmetry. 
        #    If the interior points are placed symmetrically with respect to the k-gon,
        #    the whole configuration has D_k symmetry (or a subgroup).
        #    Actually, if we place interior points on specific axes, the symmetry is reduced.
        #    However, rotating the whole configuration doesn't change the score.
        #    So offset is redundant if we fix the interior points relative to the boundary.
        #    BUT, the "axes" we choose for interior points depend on the offset if we 
        #    define them in absolute terms. 
        #    In _generate_symmetric_config, we add offset to both boundary and interior angles.
        #    So the relative geometry is invariant to offset.
        #    Thus, we can fix offset = 0 for the grid search, unless we want to 
        #    search for "shifted" symmetric configurations.
        #    Wait, if we fix offset=0, we are searching over the space of configurations 
        #    that are symmetric with respect to the axes defined by the regular k-gon 
        #    at offset 0.
        #    Is it possible that the optimal configuration is a regular k-gon rotated 
        #    by an angle that is NOT a multiple of the symmetry step?
        #    No, because the interior points are constrained to the symmetry axes of the 
        #    regular k-gon. If we rotate the k-gon, the axes rotate with it.
        #    The set of all such configurations is invariant under rotation.
        #    So we can fix offset = 0.
        
        offset = 0.0
        
        # 2. radii: The radial distances of the interior points.
        #    We need to search over r_1, ..., r_n_int.
        #    Constraints: 0 <= r_i < 1 (inside the unit circle roughly, but hull is k-gon).
        #    Actually, points must be inside the convex hull.
        #    For a regular k-gon of radius 1, the inradius is cos(pi/k).
        #    So r_i should be <= cos(pi/k) to be strictly inside? 
        #    No, the convex hull is the polygon. The interior of the polygon is the valid region.
        #    The distance from center to edge is cos(pi/k).
        #    So for a point on the axis passing through a vertex, the max radius is 1.
        #    For a point on the axis passing through an edge midpoint, the max radius is cos(pi/k).
        #    My _generate_symmetric_config doesn't distinguish these max radii.
        #    It just places them. If r > max_allowed, the point is outside the hull, 
        #    and the hull area calculation will include it, changing the geometry.
        #    This might lead to invalid "convex region" assumptions if the point pops out.
        #    However, the score function uses ConvexHull(pts), so it will correctly 
        #    calculate the hull of the resulting set.
        #    If a point pops out, the hull becomes larger, and the "boundary" points 
        #    are no longer just the k-gon. This breaks the symmetry assumption.
        #    To avoid this, we should constrain r_i to be safely inside.
        #    Let's set max_r = 0.9 * cos(pi/k) for edge-axes and 0.9 for vertex-axes?
        #    Simpler: just search r in [0, 0.8]. This keeps points well inside.
        
        if n_int == 0:
            # Regular k-gon
            pts = _generate_symmetric_config(k, [], offset)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts
            continue
        
        # Grid search over radii
        # n_int is small (1 to 7).
        # We can use a coarse grid first, then refine.
        
        # Coarse grid
        num_steps = 50
        r_vals = np.linspace(0.0, 0.8, num_steps)
        
        # If n_int is large, 50^n_int is too big.
        # For n_int=1: 50
        # For n_int=2: 2500
        # For n_int=3: 125000 (might be slow, ~125k * cost_score)
        # cost_score is vectorized, ~286 triangles. 
        # 125k * 286 * 10 ops = 350M ops. In Python/NumPy, this might take a few seconds.
        # For n_int=4: 6.25M -> too slow for Python loop.
        # We need a smarter search for n_int >= 3.
        
        # Strategy:
        # If n_int == 1: 1D search.
        # If n_int == 2: 2D search.
        # If n_int >= 3: 
        #    Assume symmetry among interior points? 
        #    If the configuration has high symmetry, maybe all interior points are at the same radius?
        #    Or maybe they form a smaller regular polygon?
        #    Let's try a "layered" approach:
        #    Assume interior points are placed in concentric regular polygons?
        #    No, they are on axes.
        #    
        #    Let's use a random restart + local polish for n_int >= 3.
        #    Or, reduce the grid size.
        
        if n_int == 1:
            for r in r_vals:
                if time.time() > _DEADLINE - 5: break
                pts = _generate_symmetric_config(k, [r], offset)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts
                    # Refine around best
                    # We'll do a local refinement after the coarse pass
            
        elif n_int == 2:
            # 2D grid
            # Use a slightly finer grid if time permits, or coarser if not.
            # 50x50 = 2500 points. Fast.
            for r1 in r_vals:
                if time.time() > _DEADLINE - 5: break
                for r2 in r_vals:
                    pts = _generate_symmetric_config(k, [r1, r2], offset)
                    s = _score(pts)
                    if s > best_score:
                        best_score = s
                        best_pts = pts
                        
        else:
            # n_int >= 3
            # Use random sampling in the box [0, 0.8]^n_int
            # plus structured guesses (e.g., all same radius)
            
            # Structured: all same radius
            for r in r_vals:
                if time.time() > _DEADLINE - 5: break
                radii = [r] * n_int
                pts = _generate_symmetric_config(k, radii, offset)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts
            
            # Random sampling
            num_random = 2000
            rng = np.random.RandomState(42)
            for _ in range(num_random):
                if time.time() > _DEADLINE - 5: break
                radii = rng.uniform(0, 0.8, n_int)
                pts = _generate_symmetric_config(k, radii, offset)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts

    # Local Polish
    # Use Nelder-Mead on the best found configuration to break the grid resolution.
    # This is safe because we start from a good symmetric point.
    
    if best_pts is not None:
        # Flatten
        x0 = best_pts.flatten()
        
        def _neg_score(x):
            pts = x.reshape(N, 2)
            return -_score(pts)
        
        try:
            from scipy.optimize import minimize
            res = minimize(
                _neg_score, 
                x0, 
                method='Nelder-Mead', 
                options={
                    'maxiter': 1000, 
                    'xatol': 1e-8, 
                    'fatol': 1e-8, 
                    'adaptive': True
                }
            )
            if -res.fun > best_score:
                best_score = -res.fun
                best_pts = res.x.reshape(N, 2)
        except Exception:
            pass

    return best_pts
# EVOLVE-BLOCK-END
