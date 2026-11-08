"""Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle.
Strategy: Critical Pair Coordinate Descent.
Instead of smooth approximations, we directly manipulate points.
We identify the triangle with the minimum area (the 'critical' triangle).
We then attempt to move one of its vertices to increase this area, 
while ensuring no other triangle becomes smaller than the current target.
This is a form of active set method / coordinate descent on the combinatorial structure.
"""
import numpy as np
import random
import time
from itertools import combinations

N = 11

# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
AREA = np.sqrt(3) / 4.0  # area of the equilateral triangle

def _area(p1, p2, p3):
    """Calculate area of triangle formed by 3 points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points, normalized by AREA."""
    n = len(points)
    min_area = float('inf')
    min_idx = (0, 1, 2)
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                a = _area(pi, pj, pk)
                if a < min_area:
                    min_area = a
                    min_idx = (i, j, k)
    return (min_area / AREA, min_idx)

def _is_inside_triangle(p):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    if y < -1e-9:
        return False
    if x < -1e-9 or x > 1.0 + 1e-9:
        return False
    max_y = np.sqrt(3) * x if x <= 0.5 else np.sqrt(3) * (1.0 - x)
    return y <= max_y + 1e-9

def _clip_to_triangle(p):
    """Clip a point to be inside the triangle."""
    if not _is_inside_triangle(p):
        # Simple projection or rejection. For coordinate descent, we just move within bounds.
        # This is a rough clip. Better to constrain steps.
        x, y = p
        if y < 0: y = 0
        if x < 0: x = 0
        if x > 1: x = 1
        max_y = np.sqrt(3) * x if x <= 0.5 else np.sqrt(3) * (1.0 - x)
        if y > max_y: y = max_y
        p = np.array([x, y])
    return p

def _generate_random_points(rng, n=N):
    """Generate n random points inside the triangle using rejection sampling."""
    points = []
    while len(points) < n:
        u = rng.random()
        v = rng.random()
        if u + v <= 1.0:
            p = V0 * (1 - u - v) + V1 * u + V2 * v
            points.append(p)
    return np.array(points)

def _generate_structured_initial():
    """Generate a structured initial configuration based on known good patterns."""
    # Vertices, midpoints, centroid, etc.
    pts = [
        V0 + np.array([0.05, 0.05]),
        V1 + np.array([-0.05, 0.05]),
        V2 + np.array([0.0, -0.05]),
        (V0 + V1) / 2.0,
        (V1 + V2) / 2.0,
        (V2 + V0) / 2.0,
        (V0 + V1 + V2) / 3.0,
        (V0 + 2*V1 + V2) / 4.0,
        (2*V0 + V1 + V2) / 4.0,
        (V0 + V1 + 2*V2) / 4.0,
        (2*V0 + 2*V1 + V2) / 5.0
    ]
    return np.array(pts)

def _optimize_point(points, idx, target_min_area, rng, max_iters=1000):
    """
    Try to improve the position of points[idx] such that the global min area 
    is at least target_min_area. 
    Returns the new position of points[idx] or None if failed to improve.
    """
    n = len(points)
    current_p = points[idx].copy()
    best_p = current_p.copy()
    best_gain = 0.0
    
    # We want to move this point to increase the minimum area of triangles involving it.
    # The limiting triangles are those with area close to the global min.
    # Strategy: Random perturbation + local search around the point.
    
    # First, identify triangles involving idx that are "tight"
    # Actually, simpler: just do a local random search / hill climb on this single point.
    
    step_size = 0.05
    for _ in range(max_iters):
        # Generate a candidate move
        # Bias moves towards directions that increase area of tight triangles
        # For simplicity, use random direction with decaying step size
        
        angle = rng.random() * 2 * np.pi
        dist = step_size * rng.random()
        delta = np.array([dist * np.cos(angle), dist * np.sin(angle)])
        
        new_p = current_p + delta
        new_p = _clip_to_triangle(new_p)
        
        # Check if this move improves the minimum area involving idx
        # And doesn't break other triangles (though we only check global min in main loop)
        
        # Calculate min area of triangles involving idx with new_p
        min_area_involving_idx = float('inf')
        for i in range(n):
            if i == idx: continue
            for j in range(i + 1, n):
                if j == idx: continue
                # Triangle (i, j, idx)
                a = _area(points[i], points[j], new_p)
                if a < min_area_involving_idx:
                    min_area_involving_idx = a
        
        # We accept if min_area_involving_idx > target_min_area
        # Or if it's significantly better than current min_area_involving_idx
        # Let's just accept if it increases the min area involving this point
        
        # To be safe, we should check if global min improves.
        # But that's expensive. Let's assume local improvement helps.
        # Actually, let's just check if min_area_involving_idx > current_min_area_involving_idx
        
        # We need current_min_area_involving_idx
        # Let's compute it once outside or pass it.
        # For now, let's just do a simple hill climb: accept if min_area_involving_idx > best_gain
        # Wait, best_gain should be relative to the *global* minimum? No, just local.
        
        # Let's refine: Accept if min_area_involving_idx > current_min_area_involving_idx
        # We need current_min_area_involving_idx
        current_min_area_involving_idx = float('inf')
        for i in range(n):
            if i == idx: continue
            for j in range(i + 1, n):
                if j == idx: continue
                a = _area(points[i], points[j], current_p)
                if a < current_min_area_involving_idx:
                    current_min_area_involving_idx = a
        
        if min_area_involving_idx > current_min_area_involving_idx:
            current_p = new_p
            best_p = new_p.copy()
            step_size *= 0.95 # Shrink step
        else:
            step_size *= 0.99
    
    return best_p

def _coordinate_descent(points, rng, max_rounds=50, time_limit=None, start_time=None):
    """
    Perform coordinate descent on the points.
    """
    if time_limit is None:
        time_limit = 140.0
    if start_time is None:
        start_time = time.time()
        
    current_min, _ = _min_triangle_area(points)
    
    for round_num in range(max_rounds):
        if time.time() > time_limit:
            break
            
        improved = False
        # Shuffle indices to avoid bias
        indices = list(range(N))
        rng.shuffle(indices)
        
        for idx in indices:
            if time.time() > time_limit:
                break
                
            # Get current min area
            current_min, _ = _min_triangle_area(points)
            
            # Try to improve point idx
            new_p = _optimize_point(points, idx, current_min * AREA, rng, max_iters=200)
            
            if new_p is not None:
                old_p = points[idx].copy()
                points[idx] = new_p
                
                # Check if global min improved
                new_min, _ = _min_triangle_area(points)
                
                if new_min > current_min + 1e-10:
                    current_min = new_min
                    improved = True
                else:
                    # Revert if no improvement
                    points[idx] = old_p
        
        if not improved:
            # No improvement in this round, try a small random perturbation to escape local minima
            # Perturb the point with the smallest "slack" or just a random one
            idx = rng.randint(0, N - 1)
            step_size = 0.01
            angle = rng.random() * 2 * np.pi
            delta = np.array([step_size * np.cos(angle), step_size * np.sin(angle)])
            new_p = points[idx] + delta
            new_p = _clip_to_triangle(new_p)
            points[idx] = new_p
            # Don't check improvement here, let next round handle it
            # Or check and revert if bad
            new_min, _ = _min_triangle_area(points)
            if new_min < current_min:
                points[idx] = points[idx] - delta # Revert roughly
                points[idx] = _clip_to_triangle(points[idx])

    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s
    
    best_points = None
    best_min = 0.0
    
    rng = random.Random(42)
    
    # Strategy: Multi-start Coordinate Descent
    # 1. Start with structured initial
    # 2. Start with several random initializations
    
    initial_configs = []
    initial_configs.append(_generate_structured_initial())
    
    # Add random starts
    for _ in range(10):
        initial_configs.append(_generate_random_points(rng))
    
    # Run Coordinate Descent on each
    for idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline:
            break
            
        # Coordinate descent
        # Allocate time based on start
        time_per_start = 10 if idx == 0 else 5
        t_limit = start_time + (deadline - start_time) * (1 - idx / (len(initial_configs) + 1))
        
        try:
            opt_pts = _coordinate_descent(init_pts.copy(), rng, max_rounds=100, time_limit=t_limit, start_time=start_time)
            current_min, _ = _min_triangle_area(opt_pts)
            
            if current_min > best_min:
                best_min = current_min
                best_points = opt_pts.copy()
        except Exception as e:
            continue

    # If no valid points found (should not happen), fallback
    if best_points is None:
        best_points = _generate_random_points(rng)

    return best_points
