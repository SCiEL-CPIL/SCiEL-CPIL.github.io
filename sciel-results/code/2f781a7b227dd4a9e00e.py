"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy: Derivative-Free Active-Set Local Search in barycentric coordinates.
We represent points in barycentric coordinates (u, v, w) where u+v+w=1, u,v,w >= 0.
This naturally enforces the triangle constraint.
The objective is the minimum area of any triangle formed by 3 points.
We use a local search that identifies the "critical" triangles (those with minimum area)
and adjusts points to increase their area, while monitoring for violations of other triangles.
A dynamic step size is used to balance exploration and exploitation.
"""
import numpy as np
import time
import random
from itertools import combinations

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

def _barycentric_to_cartesian(bary_pts):
    """
    Convert barycentric coordinates to Cartesian coordinates.
    bary_pts: (n, 3) array where each row is (u, v, w)
    Vertices of the equilateral triangle:
    A = (0, 0)
    B = (1, 0)
    C = (0.5, sqrt(3)/2)
    P = u*A + v*B + w*C
    """
    n = len(bary_pts)
    cart_pts = np.zeros((n, 2))
    # A=(0,0), B=(1,0), C=(0.5, S3H)
    # x = u*0 + v*1 + w*0.5 = v + 0.5*w
    # y = u*0 + v*0 + w*S3H = w*S3H
    cart_pts[:, 0] = bary_pts[:, 1] + 0.5 * bary_pts[:, 2]
    cart_pts[:, 1] = bary_pts[:, 2] * S3H
    return cart_pts

def _cartesian_to_bary(cart_pts):
    """
    Convert Cartesian coordinates to barycentric coordinates.
    cart_pts: (n, 2) array
    """
    n = len(cart_pts)
    bary_pts = np.zeros((n, 3))
    x = cart_pts[:, 0]
    y = cart_pts[:, 1]
    # y = w * S3H  => w = y / S3H
    w = y / S3H
    # x = v + 0.5 * w => v = x - 0.5 * w
    v = x - 0.5 * w
    # u = 1 - v - w
    u = 1.0 - v - w
    bary_pts[:, 0] = u
    bary_pts[:, 1] = v
    bary_pts[:, 2] = w
    return bary_pts

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(pts)
    min_area = float('inf')
    min_indices = None
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
                    min_indices = (i, j, k)
    return min_area, min_indices

def _all_triangle_areas(pts):
    """Compute areas of all triangles and return them along with indices."""
    n = len(pts)
    areas = []
    indices = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                areas.append(area)
                indices.append((i, j, k))
    return np.array(areas), indices

def _random_point_in_triangle(rng):
    """Generate a random point inside the equilateral triangle in barycentric coords."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    w = 1 - u - v
    return [u, v, w]

def _structured_init(rng):
    """
    Generate a structured initial set of 11 points in barycentric coordinates.
    Includes corners, center, and midpoints to provide a good starting spread.
    """
    pts = []
    # Corners
    pts.append([1.0, 0.0, 0.0])
    pts.append([0.0, 1.0, 0.0])
    pts.append([0.0, 0.0, 1.0])
    # Center
    pts.append([1.0/3, 1.0/3, 1.0/3])
    # Midpoints of sides
    pts.append([0.5, 0.5, 0.0])
    pts.append([0.5, 0.0, 0.5])
    pts.append([0.0, 0.5, 0.5])
    # Centroids of smaller triangles formed by center and midpoints
    # This creates a nice 11-point configuration
    for _ in range(4):
        # Random points near the center to break symmetry and allow optimization
        u = rng.uniform(0.2, 0.8)
        v = rng.uniform(0.2, 0.8)
        if u + v > 1:
            v = 1 - u
        w = 1 - u - v
        pts.append([u, v, w])
    
    return np.array(pts)

def _local_search(bary_pts, rng, time_limit):
    """
    Perform derivative-free local search in barycentric space.
    """
    start_time = time.time()
    step_size = 0.05
    best_area, _ = _min_triangle_area(_barycentric_to_cartesian(bary_pts))
    current_bary = bary_pts.copy()
    current_cart = _barycentric_to_cartesian(current_bary)
    current_area, _ = _min_triangle_area(current_cart)
    
    no_improve_count = 0
    max_no_improve = 10
    
    while time.time() - start_time < time_limit and no_improve_count < max_no_improve:
        # Find the minimum area triangle
        min_area, min_idx = _min_triangle_area(current_cart)
        
        # Try to improve the minimum area triangle
        # We can move one of the three points slightly to increase the area
        # Or move all three
        improved = False
        
        # Strategy: Try moving each point in the min triangle in a random direction
        # that tends to increase the area of that specific triangle, then check global min.
        i, j, k = min_idx
        
        # Calculate the normal vector to the triangle (i,j,k) to determine "outward" direction
        # Actually, simpler: just try small random perturbations to i, j, or k
        # and see if the global minimum increases.
        
        # To be more efficient, we can try to increase the area of (i,j,k) specifically.
        # The area is 0.5 * |cross(pj-pi, pk-pi)|.
        # We can try moving p_i in the direction opposite to the projection of (pj-pi) and (pk-pi).
        
        # For now, let's use a simple random perturbation approach on the active set.
        # Perturb one of the points i, j, or k.
        point_to_move = rng.choice([i, j, k])
        
        # Generate a random perturbation in barycentric space
        # Ensure it stays inside the triangle (barycentric coords sum to 1, >= 0)
        perturbation = np.zeros(3)
        # Random direction in barycentric space
        dir1 = rng.gauss(0, 1)
        dir2 = rng.gauss(0, 1)
        # We need to maintain sum=1. So we can change two coordinates and adjust the third.
        # Or just generate a random valid barycentric point near the current one.
        
        new_bary_point = current_bary[point_to_move].copy()
        # Add small random noise
        new_bary_point += step_size * np.array([rng.gauss(0,1), rng.gauss(0,1), rng.gauss(0,1)])
        # Project back to simplex
        new_bary_point = np.maximum(new_bary_point, 0)
        s = new_bary_point.sum()
        if s > 0:
            new_bary_point /= s
        else:
            new_bary_point = current_bary[point_to_move]
            
        new_bary = current_bary.copy()
        new_bary[point_to_move] = new_bary_point
        new_cart = _barycentric_to_cartesian(new_bary)
        new_min_area, _ = _min_triangle_area(new_cart)
        
        if new_min_area > current_area + 1e-12:
            current_bary = new_bary
            current_cart = new_cart
            current_area = new_min_area
            improved = True
            no_improve_count = 0
            # Increase step size slightly if we improved
            if step_size < 0.1:
                step_size *= 1.05
        else:
            no_improve_count += 1
            # Decrease step size if we didn't improve
            step_size *= 0.9
            
    return current_bary, current_area

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses derivative-free local search in barycentric coordinates with multiple restarts.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_bary = None
    best_area = -float('inf')
    
    # Initial configurations
    initial_configs = []
    
    # 1. Structured start
    initial_configs.append(_structured_init(rng))
    
    # 2. Random starts
    for _ in range(10):
        pts = []
        for _ in range(N):
            pts.append(_random_point_in_triangle(rng))
        initial_configs.append(np.array(pts))
        
    # 3. Greedy construction
    # Start with 3 corners, then add points one by one to maximize the new minimum area
    greedy_pts = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]
    for _ in range(N - 3):
        best_new_pt = None
        best_new_area = -float('inf')
        for _ in range(50): # Sample 50 candidates
            candidate = _random_point_in_triangle(rng)
            temp_pts = greedy_pts + [candidate]
            cart_pts = _barycentric_to_cartesian(np.array(temp_pts))
            area, _ = _min_triangle_area(cart_pts)
            if area > best_new_area:
                best_new_area = area
                best_new_pt = candidate
        if best_new_pt:
            greedy_pts.append(best_new_pt)
    initial_configs.append(np.array(greedy_pts))
    
    # Optimization loop
    for init_bary in initial_configs:
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Ensure feasibility (should be by construction, but just in case)
        init_bary = np.maximum(init_bary, 0)
        init_bary = init_bary / init_bary.sum(axis=1, keepdims=True)
        
        # Run local search with a time budget
        time_per_restart = min(10.0, (DEADLINE - 10 - (time.time() - start_time)) / max(1, len(initial_configs) - len([c for c in initial_configs[:len(initial_configs)] if c is init_bary])) if len(initial_configs) > 0 else 10.0)
        # Simpler time budget: just give each restart a fixed chunk, or use remaining time
        remaining = DEADLINE - 10 - (time.time() - start_time)
        if remaining < 1:
            break
        
        # Allocate time based on number of remaining restarts
        # For simplicity, let's just run for a fixed duration per restart if time allows
        restart_time = min(5.0, remaining / 2) # Heuristic: at least 2 restarts should fit
        
        polished_bary, polished_area = _local_search(init_bary, rng, restart_time)
        
        if polished_area > best_area:
            best_area = polished_area
            best_bary = polished_bary.copy()
            
            # If we found a good solution, try to refine it further with more time
            if best_area > 0.005 and time.time() - start_time < DEADLINE - 20:
                # Run another local search from this point with more time
                refine_time = min(20.0, DEADLINE - 20 - (time.time() - start_time))
                if refine_time > 1:
                    refined_bary, refined_area = _local_search(polished_bary, rng, refine_time)
                    if refined_area > best_area:
                        best_area = refined_area
                        best_bary = refined_bary.copy()

    if best_bary is None:
        # Fallback
        best_bary = _structured_init(rng)
        
    # Convert to Cartesian coordinates for output
    best_cart = _barycentric_to_cartesian(best_bary)
    
    # Final check and projection (should be inside)
    # The barycentric representation ensures points are inside the triangle defined by (0,0), (1,0), (0.5, S3H)
    # Let's double check the area
    final_area, _ = _min_triangle_area(best_cart)
    
    return best_cart
