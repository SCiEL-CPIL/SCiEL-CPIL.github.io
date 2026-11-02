import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

# Pre-compute triplet indices for vectorized area calculation
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triplets.append((i, j, k))
_triplets = np.array(_triplets)
_i_idx = _triplets[:, 0]
_j_idx = _triplets[:, 1]
_k_idx = _triplets[:, 2]

def _min_triangle_area_bary(bary_pts):
    """
    Compute the minimum area of any triangle formed by three points given in barycentric coordinates.
    bary_pts: (N, 3) array where each row sums to 1 and all elements >= 0.
    """
    # Convert to Cartesian for area calculation
    # V1 = (0,0), V2 = (1,0), V3 = (0.5, sqrt(3)/2)
    # x = bary[1] + 0.5 * bary[2]
    # y = sqrt(3)/2 * bary[2]
    b1 = bary_pts[:, 1]
    b2 = bary_pts[:, 2]
    
    x = b1 + 0.5 * b2
    y = S3H * b2
    pts = np.column_stack((x, y))
    
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_j_idx] # Wait, need p_k
    p_k = pts[_k_idx]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _objective(x):
    """
    Objective function for Nelder-Mead.
    x: (N*3,) flattened barycentric coordinates.
    We want to maximize min area, so we minimize -min_area.
    We add a penalty if points go outside the triangle (bary < 0).
    """
    bary = x.reshape(N, 3)
    
    # Penalty for barycentric coordinates being negative
    # This helps keep the optimizer in the feasible region
    neg = np.minimum(bary, 0)
    penalty = 1000.0 * np.sum(neg**2)
    
    # Normalize barycentric coordinates? 
    # Nelder-Mead doesn't handle constraints well. 
    # If we just pass raw barycentric coords, they might drift away from sum=1.
    # It is better to parameterize such that the sum is always 1.
    # Let's use a different parameterization:
    # Actually, Nelder-Mead on 33 variables is hard.
    # Let's stick to Cartesian but use the "exact projection" idea?
    # No, the prompt asks for a DIFFERENT strategy.
    # Barycentric is good, but we need to ensure sum=1.
    # We can optimize over N*2 variables (x,y) and project to barycentric?
    # Or optimize over N*3 variables and normalize?
    
    # Let's try a different approach for the "Different Strategy":
    # **Simulated Annealing** on Cartesian coordinates with a specialized cooling schedule 
    # and "active set" moves (only moving points involved in the minimum area triangle).
    
    # Wait, I started writing Barycentric. Let's pivot to **Simulated Annealing** with **Active Set** moves.
    # This is materially different from Nelder-Mead/Gradient methods.
    
    # But I already defined the function signature. Let's implement SA inside construct.
    # The _objective above is not used if I switch to SA.
    
    # Let's implement the SA logic directly in construct.
    # I will return 0 here as it's not used in the final chosen strategy to avoid confusion.
    return 0.0 

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        pts[i] = [x, y]
    return pts

def _symmetric_init(rng, noise_scale=0.05):
    """
    Generate a highly symmetric initial configuration.
    3 vertices, 3 edge midpoints, 1 centroid, 4 inner points arranged symmetrically.
    """
    pts = np.zeros((N, 2))
    
    # Vertices
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2] = [0.5, S3H]
    
    # Edge Midpoints
    pts[3] = [0.5, 0.0]
    pts[4] = [0.25, S3H/2]
    pts[5] = [0.75, S3H/2]
    
    # Centroid
    pts[6] = [0.5, S3H/3]
    
    # Inner points: 4 points. 
    # Let's place them symmetrically around the centroid.
    # Distance r from centroid, angles 0, pi/2, pi, 3pi/2? 
    # Or maybe 3-fold symmetry? 
    # Let's try 3 points on a circle around centroid + 1 at centroid? No, centroid is taken.
    # Let's try 4 points: 2 on the vertical axis, 2 off-axis?
    # To maintain 3-fold symmetry (D3), we should have orbits of size 1, 2, or 3.
    # Size 1: Centroid (index 6).
    # Size 3: Vertices (0,1,2), Midpoints (3,4,5).
    # Remaining 2 points (indices 7, 8, 9, 10). 
    # We can have one orbit of size 3 and one point at centroid? No, centroid is fixed.
    # We can have two orbits of size ... wait.
    # 11 points.
    # Orbit 1 (size 1): Centroid.
    # Orbit 2 (size 3): Vertices.
    # Orbit 3 (size 3): Midpoints.
    # Remaining 4 points. 
    # We can have one orbit of size 3 (indices 7,8,9) and one point? 
    # If we add one more point, it must be at the centroid (size 1) or part of a size 3 orbit.
    # If we add a size 3 orbit (indices 7,8,9) and another size 1 orbit (index 10) at centroid? 
    # Two points at centroid? That creates zero area triangles. Bad.
    # So we must have orbits that sum to 4.
    # 4 = 1 + 3? 
    # One point at centroid (size 1) and 3 points in a ring (size 3).
    # But we already have a point at centroid.
    # So maybe: 
    # 1 point at centroid (index 6).
    # 3 points in a ring (indices 7,8,9).
    # 1 point ... where? 
    # If we put the 11th point at the centroid, we have a duplicate.
    # Maybe the "Centroid" point is not exactly at centroid?
    # Let's assume the optimal solution has 3-fold symmetry.
    # Then the points must form orbits under 120-degree rotation.
    # Possible orbit sizes: 1 (center), 3 (ring).
    # 11 = 1 + 3 + 3 + 3 + 1? No, only one center.
    # 11 = 1 + 3 + 3 + 4? No, 4 is not an orbit size.
    # So a strict D3 symmetry with 11 points is impossible if we require distinct points?
    # Wait, 11 is not divisible by 3. 11 = 3*3 + 2.
    # So we can have 3 orbits of size 3 (9 points) and 2 points.
    # The 2 points must be fixed by the rotation? Only the center is fixed.
    # So we can have 1 point at center, and 1 point ... where?
    # If the 2nd point is not at center, it breaks symmetry.
    # So the optimal solution might NOT have full 3-fold symmetry, or I am miscounting.
    # Actually, for N=7, 7 = 1 + 3 + 3. (Center, Ring, Ring).
    # For N=10, 10 = 1 + 3 + 3 + 3.
    # For N=11, 11 = 1 + 3 + 3 + 3 + 1? No.
    # Maybe 11 = 3 + 3 + 3 + 1 + 1? No.
    # It is possible the symmetry is broken.
    
    # Let's just place the remaining 4 points randomly near the centroid or in a ring.
    # Indices 7, 8, 9, 10.
    # Let's place 3 in a ring of radius r around centroid, and 1 at centroid? No.
    # Let's place 4 points in a ring of radius r around centroid? 
    # That breaks 3-fold symmetry (4-fold instead).
    # Let's place 3 in a ring (3-fold) and 1 at the centroid? 
    # If we put 2 points at centroid, area is 0.
    # So we must move the centroid point slightly?
    # Let's just use a generic symmetric-looking init:
    # 3 vertices, 3 midpoints, 1 centroid.
    # 4 points: 3 in a ring around centroid (radius 0.1), 1 at centroid? 
    # Let's offset the centroid point slightly to avoid duplication, or just place the 4th inner point 
    # such that it doesn't duplicate.
    
    # Let's place indices 7,8,9 in a ring of radius 0.15 around centroid.
    # Index 10: place it at the centroid? 
    # If index 6 is centroid, and index 10 is centroid, they are the same point.
    # Let's move index 6 slightly? 
    # Or better: 
    # Indices 7,8,9: Ring radius 0.2.
    # Index 10: Ring radius 0.05? 
    # Let's just scatter them.
    
    cx, cy = 0.5, S3H/3
    r1 = 0.15
    for i in range(3):
        angle = 2 * np.pi * i / 3
        pts[7+i] = [cx + r1 * np.cos(angle), cy + r1 * np.sin(angle)]
    
    # Index 10: place it slightly off-center to avoid exact overlap if needed, 
    # or just at center. If at center, it's a duplicate of index 6.
    # Let's move index 6 (centroid) to be slightly off? 
    # No, let's just place index 10 at a different radius.
    r2 = 0.05
    angle2 = np.pi / 6
    pts[10] = [cx + r2 * np.cos(angle2), cy + r2 * np.sin(angle2)]
    
    # Add noise
    if noise_scale > 0:
        noise = rng.gauss(0, noise_scale, (N, 2))
        pts += noise
        pts = _project_to_triangle(pts)
        
    return pts

def _simulated_annealing(pts, rng, time_limit):
    """
    Simulated Annealing with Active Set moves.
    """
    current_pts = pts.copy()
    current_score = _min_triangle_area(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    # Cooling schedule
    T_start = 0.05
    T_end = 0.0001
    alpha = 0.9999  # Cooling rate
    
    T = T_start
    
    while time.time() < time_limit and T > T_end:
        # Calculate active set (triplets with area close to min)
        p_i = current_pts[_i_idx]
        p_j = current_pts[_j_idx]
        p_k = current_pts[_k_idx]
        dx1 = p_j - p_i
        dx2 = p_k - p_i
        cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        active_thresh = min_area * 1.01
        active_mask = areas < active_thresh
        active_triplets = _triplets[active_mask]
        
        if len(active_triplets) == 0:
            break
            
        # Pick a random active triplet
        tri = active_triplets[rng.randint(0, len(active_triplets)-1)]
        i, j, k = tri
        
        # Pick a random point from the triplet to move
        point_idx = rng.choice([i, j, k])
        orig_p = current_pts[point_idx].copy()
        
        # Generate a random perturbation
        # Step size proportional to temperature
        step_size = T * rng.uniform(0.5, 1.5)
        angle = rng.uniform(0, 2*np.pi)
        dx = step_size * np.cos(angle)
        dy = step_size * np.sin(angle)
        
        new_p = orig_p + np.array([dx, dy])
        
        # Project to triangle
        new_p = _project_to_triangle(np.array([new_p]))[0]
        
        # Calculate new score
        test_pts = current_pts.copy()
        test_pts[point_idx] = new_p
        new_score = _min_triangle_area(test_pts)
        
        delta = new_score - current_score
        
        # Metropolis criterion
        if delta > 0:
            # Always accept improvement
            current_pts = test_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            # Accept worse solution with probability exp(delta / T)
            # Note: delta is negative. We want to maximize, so we look at (new - old).
            # Probability = exp((new - old) / T)
            prob = np.exp(delta / T)
            if rng.random() < prob:
                current_pts = test_pts
                current_score = new_score
        
        # Cool down
        T *= alpha
        
    return best_pts, best_score

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses Simulated Annealing with Active Set moves, starting from symmetric configurations.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1.0
    
    # Generate diverse symmetric initializations
    initializations = []
    
    # 1. Symmetric with small noise
    initializations.append(_symmetric_init(rng, noise_scale=0.01))
    
    # 2. Symmetric with medium noise
    initializations.append(_symmetric_init(rng, noise_scale=0.05))
    
    # 3. Symmetric with large noise
    initializations.append(_symmetric_init(rng, noise_scale=0.1))
    
    # 4. Random
    p_rand = np.array([
        [rng.random(), rng.random() * S3H] for _ in range(N)
    ])
    # Project random points to triangle
    # Simple rejection sampling for random points
    valid_pts = []
    while len(valid_pts) < N:
        x = rng.random()
        y = rng.random() * S3H
        if y <= np.sqrt(3) * x and y <= np.sqrt(3) * (1 - x):
            valid_pts.append([x, y])
    initializations.append(np.array(valid_pts))
    
    # 5. Greedy
    p_greedy = np.zeros((N, 2))
    p_greedy[0] = [0.5, S3H/3] # Start with center
    for i in range(1, N):
        best_pos = None
        best_min_dist = -1
        for _ in range(100):
            x = rng.random()
            y = rng.random() * S3H
            if y <= np.sqrt(3) * x and y <= np.sqrt(3) * (1 - x):
                cand = np.array([x, y])
                dists = np.linalg.norm(p_greedy[:i] - cand, axis=1)
                min_d = np.min(dists)
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_pos = cand
        if best_pos is not None:
            p_greedy[i] = best_pos
    initializations.append(p_greedy)
    
    # Run Simulated Annealing on each initialization
    # We have ~140 seconds. 
    # Let's allocate time to each.
    time_per_init = 25.0
    
    for i, init_pts in enumerate(initializations):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        time_limit = start_time + (i+1) * time_per_init
        if time_limit > DEADLINE - 2:
            time_limit = DEADLINE - 2
            
        try:
            refined_pts, score = _simulated_annealing(init_pts, rng, time_limit)
            if score > best_score:
                best_score = score
                best_pts = refined_pts.copy()
        except Exception as e:
            pass
            
    # If no optimization succeeded, fall back to the best initial guess
    if best_pts is None:
        for init_pts in initializations:
            score = _min_triangle_area(init_pts)
            if score > best_score:
                best_score = score
                best_pts = init_pts.copy()
                
    if best_pts is None:
        best_pts = _project_to_triangle(np.array([
            [rng.random(), rng.random() * S3H] for _ in range(N)
        ]))
        
    # Final check
    if not _check_inside(best_pts):
        best_pts = _project_to_triangle(best_pts)
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")
