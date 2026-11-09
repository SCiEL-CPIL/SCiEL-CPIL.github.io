"""Rewritten program: circle-packing heuristic for unit hexagons in a hexagon."""
import numpy as np
from scipy.optimize import minimize
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Place N unit regular hexagons (side length 1) inside a regular hexagon,
    minimizing the outer side length. Each inner hexagon may translate and rotate.
    
    Approach: Model each unit hexagon by its circumcircle (radius 1).
    Approximate the outer hexagon by its circumscribed circle (radius = side_length).
    Then solve a circle-packing problem: minimize outer radius R such that all
    inner circles of radius 1 are inside the outer circle of radius R and 
    non-overlapping.
    
    The actual constraint for hexagons is slightly different, but using circumcircles
    gives a valid upper bound on the required outer side length (since a hexagon 
    fits in its circumcircle, and if circles fit in a circle, hexagons fit in 
    the corresponding hexagon if we use the circumscribed circle of the outer hexagon 
    as the boundary). However, we need to be more careful: the outer hexagon's 
    incircle radius is R_outer * sqrt(3)/2, and its circumcircle radius is R_outer.
    A unit hexagon has circumradius 1 and inradius sqrt(3)/2.
    
    For two unit hexagons to not overlap, their centers must be at least 2*sqrt(3)/2 
    = sqrt(3) apart? No, that's for inradii. Actually, the minimum distance between 
    centers of two non-overlapping unit regular hexagons depends on orientation.
    The worst case (same orientation) requires center distance >= 2 (since circumradius 
    is 1, two circles of radius 1 touching have centers 2 apart). But with rotation, 
    you might get closer? No, the convex hull of a unit hexagon has circumradius 1, 
    so any two non-overlapping unit hexagons must have centers at least 2 apart? 
    Actually no. Two unit hexagons can be closer than 2 if rotated. The minimum 
    center distance for non-overlapping unit regular hexagons is sqrt(3) ≈ 1.732 
    (when one is rotated 30° relative to the other, the inradii touch).
    
    To be safe and simple, let's use the circumcircle model: each hexagon is approximated 
    by a circle of radius 1 (its circumcircle). Two such circles must not overlap, 
    so centers must be >= 2 apart. The outer hexagon is approximated by its circumscribed 
    circle of radius R_outer. So all inner circle centers must be within distance 
    R_outer - 1 from the outer center.
    
    This gives a valid solution (if circles fit, hexagons fit in the circumscribed 
    circle of the outer hexagon, which is inside the outer hexagon? No, the circumscribed 
    circle of the outer hexagon contains the outer hexagon. So if inner hexagons are 
    inside their circumcircles, and those circumcircles are inside the outer circumscribed 
    circle, then the inner hexagons are inside the outer circumscribed circle, but 
    not necessarily inside the outer hexagon.
    
    Let me reconsider. The outer regular hexagon with side length S has:
    - Circumradius: S
    - Inradius: S * sqrt(3)/2
    
    A point is inside the outer hexagon iff its distance from center is <= S in the 
    directions of the vertices, and <= S*sqrt(3)/2 in the directions of the edges.
    
    If we place inner hexagons such that their circumcircles (radius 1) are inside 
    the outer hexagon's incircle (radius S*sqrt(3)/2), then they are definitely inside.
    So we need: distance from outer center to inner center + 1 <= S*sqrt(3)/2.
    Thus: S >= (distance + 1) / (sqrt(3)/2) = 2*(distance + 1)/sqrt(3).
    
    And for non-overlap of inner hexagons, using the circumcircle model, we need 
    center distance >= 2.
    
    So the problem reduces to: place N points inside a circle of radius r_max = S*sqrt(3)/2 - 1,
    with pairwise distances >= 2, minimizing r_max, then S = (r_max + 1) * 2/sqrt(3).
    
    Actually, let's just minimize S directly. Let the outer center be at origin WLOG.
    Each inner hexagon i has center p_i and we require:
    1. ||p_i|| <= S*sqrt(3)/2 - 1  (so circumcircle is inside incircle of outer hex)
    2. ||p_i - p_j|| >= 2 for i != j  (circumcircles don't overlap)
    
    Minimize S.
    """
    # Use a force-directed / optimization approach
    # Initial guess: arrange in a hexagonal lattice pattern
    
    # For N=11, a good initial arrangement is a central hexagon surrounded by 6,
    # plus 4 more. Let's use a hexagonal lattice.
    
    # Hexagonal lattice spacing for non-overlapping circles of radius 1:
    # Nearest neighbor distance = 2.
    # Lattice vectors: (2, 0) and (1, sqrt(3))
    
    a = 2.0  # nearest neighbor distance
    sqrt3 = math.sqrt(3)
    
    # Generate hexagonal lattice points
    # Center at (0,0), then rings
    def hex_lattice_points(n):
        """Generate n points in hexagonal lattice order."""
        points = [(0.0, 0.0)]
        if n <= 1:
            return points
        
        ring = 1
        while len(points) < n:
            # Ring of hexagonal lattice
            for i in range(6):
                angle = i * math.pi / 3
                # Six sides of the ring
                for j in range(ring):
                    if len(points) >= n:
                        break
                    # Position on ring i
                    # Starting from vertex, going along edge
                    if i == 0:
                        x = a * ring
                        y = 0
                    elif i == 1:
                        x = a * ring * math.cos(math.pi/3)
                        y = a * ring * math.sin(math.pi/3)
                    elif i == 2:
                        x = 0
                        y = a * ring
                    elif i == 3:
                        x = -a * ring * math.cos(math.pi/3)
                        y = a * ring * math.sin(math.pi/3)
                    elif i == 4:
                        x = -a * ring
                        y = 0
                    else:
                        x = -a * ring * math.cos(math.pi/3)
                        y = -a * ring * math.sin(math.pi/3)
                    
                    # Move along the edge
                    # Edge direction for side i
                    if i == 0:
                        dx, dy = 0, -a * sqrt3 / 2
                    elif i == 1:
                        dx, dy = -a/2, -a * sqrt3 / 2
                    elif i == 2:
                        dx, dy = -a/2, a * sqrt3 / 2
                    elif i == 3:
                        dx, dy = 0, a * sqrt3 / 2
                    elif i == 4:
                        dx, dy = a/2, a * sqrt3 / 2
                    else:
                        dx, dy = a/2, -a * sqrt3 / 2
                    
                    px = x + j * dx
                    py = y + j * dy
                    points.append((px, py))
                if len(points) >= n:
                    break
            ring += 1
            if ring > 10:  # safety
                break
        return points[:n]
    
    # Actually, let me use a simpler initialization: grid search over a small region
    # and use scipy.optimize to refine.
    
    # Let's try a different approach: use a known packing configuration.
    # For 11 circles of radius 1 in a circle, the optimal packing has radius ~3.4 (approx).
    # Let's just use optimization.
    
    def objective(x):
        """x: 2*N array of center coordinates. Minimize max distance + 1, 
        subject to pairwise distances >= 2."""
        centers = x.reshape(N, 2)
        # Compute max distance from origin
        dists = np.linalg.norm(centers, axis=1)
        max_dist = np.max(dists)
        
        # Penalty for overlapping circles
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < 2.0:
                    penalty += (2.0 - d) ** 2
        
        # Also penalty if any circle goes too far (shouldn't happen if we minimize max_dist)
        return max_dist + 100.0 * penalty
    
    # Initial guess: hexagonal lattice
    # For N=11: center + 6 in first ring + 4 in second ring
    # Let's place them manually for a good start
    
    # Better: use a random restart + optimization
    best_S = float('inf')
    best_centers = None
    
    np.random.seed(42)
    
    for restart in range(5):
        # Random initial positions in a disk of radius 5
        if restart == 0:
            # Hexagonal lattice initial
            centers_init = []
            # Center
            centers_init.append([0.0, 0.0])
            # Ring 1: 6 points at distance 2
            for k in range(6):
                angle = k * math.pi / 3
                centers_init.append([2.0 * math.cos(angle), 2.0 * math.sin(angle)])
            # Ring 2: 4 more points
            # Place at distance ~3.46 (2*sqrt(3)) at angles 30, 90, 150, 210
            for angle_deg in [30, 90, 150, 210]:
                angle = math.radians(angle_deg)
                r = 2.0 * sqrt3
                centers_init.append([r * math.cos(angle), r * math.sin(angle)])
            x0 = np.array(centers_init).flatten()
        else:
            # Random
            angles = np.random.uniform(0, 2*math.pi, N)
            radii = np.random.uniform(0, 4, N)
            x0 = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]).flatten()
        
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 500, 'ftol': 1e-8})
        
        if result.success or result.fun < best_S:
            centers = result.x.reshape(N, 2)
            max_dist = np.max(np.linalg.norm(centers, axis=1))
            # Check pairwise distances
            min_pair_dist = float('inf')
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(centers[i] - centers[j])
                    min_pair_dist = min(min_pair_dist, d)
            
            if min_pair_dist >= 1.999:  # Allow small numerical error
                S_candidate = (max_dist + 1.0) * 2.0 / sqrt3
                if S_candidate < best_S:
                    best_S = S_candidate
                    best_centers = centers
    
    if best_centers is None:
        # Fallback: simple linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Convert to output format
    # All hexagons have angle 0 (no rotation needed in our model)
    inner = []
    for i in range(N):
        x, y = best_centers[i]
        inner.append([float(x), float(y), 0.0])
    
    outer_center = (0.0, 0.0)
    side_length = float(best_S)
    outer_angle = 0.0
    
    return inner, outer_center, side_length, outer_angle

# EVOLVE-BLOCK-END
