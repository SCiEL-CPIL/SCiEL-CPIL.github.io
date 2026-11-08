import numpy as np
import time
import math
from itertools import product

N = 11

def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of regular hexagon centered at (cx,cy), side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append([cx + s * np.cos(theta), cy + s * np.sin(theta)])
    return np.array(verts)

def sat_overlap(v1, v2):
    """
    Check if two convex polygons (hexagons) overlap using Separating Axis Theorem.
    v1, v2 are arrays of shape (6, 2).
    Returns True if they overlap (including touching), False if disjoint.
    """
    # Get all edges of both polygons
    # For each edge, compute the normal
    # Project both polygons onto the normal
    # If projections are disjoint, no overlap
    
    axes = []
    for i in range(6):
        # Edge from v1[i] to v1[(i+1)%6]
        p1 = v1[i]
        p2 = v1[(i+1)%6]
        edge = p2 - p1
        # Normal to edge (rotate 90 deg)
        # Normalizing is optional for SAT if we just check sign, but better for consistency
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-9:
            axes.append(normal / norm)
            
    for i in range(6):
        p1 = v2[i]
        p2 = v2[(i+1)%6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-9:
            axes.append(normal / norm)
            
    # Check each axis
    for axis in axes:
        # Project v1
        proj1 = v1 @ axis
        min1, max1 = proj1.min(), proj1.max()
        # Project v2
        proj2 = v2 @ axis
        min2, max2 = proj2.min(), proj2.max()
        
        # Check for separation
        if max1 < min2 or max2 < min1:
            return False
            
    return True

def min_enclosing_side(centers, side=1.0, angle=0.0):
    """
    Calculate the minimum side length of a regular hexagon with given center (0,0 for now, we'll shift later)
    and rotation 'angle' that contains all vertices of hexagons at 'centers'.
    Actually, it's better to compute the required side length for a hexagon centered at (0,0) 
    containing the points. Then we can shift the whole system.
    """
    # Collect all vertices
    all_verts = []
    for cx, cy in centers:
        verts = hex_vertices(cx, cy, side, angle)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # The outer hexagon has 6 face normals.
    # Let the outer hexagon be centered at (0,0) for now.
    # Rotation 'angle' defines the orientation.
    # Normals are at angles: angle + i*pi/3 + pi/6 (for flat-top? No, depends on definition)
    # Let's define outer hexagon vertices at angles theta_i = angle + i*pi/3.
    # Then the face normals are at angles theta_i + pi/6? 
    # For a regular polygon, the normal to the edge between vertex i and i+1 is at angle (theta_i + theta_{i+1})/2 + pi/2?
    # No, simpler:
    # A regular hexagon with circumradius R (side s = R) and center (0,0) can be defined by 6 half-planes.
    # The distance from center to each face is apothem = s * sqrt(3)/2.
    # The normals to the faces are at angles: alpha, alpha+60, ..., alpha+300.
    # We want to find alpha and s such that all points are inside.
    # Point (x,y) is inside if for all i: x*cos(n_i) + y*sin(n_i) <= s*sqrt(3)/2.
    # So s >= max_i (x*cos(n_i) + y*sin(n_i)) / (sqrt(3)/2).
    # To minimize s, we should choose alpha such that the max projection is minimized.
    # This is equivalent to finding the minimum width hexagon enclosing the points.
    
    # We can iterate over possible outer rotations.
    # Since the system is symmetric, we can fix the outer hexagon's rotation to 0 
    # and rotate the inner points, OR fix inner and rotate outer.
    # Let's try rotating the outer hexagon.
    
    best_side = float('inf')
    sqrt3 = math.sqrt(3)
    
    # Try 100 rotations
    for rot_deg in np.linspace(0, 60, 101):
        a = math.radians(rot_deg)
        # Normals for outer hexagon faces
        # If vertices are at a + i*pi/3, faces are perpendicular to normals at a + pi/6 + i*pi/3
        normals = []
        for i in range(6):
            theta = a + math.pi/6 + i * math.pi / 3
            normals.append([math.cos(theta), math.sin(theta)])
        normals = np.array(normals)
        
        # Project all vertices onto normals
        # proj[i, j] = dot(v_i, n_j)
        proj = all_verts @ normals.T
        max_proj = proj.max(axis=0)
        # The required side length is determined by the maximum projection distance
        # s * sqrt(3)/2 >= max_proj
        req_side = max_proj.max() / (sqrt3 / 2)
        
        if req_side < best_side:
            best_side = req_side
            
    return best_side

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin
    
    # Initial configuration: Hex lattice
    sqrt3 = math.sqrt(3)
    hex_positions_axial = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (2, -1), (1, -2), (0, -2)
    ]
    
    centers = []
    for q, r in hex_positions_axial:
        x = 1.5 * q
        y = sqrt3 * (r + 0.5 * q)
        centers.append([x, y])
    centers = np.array(centers)
    
    # Pre-compute vertices for initial state? No, they change.
    
    # Calculate initial cost
    # We need to ensure no overlaps. The lattice has no overlaps.
    # Cost = min_enclosing_side(centers)
    
    current_centers = centers.copy()
    current_side = min_enclosing_side(current_centers)
    
    best_centers = current_centers.copy()
    best_side = current_side
    
    # Simulated Annealing
    T = 0.5 # Initial temperature
    T_min = 1e-4
    alpha = 0.995 # Cooling rate
    steps = 50000
    
    # Pre-calculate normals for inner hexagons for SAT?
    # Inner hexagons are always axis-aligned (angle=0).
    # So their edge normals are fixed.
    # Edge normals for unit hex at (0,0):
    # Vertices: (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # Edges:
    # 1. (1,0) to (0.5, sqrt(3)/2): vec (-0.5, sqrt(3)/2). Normal (sqrt(3)/2, 0.5) -> (sqrt(3), 1)/2
    # 2. (0.5, sqrt(3)/2) to (-0.5, sqrt(3)/2): vec (-1, 0). Normal (0, 1)
    # 3. (-0.5, sqrt(3)/2) to (-1, 0): vec (-0.5, -sqrt(3)/2). Normal (-sqrt(3)/2, 0.5) -> (-sqrt(3), 1)/2
    # 4. (-1,0) to (-0.5, -sqrt(3)/2): vec (0.5, -sqrt(3)/2). Normal (sqrt(3)/2, 0.5) -> (sqrt(3), 1)/2 ... wait
    # Let's just use the 6 unique normals for a unit hexagon.
    # Normals are at angles 30, 90, 150, 210, 270, 330 degrees.
    inner_normals = []
    for i in range(6):
        theta = math.pi/6 + i * math.pi / 3
        inner_normals.append([math.cos(theta), math.sin(theta)])
    inner_normals = np.array(inner_normals)
    
    # Helper to check overlap between two centers with unit hexagons
    def check_overlap(c1, c2):
        # SAT on relative positions
        # Project c1 and c2 onto inner_normals
        # Actually, since hexagons are identical and axis-aligned,
        # we can just check if the distance vector d = c1 - c2 satisfies:
        # |d . n_i| <= apothem + apothem = sqrt(3) for all i?
        # No, that's for Minkowski sum.
        # The Minkowski sum of two unit hexagons is a hexagon of side 2.
        # So c1 and c2 do not overlap if c1 - c2 is OUTSIDE the Minkowski sum hexagon.
        # The Minkowski sum hexagon has apothem = 2 * (sqrt(3)/2) = sqrt(3).
        # So overlap if max_i | (c1-c2) . n_i | < sqrt(3) - epsilon?
        # Let's verify.
        # Distance between centers of touching hexagons is sqrt(3).
        # Projection of vector of length sqrt(3) along normal is sqrt(3).
        # So if max projection < sqrt(3), they overlap.
        # If max projection >= sqrt(3), they are disjoint (or touching).
        
        d = c1 - c2
        projs = np.abs(d @ inner_normals.T)
        max_proj = projs.max()
        # If max_proj < sqrt(3), they overlap.
        # Note: This assumes the normals are correct for the "width" in that direction.
        # Yes, for a convex polygon, the width in direction n is max(v.n) - min(v.n).
        # For two polygons P and Q, they are disjoint if there exists a direction n such that
        # max(P.n) < min(Q.n) or max(Q.n) < min(P.n).
        # This is equivalent to max((P-Q).n) < 0 or min((P-Q).n) > 0.
        # (P-Q) is the Minkowski difference, which for identical convex shapes is the Minkowski sum of P and -Q.
        # For regular hexagons, -Q is a rotated Q. The Minkowski sum of two hexagons is a hexagon.
        # The condition "disjoint" is that the center difference vector lies OUTSIDE the Minkowski sum of one hexagon and the reflection of the other.
        # Since hexagons are centrally symmetric, -Q is congruent to Q.
        # So we check if d is outside the hexagon defined by apothem sqrt(3).
        
        return max_proj < sqrt3 - 1e-9
        
    # Optimize
    for step in range(steps):
        if time.time() > deadline:
            break
            
        # Generate new candidate
        new_centers = current_centers.copy()
        # Perturb one or two centers
        idx = np.random.randint(N)
        angle = np.random.uniform(0, 2*math.pi)
        magnitude = np.random.uniform(0, T)
        new_centers[idx, 0] += magnitude * math.cos(angle)
        new_centers[idx, 1] += magnitude * math.sin(angle)
        
        # Check validity (overlaps)
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if check_overlap(new_centers[i], new_centers[j]):
                    valid = False
                    break
            if not valid:
                break
                
        if not valid:
            continue
            
        # Calculate new cost
        new_side = min_enclosing_side(new_centers)
        
        # Acceptance probability
        delta = new_side - current_side
        if delta < 0:
            current_centers = new_centers
            current_side = new_side
            if new_side < best_side:
                best_side = new_side
                best_centers = new_centers.copy()
        else:
            prob = math.exp(-delta / T)
            if np.random.random() < prob:
                current_centers = new_centers
                current_side = new_side
                
        # Cool
        T = max(T * alpha, T_min)
        
    # Final polish: optimize outer hexagon rotation more precisely for best_centers
    # min_enclosing_side already does this, but we can do a finer search.
    final_side = best_side
    
    # Construct return values
    inner = [[float(cx), float(cy), 0.0] for cx, cy in best_centers]
    
    # Find best outer hexagon center and rotation
    # We used (0,0) as center in min_enclosing_side.
    # We should center the outer hexagon at the centroid of the inner vertices to minimize side?
    # Actually, min_enclosing_side calculated the side required for a hexagon at (0,0).
    # If we shift the outer hexagon, the side length might change?
    # No, the minimal enclosing hexagon side length is invariant to translation of the set, 
    # but the *center* of the minimal hexagon is specific.
    # My min_enclosing_side function assumes the outer hexagon is at (0,0).
    # This is a constraint. We should optimize the center of the outer hexagon too.
    # However, for a set of points, the minimal width in 6 directions is what matters.
    # The side length is determined by the maximum width.
    # The center of the minimal hexagon is the intersection of the midlines of the supporting lines.
    # For now, let's assume the center is the mean of the vertices.
    
    all_verts = []
    for cx, cy in best_centers:
        verts = hex_vertices(cx, cy, 1.0, 0.0)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    center_x = all_verts[:, 0].mean()
    center_y = all_verts[:, 1].mean()
    
    # Re-calculate side with this center?
    # The side length calculated by min_enclosing_side(centers) is the correct minimal side length 
    # for a hexagon that *can* be placed anywhere. 
    # Wait, min_enclosing_side(centers) calculates the side length of a hexagon at (0,0) containing the points.
    # If the points are not centered at (0,0), this will be larger than necessary.
    # I need to modify min_enclosing_side to also optimize the center, or just use the center (0,0) 
    # and shift the points to be centered at (0,0) before calling it?
    # Yes, shift points to origin, calculate side, then shift back.
    
    # Let's do a final precise calculation for the best_centers
    # 1. Compute centroid of vertices
    # 2. Shift vertices to centroid
    # 3. Find min side for hexagon at (0,0) containing shifted vertices
    # 4. The outer hexagon center is the centroid.
    
    cx_mean = all_verts[:, 0].mean()
    cy_mean = all_verts[:, 1].mean()
    
    shifted_verts = all_verts - np.array([cx_mean, cy_mean])
    
    # Find min side for shifted_verts
    best_side_final = float('inf')
    best_angle_final = 0.0
    sqrt3 = math.sqrt(3)
    
    for rot_deg in np.linspace(0, 60, 201):
        a = math.radians(rot_deg)
        normals = []
        for i in range(6):
            theta = a + math.pi/6 + i * math.pi / 3
            normals.append([math.cos(theta), math.sin(theta)])
        normals = np.array(normals)
        
        proj = shifted_verts @ normals.T
        max_proj = proj.max(axis=0)
        req_side = max_proj.max() / (sqrt3 / 2)
        
        if req_side < best_side_final:
            best_side_final = req_side
            best_angle_final = rot_deg
            
    # Add safety margin
    best_side_final *= 1.001
    
    return inner, (float(cx_mean), float(cy_mean)), float(best_side_final), float(best_angle_final)
