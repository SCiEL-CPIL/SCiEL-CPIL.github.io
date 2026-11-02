"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
import random

N = 12
DEADLINE = 140.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return np.array(verts)

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    a = math.radians(angle_deg)
    # Transform to hexagon's local frame
    dx, dy = px - cx, py - cy
    # Rotate by -angle
    ca, sa = math.cos(-a), math.sin(-a)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    # For a regular hexagon with flat-top orientation (angle=0 means vertex at right),
    # the hexagon has vertices at angles 0, 60, 120, 180, 240, 300
    # In local frame, the hexagon is defined by |x| <= side and constraints on y
    # Actually for a pointy-top vs flat-top... let's use the standard form.
    # A regular hexagon with side s, centered at origin, with one vertex at (s, 0):
    # Vertices: (s,0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s,0), (-s/2,-s*sqrt(3)/2), (s/2,-s*sqrt(3)/2)
    # The hexagon is the intersection of 6 half-planes.
    # For vertex at angle 0: the hexagon has "flat" sides perpendicular to the directions 30, 90, 150, 210, 270, 330
    # The distance from center to each flat side is s * sqrt(3)/2
    # The 6 normal directions (pointing outward) are at angles 30, 90, 150, 210, 270, 330
    # So: dot((lx,ly), n_i) <= s*sqrt(3)/2 for each normal
    sqrt3 = math.sqrt(3)
    inradius = side * sqrt3 / 2.0
    for j in range(6):
        na = math.radians(30 + j * 60)
        nx, ny = math.cos(na), math.sin(na)
        if lx * nx + ly * ny > inradius:
            return False
    return True

def _hexagon_overlap_penalty(h1_verts, h2_verts, margin=0.01):
    """
    Compute a penalty if two hexagons overlap.
    Uses support function / SAT-like approach.
    Returns 0 if separated, positive penalty if overlapping.
    """
    # For convex polygons, check separation along all edge normals
    all_verts = np.vstack([h1_verts, h2_verts])
    # Get all edge normals from both polygons
    edges = []
    for i in range(6):
        v1 = h1_verts[i]
        v2 = h1_verts[(i + 1) % 6]
        edges.append(v2 - v1)
    for i in range(6):
        v1 = h2_verts[i]
        v2 = h2_verts[(i + 1) % 6]
        edges.append(v2 - v1)
    
    min_separation = float('inf')
    for edge in edges:
        # Normal perpendicular to edge
        ex, ey = edge
        length = math.sqrt(ex*ex + ey*ey)
        if length < 1e-12:
            continue
        nx, ny = -ey / length, ex / length
        # Project all vertices onto this normal
        proj1 = h1_verts @ np.array([nx, ny])
        proj2 = h2_verts @ np.array([nx, ny])
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        # Separation: if max1 < min2 or max2 < min1, they're separated
        sep = min(max1, max2) - max(min1, min2)
        if sep < min_separation:
            min_separation = sep
    
    # If min_separation < -margin, they overlap by (margin - min_separation)
    if min_separation < -margin:
        return (margin - min_separation) ** 2
    elif min_separation < 0:
        return min_separation ** 2
    else:
        return 0.0

def _min_enclosing_hexagon_side(verts):
    """
    Given a set of 2D points, find the minimum side length of a regular hexagon 
    that can contain all points, and return the center and angle.
    We optimize over center (cx, cy) and angle.
    For a given center and angle, the required side length is determined by the 
    maximum "support" in the 6 hexagon normal directions.
    """
    # The hexagon with center c, angle a, side s contains point p iff:
    # For each of 6 outward normals n_j (at angles a + 30 + j*60):
    #   dot(p - c, n_j) <= s * sqrt(3)/2
    # So s >= max_j dot(p - c, n_j) / (sqrt(3)/2) for all points p
    # And also s >= max_j dot(c - p, n_j) / (sqrt(3)/2) ... wait no.
    # Actually: dot(p - c, n_j) <= inradius = s * sqrt(3)/2
    # So s >= max over j, p of dot(p - c, n_j) / (sqrt(3)/2)
    # But we also need dot(c - p, -n_j) <= inradius which is the same thing.
    # Wait, the hexagon is symmetric, so we need |dot(p - c, n_j)| <= inradius for all j?
    # No! The hexagon is defined by 6 half-planes: dot(p - c, n_j) <= inradius for j=0..5
    # where n_j are the 6 outward normals. This automatically ensures containment.
    # So for given c and angle a:
    #   s_min = max over j=0..5, p in verts of dot(p - c, n_j(a)) / (sqrt(3)/2)
    
    best_side = float('inf')
    best_c = (0, 0)
    best_a = 0.0
    
    # Try various angles (hexagon has 6-fold symmetry, so we only need 0 to 60 degrees)
    # But the inner hexagons can be at any angle, so the enclosing hexagon angle matters
    for a_deg in np.linspace(0, 60, 13):  # 13 angles
        a = math.radians(a_deg)
        normals = []
        for j in range(6):
            na = a + math.radians(30 + j * 60)
            normals.append((math.cos(na), math.sin(na)))
        normals = np.array(normals)
        
        # For a given angle, we need to find center c that minimizes:
        #   max_j max_p dot(p - c, n_j) = max_j (max_p dot(p, n_j) - dot(c, n_j))
        # = max_j (max_p dot(p, n_j)) - min_j dot(c, n_j) ... no
        # Actually: max_j [max_p dot(p, n_j) - dot(c, n_j)]
        # Let M_j = max_p dot(p, n_j)
        # We want to minimize max_j (M_j - dot(c, n_j))
        # This is: minimize t such that M_j - dot(c, n_j) <= t for all j
        # i.e., dot(c, n_j) >= M_j - t for all j
        # This is a linear programming problem in (c_x, c_y, t)
        # Variables: c_x, c_y, t (3 vars)
        # Constraints: for each j: -n_jx * c_x - n_jy * c_y + t >= M_j
        #   i.e., -n_jx * c_x - n_jy * c_y + t - M_j >= 0
        # Minimize t
        
        # Let's use scipy's linprog or just do a simple approach
        # Since we have 6 constraints and 3 variables, we can solve this analytically
        # or use a simple iterative method
        
        M = np.array([np.max(verts @ np.array([nx, ny])) for nx, ny in normals])
        
        # Solve: minimize t s.t. n_j · c <= M_j - t for all j
        # Equivalently: t >= M_j - n_j · c for all j
        # So t = max_j (M_j - n_j · c)
        # Minimize over c: f(c) = max_j (M_j - n_j · c)
        
        # This is a convex piecewise linear function. We can use subgradient method
        # or just try the centroid and nearby points
        
        # The optimal c is where the subgradient contains 0, i.e., the weighted average
        # of the active normals sums to 0.
        
        # Simple approach: try centroid and a few other points
        centroid = np.mean(verts, axis=0)
        
        candidates = [centroid]
        # Also try vertices of the convex hull
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(verts)
            hull_verts = verts[hull.vertices]
            candidates.extend(list(hull_verts))
            # Midpoints of hull edges
            for i in range(len(hull_verts)):
                mid = (hull_verts[i] + hull_verts[(i+1) % len(hull_verts)]) / 2
                candidates.append(mid)
        except:
            pass
        
        best_t = float('inf')
        best_c_local = centroid
        for c in candidates:
            t = max(M[j] - c @ normals[j] for j in range(6))
            if t < best_t:
                best_t = t
                best_c_local = c
        
        # Refine with a few iterations of subgradient
        c = np.array(best_c_local, dtype=float)
        for _ in range(50):
            # Compute active constraints
            vals = M - c @ normals
            t = vals.max()
            # Active constraints: those where vals[j] is close to t
            active = np.where(vals > t - 1e-6)[0]
            if len(active) == 0:
                break
            # Subgradient of max_j (M_j - n_j·c) with respect to c
            # = -sum_{j in active} n_j / |active| ... actually the subgradient is the convex hull
            # of {-n_j for j in active}
            # Use the negative of the average of active normals as descent direction
            grad = -np.mean(normals[active], axis=0)
            # Step size
            step = 0.1 / (1 + _ * 0.01)
            c_new = c + step * grad
            t_new = max(M - c_new @ normals)
            if t_new < t:
                c = c_new
            else:
                step *= 0.5
                c_new = c + step * grad
                t_new = max(M - c_new @ normals)
                if t_new < t:
                    c = c_new
        
        side = best_t / (math.sqrt(3) / 2)
        # Wait, t = max_j (M_j - n_j·c) is the required inradius
        # inradius = s * sqrt(3)/2, so s = inradius / (sqrt(3)/2) = 2*inradius/sqrt(3)
        side = 2.0 * t / math.sqrt(3)
        
        if side < best_side:
            best_side = side
            best_c = (c[0], c[1])
            best_a = a_deg
    
    return best_side, best_c, best_a

def _objective(positions, angles):
    """
    Compute the objective: outer hexagon side length + penalty for overlaps.
    positions: (N, 2) array of centers
    angles: (N,) array of angles in degrees
    """
    # Compute vertices for each inner hexagon
    hex_verts = []
    for i in range(N):
        verts = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
        hex_verts.append(verts)
    
    # Check overlaps
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            overlap_penalty += _hexagon_overlap_penalty(hex_verts[i], hex_verts[j])
    
    # Find minimum enclosing hexagon
    all_verts = np.vstack(hex_verts)
    side, center, angle = _min_enclosing_hexagon_side(all_verts)
    
    return side + 10.0 * overlap_penalty, side, center, angle

def construct():
    start_time = time.time()
    
    # Initial arrangement: hexagonal lattice pattern
    # Place 12 hexagons in a roughly hexagonal cluster
    # Hexagonal lattice spacing: for unit hexagons, centers need to be at least
    # sqrt(3) apart for non-overlapping (inradius = sqrt(3)/2, so center distance >= 2*inradius = sqrt(3))
    # But with rotations, we can do better
    
    sqrt3 = math.sqrt(3)
    
    # Start with a 3-2-3-2-3 pattern or similar
    # Row-based hexagonal packing
    # Let's try: rows of 3, 2, 3, 2, 3 = 13, too many
    # 3, 2, 3, 2, 2 = 12? Or 4, 3, 2, 3 = 12?
    # Let's do a compact arrangement:
    # Row 0: 4 hexagons
    # Row 1: 3 hexagons (offset)
    # Row 2: 2 hexagons (offset)
    # Row 3: 3 hexagons (offset)
    # Total: 12
    
    positions = []
    angles = []
    
    # Hexagonal lattice: neighboring centers at distance sqrt(3)
    dx = sqrt3  # horizontal spacing
    dy = 1.5    # vertical spacing (sqrt(3)/2 * sqrt(3) = 1.5)
    
    # Row 0: 4 hexagons at y=0
    for k in range(4):
        x = (k - 1.5) * dx
        positions.append([x, 0.0])
        angles.append(0.0)
    
    # Row 1: 3 hexagons at y=dy, offset by dx/2
    for k in range(3):
        x = (k - 1.0) * dx
        positions.append([x, dy])
        angles.append(0.0)
    
    # Row 2: 2 hexagons at y=2*dy, offset by dx/2
    for k in range(2):
        x = (k - 0.5) * dx
        positions.append([x, 2 * dy])
        angles.append(0.0)
    
    # Row 3: 3 hexagons at y=3*dy, offset
    for k in range(3):
        x = (k - 1.0) * dx
        positions.append([x, 3 * dy])
        angles.append(0.0)
    
    positions = np.array(positions, dtype=float)
    angles = np.array(angles, dtype=float)
    
    # Compute initial objective
    obj, side, center, angle = _objective(positions, angles)
    best_obj = obj
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_side = side
    best_center = center
    best_angle = angle
    
    # Simulated annealing
    current_temp = 2.0
    cooling_rate = 0.997
    min_temp = 0.001
    
    elapsed = time.time() - start_time
    while current_temp > min_temp and elapsed < DEADLINE - 10:
        # Generate perturbation
        new_positions = positions.copy()
        new_angles = angles.copy()
        
        # Perturb a random subset
        n_perturb = random.randint(1, 3)
        idx = random.choice(range(N))
        new_positions[idx] += np.array([random.gauss(0, 0.5), random.gauss(0, 0.5)])
        new_angles[idx] += random.gauss(0, 15)
        
        for _ in range(n_perturb - 1):
            idx = random.choice(range(N))
            new_positions[idx] += np.array([random.gauss(0, 0.3), random.gauss(0, 0.3)])
            new_angles[idx] += random.gauss(0, 10)
        
        new_obj, new_side, new_center, new_angle = _objective(new_positions, new_angles)
        
        # Accept or reject
        if new_obj < obj or random.random() < math.exp(-(new_obj - obj) / current_temp):
            positions = new_positions
            angles = new_angles
            obj = new_obj
            
            if obj < best_obj:
                best_obj = obj
                best_positions = positions.copy()
                best_angles = angles.copy()
                best_side = new_side
                best_center = new_center
                best_angle = new_angle
        
        current_temp *= cooling_rate
        elapsed = time.time() - start_time
    
    # Final polish: Nelder-Mead on best solution
    if time.time() - start_time < DEADLINE - 5:
        from scipy.optimize import minimize
        
        def nm_objective(x):
            pos = x[:2*N].reshape(N, 2)
            ang = x[2*N:]
            o, s, c, a = _objective(pos, ang)
            return o
        
        x0 = np.concatenate([best_positions.flatten(), best_angles])
        
        result = minimize(nm_objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if result.fun < best_obj:
            best_positions = result.x[:2*N].reshape(N, 2)
            best_angles = result.x[2*N:]
            best_obj = result.fun
            _, best_side, best_center, best_angle = _objective(best_positions, best_angles)
    
    # Build return values
    inner = []
    for i in range(N):
        inner.append([best_positions[i, 0], best_positions[i, 1], best_angles[i]])
    
    return inner, best_center, best_side, best_angle

# EVOLVE-BLOCK-END
