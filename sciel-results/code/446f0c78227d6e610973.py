"""Hexagon packing: 11 unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(angle_deg=0.0, center=None):
    """Return 6 vertices of a unit regular hexagon rotated by angle_deg and translated by center."""
    if center is None:
        center = [0.0, 0.0]
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append([center[0] + math.cos(theta), center[1] + math.sin(theta)])
    return np.array(verts)

def _point_in_polygon(pt, polygon):
    """Ray casting point-in-polygon test. polygon is (n,2)."""
    x, y = pt
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
            inside = not inside
        j = i
    return inside

def _segment_distance(p1, p2, p3, p4):
    """Minimum distance between segments p1-p2 and p3-p4."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-10 and e < 1e-10:
        return np.linalg.norm(r)
    elif a < 1e-10:
        t2 = f / e
        t2 = max(0, min(1, t2))
        return np.linalg.norm(r + d2 * t2)
    elif e < 1e-10:
        t1 = -np.dot(d1, r) / a
        t1 = max(0, min(1, t1))
        return np.linalg.norm(r + d1 * t1)
    
    c = np.dot(d1, r)
    b = np.dot(d1, d2)
    denom = a * e - b * b
    if denom < 1e-10:
        t1 = 0.0
    else:
        t1 = (b * f - c * e) / denom
        t1 = max(0, min(1, t1))
    
    t2 = (b * t1 + f) / e
    t2 = max(0, min(1, t2))
    if t2 < 0:
        t2 = 0
        t1 = max(0, min(1, -c / a))
    elif t2 > 1:
        t2 = 1
        t1 = max(0, min(1, (b - c) / a))
    
    return np.linalg.norm(r + d1 * t1 - d2 * t2)

def _poly_distance(poly1, poly2):
    """Minimum distance between two convex polygons. Returns 0 if overlapping."""
    # Check if any vertex of poly1 is inside poly2 or vice versa
    for v in poly1:
        if _point_in_polygon(v, poly2):
            return 0.0
    for v in poly2:
        if _point_in_polygon(v, poly1):
            return 0.0
    
    # Check all edge pairs
    n1 = len(poly1)
    n2 = len(poly2)
    min_dist = float('inf')
    for i in range(n1):
        for j in range(n2):
            d = _segment_distance(poly1[i], poly1[(i+1)%n1], poly2[j], poly2[(j+1)%n2])
            if d < min_dist:
                min_dist = d
    return min_dist

def _compute_side_length(centers, angles, outer_angle_deg=0.0):
    """
    Compute the minimum side length of an outer hexagon (at given rotation)
    that contains all inner hexagons.
    """
    outer_normals = []
    a = math.radians(outer_angle_deg)
    for k in range(6):
        theta = a + k * math.pi / 3.0
        outer_normals.append([math.cos(theta), math.sin(theta)])
    outer_normals = np.array(outer_normals)  # (6, 2)
    
    max_extent = 0.0
    for i in range(N):
        verts = _hex_vertices(angles[i], centers[i])
        projections = verts @ outer_normals.T  # (6, 6)
        abs_proj = np.abs(projections)
        max_extent = max(max_extent, np.max(abs_proj))
    
    S = max_extent / (math.sqrt(3) / 2.0)
    return S

def _is_valid_config(centers, angles, outer_side, outer_angle_deg=0.0):
    """
    Check if the configuration is valid:
    1. No inner hexagons overlap.
    2. All inner hexagons are inside the outer hexagon of side 'outer_side'.
    """
    polys = []
    for i in range(N):
        verts = _hex_vertices(angles[i], centers[i])
        polys.append(verts)
    
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            if _poly_distance(polys[i], polys[j]) < 1e-9:
                return False
    
    # Check containment
    # We can optimize this by checking vertices against the outer hexagon
    # The outer hexagon is defined by 6 half-planes.
    a = math.radians(outer_angle_deg)
    # Normals of the outer hexagon sides
    normals = np.array([[math.cos(a + k * math.pi / 3.0), math.sin(a + k * math.pi / 3.0)] for k in range(6)])
    # The distance from center to side is apothem = S * sqrt(3) / 2
    apothem = outer_side * math.sqrt(3) / 2.0
    
    for i in range(N):
        verts = polys[i]
        # Project vertices onto normals
        # If any projection exceeds apothem, it's outside
        # Note: The normals point OUTWARDS from the center of the outer hexagon (which is 0,0)
        # So for a point p, dot(p, normal) <= apothem must hold.
        projs = verts @ normals.T
        if np.any(projs > apothem + 1e-9):
            return False
            
    return True

def _get_initial_solution():
    """
    Generate a valid initial solution.
    We use a hexagonal lattice pattern.
    """
    d = math.sqrt(3) # Distance between centers of touching hexagons
    centers = []
    angles = []
    
    # Center
    centers.append([0.0, 0.0])
    angles.append(0.0)
    
    # 6 surrounding
    for i in range(6):
        theta = i * math.pi / 3.0
        centers.append([d * math.cos(theta), d * math.sin(theta)])
        angles.append(0.0)
        
    # 4 more from second ring
    # To ensure validity, we place them further out or in gaps.
    # A simple valid configuration for 11 hexagons:
    # 1 center, 6 ring 1, 4 from ring 2.
    # Ring 2 positions are at distance 2*sqrt(3) in directions 0, 60, 120...
    # But we need to pick 4 that don't overlap with ring 1.
    # Actually, a known compact packing for 11 hexagons is roughly S ~ 3.4-3.5.
    # Let's just use a slightly expanded lattice to guarantee validity.
    
    # Expand the lattice slightly to ensure no overlaps (gap)
    gap = 1.05
    d_eff = d * gap
    
    # Re-centering might be needed, but let's just compute S and use that.
    # We'll create a "safe" initial state.
    
    # Let's use the positions from the seed solution in the prompt but ensure validity.
    # The prompt's seed_solution used ring 2 positions.
    ring2_positions = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d_eff * (q + r / 2.0)
                y = d_eff * (math.sqrt(3) / 2.0) * r
                ring2_positions.append([x, y])
                
    # Pick 4 well-separated
    # Indices 0, 3, 6, 9 in the list of 12 ring-2 positions
    selected = [ring2_positions[i] for i in [0, 3, 6, 9]]
    for pos in selected:
        centers.append(pos)
        angles.append(0.0)
        
    centers = np.array(centers)
    angles = np.array(angles)
    
    # Compute S
    S = _compute_side_length(centers, angles)
    # Add margin to ensure validity
    S *= 1.05
    
    return centers, angles, S

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    # Initialize
    centers, angles, S_init = _get_initial_solution()
    
    # Verify initial validity
    if not _is_valid_config(centers, angles, S_init):
        # If initial is invalid (shouldn't happen with margin), expand more
        S_init *= 1.1
        while not _is_valid_config(centers, angles, S_init):
            S_init *= 1.01
            if time.time() > deadline - 10:
                break
                
    best_centers = centers.copy()
    best_angles = angles.copy()
    best_S = S_init
    
    # Simulated Annealing
    # Parameters
    T = 0.5  # Initial temperature
    T_min = 1e-5
    cooling_rate = 0.9995
    max_steps = 2000000  # Large number, limited by time
    
    step = 0
    np.random.seed(42)
    
    while time.time() < deadline - 5:
        if step % 1000 == 0 and T < T_min:
            break
            
        # Choose a random hexagon to move
        i = np.random.randint(N)
        
        # Generate a move
        # Move magnitude depends on temperature
        move_size = 0.1 * T + 0.005
        
        new_centers = best_centers.copy()
        new_angles = best_angles.copy()
        
        # Perturb center
        new_centers[i][0] += np.random.uniform(-move_size, move_size)
        new_centers[i][1] += np.random.uniform(-move_size, move_size)
        
        # Perturb angle (probabilistic)
        if np.random.random() < 0.5:
            new_angles[i] += np.random.uniform(-10 * T - 1, 10 * T + 1)
            # Keep angle in [0, 180) for uniqueness
            new_angles[i] %= 180.0
        
        # Calculate new S
        S_new = _compute_side_length(new_centers, new_angles)
        
        # Check validity
        # We check validity against S_new. If it's valid in S_new, it's a candidate.
        # But we want to minimize S.
        # If S_new < best_S, we must check if it's valid in S_new.
        # If S_new >= best_S, we might accept it if it's valid in S_new (to escape local minima)
        # or if it's valid in best_S (which implies it's valid in S_new if S_new > best_S? No, S_new is the *minimum* required.
        # If valid in S_new, it is valid in any S >= S_new.
        
        # So:
        # 1. Is the configuration geometrically valid (no overlaps)?
        # 2. If yes, what is the min S? (S_new)
        # 3. If S_new < best_S: Accept if valid in S_new.
        # 4. If S_new >= best_S: Accept with probability exp(-(S_new - best_S)/T) IF valid in S_new.
        
        # To save time, we first check overlaps.
        valid = True
        polys_new = []
        for k in range(N):
            polys_new.append(_hex_vertices(new_angles[k], new_centers[k]))
            
        for k in range(N):
            for m in range(k + 1, N):
                if _poly_distance(polys_new[k], polys_new[m]) < 1e-9:
                    valid = False
                    break
            if not valid:
                break
                
        if not valid:
            # Reject
            step += 1
            T *= cooling_rate
            continue
            
        # Now check containment in S_new
        # We can use _is_valid_config but we already have polys_new.
        # Let's just call _is_valid_config for simplicity, it's not too slow.
        if not _is_valid_config(new_centers, new_angles, S_new):
            valid = False
            step += 1
            T *= cooling_rate
            continue
            
        # If we reach here, the configuration is valid in S_new.
        
        if S_new < best_S:
            # Accept, update best
            best_centers = new_centers
            best_angles = new_angles
            best_S = S_new
            # Optional: Reset T slightly to explore new basin?
            # T = max(T, 0.1) # Un-comment to restart cooling
        else:
            # S_new >= best_S
            delta = S_new - best_S
            if np.random.random() < math.exp(-delta / T):
                # Accept (Metropolis criterion)
                best_centers = new_centers
                best_angles = new_angles
                # Do not update best_S, as we only care about the global minimum
                # But we use the new state as the "current" state for future moves
                # Note: best_S remains the same, but we are now in a "worse" state
                # This allows escaping local minima.
                pass
        
        step += 1
        T *= cooling_rate
        
        # Periodic check
        if step % 10000 == 0:
            # We can try a local refinement here if time permits
            # But for now, just continue SA
            pass
            
    # Final Refinement
    # Once SA is done, we can try to tighten the fit further.
    # We can run a few iterations of "tightening" where we only accept moves that reduce S.
    T = 0.01
    for _ in range(50000):
        if time.time() > deadline - 2:
            break
        i = np.random.randint(N)
        move_size = 0.005
        
        new_centers = best_centers.copy()
        new_angles = best_angles.copy()
        
        new_centers[i][0] += np.random.uniform(-move_size, move_size)
        new_centers[i][1] += np.random.uniform(-move_size, move_size)
        if np.random.random() < 0.5:
            new_angles[i] += np.random.uniform(-2, 2)
            new_angles[i] %= 180.0
            
        S_new = _compute_side_length(new_centers, new_angles)
        
        if S_new < best_S:
            if _is_valid_config(new_centers, new_angles, S_new):
                best_centers = new_centers
                best_angles = new_angles
                best_S = S_new
        else:
            # Reject if worse
            pass

    # Final verification
    if not _is_valid_config(best_centers, best_angles, best_S):
        # Fallback to initial if something went wrong
        best_centers, best_angles, best_S = _get_initial_solution()
        if not _is_valid_config(best_centers, best_angles, best_S):
             # Expand until valid
             while not _is_valid_config(best_centers, best_angles, best_S):
                 best_S *= 1.01

    # Format output
    inner = []
    for i in range(N):
        inner.append([float(best_centers[i][0]), float(best_centers[i][1]), float(best_angles[i])])
    
    return inner, (0.0, 0.0), float(best_S), 0.0

# EVOLVE-BLOCK-END
