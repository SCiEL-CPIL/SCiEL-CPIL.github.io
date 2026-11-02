import numpy as np
from scipy.optimize import minimize
import time
import random
import math

N = 12
DEADLINE = 145.0

# Unit hexagon vertices (pointy-top, angle 0 = vertex at top)
# For flat-top: vertices at 0, 60, 120, 180, 240, 300 degrees
UNIT_VERTS = np.array([
    [np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))] for i in range(6)
])

def get_hex_verts(cx, cy, angle_deg):
    """Get 6 vertices of a unit hexagon centered at (cx,cy) with rotation angle_deg."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s], [s, c]])
    return UNIT_VERTS @ R.T + np.array([cx, cy])

def polygon_area_signed(verts):
    """Signed area of a polygon."""
    n = len(verts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += verts[i][0] * verts[j][1] - verts[j][0] * verts[i][1]
    return area / 2.0

def point_in_polygon(pt, poly):
    """Ray casting point-in-polygon test."""
    x, y = pt
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def polygons_overlap(v1, v2):
    """Check if two convex polygons overlap using SAT."""
    edges = []
    for v in [v1, v2]:
        for i in range(len(v)):
            j = (i + 1) % len(v)
            edge = v[j] - v[i]
            # Normal
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-15:
                edges.append(normal / norm)
    
    for axis in edges:
        proj1 = v1 @ axis
        proj2 = v2 @ axis
        if proj1.max() < proj2.min() or proj2.max() < proj1.min():
            return False
    return True

def min_separation(v1, v2):
    """Minimum separation between two convex polygons via SAT. Negative if overlapping."""
    edges = []
    for v in [v1, v2]:
        for i in range(len(v)):
            j = (i + 1) % len(v)
            edge = v[j] - v[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-15:
                edges.append(normal / norm)
    
    min_sep = float('inf')
    for axis in edges:
        proj1 = v1 @ axis
        proj2 = v2 @ axis
        sep1 = proj1.min() - proj2.max()
        sep2 = proj2.min() - proj1.max()
        sep = max(sep1, sep2)
        min_sep = min(min_sep, sep)
    return min_sep

def outer_hex_verts(cx, cy, side, angle_deg):
    """Vertices of outer hexagon. Flat-top: vertices at 0,60,...,300 + angle."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s], [s, c]])
    verts = np.array([[side * np.cos(np.radians(60 * i)), side * np.sin(np.radians(60 * i))] for i in range(6)])
    return verts @ R.T + np.array([cx, cy])

def compute_min_side(centers, angles, outer_center=(0,0), outer_angle=0.0):
    """Compute minimum side length of outer hexagon to contain all inner hexagons."""
    outer_center = np.array(outer_center)
    # Outer hexagon normals (outward) for flat-top: at 30, 90, 150, 210, 270, 330 + angle
    oa = np.radians(outer_angle)
    ca, sa = np.cos(oa), np.sin(oa)
    Ro = np.array([[ca, -sa], [sa, ca]])
    
    max_apothem = 0.0
    for i in range(len(centers)):
        v = get_hex_verts(centers[i][0], centers[i][1], angles[i])
        v_rel = v - outer_center
        v_rot = v_rel @ Ro.T
        # Project onto 6 outward normal directions
        for k in range(6):
            theta = np.radians(60 * k + 30 + outer_angle)
            n = np.array([np.cos(theta), np.sin(theta)])
            proj = np.max(v_rot @ n)
            # The constraint is: proj <= apothem (for this normal direction)
            # But we need to check all 6 directions
            # Actually for a regular hexagon, the support function in direction n gives the distance
            # from center to the edge perpendicular to n.
            # For containment: max over all vertices of (v . n) <= apothem for each of 6 normals
            # But the normals are at 30,90,... relative to the hexagon orientation
            pass
        
        # Better: compute the support function of the inner hexagon in each of the 6 outer normal directions
        for k in range(6):
            theta = np.radians(60 * k + 30)  # relative to outer hex orientation
            # Rotate to world frame
            theta_world = theta + np.radians(outer_angle)
            n = np.array([np.cos(theta_world), np.sin(theta_world)])
            proj = np.max(v @ n)  # max projection of inner hex vertices onto this normal
            # Distance from outer center:
            proj_centered = np.max((v - outer_center) @ n)
            max_apothem = max(max_apothem, proj_centered)
    
    # Apothem = side * sqrt(3)/2
    return max_apothem / (np.sqrt(3) / 2)

def evaluate_packing(centers, angles, outer_center=(0,0), outer_angle=0.0):
    """
    Evaluate a packing configuration.
    Returns (side_length, is_valid).
    """
    # Check pairwise non-overlap
    verts_list = []
    for i in range(len(centers)):
        v = get_hex_verts(centers[i][0], centers[i][1], angles[i])
        verts_list.append(v)
    
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            sep = min_separation(verts_list[i], verts_list[j])
            if sep < -1e-9:
                return float('inf'), False
    
    # Compute required outer side
    side = compute_min_side(centers, angles, outer_center, outer_angle)
    return side, True

def generate_symmetric_config(r1, r2, r3, angle_offset1, angle_offset2, angle_offset3, inner_angles):
    """
    Generate a configuration with approximate 3-fold or 6-fold symmetry.
    Pattern: 3 hexagons on ring 1, 3 on ring 2, 3 on ring 3, or other splits.
    Actually let's use: 1 center + 5 ring1 + 6 ring2, or 2+4+6, etc.
    
    For simplicity and symmetry: use 6-fold symmetry where possible.
    12 = 1 + 5 + 6 or 2 + 4 + 6 or 3 + 3 + 6 or 6 + 6
    """
    centers = []
    angles = []
    
    # Pattern: 6 on inner ring, 6 on outer ring (6-fold symmetry)
    # Or: 1 center + 5 inner + 6 outer
    # Let's use: 6 inner + 6 outer for maximum symmetry
    
    # Inner ring: 6 hexagons at radius r1, angles 0, 60, 120, ...
    for i in range(6):
        theta = np.radians(60 * i + angle_offset1)
        centers.append([r1 * np.cos(theta), r1 * np.sin(theta)])
        angles.append(inner_angles[0] if len(inner_angles) > 0 else 0.0)
    
    # Outer ring: 6 hexagons at radius r2, angles 30, 90, 150, ... (offset by 30)
    for i in range(6):
        theta = np.radians(60 * i + 30 + angle_offset2)
        centers.append([r2 * np.cos(theta), r2 * np.sin(theta)])
        angles.append(inner_angles[1] if len(inner_angles) > 1 else 0.0)
    
    return centers, angles

def generate_config_1_5_6(r1, r2, off1, off2, ang1, ang2):
    """1 center + 5 inner ring + 6 outer ring."""
    centers = [[0.0, 0.0]]
    angles = [ang1]
    
    for i in range(5):
        theta = np.radians(72 * i + off1)
        centers.append([r1 * np.cos(theta), r1 * np.sin(theta)])
        angles.append(ang1)
    
    for i in range(6):
        theta = np.radians(60 * i + off2)
        centers.append([r2 * np.cos(theta), r2 * np.sin(theta)])
        angles.append(ang2)
    
    return centers, angles

def generate_config_2_4_6(r1, r2, r3, off1, off2, off3, ang1, ang2, ang3):
    """2 center-ish + 4 inner + 6 outer."""
    centers = []
    angles = []
    
    # 2 hexagons near center (slightly offset)
    centers.append([-0.3, 0.0])
    angles.append(ang1)
    centers.append([0.3, 0.0])
    angles.append(ang1)
    
    # 4 on ring r1
    for i in range(4):
        theta = np.radians(90 * i + off1)
        centers.append([r1 * np.cos(theta), r1 * np.sin(theta)])
        angles.append(ang2)
    
    # 6 on ring r2
    for i in range(6):
        theta = np.radians(60 * i + off2)
        centers.append([r2 * np.cos(theta), r2 * np.sin(theta)])
        angles.append(ang3)
    
    return centers, angles

def generate_config_3_3_6(r1, r2, r3, off1, off2, off3, ang1, ang2, ang3):
    """3 inner + 3 middle + 6 outer."""
    centers = []
    angles = []
    
    for i in range(3):
        theta = np.radians(120 * i + off1)
        centers.append([r1 * np.cos(theta), r1 * np.sin(theta)])
        angles.append(ang1)
    
    for i in range(3):
        theta = np.radians(120 * i + 60 + off2)
        centers.append([r2 * np.cos(theta), r2 * np.sin(theta)])
        angles.append(ang2)
    
    for i in range(6):
        theta = np.radians(60 * i + off3)
        centers.append([r3 * np.cos(theta), r3 * np.sin(theta)])
        angles.append(ang3)
    
    return centers, angles

def full_evaluate(centers, angles, outer_center=(0,0), outer_angle=0.0):
    """Full evaluation with overlap check and containment."""
    n = len(centers)
    verts_list = []
    for i in range(n):
        v = get_hex_verts(centers[i][0], centers[i][1], angles[i])
        verts_list.append(v)
    
    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            sep = min_separation(verts_list[i], verts_list[j])
            if sep < -1e-8:
                return float('inf'), False
    
    # Compute minimum outer side
    side = compute_min_side(centers, angles, outer_center, outer_angle)
    return side, True

def optimize_from_config(centers, angles, time_budget=3.0):
    """
    Optimize a configuration using Nelder-Mead on the full parameter vector.
    Parameters: all x, y, angle for each hexagon (36 params).
    """
    start = time.time()
    
    x0 = np.array(centers).flatten()
    a0 = np.array(angles).flatten()
    x0_full = np.concatenate([x0, a0])
    
    def objective(params):
        if time.time() - start > time_budget:
            return 1e10
        x, a = params[:2*N], params[2*N:]
        c = x.reshape(N, 2)
        ang = a
        
        # Quick overlap check
        verts_list = []
        for i in range(N):
            v = get_hex_verts(c[i][0], c[i][1], ang[i])
            verts_list.append(v)
        
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                sep = min_separation(verts_list[i], verts_list[j])
                if sep < 0:
                    overlap_penalty += (-sep) ** 2 * 100
        
        side = compute_min_side(c, ang)
        return side + overlap_penalty
    
    try:
        res = minimize(objective, x0_full, method='Nelder-Mead',
                      options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5})
        x_opt = res.x[:2*N].reshape(N, 2)
        a_opt = res.x[2*N:]
        
        # Validate
        side, valid = full_evaluate(x_opt, a_opt)
        if valid and side < 1e9:
            return x_opt, a_opt, side
    except:
        pass
    return None

def simulated_annealing_search(time_budget=30.0):
    """
    Simulated annealing over symmetric parameter space.
    """
    start = time.time()
    best_side = float('inf')
    best_centers = None
    best_angles = None
    
    # Initial configuration: 6+6 rings
    r1_init = np.sqrt(3) * 1.05
    r2_init = np.sqrt(3) * 2.0
    T = 0.5  # Initial temperature
    
    current_r1 = r1_init
    current_r2 = r2_init
    current_off1 = 0.0
    current_off2 = 30.0
    current_ang1 = 0.0
    current_ang2 = 0.0
    
    centers, angles = generate_symmetric_config(current_r1, current_r2, 0, current_off1, current_off2, current_off2, [current_ang1, current_ang2])
    side, valid = full_evaluate(centers, angles)
    current_side = side if valid else 1e6
    
    best_side = current_side
    best_centers = [c[:] for c in centers]
    best_angles = angles[:]
    
    n_iter = 0
    while time.time() - start < time_budget:
        n_iter += 1
        # Cooling
        T *= 0.995
        
        # Random perturbation
        new_r1 = current_r1 + random.gauss(0, 0.1)
        new_r2 = current_r2 + random.gauss(0, 0.1)
        new_off1 = current_off1 + random.gauss(0, 5.0)
        new_off2 = current_off2 + random.gauss(0, 5.0)
        new_ang1 = current_ang1 + random.gauss(0, 10.0)
        new_ang2 = current_ang2 + random.gauss(0, 10.0)
        
        new_centers, new_angles = generate_symmetric_config(new_r1, new_r2, 0, new_off1, new_off2, new_off2, [new_ang1, new_ang2])
        new_side, new_valid = full_evaluate(new_centers, new_angles)
        if not new_valid:
            new_side = 1e6
        
        delta = new_side - current_side
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            current_r1, current_r2 = new_r1, new_r2
            current_off1, current_off2 = new_off1, new_off2
            current_ang1, current_ang2 = new_ang1, new_ang2
            current_side = new_side
            centers, angles = new_centers, new_angles
            
            if new_side < best_side and new_valid:
                best_side = new_side
                best_centers = [c[:] for c in new_centers]
                best_angles = new_angles[:]
        
        # Periodically try asymmetric optimization
        if n_iter % 50 == 0 and time.time() - start < time_budget - 5:
            opt_result = optimize_from_config(centers, angles, time_budget=min(2.0, time_budget - (time.time() - start)))
            if opt_result is not None:
                oc, oa, oside = opt_result
                if oside < best_side:
                    best_side = oside
                    best_centers = [c[:] for c in oc]
                    best_angles = oa[:]
                # Update current to optimized position
                current_side = oside
                # Re-extract symmetric params approximately (just use as-is for next SA step)
                # For simplicity, reset to symmetric
                current_r1 = r1_init
                current_r2 = r2_init
    
    return best_centers, best_angles, best_side

def honeycomb_packing():
    """
    Generate a dense honeycomb-like packing of 12 unit hexagons.
    In a perfect honeycomb, hexagons tile the plane.
    For 12 hexagons, we can arrange them in a hexagonal shape.
    
    The optimal arrangement for minimizing the bounding hexagon:
    - A hexagonal cluster of hexagons (like a "flower" pattern)
    - 1 center + 6 around + 5 more on the outside
    """
    D = np.sqrt(3)  # Center-to-center distance for touching flat-top hexagons
    
    centers = []
    angles = []
    
    # Center
    centers.append([0.0, 0.0])
    angles.append(0.0)
    
    # 6 around center
    for i in range(6):
        theta = np.radians(60 * i)
        centers.append([D * np.cos(theta), D * np.sin(theta)])
        angles.append(0.0)
    
    # 5 more on the outer ring
    # Place them in the gaps between the 6 surrounding hexagons
    # The outer ring has 6 gaps, we fill 5 of them
    r_outer = D * 2.0  # Approximately
    for i in range(5):
        theta = np.radians(60 * i + 30)  # Offset by 30 to fit in gaps
        centers.append([r_outer * np.cos(theta), r_outer * np.sin(theta)])
        angles.append(0.0)
    
    return centers, angles

def optimize_honeycomb(time_budget=10.0):
    """Optimize the honeycomb packing."""
    centers, angles = honeycomb_packing()
    return optimize_from_config(centers, angles, time_budget)

def construct():
    start_time = time.time()
    best_side = float('inf')
    best_centers = None
    best_angles = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Strategy 1: Honeycomb packing with optimization
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 10:
        result = optimize_honeycomb(time_budget=min(15.0, DEADLINE - elapsed - 5))
        if result is not None:
            oc, oa, oside = result
            if oside < best_side:
                best_side = oside
                best_centers = [c[:] for c in oc]
                best_angles = oa[:]
    
    # Strategy 2: Simulated annealing over symmetric 6+6 config
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 10:
        bc, ba, bs = simulated_annealing_search(time_budget=min(25.0, DEADLINE - elapsed - 5))
        if bs < best_side:
            best_side = bs
            best_centers = bc
            best_angles = ba
    
    # Strategy 3: Try 1-5-6 pattern
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 10:
        for trial in range(5):
            r1 = np.sqrt(3) * (1.0 + random.uniform(-0.1, 0.1))
            r2 = np.sqrt(3) * (1.8 + random.uniform(-0.2, 0.2))
            off1 = random.uniform(-10, 10)
            off2 = random.uniform(-10, 10)
            ang1 = random.uniform(-15, 15)
            ang2 = random.uniform(-15, 15)
            
            centers, angles = generate_config_1_5_6(r1, r2, off1, off2, ang1, ang2)
            result = optimize_from_config(centers, angles, time_budget=min(3.0, DEADLINE - (time.time() - start_time) - 5))
            if result is not None:
                oc, oa, oside = result
                if oside < best_side:
                    best_side = oside
                    best_centers = [c[:] for c in oc]
                    best_angles = oa[:]
            
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 10:
                break
    
    # Strategy 4: Try 3-3-6 pattern
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 10:
        for trial in range(3):
            r1 = np.sqrt(3) * (0.8 + random.uniform(-0.1, 0.1))
            r2 = np.sqrt(3) * (1.5 + random.uniform(-0.1, 0.1))
            r3 = np.sqrt(3) * (2.2 + random.uniform(-0.1, 0.1))
            off1 = random.uniform(-10, 10)
            off2 = random.uniform(-10, 10)
            off3 = random.uniform(-10, 10)
            
            centers, angles = generate_config_3_3_6(r1, r2, r3, off1, off2, off3, 0, 0, 0)
            result = optimize_from_config(centers, angles, time_budget=min(3.0, DEADLINE - (time.time() - start_time) - 5))
            if result is not None:
                oc, oa, oside = result
                if oside < best_side:
                    best_side = oside
                    best_centers = [c[:] for c in oc]
                    best_angles = oa[:]
            
            elapsed = time.time() - start_time
            if elapsed > DEADLINE - 10:
                break
    
    # Strategy 5: Final polish with more iterations on best config
    elapsed = time.time() - start_time
    if best_centers is not None and elapsed < DEADLINE - 5:
        result = optimize_from_config(best_centers, best_angles, time_budget=min(10.0, DEADLINE - elapsed - 2))
        if result is not None:
            oc, oa, oside = result
            if oside < best_side:
                best_side = oside
                best_centers = [c[:] for c in oc]
                best_angles = oa[:]
    
    # Fallback if nothing worked
    if best_centers is None:
        centers, angles = honeycomb_packing()
        best_side, valid = full_evaluate(centers, angles)
        best_centers = centers
        best_angles = angles
    
    # Format output
    inner = []
    for i in range(N):
        inner.append([best_centers[i][0], best_centers[i][1], best_angles[i]])
    
    return (inner, best_outer_center, best_side, best_outer_angle)

if __name__ == "__main__":
    result = construct()
    print(f"Best Side: {result[2]}")
