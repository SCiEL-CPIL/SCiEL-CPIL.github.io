"""Hexagon packing: 12 unit hexagons in minimal regular hexagon."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time

N = 12
TIME_LIMIT = 140.0  # safety margin below 150s


def hexagon_vertices(cx, cy, side, angle_deg=0.0):
    """Return vertices of a regular hexagon (side length `side`) centered at (cx,cy).
    angle_deg rotates the hexagon. Vertices ordered CCW."""
    a = np.radians(angle_deg)
    # Regular hexagon: vertices at angles 0, 60, 120, 180, 240, 300 (flat-top)
    # Actually for pointy-top (default), vertices at 30, 90, 150, 210, 270, 330
    # Let's use: vertex i at angle = a + i*pi/3
    angles = a + np.arange(6) * np.pi / 3.0
    # For a regular hexagon with side s, circumradius = s
    r = side
    xs = cx + r * np.cos(angles)
    ys = cy + r * np.sin(angles)
    return np.column_stack([xs, ys])


def min_distance_hexagons(v1, v2):
    """Minimum distance between two convex polygons (hexagons).
    Returns the minimum distance between any edge of v1 and any edge of v2.
    If polygons overlap, returns negative value."""
    # For two convex polygons, min distance is min over all edge pairs
    # of the minimum distance between the two line segments
    # But we need to handle overlap: if one vertex is inside the other, distance is negative
    
    n1, n2 = len(v1), len(v2)
    min_d = np.inf
    
    for i in range(n1):
        for j in range(n2):
            # Distance between edge (v1[i], v1[i+1]) and (v2[j], v2[j+1])
            d = _seg_seg_dist(v1[i], v1[(i+1) % n1], v2[j], v2[(j+1) % n2])
            if d < min_d:
                min_d = d
    return min_d


def _seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between two line segments p1-p2 and p3-p4."""
    # Check if segments intersect
    if _segments_intersect(p1, p2, p3, p4):
        return 0.0
    
    # Minimum of distances from each endpoint to the other segment
    d1 = _pt_seg_dist(p1, p3, p4)
    d2 = _pt_seg_dist(p2, p3, p4)
    d3 = _pt_seg_dist(p3, p1, p2)
    d4 = _pt_seg_dist(p4, p1, p2)
    return min(d1, d2, d3, d4)


def _segments_intersect(p1, p2, p3, p4):
    """Check if segments p1-p2 and p3-p4 intersect."""
    def orient(a, b, c):
        val = (b[1]-a[1])*(c[0]-a[0]) - (b[0]-a[0])*(c[1]-a[1])
        if abs(val) < 1e-10:
            return 0
        return 1 if val > 0 else -1
    
    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)
    
    if o1 != o2 and o3 != o4:
        return True
    
    # Collinear cases
    if o1 == 0 and _on_segment(p1, p3, p2): return True
    if o2 == 0 and _on_segment(p1, p4, p2): return True
    if o3 == 0 and _on_segment(p3, p1, p4): return True
    if o4 == 0 and _on_segment(p3, p2, p4): return True
    
    return False


def _on_segment(a, b, c):
    """Check if point b lies on segment a-c (collinear)."""
    if (min(a[0], c[0]) - 1e-10 <= b[0] <= max(a[0], c[0]) + 1e-10 and
        min(a[1], c[1]) - 1e-10 <= b[1] <= max(a[1], c[1]) + 1e-10):
        return True
    return False


def _pt_seg_dist(p, a, b):
    """Distance from point p to segment a-b."""
    ab = b - a
    ap = p - a
    ab_sq = np.dot(ab, ab)
    if ab_sq < 1e-20:
        return np.linalg.norm(p - a)
    t = np.dot(ap, ab) / ab_sq
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return np.linalg.norm(p - proj)


def point_in_hexagon(p, cx, cy, side, angle_deg=0.0):
    """Check if point p is inside or on the boundary of the hexagon."""
    verts = hexagon_vertices(cx, cy, side, angle_deg)
    # Use winding number / ray casting for convex polygon
    n = len(verts)
    inside = True
    for i in range(n):
        v1 = verts[i]
        v2 = verts[(i + 1) % n]
        cross = (v2[0] - v1[0]) * (p[1] - v1[1]) - (v2[1] - v1[1]) * (p[0] - v1[0])
        if cross < -1e-9:
            inside = False
            break
    return inside


def hexagon_in_hexagon(inner_verts, outer_verts):
    """Check if all inner vertices are inside or on the outer hexagon."""
    for p in inner_verts:
        if not point_in_hexagon(p, *outer_center_side(outer_verts)):
            return False
    return True


def outer_center_side(outer_verts):
    """Extract center and side length from outer hexagon vertices."""
    center = np.mean(outer_verts, axis=0)
    # Side length = distance between adjacent vertices
    side = np.linalg.norm(outer_verts[1] - outer_verts[0])
    return center[0], center[1], side


def compute_min_outer_side(inner_centers, inner_angles, outer_center, outer_angle, outer_side):
    """Compute the minimum outer side length needed to contain all inner hexagons,
    given fixed outer center and angle. Returns the required side length."""
    # For each inner hexagon, find the maximum "radius" in each of the 6 normal directions
    # of the outer hexagon, then the outer side must be large enough to contain all.
    
    # The outer hexagon with center C, angle A, side S has 6 faces.
    # Face i has outward normal at angle A + i*pi/3 + pi/6 (for pointy-top)
    # Actually, let's think differently.
    
    # A regular hexagon with side S can be described as the intersection of 6 half-planes.
    # The distance from center to each face (inradius) = S * sqrt(3)/2
    # The distance from center to each vertex (circumradius) = S
    
    # For a hexagon centered at C with rotation A, the 6 face normals are at angles:
    # A + pi/6 + k*pi/3 for k=0..5 (if we define the hexagon appropriately)
    
    # Let me use a cleaner formulation. A regular hexagon with circumradius R (= side length S)
    # centered at origin, with one vertex at angle theta0, has vertices at theta0 + k*pi/3.
    # The 6 face normals (outward) are at angles theta0 + pi/6 + k*pi/3.
    # The distance from center to each face is R * cos(pi/6) = R * sqrt(3)/2.
    
    # So the hexagon = { p : for all k, dot(p - C, n_k) <= R * sqrt(3)/2 }
    # where n_k = (cos(theta0 + pi/6 + k*pi/3), sin(theta0 + pi/6 + k*pi/3))
    
    # For each inner hexagon vertex, we need:
    # dot(v - C, n_k) <= R * sqrt(3)/2 for all k
    # => R >= max over all inner vertices v and all k of dot(v - C, n_k) / (sqrt(3)/2)
    
    theta0 = np.radians(outer_angle)
    normals = []
    for k in range(6):
        ang = theta0 + np.pi/6 + k * np.pi / 3
        normals.append(np.array([np.cos(ang), np.sin(ang)]))
    
    max_val = 0.0
    for ic, ia in zip(inner_centers, inner_angles):
        verts = hexagon_vertices(ic[0], ic[1], 1.0, ia)
        for v in verts:
            rel = v - np.array(outer_center)
            for n in normals:
                proj = np.dot(rel, n)
                if proj > max_val:
                    max_val = proj
    
    # R * sqrt(3)/2 >= max_val => R >= max_val / (sqrt(3)/2)
    required_R = max_val / (np.sqrt(3) / 2)
    return required_R


def initial_placement():
    """Create initial hexagonal grid placement of 12 unit hexagons."""
    # Hexagonal grid: arrange in a 4x3 or 3x4 pattern
    # For 12 hexagons, a nice arrangement is 4 columns x 3 rows with offset
    centers = []
    angles = []
    
    # Hex grid spacing: for unit hexagons (side=1), the distance between centers
    # in a hex grid is sqrt(3) for adjacent hexagons (flat-top orientation)
    # or 2 for pointy-top... let me just use a reasonable spacing.
    
    # Let's use a 4x3 grid with hex offset
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons (offset by half)
    # Row 2: 4 hexagons
    
    # Actually for 12, let's do:
    # 3 rows of 4, with alternating rows offset
    dx = 1.75  # horizontal spacing (slightly more than sqrt(3) for safety)
    dy = 1.5   # vertical spacing
    
    for row in range(3):
        for col in range(4):
            x = col * dx + (row % 2) * dx * 0.5
            y = row * dy
            centers.append([x - 1.5 * dx, y - dy])  # center the arrangement
            angles.append(0.0)
    
    return centers, angles


def objective_separation(params, outer_center, outer_angle, outer_side):
    """Objective: negative of minimum separation (to maximize it).
    params = [x1,y1,ang1, x2,y2,ang2, ..., x12,y12,ang12]"""
    n = N
    centers = []
    angles = []
    for i in range(n):
        centers.append(params[3*i:3*i+2])
        angles.append(params[3*i+2])
    
    min_sep = np.inf
    
    # Separation between pairs of inner hexagons
    for i in range(n):
        for j in range(i+1, n):
            v1 = hexagon_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            v2 = hexagon_vertices(centers[j][0], centers[j][1], 1.0, angles[j])
            d = min_distance_hexagons(v1, v2)
            if d < min_sep:
                min_sep = d
    
    # Separation from outer boundary
    outer_verts = hexagon_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    for i in range(n):
        v1 = hexagon_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
        # Distance from inner hex to outer boundary
        # Min distance from any inner vertex to any outer edge
        for v in v1:
            for k in range(6):
                d = _pt_seg_dist(v, outer_verts[k], outer_verts[(k+1) % 6])
                if d < min_sep:
                    min_sep = d
    
    # We want to maximize min_sep, so minimize -min_sep
    # Add a small penalty if any overlap (min_sep < 0)
    return -min_sep


def construct():
    start_time = time.time()
    
    # Initial placement
    centers, angles = initial_placement()
    n = N
    
    # Initial outer hexagon: estimate from bounding box
    all_pts = []
    for c, a in zip(centers, angles):
        verts = hexagon_vertices(c[0], c[1], 1.0, a)
        all_pts.extend(verts.tolist())
    all_pts = np.array(all_pts)
    
    cx = np.mean(all_pts[:, 0])
    cy = np.mean(all_pts[:, 1])
    
    # Estimate side length from max extent
    max_r = np.max(np.linalg.norm(all_pts - np.array([cx, cy]), axis=1))
    outer_side = max_r * 1.2  # add margin
    outer_center = [cx, cy]
    outer_angle = 0.0
    
    best_outer_side = outer_side
    best_params = None
    
    # Phase 1: Optimize inner hexagon positions and angles
    # Parameters: [x1, y1, ang1, x2, y2, ang2, ..., x12, y12, ang12]
    x0 = []
    for c, a in zip(centers, angles):
        x0.extend([c[0], c[1], np.radians(a)])
    
    # Bound constraints: keep hexagons in a reasonable region
    bounds = []
    for i in range(n):
        bounds.append((-20, 20))  # x
        bounds.append((-20, 20))  # y
        bounds.append((-np.pi, np.pi))  # angle
    
    # Optimize to maximize minimum separation
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 30:
        res = minimize(
            objective_separation, x0,
            args=(outer_center, outer_angle, outer_side),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-6}
        )
        if res.success or res.fun < 0:
            x_opt = res.x
        else:
            x_opt = x0
    else:
        x_opt = x0
    
    # Extract optimized positions
    opt_centers = []
    opt_angles = []
    for i in range(n):
        opt_centers.append(x_opt[3*i:3*i+2])
        opt_angles.append(np.degrees(x_opt[3*i+2]))
    
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 20:
        # Phase 2: Optimize outer hexagon to tightly fit the inner arrangement
        # For each candidate outer angle, compute the minimal side length
        best_side = np.inf
        best_outer = None
        
        for trial_angle in [0, 15, 30, 45, 60]:
            # For this outer angle, find the best center and side
            # The center should be near the centroid of inner hexagons
            centroid = np.mean(opt_centers, axis=0)
            
            # Try a few center offsets
            for dx in [-0.5, 0, 0.5]:
                for dy in [-0.5, 0, 0.5]:
                    oc = [centroid[0] + dx, centroid[1] + dy]
                    side = compute_min_outer_side(opt_centers, opt_angles, oc, trial_angle, 100)
                    if side < best_side:
                        best_side = side
                        best_outer = (oc, trial_angle)
        
        if best_outer is not None:
            outer_center, outer_angle = best_outer
            outer_side = best_side
            best_outer_side = outer_side
    
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 15:
        # Phase 3: Joint refinement - alternate inner and outer optimization
        for iteration in range(3):
            if time.time() - start_time > TIME_LIMIT - 15:
                break
            
            # Recompute optimal outer for current inner
            centroid = np.mean(opt_centers, axis=0)
            best_side = np.inf
            best_outer = None
            for trial_angle in [0, 15, 30, 45, 60]:
                for dx in [-0.3, 0, 0.3]:
                    for dy in [-0.3, 0, 0.3]:
                        oc = [centroid[0] + dx, centroid[1] + dy]
                        side = compute_min_outer_side(opt_centers, opt_angles, oc, trial_angle, 100)
                        if side < best_side:
                            best_side = side
                            best_outer = (oc, trial_angle)
            
            if best_outer is not None:
                outer_center, outer_angle = best_outer
                outer_side = best_side
            
            # Re-optimize inner with new outer
            x0 = []
            for c, a in zip(opt_centers, opt_angles):
                x0.extend([c[0], c[1], np.radians(a)])
            
            res = minimize(
                objective_separation, x0,
                args=(outer_center, outer_angle, outer_side),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-7}
            )
            if res.success:
                x_opt = res.x
                for i in range(n):
                    opt_centers[i] = x_opt[3*i:3*i+2]
                    opt_angles[i] = np.degrees(x_opt[3*i+2])
            
            # Update best
            centroid = np.mean(opt_centers, axis=0)
            for trial_angle in [0, 15, 30, 45, 60]:
                oc = [centroid[0], centroid[1]]
                side = compute_min_outer_side(opt_centers, opt_angles, oc, trial_angle, 100)
                if side < best_outer_side:
                    best_outer_side = side
                    outer_center = oc
                    outer_angle = trial_angle
    
    # Final: compute the true minimal outer side
    centroid = np.mean(opt_centers, axis=0)
    final_best = np.inf
    for trial_angle in range(0, 360, 10):
        for dx in np.linspace(-0.5, 0.5, 5):
            for dy in np.linspace(-0.5, 0.5, 5):
                oc = [centroid[0] + dx, centroid[1] + dy]
                side = compute_min_outer_side(opt_centers, opt_angles, oc, trial_angle, 100)
                if side < final_best:
                    final_best = side
                    outer_center = oc
                    outer_angle = trial_angle
    
    outer_side = final_best
    
    # Build inner list
    inner = []
    for c, a in zip(opt_centers, opt_angles):
        inner.append([float(c[0]), float(c[1]), float(a)])
    
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)


if __name__ == '__main__':
    result = construct()
    print(f"Outer side length: {result[2]:.6f}")
    print(f"Outer center: {result[1]}, angle: {result[3]}")
    for i, row in enumerate(result[0]):
        print(f"  Inner {i}: ({row[0]:.4f}, {row[1]:.4f}, {row[2]:.2f} deg)")
