import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12
DEADLINE = 145.0  # Safety margin before 150s wall-clock kill

def construct():
    t0 = time.time()
    
    # Precompute unit hexagon vertices (angle 0)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    angles_hex = np.radians(np.arange(0, 360, 60))
    unit_hex_verts = np.column_stack((np.cos(angles_hex), np.sin(angles_hex)))
    
    # Helper: Get vertices of a single hexagon
    def get_hex_verts(cx, cy, angle_rad):
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        ux = unit_hex_verts[:, 0]
        uy = unit_hex_verts[:, 1]
        vx = cx + ux * c - uy * s
        vy = cy + ux * s + uy * c
        return np.column_stack((vx, vy))
    
    # Helper: Fast SAT overlap check for two sets of vertices
    # v1, v2: (6, 2) arrays
    def sat_overlap_depth(v1, v2):
        # Compute normals for v1
        edges1 = np.roll(v1, -1, axis=0) - v1
        normals1 = np.column_stack((edges1[:, 1], -edges1[:, 0]))
        norms1 = np.linalg.norm(normals1, axis=1)
        normals1 = normals1 / (norms1[:, np.newaxis] + 1e-15)
        
        # Compute normals for v2
        edges2 = np.roll(v2, -1, axis=0) - v2
        normals2 = np.column_stack((edges2[:, 1], -edges2[:, 0]))
        norms2 = np.linalg.norm(normals2, axis=1)
        normals2 = normals2 / (norms2[:, np.newaxis] + 1e-15)
        
        # Combine normals
        normals = np.vstack([normals1, normals2])
        
        # Project both polygons onto all normals
        # p1: (6, 12), p2: (6, 12)
        p1 = v1 @ normals.T
        p2 = v2 @ normals.T
        
        min1 = np.min(p1, axis=0)
        max1 = np.max(p1, axis=0)
        min2 = np.min(p2, axis=0)
        max2 = np.max(p2, axis=0)
        
        # Overlap depth on each axis
        # If min(max1, max2) < max(min1, min2), there is a gap (negative overlap)
        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        
        # The minimum overlap depth is the maximum gap
        # If min_overlap > 0, they are disjoint. If <= 0, they overlap.
        min_overlap = np.min(overlaps)
        
        return min_overlap

    # Helper: Compute minimum outer hexagon side length
    # hexes: list of 12 (6, 2) arrays
    def compute_min_side(hexes):
        # Outer hexagon normals (fixed orientation for optimization, we can rotate result later or just report side)
        # Normals of a regular hexagon at angles 30, 90, 150, 210, 270, 330
        normal_angles = np.radians([30, 90, 150, 210, 270, 330])
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        
        max_proj = 0.0
        for h in hexes:
            proj = h @ normals.T
            # We need the maximum absolute projection if centered at 0?
            # Actually, the outer hexagon center is not necessarily 0 in the local frame, 
            # but for "minimizing side length", we can assume the outer hexagon is centered 
            # such that it contains the points.
            # The width of the bounding hexagon is determined by the max extent in each normal direction.
            # For a regular hexagon of side s, the distance from center to side is s*sqrt(3)/2.
            # The extent in direction n is (max_proj - min_proj) / 2 + center_offset?
            # Simpler: The minimal enclosing regular hexagon of fixed orientation has side length 
            # determined by the maximum of the "widths" in the 3 main directions? No, 6 directions.
            
            # Let's find the minimal s such that all points are inside.
            # If we center the outer hexagon at the centroid of the points, 
            # the required half-width in direction n is max(|proj - centroid_proj|).
            # But we can shift the center. The minimal side length is achieved when the center is 
            # positioned to minimize the max extent.
            
            # Actually, for a fixed orientation, the minimal enclosing regular hexagon side length S
            # is given by: S = (2/sqrt(3)) * max_k ( max_i (v_i . n_k) - min_i (v_i . n_k) ) / 2 ?
            # No, that's for the bounding box.
            
            # Let's use the property: A regular hexagon is the intersection of 6 half-planes.
            # The minimal side length is determined by the largest "span" in any of the 6 normal directions?
            # No, the span in direction n_k is max(v.n_k) - min(v.n_k).
            # The distance between parallel sides is the span.
            # For a regular hexagon of side s, the distance between parallel sides is s*sqrt(3).
            # So s = span / sqrt(3).
            # We need s such that span_k <= s*sqrt(3) for all k.
            # So s >= max_k(span_k) / sqrt(3).
            
            spans = np.max(proj, axis=0) - np.min(proj, axis=0)
            max_span = np.max(spans)
            if max_span > max_proj:
                max_proj = max_span
                
        s = max_proj / math.sqrt(3.0)
        return s

    def objective(params):
        if time.time() - t0 > DEADLINE:
            return 1e10
            
        x1, y1, a1, x2, y2, a2 = params
        
        # Generate 12 hexagons using 6-fold symmetry
        # Base hexagons
        base1 = get_hex_verts(x1, y1, a1)
        base2 = get_hex_verts(x2, y2, a2)
        
        hexes = []
        for k in range(6):
            angle = np.radians(60 * k)
            c = np.cos(angle)
            s_ = np.sin(angle)
            rot = np.array([[c, -s_], [s_, c]])
            
            # Rotate base1 and base2
            h1 = base1 @ rot.T
            h2 = base2 @ rot.T
            
            hexes.append(h1)
            hexes.append(h2)
        
        # Calculate minimal outer side
        s_min = compute_min_side(hexes)
        
        penalty = 0.0
        
        # Check overlaps
        for i in range(12):
            for j in range(i + 1, 12):
                # Quick bounding box check
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap_depth = sat_overlap_depth(hexes[i], hexes[j])
                if overlap_depth <= 0:
                    # Overlapping, penalty proportional to depth
                    penalty += (-overlap_depth) * 10.0
                    
        return s_min + penalty

    # Bounds for the 2 base hexagons
    # x, y in [-4, 4]
    # angle in [0, pi/3] (symmetry of hexagon)
    bounds = [
        (-4.0, 4.0),  # x1
        (-4.0, 4.0),  # y1
        (0.0, np.pi/3), # a1
        (-4.0, 4.0),  # x2
        (-4.0, 4.0),  # y2
        (0.0, np.pi/3), # a2
    ]
    
    best_s = float('inf')
    best_params = None
    
    # 1. Differential Evolution for global search
    try:
        res = differential_evolution(
            objective,
            bounds,
            maxiter=200,
            popsize=15,
            tol=1e-5,
            seed=42,
            polish=False, # We will polish manually to control time
            init='sobol'
        )
        
        if res.fun < best_s:
            best_s = res.fun
            best_params = res.x
            
    except Exception:
        pass
    
    # 2. Local Refinement with L-BFGS-B
    if best_params is not None:
        try:
            res_local = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if res_local.fun < best_s:
                best_s = res_local.fun
                best_params = res_local.x
        except Exception:
            pass
            
    # 3. Try a few random starts if time permits or if DE failed
    if time.time() - t0 < DEADLINE - 10:
        for _ in range(5):
            if time.time() - t0 > DEADLINE - 5:
                break
            p = np.array([
                np.random.uniform(1.0, 3.0), np.random.uniform(-0.5, 0.5), np.random.uniform(0, np.pi/3),
                np.random.uniform(2.0, 4.0), np.random.uniform(-0.5, 0.5), np.random.uniform(0, np.pi/3)
            ])
            val = objective(p)
            if val < best_s:
                best_s = val
                best_params = p
            try:
                res_local = minimize(
                    objective,
                    p,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50}
                )
                if res_local.fun < best_s:
                    best_s = res_local.fun
                    best_params = res_local.x
            except Exception:
                pass

    if best_params is None:
        # Fallback: Simple grid
        x1, y1, a1, x2, y2, a2 = 2.0, 0.0, 0.0, 4.0, 0.0, 0.0
        best_params = np.array([x1, y1, a1, x2, y2, a2])
        best_s = objective(best_params)

    x1, y1, a1, x2, y2, a2 = best_params
    
    # Generate the 12 inner hexagons
    inner = []
    for k in range(6):
        angle = np.radians(60 * k)
        c = np.cos(angle)
        s_ = np.sin(angle)
        
        # Rotate (x1, y1)
        x1_rot = x1 * c - y1 * s_
        y1_rot = x1 * s_ + y1 * c
        a1_rot = a1 + angle
        
        # Rotate (x2, y2)
        x2_rot = x2 * c - y2 * s_
        y2_rot = x2 * s_ + y2 * c
        a2_rot = a2 + angle
        
        inner.append([float(x1_rot), float(y1_rot), float(np.degrees(a1_rot))])
        inner.append([float(x2_rot), float(y2_rot), float(np.degrees(a2_rot))])
        
    # Calculate final outer side length and center
    # We need to return the center of the outer hexagon.
    # The outer hexagon is centered at the origin in our calculation?
    # No, `compute_min_side` assumed the outer hexagon could be shifted.
    # The side length is correct. The center of the minimal enclosing hexagon 
    # is the midpoint of the min and max projections in each direction?
    # Actually, for a regular hexagon, if we align it with the normals, 
    # the center is the average of the extreme points?
    
    # Let's compute the center of the bounding hexagon.
    # The bounding hexagon is defined by the constraints:
    # n_k . (v - c) <= s*sqrt(3)/2
    # To minimize s, we chose s based on max span.
    # The center c should be such that the hexagon is "centered" on the points.
    # A good approximation is the centroid of all vertices.
    
    all_verts = []
    for h in [get_hex_verts(i[0], i[1], np.radians(i[2])) for i in inner]:
        all_verts.append(h)
    all_verts = np.vstack(all_verts)
    center = np.mean(all_verts, axis=0)
    
    # Recompute side length with this specific center to be precise?
    # The `compute_min_side` function effectively finds the side length of the 
    # hexagon of fixed orientation that contains the points, assuming optimal centering.
    # The value returned is the side length.
    
    # The angle of the outer hexagon is 0 (or 30 deg depending on normal def).
    # In `compute_min_side`, normals were at 30, 90, etc.
    # This corresponds to a hexagon with flat top/bottom?
    # Normal at 90 is vertical. So the side is horizontal.
    # Angle of rotation for the outer hexagon:
    # If vertices are at 0, 60..., normals are at 30, 90...
    # So the hexagon is "pointy" on the sides? No, normal 90 means top side is horizontal.
    # Standard hexagon angle 0 usually means vertex at (1,0).
    # If vertex at (1,0), normal to side (0, 30 deg) is at 90 deg?
    # Side from (1,0) to (0.5, 0.866). Midpoint (0.75, 0.433). Normal is perpendicular.
    # Vector (1,0) to (0.5, 0.866) is (-0.5, 0.866). Normal is (0.866, 0.5) -> 30 deg.
    # So yes, angle 0 hexagon has normals at 30, 90, 150...
    # So the outer hexagon angle is 0.
    
    side = float(best_s)
    angle = 0.0
    
    return inner, (float(center[0]), float(center[1])), side, angle

# EVOLVE-BLOCK-END
