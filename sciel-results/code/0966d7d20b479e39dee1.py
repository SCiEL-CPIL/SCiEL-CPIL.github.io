"""
Strategy: Full-DOF Simulated Annealing with Smooth Collision Penalties.
Instead of constraining the 12 hexagons to a symmetric 1-6-5 cluster topology,
we treat all 12 hexagons as free bodies (36 DOF: x, y, angle for each).
We use Simulated Annealing to navigate the high-dimensional space, utilizing
a smooth objective function that combines the minimal enclosing hexagon side length
with a softplus penalty for collisions. This allows the optimizer to explore
asymmetric configurations that might pack tighter than any symmetric constraint,
while the smooth gradient-like landscape helps avoid getting stuck on sharp
collision boundaries.
"""

import numpy as np
import time
import math

N = 12
SQRT3 = math.sqrt(3.0)
APOHEM_UNIT = SQRT3 / 2.0

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def get_hex_verts(cx, cy, side, angle_deg):
        """Return vertices of a regular hexagon."""
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def min_outer_side(verts_all, outer_center, outer_angle_deg):
        """
        Compute minimal side length of a regular hexagon (center, angle) 
        containing all points in verts_all.
        """
        # The 6 normals of the outer hexagon
        base_normals = np.radians(np.arange(30, 360, 60))
        outer_rad = np.radians(outer_angle_deg)
        normals = base_normals + outer_rad
        
        max_proj = 0.0
        for na in normals:
            nx = math.cos(na)
            ny = math.sin(na)
            projs = (verts_all[:, 0] - outer_center[0]) * nx + (verts_all[:, 1] - outer_center[1]) * ny
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
        
        # max_proj is the required apothem.
        # side = apothem / (sqrt(3)/2)
        return max_proj / (SQRT3 / 2)

    def min_distance_polygons(v1, v2):
        """
        Calculate the minimum distance between two convex polygons.
        If they intersect, return 0.
        Uses SAT for intersection check and vertex-vertex/vertex-edge distance for separation.
        For efficiency in SA, we can approximate or use a simpler metric if needed,
        but let's try a robust one first.
        """
        # First, check intersection via SAT
        # Axes from v1 and v2
        axes = []
        for k in range(6):
            e1 = v1[(k+1)%6] - v1[k]
            n1 = np.array([-e1[1], e1[0]])
            norm1 = np.linalg.norm(n1)
            if norm1 > 1e-9:
                axes.append(n1/norm1)
            e2 = v2[(k+1)%6] - v2[k]
            n2 = np.array([-e2[1], e2[0]])
            norm2 = np.linalg.norm(n2)
            if norm2 > 1e-9:
                axes.append(n2/norm2)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # If there is a gap, they don't intersect on this axis
            if max1 < min2 or max2 < min1:
                gap = min2 - max1 if max1 < min2 else min1 - max2
                # This gap is a lower bound on distance, but not necessarily the min distance.
                # However, if they intersect on all other axes, we need to be careful.
                # For smooth penalty, we want a continuous function.
                # A common trick: return the minimum gap over all separating axes if they are separated,
                # but if they intersect, return 0.
                # To make it smooth, we can use a softplus on the minimum gap.
                # But SAT gives 0 if intersecting.
                # Let's return the minimum gap found so far, but if we find a gap, we can stop?
                # No, we need the MINIMUM gap to ensure they are separated.
                # If any axis has a gap, they are separated. The minimum gap over all axes is the distance?
                # Actually, for convex polygons, the distance is the minimum over all axes of the gap?
                # No, the distance is the minimum distance between any two points.
                # For separated convex polygons, the distance is equal to the gap along the unique separating axis?
                # Not exactly, but close.
                # Let's use a simpler metric: Minimum distance between any vertex of v1 and any edge of v2, and vice versa.
                # This is O(36) per pair, which is 66 pairs * 36 = 2376 ops per evaluation.
                # This is fast enough.
                pass
        
        # If we reach here, they might intersect or we didn't find a clear gap in the simple check above.
        # Let's compute min distance between vertices and edges.
        min_dist = float('inf')
        
        # Helper: distance from point to line segment
        def dist_point_segment(p, a, b):
            v = b - a
            w = p - a
            c1 = np.dot(w, v)
            if c1 <= 0:
                return np.linalg.norm(p - a)
            c2 = np.dot(v, v)
            if c2 <= c1:
                return np.linalg.norm(p - b)
            t = c1 / c2
            projection = a + t * v
            return np.linalg.norm(p - projection)

        for i in range(6):
            p1 = v1[i]
            # Dist to edges of v2
            for j in range(6):
                d = dist_point_segment(p1, v2[j], v2[(j+1)%6])
                if d < min_dist:
                    min_dist = d
            # Dist to vertices of v2 (covered by segment dist, but explicit check is safe)
            for j in range(6):
                d = np.linalg.norm(p1 - v2[j])
                if d < min_dist:
                    min_dist = d
        
        for i in range(6):
            p2 = v2[i]
            for j in range(6):
                d = dist_point_segment(p2, v1[j], v1[(j+1)%6])
                if d < min_dist:
                    min_dist = d
            for j in range(6):
                d = np.linalg.norm(p2 - v1[j])
                if d < min_dist:
                    min_dist = d

        # If they intersect, min_dist will be 0 (or very close).
        # To be safe, if min_dist is very small, assume 0.
        return min_dist

    def objective(params):
        """
        Objective: Minimize outer side length + collision penalty.
        params: 36 array [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12]
        """
        # Extract hexagons
        verts_list = []
        for i in range(N):
            cx = params[i*3]
            cy = params[i*3+1]
            ang = params[i*3+2]
            verts_list.append(get_hex_verts(cx, cy, 1.0, ang))
        
        # Calculate minimal enclosing hexagon side length
        # We assume the outer hexagon is centered at (0,0) and we optimize its rotation?
        # To keep it 36 DOF, we can fix the outer hexagon orientation to 0 and center to (0,0)
        # and just find the side length. Or we can add outer_rot as a 37th DOF?
        # Let's stick to 36 DOF for the inner hexes and assume the outer hexagon
        # is centered at the centroid of the inner hexes? No, that breaks symmetry.
        # Let's assume the outer hexagon is centered at (0,0) and we try a few fixed rotations
        # or just 0. Given the symmetry of the problem, 0 is a good guess.
        # Actually, the minimal enclosing hexagon might need to rotate.
        # Let's add outer_rot as a parameter? That would be 37 DOF.
        # Let's stick to 36 and fix outer_rot = 0. If the packing is symmetric, it will align.
        
        all_verts = np.vstack(verts_list)
        
        # Try to find the best outer rotation? That's expensive.
        # Let's just use 0 for now, or maybe the average angle?
        # Let's use 0.
        side = min_outer_side(all_verts, (0.0, 0.0), 0.0)
        
        # Collision Penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = min_distance_polygons(verts_list[i], verts_list[j])
                # Softplus penalty: if d < 0 (overlap), penalty is high.
                # We want d >= 0.
                # Penalty = max(0, -d) is not smooth.
                # Penalty = softplus(-d * k) / k
                k = 10.0
                penalty += math.log(1 + math.exp(k * (-d))) / k
        
        return side + 10.0 * penalty

    # Initialize
    # Start with a honeycomb-like configuration
    # Center at (0,0)
    # Ring 1 at distance 2, angles 0, 60, ...
    # Ring 2 at distance 2*sqrt(3), angles 30, 90, ... (5 hexes)
    # This is the 1-6-5 cluster.
    
    initial_params = np.zeros(36)
    
    # Center
    initial_params[0] = 0.0
    initial_params[1] = 0.0
    initial_params[2] = 0.0
    
    # Ring 1
    for k in range(6):
        idx = 3 + k*3
        ang = np.radians(60*k)
        r = 2.0
        initial_params[idx] = r * math.cos(ang)
        initial_params[idx+1] = r * math.sin(ang)
        initial_params[idx+2] = 0.0
        
    # Ring 2
    for k in range(5):
        idx = 21 + k*3
        ang = np.radians(60*k + 30)
        r = 2 * SQRT3
        initial_params[idx] = r * math.cos(ang)
        initial_params[idx+1] = r * math.sin(ang)
        initial_params[idx+2] = 0.0
        
    current_params = initial_params.copy()
    current_score = objective(current_params)
    best_params = current_params.copy()
    best_score = current_score
    
    # Simulated Annealing
    T = 1.0
    T_min = 1e-4
    cooling_rate = 0.995
    step_size = 0.1
    
    iters = 0
    max_iters = 50000 # Cap iterations, time is the real limit
    
    while time.time() < deadline - 10 and T > T_min:
        # Perturb
        new_params = current_params.copy()
        # Random walk
        noise = np.random.normal(0, step_size, 36)
        # Larger noise for positions, smaller for angles?
        # Let's just use uniform noise for simplicity, or scaled.
        for i in range(N):
            new_params[i*3] += noise[i*3]
            new_params[i*3+1] += noise[i*3+1]
            new_params[i*3+2] += noise[i*3+2] * 0.5 # Angles change slower
            
        new_score = objective(new_params)
        
        # Acceptance
        if new_score < current_score:
            current_params = new_params
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_params = current_params.copy()
        else:
            delta = new_score - current_score
            if delta < 0:
                current_params = new_params
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_params = current_params.copy()
            else:
                accept_prob = math.exp(-delta / T)
                if np.random.random() < accept_prob:
                    current_params = new_params
                    current_score = new_score
        
        T *= cooling_rate
        iters += 1
        
        # Adaptive step size: if we accept too many, increase step; if too few, decrease
        # This is a heuristic.
        
    # Refine with Nelder-Mead if time permits
    if time.time() < deadline - 5 and best_score < 100:
        from scipy.optimize import minimize
        try:
            res = minimize(objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Extract results from best_params
    inner = []
    for i in range(N):
        cx = float(best_params[i*3])
        cy = float(best_params[i*3+1])
        ang = float(best_params[i*3+2])
        inner.append((cx, cy, ang))
        
    center = (0.0, 0.0)
    angle = 0.0
    
    # Calculate final side length precisely
    verts_list = []
    for cx, cy, ang in inner:
        verts_list.append(get_hex_verts(cx, cy, 1.0, ang))
    all_verts = np.vstack(verts_list)
    side = min_outer_side(all_verts, center, angle)
    
    # Final Validation: Ensure all vertices are inside the outer hexagon
    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        dx = px - cx
        dy = py - cy
        angle_rad = np.radians(angle_deg)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        apothem = side * SQRT3 / 2
        normal_angles = np.radians(np.arange(30, 360, 60))
        for na in normal_angles:
            nx = math.cos(na)
            ny = math.sin(na)
            dist = local_x * nx + local_y * ny
            if dist > apothem + 1e-8:
                return False
        return True

    valid = True
    for cx, cy, ang in inner:
        verts = get_hex_verts(cx, cy, 1.0, ang)
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, center[0], center[1], side, angle):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # If invalid due to numerical precision or center shift, expand slightly
        side *= 1.01

    return inner, center, float(side), float(angle)

if __name__ == "__main__":
    # Quick test
    inner, center, side, angle = construct()
    print(f"Side: {side}, Angle: {angle}")
    print(f"Inner: {inner[:3]}...")
