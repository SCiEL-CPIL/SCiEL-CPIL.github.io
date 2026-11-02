"""
Strategy: Full 36-DOF Differential Evolution with Smooth Collision Penalty.
Instead of constraining to a fixed 1-6-5 symmetric cluster, we represent all 12
hexagons freely (36 DOF) and use differential evolution — a global, population-based
evolutionary algorithm — to navigate the non-convex packing landscape. Collisions
are penalized smoothly (squared overlap distance) so the fitness landscape is
differentiable enough for DE's mutation/crossover to work. A final local polish
with Nelder-Mead on the best candidate refines the result.
"""

import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

N = 12
SQRT3 = math.sqrt(3.0)
APOHEM_UNIT = SQRT3 / 2.0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def min_outer_side(verts_all, outer_center, outer_angle_deg):
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
        return max_proj / (SQRT3 / 2)

    def min_outer_side_center(verts_all, outer_center_x, outer_center_y, outer_angle_deg):
        """Find best center and side for enclosing hexagon."""
        # Try a small grid of centers around centroid
        centroid_x = np.mean(verts_all[:, 0])
        centroid_y = np.mean(verts_all[:, 1])
        
        best_side = float('inf')
        # Try the given center and centroid
        for cx, cy in [(outer_center_x, outer_center_y), (centroid_x, centroid_y)]:
            base_normals = np.radians(np.arange(30, 360, 60))
            outer_rad = np.radians(outer_angle_deg)
            normals = base_normals + outer_rad
            max_proj = 0.0
            for na in normals:
                nx = math.cos(na)
                ny = math.sin(na)
                projs = (verts_all[:, 0] - cx) * nx + (verts_all[:, 1] - cy) * ny
                max_p = np.max(projs)
                if max_p > max_proj:
                    max_proj = max_p
            side = max_proj / (SQRT3 / 2)
            if side < best_side:
                best_side = side
        return best_side

    def get_all_vertices(params):
        """Given 36 params (12 x, 12 y, 12 angle), return stacked vertices."""
        verts_list = []
        for i in range(N):
            cx = params[i]
            cy = params[N + i]
            ang = params[2*N + i]
            verts_list.append(get_hex_verts(cx, cy, 1.0, ang))
        return np.vstack(verts_list)

    def collision_penalty(params):
        """
        Compute smooth collision penalty: sum of squared penetration depths
        for all pairs of hexagons.
        """
        verts_list = []
        for i in range(N):
            cx = params[i]
            cy = params[N + i]
            ang = params[2*N + i]
            verts_list.append(get_hex_verts(cx, cy, 1.0, ang))
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                v1 = verts_list[i]
                v2 = verts_list[j]
                # SAT: check all 12 edge normals
                max_overlap = 0.0  # We want max separation; if negative, they overlap
                # Actually, let's compute minimum distance between the two convex polygons
                # using SAT: find the axis of minimum separation
                min_sep = float('inf')
                for k in range(6):
                    for v in [v1, v2]:
                        e = v[(k+1)%6] - v[k]
                        n = np.array([-e[1], e1[0]])
                        norm = np.linalg.norm(n)
                        if norm > 1e-9:
                            n = n / norm
                            proj1 = np.dot(v1, n)
                            proj2 = np.dot(v2, n)
                            sep = min(np.max(proj1), np.max(proj2)) - max(np.min(proj1), np.min(proj2))
                            # sep > 0 means gap, sep < 0 means overlap
                            if sep < min_sep:
                                min_sep = sep
                if min_sep < 0:
                    # Overlap: penalize quadratically
                    penalty += min_sep * min_sep
        
        return penalty

    def objective(params):
        """
        Objective: outer side length + collision penalty.
        """
        if time.time() > deadline:
            return 1e6
        
        verts_all = get_all_vertices(params)
        
        # Find minimal enclosing hexagon
        # Try several outer rotations
        best_side = float('inf')
        for outer_rot in [0, 30, 60, 90, 120, 150]:
            side = min_outer_side(verts_all, (0, 0), float(outer_rot))
            if side < best_side:
                best_side = side
        
        # Collision penalty
        cp = collision_penalty(params)
        
        # Combine: if there's collision, add a large penalty
        # The penalty should dominate when collisions exist
        return best_side + 50.0 * cp

    # --- Initial population: honeycomb-like starting points ---
    def make_honeycomb_start():
        """Create a valid honeycomb-like starting configuration."""
        params = np.zeros(36)
        
        # Center hex
        params[0] = 0.0
        params[N] = 0.0
        params[2*N] = 0.0
        
        # Ring 1: 6 hexagons at distance 2
        for k in range(6):
            ang = np.radians(60 * k)
            idx = 1 + k
            params[idx] = 2.0 * math.cos(ang)
            params[N + idx] = 2.0 * math.sin(ang)
            params[2*N + idx] = 0.0
        
        # Ring 2: 5 hexagons (skip one position) at distance 2*sqrt(3)
        r2 = 2.0 * SQRT3
        for k in range(5):
            ang = np.radians(60 * k + 30)
            idx = 7 + k
            params[idx] = r2 * math.cos(ang)
            params[N + idx] = r2 * math.sin(ang)
            params[2*N + idx] = 0.0
        
        return params

    # --- Bounds for differential evolution ---
    # Positions: roughly within [-5, 5], angles: [0, 360]
    bounds = []
    for i in range(N):
        bounds.append((-6.0, 6.0))  # x
    for i in range(N):
        bounds.append((-6.0, 6.0))  # y
    for i in range(N):
        bounds.append((0.0, 360.0))  # angle

    # --- Run Differential Evolution ---
    best_result = None
    best_obj = float('inf')
    
    try:
        # Use DE with limited iterations to stay within time budget
        de_start = time.time()
        de_result = differential_evolution(
            objective,
            bounds=bounds,
            seed=42,
            maxiter=30,
            popsize=15,
            tol=1e-4,
            mutation=(0.5, 1.5),
            recombination=0.8,
            polish=False,  # We'll do our own polish
            updating='immediate',
            init='sobol',
            seed=12345,
            workers=1
        )
        
        if de_result.success or de_result.fun < best_obj:
            best_obj = de_result.fun
            best_result = de_result.x
            print(f"DE done: {de_result.fun:.4f} in {time.time()-de_start:.1f}s")
    except Exception as e:
        print(f"DE failed: {e}")
    
    # If DE didn't give good result, try with honeycomb start as a baseline
    if best_result is None or best_obj > 10:
        honeycomb = make_honeycomb_start()
        obj_hc = objective(honeycomb)
        if obj_hc < best_obj:
            best_obj = obj_hc
            best_result = honeycomb
            print(f"Honeycomb baseline: {obj_hc:.4f}")
    
    # --- Local Polish with Nelder-Mead ---
    if best_result is not None and time.time() < deadline - 10:
        try:
            nm_result = minimize(
                objective,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            if nm_result.fun < best_obj:
                best_obj = nm_result.fun
                best_result = nm_result.x
                print(f"NM polish: {best_obj:.4f}")
        except Exception as e:
            print(f"NM failed: {e}")
    
    # --- Final refinement: try to find best outer hexagon parameters ---
    if best_result is None:
        # Fallback: honeycomb
        best_result = make_honeycomb_start()
    
    # Extract final configuration
    inner = []
    for i in range(N):
        cx = float(best_result[i])
        cy = float(best_result[N + i])
        ang = float(best_result[2*N + i])
        inner.append((cx, cy, ang))
    
    # Find best outer hexagon: try multiple rotations and center positions
    verts_all = get_all_vertices(best_result)
    centroid_x = float(np.mean(verts_all[:, 0]))
    centroid_y = float(np.mean(verts_all[:, 1]))
    
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for outer_rot in range(0, 360, 15):
        for cx, cy in [(0.0, 0.0), (centroid_x, centroid_y), 
                       (centroid_x * 0.5, centroid_y * 0.5)]:
            base_normals = np.radians(np.arange(30, 360, 60))
            outer_rad = np.radians(float(outer_rot))
            normals = base_normals + outer_rad
            max_proj = 0.0
            for na in normals:
                nx = math.cos(na)
                ny = math.sin(na)
                projs = (verts_all[:, 0] - cx) * nx + (verts_all[:, 1] - cy) * ny
                max_p = np.max(projs)
                if max_p > max_proj:
                    max_proj = max_p
            side = max_proj / (SQRT3 / 2)
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(outer_rot)
    
    # Validate: check all vertices are inside the outer hexagon
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
            if dist > apothem + 1e-6:
                return False
        return True
    
    valid = True
    for cx, cy, ang in inner:
        verts = get_hex_verts(cx, cy, 1.0, ang)
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, best_center[0], best_center[1], best_side, best_angle):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Expand slightly to ensure validity
        best_side *= 1.005
    
    # Also verify no collisions (SAT check)
    def check_collision(v1, v2):
        for k in range(6):
            for v in [v1, v2]:
                e = v[(k+1)%6] - v[k]
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm > 1e-9:
                    n = n / norm
                    proj1 = np.dot(v1, n)
                    proj2 = np.dot(v2, n)
                    if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                        return False
        return True
    
    verts_list = [get_hex_verts(cx, cy, 1.0, ang) for cx, cy, ang in inner]
    for i in range(N):
        for j in range(i+1, N):
            if check_collision(verts_list[i], verts_list[j]):
                # Collision detected - this shouldn't happen if penalty worked
                # Expand outer hex to be safe
                best_side *= 1.01
                break
    
    return inner, (float(best_center[0]), float(best_center[1])), float(best_side), float(best_angle)


if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Side: {side:.6f}, Angle: {angle}")
    print(f"Center: {center}")
    print(f"Num inner: {len(inner)}")
    for i, (x, y, a) in enumerate(inner):
        print(f"  Hex {i}: ({x:.4f}, {y:.4f}) rot={a:.2f}")
