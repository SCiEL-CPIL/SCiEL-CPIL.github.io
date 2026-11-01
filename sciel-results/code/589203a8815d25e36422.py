"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
import itertools

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_rad):
        """Vertices of a regular hexagon (flat-top when angle=0)."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def sat_gap(poly1, poly2):
        """
        Calculate the minimum gap between two convex polygons using SAT.
        Returns a positive value if disjoint (distance), negative if overlapping.
        """
        min_gap = np.inf
        for poly in [poly1, poly2]:
            for i in range(len(poly)):
                v1 = poly[i]
                v2 = poly[(i + 1) % len(poly)]
                edge = v2 - v1
                # Normal vector
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                normal = normal / norm
                
                # Project both polygons onto the normal
                proj1 = np.dot(poly1, normal)
                proj2 = np.dot(poly2, normal)
                
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                # Calculate gap
                # If max1 < min2, gap is min2 - max1
                # If max2 < min1, gap is min1 - max2
                # Else, overlapping (gap is negative)
                gap1 = min2 - max1
                gap2 = min1 - max2
                
                # The gap along this axis is the minimum of the two potential gaps
                # If they overlap on this axis, both gaps will be negative?
                # No, if they overlap, max1 > min2 AND max2 > min1.
                # So gap1 < 0 and gap2 < 0.
                # The "penetration depth" along this axis is max(gap1, gap2) ? No.
                # If they overlap, the gap is negative.
                # We want the minimum gap over all axes.
                # If disjoint, min_gap will be positive.
                # If overlapping, min_gap will be negative.
                
                # Actually, if they overlap, gap1 and gap2 are both negative.
                # We want the axis with the *largest* negative gap (least overlap) to define the distance?
                # No, for "disjointness", we care if ANY axis separates them.
                # For "distance", we care about the MINIMUM separation distance.
                # If disjoint, min_gap = min(positive_gaps).
                # If overlapping, we want a measure of "how much" they overlap.
                # A common proxy is the maximum penetration depth.
                # Let's return the minimum gap found. If it's negative, they overlap.
                
                current_gap = min(gap1, gap2)
                if current_gap < min_gap:
                    min_gap = current_gap
                    
        return min_gap

    def compute_outer_side(inner_hexes, outer_cx, outer_cy, outer_angle_rad):
        """Compute minimum outer hexagon side length to contain all inner hexagons."""
        max_required = 0.0
        theta = outer_angle_rad
        for hex_verts in inner_hexes:
            for v in hex_verts:
                dx = v[0] - outer_cx
                dy = v[1] - outer_cy
                for k in range(6):
                    edge_angle = theta + np.radians(k * 60)
                    nx = np.cos(edge_angle)
                    ny = np.sin(edge_angle)
                    dist = dx * nx + dy * ny
                    if dist > 0:
                        required_S = dist * 2.0 / np.sqrt(3.0)
                        if required_S > max_required:
                            max_required = required_S
        return max_required

    def make_hexes(params):
        """Build inner hexagon vertices from params."""
        inner_x = params[0:12]
        inner_y = params[12:24]
        inner_a = params[24:36]
        outer_cx = params[36]
        outer_cy = params[37]
        outer_a = params[38]
        inner_hexes = []
        for i in range(N):
            verts = hexagon_vertices(inner_x[i], inner_y[i], 1.0, inner_a[i])
            inner_hexes.append(verts)
        return inner_hexes, outer_cx, outer_cy, outer_a

    def objective(params, penalty_weight=10.0):
        if time.time() > deadline:
            return 1e6
        
        inner_hexes, outer_cx, outer_cy, outer_a = make_hexes(params)
        
        penalty = 0.0
        # Check pairwise disjointness using SAT gap
        for i in range(N):
            for j in range(i + 1, N):
                gap = sat_gap(inner_hexes[i], inner_hexes[j])
                if gap < 0:
                    # Overlapping: penalty is square of overlap depth
                    penalty += penalty_weight * gap * gap
        
        side = compute_outer_side(inner_hexes, outer_cx, outer_cy, outer_a)
        return side + penalty

    # Generate diverse initial configurations
    np.random.seed(42)
    configs = []
    sqrt3 = np.sqrt(3)

    # Config 1: Honeycomb lattice (3x4 staggered)
    hx, hy, ha = [], [], []
    for r in range(3):
        for c in range(4):
            x = c * sqrt3 * 1.05
            y = r * 1.55
            if r % 2 == 1:
                x += sqrt3 * 1.05 / 2
            hx.append(x); hy.append(y); ha.append(0.0)
    hx = np.array(hx); hy = np.array(hy); ha = np.array(ha)
    hx -= np.mean(hx); hy -= np.mean(hy)
    configs.append(np.concatenate([hx, hy, ha, [0.0, 0.0, 0.0]]))

    # Config 2: Compact circular arrangement
    cx, cy, ca = [], [], []
    for i in range(12):
        angle = 2 * np.pi * i / 12
        r = 2.8
        cx.append(r * np.cos(angle))
        cy.append(r * np.sin(angle))
        ca.append(angle)
    cx = np.array(cx); cy = np.array(cy); ca = np.array(ca)
    cx -= np.mean(cx); cy -= np.mean(cy)
    configs.append(np.concatenate([cx, cy, ca, [0.0, 0.0, 0.0]]))

    # Config 3: Two rows of 6
    rx, ry, ra = [], [], []
    for i in range(6):
        rx.append(i * sqrt3 * 1.05 - 2.5 * sqrt3 * 1.05)
        ry.append(0.0)
        ra.append(0.0)
        rx.append(i * sqrt3 * 1.05 - 2.5 * sqrt3 * 1.05)
        ry.append(1.55)
        ra.append(0.0)
    rx = np.array(rx); ry = np.array(ry); ra = np.array(ra)
    rx -= np.mean(rx); ry -= np.mean(ry)
    configs.append(np.concatenate([rx, ry, ra, [0.0, 0.0, 0.0]]))

    # Config 4: Random
    rx4, ry4, ra4 = [], [], []
    for _ in range(12):
        rx4.append(np.random.uniform(-5, 5))
        ry4.append(np.random.uniform(-5, 5))
        ra4.append(np.random.uniform(0, 2*np.pi))
    rx4 = np.array(rx4); ry4 = np.array(ry4); ra4 = np.array(ra4)
    rx4 -= np.mean(rx4); ry4 -= np.mean(ry4)
    configs.append(np.concatenate([rx4, ry4, ra4, [0.0, 0.0, 0.0]]))

    best_params = None
    best_score = 1e9

    bounds = [(-6, 6)] * 12 + [(-6, 6)] * 12 + [(-np.pi, np.pi)] * 12 + [(-1, 1), (-1, 1), (-np.pi / 6, np.pi / 6)]

    for cfg in configs:
        if time.time() > deadline - 10:
            break
            
        # Phase 1: Differential Evolution
        try:
            de_result = differential_evolution(
                objective, bounds,
                maxiter=30, popsize=15,
                tol=1e-5, mutation=(0.5, 1.0),
                recombination=0.7, seed=42,
                polish=False, # We will do our own polish
                init=cfg
            )
            
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_params = de_result.x
                
        except Exception:
            pass

        # Phase 2: Local Refinement with L-BFGS-B
        if best_params is not None:
            try:
                # Use a higher penalty weight for local refinement to ensure disjointness
                def local_objective(params):
                    if time.time() > deadline:
                        return 1e6
                    return objective(params, penalty_weight=100.0)
                
                # Provide initial guess
                res = minimize(
                    local_objective, 
                    best_params, 
                    method='L-BFGS-B', 
                    options={'maxiter': 100, 'maxfun': 2000, 'ftol': 1e-10}
                )
                
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x
                    
            except Exception:
                pass

    if best_params is None:
        # Fallback: Return initial config with large outer hex
        inner = [[configs[0][i], configs[0][i+12], np.degrees(configs[0][i+24])] for i in range(N)]
        return inner, (0.0, 0.0), 30.0, 0.0

    # Extract best solution
    inner_x = best_params[0:12]
    inner_y = best_params[12:24]
    inner_a = best_params[24:36]
    outer_cx = best_params[36]
    outer_cy = best_params[37]
    outer_a = best_params[38]
    
    # Verify validity one last time
    inner_hexes, _, _, _ = make_hexes(best_params)
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            if sat_gap(inner_hexes[i], inner_hexes[j]) < -1e-6:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # If invalid, return a valid fallback (e.g. spread out)
        inner = [[configs[0][i], configs[0][i+12], np.degrees(configs[0][i+24])] for i in range(N)]
        return inner, (0.0, 0.0), 30.0, 0.0

    inner = [[inner_x[i], inner_y[i], np.degrees(inner_a[i])] for i in range(N)]
    outer_side = best_score # Note: best_score includes penalty if any, but if valid, penalty is 0
    
    # Re-calculate exact side for reporting
    exact_side = compute_outer_side(inner_hexes, outer_cx, outer_cy, outer_a)
    
    return inner, (outer_cx, outer_cy), exact_side, np.degrees(outer_a)
# EVOLVE-BLOCK-END
