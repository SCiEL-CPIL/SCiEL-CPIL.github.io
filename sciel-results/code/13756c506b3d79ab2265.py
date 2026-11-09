import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    sqrt3 = math.sqrt(3.0)
    
    # Outer hexagon normals (flat-top, side L, centered at 0,0)
    # Normals point outward
    # For a regular hexagon with flat top, the sides are at angles 0, 60, ...
    # The normals to the sides are at 30, 90, 150, 210, 270, 330
    normal_angles = np.array([math.radians(30 + 60 * j) for j in range(6)])
    normal_cos = np.cos(normal_angles)
    normal_sin = np.sin(normal_angles)
    
    def get_vertices(cx, cy, theta):
        """Get vertices of a unit hexagon centered at (cx, cy) with rotation theta."""
        angles = np.array([math.radians(60 * k) + theta for k in range(6)])
        vx = cx + np.cos(angles)
        vy = cy + np.sin(angles)
        return np.column_stack((vx, vy))
        
    def max_projection(vertices, cos_n, sin_n):
        """Max projection of vertices onto a normal direction."""
        return np.max(vertices[:, 0] * cos_n + vertices[:, 1] * sin_n)
        
    def min_distance_between_hexagons(v1, v2):
        """
        Calculate the minimum distance between two convex polygons (hexagons).
        Returns 0 if they overlap or touch, >0 if separated.
        Uses Separating Axis Theorem (SAT) logic for gap calculation.
        """
        # Get all edges normals for both polygons
        # For a convex polygon, the separating axes are the normals to the edges.
        def get_edge_normals(vertices):
            edges = np.roll(vertices, -1, axis=0) - vertices
            # Normals to edges
            normals = np.column_stack((-edges[:, 1], edges[:, 0]))
            # Normalize
            norms = np.linalg.norm(normals, axis=1)
            norms[norms == 0] = 1.0
            normals = normals / norms[:, np.newaxis]
            return normals
            
        n1 = get_edge_normals(v1)
        n2 = get_edge_normals(v2)
        all_normals = np.vstack([n1, n2])
        
        min_gap = float('inf')
        for n in all_normals:
            proj1 = np.dot(v1, n)
            proj2 = np.dot(v2, n)
            # The gap is max(proj2) - min(proj1) or max(proj1) - min(proj2)
            # We want the minimum distance, which is the minimum of the positive gaps.
            # If intervals overlap, gap is negative (overlap).
            gap = max(proj2.min() - proj1.max(), proj1.min() - proj2.max())
            # Actually, the separation is:
            # If proj1.max() < proj2.min(), gap = proj2.min() - proj1.max()
            # If proj2.max() < proj1.min(), gap = proj1.min() - proj2.max()
            # If they overlap, gap is negative.
            # We want the minimum distance, so we take the max of the "overlap" amount?
            # No, min distance is max(0, min_gap_over_axes).
            # Let's calculate the "signed gap" where positive means separated.
            # gap_axis = min(proj2) - max(proj1)
            # If gap_axis > 0, separated by gap_axis.
            # If gap_axis < 0, overlapping.
            # The minimum distance is the minimum of gap_axis over all axes?
            # No, if they overlap, min distance is 0.
            # If they are separated, min distance is min(gap_axis) over separating axes?
            # Actually, for convex sets, min distance is min over all axes of (max projection of A - min projection of B) if B is to the right, etc.
            # Simpler: The minimum distance between two convex polygons is the minimum of the gaps along the separating axes.
            # If no separating axis exists (they intersect), distance is 0.
            # So, calculate gap for each axis. If any gap is > 0, the distance is the minimum of those positive gaps.
            # If all gaps are <= 0, they intersect, distance is 0.
            
            # Let's compute the "penetration depth" or gap.
            # For a specific axis n, the interval of v1 is [min1, max1], v2 is [min2, max2].
            # If max1 < min2, gap = min2 - max1.
            # If max2 < min1, gap = min1 - max2.
            # Else, gap = 0 (overlap).
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2:
                gap = min2 - max1
            elif max2 < min1:
                gap = min1 - max2
            else:
                gap = 0.0
                
            if gap < min_gap:
                min_gap = gap
                
        return min_gap

    def constraints(x):
        """
        x: array of 36 elements: [x1, y1, theta1, x2, y2, theta2, ...]
        Returns list of constraint values (all must be >= 0 for SLSQP)
        """
        # We need to introduce L as a variable? 
        # No, we minimize L. 
        # But L is not in x. 
        # We can make L the objective, and constraints depend on L?
        # SLSQP minimizes f(x). 
        # We can add L as the last variable in x.
        # x = [x1, y1, theta1, ..., x12, y12, theta12, L]
        # Objective: L
        # Constraints: 
        # 1. Fit in outer hex: max_proj <= L * sqrt3/2  => L * sqrt3/2 - max_proj >= 0
        # 2. Disjoint: min_dist >= 0
        
        L = x[-1]
        cons = []
        
        # 1. Outer hexagon constraints
        for i in range(N):
            xi = x[3*i]
            yi = x[3*i + 1]
            theta_i = x[3*i + 2]
            verts = get_vertices(xi, yi, theta_i)
            for j in range(6):
                max_p = max_projection(verts, normal_cos[j], normal_sin[j])
                # Constraint: L * sqrt3/2 - max_p >= 0
                cons.append(L * (sqrt3 / 2.0) - max_p)
                
        # 2. Disjointness constraints
        # Only check pairs that are likely close? For N=12, 66 pairs.
        # 66 distance calculations per iteration might be slow but acceptable.
        for i in range(N):
            xi = x[3*i]
            yi = x[3*i + 1]
            theta_i = x[3*i + 2]
            verts_i = get_vertices(xi, yi, theta_i)
            for j in range(i + 1, N):
                xj = x[3*j]
                yj = x[3*j + 1]
                theta_j = x[3*j + 2]
                verts_j = get_vertices(xj, yj, theta_j)
                dist = min_distance_between_hexagons(verts_i, verts_j)
                # We want dist >= 0. 
                # To avoid numerical issues, we might want dist >= epsilon?
                # No, just >= 0.
                cons.append(dist)
                
        return cons

    def objective(x):
        return x[-1]

    # Initial guess
    # 1 at center
    # 6 in ring 1
    # 5 in ring 2
    # Scale factor to ensure initial non-overlap? 
    # In hex grid, distance is sqrt(3). 
    # If we place them at hex grid positions, they touch.
    # We need to start slightly separated or let the optimizer push them.
    # Starting exactly touching might cause issues with gradient calculation if non-smooth.
    # Let's scale initial positions by 1.01 to ensure initial feasibility.
    
    initial_x = []
    # Center
    initial_x.extend([0.0, 0.0, 0.0])
    
    # Ring 1: 6 hexagons
    for k in range(6):
        angle = math.radians(60 * k)
        dist = sqrt3 * 1.01
        x = dist * math.cos(angle)
        y = dist * math.sin(angle)
        # Rotate hexagons to align with grid? 
        # For flat-top, neighbors are at 0, 60, etc.
        # Let's just use 0 rotation for all initially.
        initial_x.extend([x, y, 0.0])
        
    # Ring 2: 5 hexagons
    # Pick 5 from the 12 positions in ring 2?
    # Ring 2 positions in hex grid (axial):
    # (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1) ... 
    # Let's just place them at angles 0, 60, 120, 180, 240 with distance 2*sqrt3 * 1.01
    # This might overlap with ring 1?
    # Distance from center is 2*sqrt3. Distance from ring 1 neighbor is sqrt(3).
    # So they touch ring 1 neighbors.
    for k in range(5):
        angle = math.radians(60 * k)
        dist = 2.0 * sqrt3 * 1.01
        x = dist * math.cos(angle)
        y = dist * math.sin(angle)
        initial_x.extend([x, y, 0.0])
        
    # Initial L
    # Max projection of the outermost hexagons
    # Approx L = max_dist / (sqrt3/2) + 1
    # max_dist = 2*sqrt3
    # L_init = 2*sqrt3 / (sqrt3/2) + 2 = 4 + 2 = 6?
    # Let's calculate properly.
    max_proj = 0
    for i in range(N):
        xi = initial_x[3*i]
        yi = initial_x[3*i + 1]
        theta_i = initial_x[3*i + 2]
        verts = get_vertices(xi, yi, theta_i)
        for j in range(6):
            p = max_projection(verts, normal_cos[j], normal_sin[j])
            if p > max_proj:
                max_proj = p
    L_init = max_proj / (sqrt3 / 2.0) + 0.1
    initial_x.append(L_init)
    
    x0 = np.array(initial_x)
    
    # Bounds: L > 0, angles can be anything, positions bounded by L
    # We don't need tight bounds if constraints handle it.
    # But SLSQP might wander.
    bounds = [(None, None)] * (3 * N) + [(1.0, None)]
    
    # Run optimization
    best_x = x0
    best_L = L_init
    
    try:
        # Use SLSQP
        res = minimize(objective, x0, method='SLSQP', 
                       constraints=[{'type': 'ineq', 'fun': constraints}],
                       bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-6})
        
        if res.success:
            best_x = res.x
            best_L = res.x[-1]
        else:
            # If failed, maybe try Nelder-Mead on the penalty function?
            # Or just use the best found?
            # SLSQP usually returns the best point even if it fails to converge perfectly.
            # Let's check if res.x is better.
            if res.x[-1] < best_L:
                best_x = res.x
                best_L = res.x[-1]
                
    except Exception as e:
        pass # Keep initial guess
    
    # Extract result
    inner = []
    for i in range(N):
        x = best_x[3*i]
        y = best_x[3*i + 1]
        theta = best_x[3*i + 2]
        inner.append([x, y, math.degrees(theta)])
        
    final_L = best_L
    return inner, (0.0, 0.0), final_L, 0.0

