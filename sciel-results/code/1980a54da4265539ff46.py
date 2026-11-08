import numpy as np
import time

N = 12

def _hex_vertices(cx, cy, angle_deg=0.0):
    """Return 6 vertices of a unit regular hexagon centered at (cx,cy) rotated by angle_deg."""
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    # Unit hexagon vertices (flat-top orientation, side=1)
    base = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2],
    ])
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    verts = base @ R.T + np.array([cx, cy])
    return verts

def _min_enclosing_hexagon(inner_hexes):
    """
    Given list of (cx, cy, angle_deg) for unit hexagons,
    find the minimal regular hexagon (center, side, rotation) that contains all of them.
    Returns (center_x, center_y, side_length, rotation_deg).
    """
    # Collect all vertices of all inner hexagons
    all_verts = []
    for cx, cy, ang in inner_hexes:
        v = _hex_vertices(cx, cy, ang)
        all_verts.append(v)
    all_verts = np.vstack(all_verts)  # shape (36, 2)
    
    # For a regular hexagon with center c, side s, rotation theta,
    # the 6 outward normals are at angles theta, theta+60, ..., theta+300
    # The constraint: for each normal direction n_i, max over all vertices of (v - c) . n_i <= s * sqrt(3) / 2 * 2 = s * sqrt(3)
    # Actually for a regular hexagon with side s, the apothem (distance from center to side) = s * sqrt(3)/2
    # So the constraint is: max_i (max_v (v-c).n_i) <= s * sqrt(3)/2 for all 6 normals
    # And s must satisfy this for all 6 directions, so s = max_i (max_v (v-c).n_i) / (sqrt(3)/2)
    
    # We need to minimize s over c and theta.
    # For fixed theta, optimal c is the centroid that minimizes the max projection.
    # Actually, for a given set of points and a fixed set of 6 directions,
    # the minimal enclosing hexagon with those directions has c at the center of the 
    # minimal bounding box in the transformed coordinates.
    
    # Let's try: for a given rotation theta, transform all points into the hexagon's coordinate system
    # The 6 normals are at angles theta + k*60 for k=0..5
    # In the hexagon frame, the constraints are on projections onto these 6 directions.
    
    # Simpler approach: for a regular hexagon containing points, the minimal s for a given center c and rotation theta
    # is: s = (2/sqrt(3)) * max over 6 normals of (max_v (v-c).n_i)
    # We want to minimize over c and theta.
    
    # For fixed theta, the optimal c minimizes the maximum of the 6 support functions.
    # This is equivalent to finding the center of the minimal enclosing circle in a transformed space,
    # but let's just do a grid search over c and theta for robustness.
    
    # First, find a good center: centroid of all vertices
    c0 = np.mean(all_verts, axis=0)
    
    best_s = float('inf')
    best_center = c0
    best_theta = 0.0
    
    # Try different rotations
    for theta_deg in np.linspace(0, 60, 61):  # 0 to 60 degrees, 61 steps
        theta = np.radians(theta_deg)
        # 6 normal directions
        normals = []
        for k in range(6):
            ang = theta + k * np.pi / 3
            normals.append(np.array([np.cos(ang), np.sin(ang)]))
        normals = np.array(normals)  # (6, 2)
        
        # For this theta, find optimal center c
        # The constraint is: for each normal n_i, max_v (v-c).n_i <= s*sqrt(3)/2
        # i.e., c.n_i >= max_v (v.n_i) - s*sqrt(3)/2 and c.n_i <= min_v (v.n_i) + s*sqrt(3)/2
        # This is a linear feasibility problem in c for given s.
        # To find minimal s, we can binary search on s and check feasibility.
        
        # Project all vertices onto each normal
        # proj[i][j] = v_j . n_i
        proj = all_verts @ normals.T  # (36, 6)
        
        # For feasibility with center c and side s:
        # For each normal i: c.n_i >= proj[:,i].max() - s*sqrt(3)/2
        #                  c.n_i <= proj[:,i].min() + s*sqrt(3)/2
        # So we need: proj[:,i].max() - s*sqrt(3)/2 <= c.n_i <= proj[:,i].min() + s*sqrt(3)/2
        # This requires: proj[:,i].max() - s*sqrt(3)/2 <= proj[:,i].min() + s*sqrt(3)/2
        # i.e., proj[:,i].max() - proj[:,i].min() <= s*sqrt(3)
        # So s >= (proj[:,i].max() - proj[:,i].min()) / sqrt(3) for all i
        # But c also needs to satisfy all 6 constraints simultaneously.
        
        # The minimal s for a given theta is determined by both the individual width constraints
        # and the joint feasibility of c.
        
        # Let's find the minimal s such that there exists c satisfying all 6 pairs of inequalities.
        # This is a linear programming problem. Let's use a simple approach:
        # For a candidate s, check if the intersection of the 6 strips is non-empty.
        
        # Actually, let's just compute the required s from the widths and then verify.
        # The width in direction i is w_i = proj[:,i].max() - proj[:,i].min()
        # We need s >= w_i / sqrt(3) for all i, so s >= max_i(w_i) / sqrt(3)
        # But we also need the center c to exist.
        
        # Let me just do a binary search on s and for each s, check if there's a valid c.
        
        def is_feasible(s, proj, normals):
            """Check if there exists center c such that all constraints are satisfied."""
            # Constraints: for each normal i,
            # c.n_i >= proj[:,i].max() - s*sqrt(3)/2
            # c.n_i <= proj[:,i].min() + s*sqrt(3)/2
            # This is a system of linear inequalities. We can solve it with scipy or manually.
            # Since we have 6 constraints in 2D, let's use a simple LP.
            from scipy.optimize import linprog
            # Minimize 0 subject to:
            # normals @ c >= proj.max(axis=0) - s*sqrt(3)/2
            # -normals @ c >= -(proj.min(axis=0) + s*sqrt(3)/2) = -proj.min(axis=0) - s*sqrt(3)/2
            A = np.vstack([
                -normals,  # -n_i . c >= -(proj_max_i - s*sqrt(3)/2) => n_i . c <= proj_max_i - s*sqrt(3)/2 ... wait
                normals    # n_i . c >= proj_max_i - s*sqrt(3)/2 => -n_i . c <= -(proj_max_i - s*sqrt(3)/2)
            ])
            # Let me redo this.
            # Constraint 1: n_i . c >= proj[:,i].max() - s*sqrt(3)/2
            #   => -n_i . c <= -(proj[:,i].max() - s*sqrt(3)/2)
            # Constraint 2: n_i . c <= proj[:,i].min() + s*sqrt(3)/2
            #   => n_i . c <= proj[:,i].min() + s*sqrt(3)/2
            
            A_ub = np.vstack([
                -normals,  # for constraint 1
                normals    # for constraint 2
            ])
            b_ub = np.vstack([
                -(proj.max(axis=0) - s * np.sqrt(3) / 2),  # for constraint 1
                proj.min(axis=0) + s * np.sqrt(3) / 2      # for constraint 2
            ])
            res = linprog(np.zeros(2), A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None)], method='highs')
            return res.status == 0
        
        # Binary search on s
        lo = 0.0
        hi = 100.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if is_feasible(mid, proj, normals):
                hi = mid
            else:
                lo = mid
        
        if hi < best_s:
            best_s = hi
            # Find the center c
            def find_center(s, proj, normals):
                from scipy.optimize import linprog
                A_ub = np.vstack([
                    -normals,
                    normals
                ])
                b_ub = np.vstack([
                    -(proj.max(axis=0) - s * np.sqrt(3) / 2),
                    proj.min(axis=0) + s * np.sqrt(3) / 2
                ])
                res = linprog(np.zeros(2), A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None)], method='highs')
                return res.x
            c = find_center(hi, proj, normals)
            best_center = c
            best_theta = theta_deg
    
    return best_center[0], best_center[1], best_s, best_theta

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Optimal configuration: hexagonal lattice arrangement of 12 unit hexagons
    # A centered hexagonal arrangement: 1 in center, 6 in first ring, 5 in second ring (partial)
    # For 12 hexagons, the best is a hexagonal close-packed cluster.
    
    # Hexagonal lattice: positions at (i * 1.5, j * sqrt(3) + (i % 2) * sqrt(3)/2) for integer i,j
    # But for side-1 hexagons in a hexagonal lattice, the center-to-center distance is sqrt(3)
    
    # Let me use a simpler approach: place 12 hexagons in a hexagonal pattern
    # Center at origin, with hexagons at positions that form a compact cluster.
    
    # For a hexagonal lattice of side-1 hexagons, adjacent centers are sqrt(3) apart.
    # Let's place them in a hexagonal arrangement:
    # Ring 0: 1 hexagon at center
    # Ring 1: 6 hexagons around it
    # Ring 2: 6 more (partial) to get to 12
    
    # Actually, let's just try a few configurations and pick the best.
    
    # Configuration 1: 3x4 grid with hexagonal offset
    # Configuration 2: Hexagonal cluster
    
    # Let me try a hexagonal close-packed arrangement:
    # Positions in axial coordinates (q, r) for hexagonal grid
    # For side-1 hexagons, center distance = sqrt(3)
    
    sqrt3 = np.sqrt(3)
    
    # Hexagonal lattice positions: for axial coords (q, r), pixel coords:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r  (for pointy-top) or similar
    
    # Let me use a different parameterization. For flat-top hexagons with side 1:
    # Width = 2, height = sqrt(3)
    # Horizontal spacing = 1.5, vertical spacing = sqrt(3)
    # Offset for odd columns = sqrt(3)/2
    
    # Let's try a 3x4 arrangement with hexagonal offset
    positions = []
    for col in range(3):
        for row in range(4):
            x = 1.5 * col
            y = sqrt3 * row + (sqrt3 / 2 if col % 2 else 0)
            positions.append((x, y, 0.0))
    
    # That's 12 hexagons. Let me compute the enclosing hexagon.
    inner1 = positions
    
    # Also try a more symmetric hexagonal arrangement
    # Center + 6 ring + 5 from next ring
    positions2 = [(0.0, 0.0, 0.0)]
    # Ring 1: 6 hexagons at distance sqrt(3) in 6 directions
    for k in range(6):
        ang = k * np.pi / 3
        x = sqrt3 * np.cos(ang)
        y = sqrt3 * np.sin(ang)
        positions2.append((x, y, 0.0))
    # Ring 2: 6 positions, take 5 of them
    for k in range(6):
        ang = k * np.pi / 3 + np.pi / 6
        x = 2 * sqrt3 * np.cos(ang)
        y = 2 * sqrt3 * np.sin(ang)
        if k < 5:  # Take only 5
            positions2.append((x, y, 0.0))
    
    inner2 = positions2[:12]  # Ensure exactly 12
    
    # Compute enclosing hexagons for both
    results = []
    for inner in [inner1, inner2]:
        if len(inner) != 12:
            continue
        cx, cy, s, theta = _min_enclosing_hexagon(inner)
        results.append((s, inner, cx, cy, theta))
    
    if not results:
        # Fallback
        return [[4.0 * k, 0.0, 0.0] for k in range(-6, 6)], (0.0, 0.0), 40.0, 0.0
    
    # Pick the best
    results.sort(key=lambda x: x[0])
    best_s, best_inner, best_cx, best_cy, best_theta = results[0]
    
    return best_inner, (best_cx, best_cy), best_s, best_theta
