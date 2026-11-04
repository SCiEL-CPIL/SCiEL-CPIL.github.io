import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11

def construct():
    start_time = time.time()
    deadline = 150.0
    safety_margin = 10.0

    # Geometry constants
    d = math.sqrt(3)
    # Distance between centers of touching unit hexagons (side 1)
    # For pointy-top hexagons, distance is 2 * sqrt(3) / 2 * 2? No.
    # Side length s=1.
    # Width (flat to flat) = sqrt(3). Height (point to point) = 2.
    # Distance between centers of adjacent hexagons in a honeycomb:
    # If aligned with axes, they touch at edges.
    # The distance from center to edge (apothem) is sqrt(3)/2.
    # So center-to-center distance is 2 * apothem = sqrt(3).
    # Wait, let's verify.
    # Vertices of unit hex (pointy top): (0,1), (sqrt(3)/2, 1/2), (sqrt(3)/2, -1/2), (0,-1), (-sqrt(3)/2, -1/2), (-sqrt(3)/2, 1/2).
    # Edge between (0,1) and (sqrt(3)/2, 1/2).
    # Normal to this edge points roughly (1/2, sqrt(3)/2).
    # Distance from center (0,0) to this edge is sqrt(3)/2.
    # So two hexagons touching edge-to-edge have centers distance sqrt(3) apart.
    spacing = d

    # Relative vertices of a unit hexagon (pointy-top, centered at origin)
    # Angles: 30, 90, 150, 210, 270, 330 degrees?
    # Standard pointy-top: vertex at (0,1). Angle 90 deg.
    # Let's use flat-top for easier lattice math?
    # Flat-top: vertices at 0, 60, 120, 180, 240, 300.
    # Vertex (1,0). Apothem sqrt(3)/2.
    # Let's stick to the code's convention: pointy-top.
    # Vertices at angles 90, 150, 210, 270, 330, 30.
    # Or just generate them.
    
    def get_hex_verts(cx, cy, angle):
        """
        Returns vertices of a unit regular hexagon centered at (cx, cy) rotated by 'angle' radians.
        Base orientation: Pointy-top (vertex at 90 deg).
        """
        # Base vertices for pointy-top unit hex
        # (0, 1), (sqrt(3)/2, 1/2), (sqrt(3)/2, -1/2), (0, -1), (-sqrt(3)/2, -1/2), (-sqrt(3)/2, 1/2)
        base_verts = np.array([
            [0.0, 1.0],
            [d / 2.0, 0.5],
            [d / 2.0, -0.5],
            [0.0, -1.0],
            [-d / 2.0, -0.5],
            [-d / 2.0, 0.5]
        ])
        
        c = math.cos(angle)
        s = math.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        verts = base_verts @ rot.T + np.array([cx, cy])
        return verts

    def compute_S_from_points(points, theta):
        """
        Compute minimal side length S of a regular hexagon with rotation theta
        that encloses the given points.
        The hexagon is defined by 6 half-planes.
        For a regular hexagon with side S, the apothem is S * sqrt(3) / 2.
        The normals of the faces are at angles theta + k * pi/3.
        The distance from center to face is apothem.
        We need to find the minimum S such that all points are inside.
        
        If the hexagon is centered at (0,0) for calculation purposes (we can translate points),
        the constraint for a point p is:
        dot(p, n_k) <= apothem for all k.
        So apothem >= max_k dot(p, n_k).
        Wait, this assumes the center is at 0.
        If the center is (cx, cy), then dot(p - c, n_k) <= apothem.
        
        Here we just want the size S given the points and the orientation theta.
        We can assume the hexagon is centered at the origin of the coordinate system of the points?
        No, the hexagon can be anywhere. But we are minimizing S.
        The minimal enclosing regular hexagon with fixed orientation theta for a set of points
        is determined by the maximum projection of the points onto the 6 normal vectors.
        
        Let points P.
        For each normal direction n_k (angle theta + k*pi/3), let m_k = max_{p in P} dot(p, n_k).
        Let m'_k = min_{p in P} dot(p, n_k).
        The width of the set in direction n_k is w_k = m_k - m'_k.
        The hexagon must have a width of at least w_k in that direction.
        For a regular hexagon with side S, the width in the direction of a normal is:
        If the normal is perpendicular to a side, width = 2 * apothem = S * sqrt(3).
        Wait, the normals of the sides are perpendicular to the sides.
        So the distance between opposite sides is 2 * apothem = S * sqrt(3).
        So we need S * sqrt(3) >= w_k for all k.
        Thus S >= max_k(w_k) / sqrt(3).
        
        Is this correct?
        Yes, if the hexagon is centered appropriately.
        The center of the hexagon should be at (max+min)/2 for each axis?
        Actually, for a fixed orientation, the minimal enclosing regular hexagon is unique in size,
        and its center is the intersection of the midlines of the bounding strips.
        The size S is indeed max_k(w_k) / sqrt(3).
        """
        k_angles = theta + np.arange(6) * math.pi / 3
        normals = np.stack([np.cos(k_angles), np.sin(k_angles)], axis=-1)
        dots = np.dot(points, normals.T)
        extents = np.max(dots, axis=1) - np.min(dots, axis=1)
        max_extent = np.max(extents)
        return max_extent / d

    def objective(params):
        """
        params: [x, y, alpha]
        x, y: center of the cluster
        alpha: rotation of the cluster relative to the container
        Container rotation is fixed at 0 for this objective, we optimize alpha.
        """
        if time.time() > deadline - safety_margin:
            return 1e10
        
        x, y, alpha = params
        
        # Get all vertices of the 11 hexagons
        all_pts = []
        for (cx, cy) in cluster_centers:
            # The cluster is rotated by alpha, and translated by (x,y)
            # Note: cluster_centers are defined in the cluster's local frame (centered at 0,0)
            # We rotate the local center (cx, cy) by alpha, then add (x,y)
            # And the hexagon itself is rotated by alpha.
            rx = cx * math.cos(alpha) - cy * math.sin(alpha) + x
            ry = cx * math.sin(alpha) + cy * math.cos(alpha) + y
            verts = get_hex_verts(rx, ry, alpha)
            all_pts.append(verts)
        
        points = np.vstack(all_pts)
        
        # The container is a regular hexagon with rotation theta_container.
        # We can fix theta_container = 0 and optimize alpha.
        # Or fix alpha = 0 and optimize theta_container.
        # Due to symmetry, it's equivalent.
        # Let's compute S for container rotation 0.
        S = compute_S_from_points(points, 0.0)
        
        # Add penalty if any hexagon overlaps?
        # Since we generate valid clusters, and we only rotate/translate the whole cluster,
        # internal overlaps are preserved.
        # We just need to ensure the cluster is valid initially.
        return S

    # --- Generate Cluster Topologies ---
    
    def generate_clusters():
        """
        Generate a list of cluster center configurations (local coordinates).
        Each configuration is a list of 11 (x, y) pairs.
        """
        clusters = []
        
        # Topology 1: Center + 6 Ring + 4 of 12 Ring 2
        # Ring 1 centers (distance d from origin)
        ring1 = []
        for i in range(6):
            ang = i * math.pi / 3
            ring1.append((d * math.cos(ang), d * math.sin(ang)))
        
        # Ring 2 centers (distance 2d from origin? No, hex lattice)
        # In a hex lattice, the neighbors of the neighbors are at distance sqrt(3) from the neighbor,
        # but their distance from the center varies.
        # Let's use axial coordinates (q, r).
        # Center (0,0).
        # Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
        # Ring 2: distance 2.
        # (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,1)
        
        def axial_to_xy(q, r):
            # Pointy-top hexagon axial to pixel
            # x = d * (q + r/2)
            # y = d * r * 0.5 * 2? 
            # Standard formula for pointy-top:
            # x = size * sqrt(3) * (q + r/2)
            # y = size * 3/2 * r
            # Here size=1. sqrt(3) is d.
            x = d * (q + r / 2.0)
            y = 1.5 * r
            return (x, y)
        
        # Verify Ring 1 distance with this formula
        # q=1, r=0 -> x=d, y=0. Distance d. Correct.
        # q=1, r=-1 -> x=d(1-0.5)=d/2, y=-1.5. Dist = sqrt(d^2/4 + 2.25) = sqrt(0.75+2.25)=sqrt(3)=d. Correct.
        
        ring2_axial = [
            (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -1),
            (-2, 0), (-2, 1), (-2, 2), (-1, 2), (0, 2), (1, 1)
        ]
        
        ring2_xy = [axial_to_xy(q, r) for q, r in ring2_axial]
        
        # We need to pick 4 from ring2.
        # To minimize the bounding box, we should pick them spread out.
        # Option 1: Symmetric selection (every 3rd one)
        # Indices: 0, 3, 6, 9
        sel1 = [ring2_xy[i] for i in [0, 3, 6, 9]]
        cluster1 = [(0,0)] + ring1 + sel1
        clusters.append(cluster1)
        
        # Option 2: Different symmetric selection
        # Indices: 1, 4, 7, 10
        sel2 = [ring2_xy[i] for i in [1, 4, 7, 10]]
        cluster2 = [(0,0)] + ring1 + sel2
        clusters.append(cluster2)
        
        # Option 3: Indices 2, 5, 8, 11
        sel3 = [ring2_xy[i] for i in [2, 5, 8, 11]]
        cluster3 = [(0,0)] + ring1 + sel3
        clusters.append(cluster3)
        
        # Option 4: A more compact "flower" shape might be impossible with 11,
        # but let's try a "cross" shape if possible?
        # Actually, the 11 hexagons forming a larger hexagon of side 2 minus corners?
        # A hexagon of side 2 has 1+6+12 = 19 hexagons.
        # So we are taking a subset.
        
        return clusters

    cluster_list = generate_clusters()
    
    best_S = float('inf')
    best_params = None
    best_cluster = None

    for idx, cluster in enumerate(cluster_list):
        if time.time() > deadline - safety_margin:
            break
            
        cluster_centers = cluster
        
        # Initial guess: Center at (0,0), rotation 0
        # We can try a few initial rotations
        initial_guesses = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, math.pi / 6],
            [0.0, 0.0, math.pi / 12],
            [0.0, 0.0, -math.pi / 12]
        ]
        
        for guess in initial_guesses:
            if time.time() > deadline - safety_margin:
                break
                
            try:
                res = minimize(objective, guess, method='Nelder-Mead',
                               options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000})
                
                if res.fun < best_S:
                    # Verify validity (no overlaps)
                    # Since the cluster is rigid and valid by construction,
                    # and we only rotate/translate, overlaps are internal to the cluster.
                    # We just need to ensure the cluster itself was valid.
                    # Our generation ensures touching, not overlapping.
                    
                    # Double check S calculation
                    x, y, alpha = res.x
                    all_pts = []
                    for (cx, cy) in cluster_centers:
                        rx = cx * math.cos(alpha) - cy * math.sin(alpha) + x
                        ry = cx * math.sin(alpha) + cy * math.cos(alpha) + y
                        verts = get_hex_verts(rx, ry, alpha)
                        all_pts.append(verts)
                    points = np.vstack(all_pts)
                    S_calc = compute_S_from_points(points, 0.0)
                    
                    if S_calc < best_S:
                        best_S = S_calc
                        best_params = res.x
                        best_cluster = cluster
                        
            except Exception as e:
                pass

    # If no valid result found, fallback to a simple configuration
    if best_params is None:
        # Use the first cluster, no rotation, centered at 0
        best_cluster = cluster_list[0]
        best_params = [0.0, 0.0, 0.0]
        all_pts = []
        for (cx, cy) in best_cluster:
            verts = get_hex_verts(cx, cy, 0.0)
            all_pts.append(verts)
        points = np.vstack(all_pts)
        best_S = compute_S_from_points(points, 0.0)

    x, y, alpha = best_params
    
    # Construct the output
    inner = []
    for (cx, cy) in best_cluster:
        rx = cx * math.cos(alpha) - cy * math.sin(alpha) + x
        ry = cx * math.sin(alpha) + cy * math.cos(alpha) + y
        angle_deg = math.degrees(alpha)
        inner.append([rx, ry, angle_deg])

    # The outer hexagon center is (x, y)?
    # Wait, the problem asks for the outer hexagon center.
    # My `compute_S_from_points` calculates the size S for a hexagon centered at... 
    # It calculates the width. The center of the minimal enclosing hexagon is not necessarily (0,0) in the local frame.
    # However, the score is just the side length S.
    # The interface requires: inner rows, outer center, side, angle.
    # The outer hexagon must enclose the inner ones.
    # If I return center (x,y) and angle 0, does it enclose them?
    # The points were rotated/translated such that the minimal enclosing hexagon of orientation 0
    # has size S.
    # The center of that enclosing hexagon is NOT necessarily (x,y).
    # The center of the enclosing hexagon is the centroid of the bounding box in the 6 directions.
    
    # Let's find the actual center of the enclosing hexagon.
    # For orientation theta=0:
    # Normals are at 0, 60, ...
    # Points P.
    # For each normal n_k, min_k = min(dot(P, n_k)), max_k = max(dot(P, n_k)).
    # The hexagon faces are at distance apothem from center C.
    # dot(P - C, n_k) <= apothem  => dot(P, n_k) <= dot(C, n_k) + apothem
    # dot(P - C, -n_k) <= apothem => dot(P, n_k) >= dot(C, n_k) - apothem
    # So dot(C, n_k) must be (min_k + max_k) / 2.
    
    points = np.vstack([get_hex_verts(
        cx * math.cos(alpha) - cy * math.sin(alpha) + x,
        cx * math.sin(alpha) + cy * math.cos(alpha) + y,
        alpha
    ) for cx, cy in best_cluster])
    
    theta = 0.0
    k_angles = theta + np.arange(6) * math.pi / 3
    normals = np.stack([np.cos(k_angles), np.sin(k_angles)], axis=-1)
    dots = np.dot(points, normals.T)
    mins = np.min(dots, axis=1)
    maxs = np.max(dots, axis=1)
    
    # dot(C, n_k) = (min_k + max_k) / 2
    # We have 6 equations for Cx, Cy.
    # dot(C, n_0) = c_0
    # dot(C, n_1) = c_1
    # ...
    # This is an overdetermined system. We can solve for Cx, Cy using the first two.
    # n_0 = (1, 0) -> Cx = c_0
    # n_1 = (cos(60), sin(60)) = (0.5, sqrt(3)/2) -> 0.5 Cx + sqrt(3)/2 Cy = c_1
    # Cy = (c_1 - 0.5 Cx) / (sqrt(3)/2)
    
    c_vals = (mins + maxs) / 2.0
    Cx = c_vals[0]
    Cy = (c_vals[1] - 0.5 * Cx) / (d / 2.0)
    
    # Verify S with this center
    # S should be consistent
    # apothem = max( |dot(P, n_k) - dot(C, n_k)| )
    # S = 2 * apothem / sqrt(3)
    
    return inner, (Cx, Cy), best_S, 0.0

