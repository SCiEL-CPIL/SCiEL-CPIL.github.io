import numpy as np
import time
import itertools

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def get_cluster_vertices(positions, angles):
        """
        Get vertices for a set of hexagons at given positions and angles.
        positions: (N, 2)
        angles: (N,) in radians
        Returns: (N*6, 2)
        """
        verts_list = []
        for i in range(N):
            v = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, np.degrees(angles[i]))
            verts_list.append(v)
        return np.vstack(verts_list)

    def required_outer_side(verts):
        """
        Compute the side length of the smallest regular hexagon (centered at origin, angle 0)
        that contains all vertices in 'verts'.
        """
        x = verts[:, 0]
        y = verts[:, 1]
        sqrt3 = np.sqrt(3.0)
        
        # Projections onto the 6 axes of the hexagon
        # Axis 0: (1, 0) -> x
        # Axis 1: (1/2, sqrt(3)/2) -> x/2 + y*sqrt(3)/2
        # ... wait, standard hexagon side length S has vertices at distance S from center?
        # No, regular hexagon side length S has circumradius S.
        # The distance from center to edge (apothem) is S * sqrt(3)/2.
        # The support function of a regular hexagon of side S centered at origin is:
        # h(u) = S * max( |u_x|, |u_x/2 + u_y*sqrt(3)/2|, |-u_x/2 + u_y*sqrt(3)/2| ) ?
        # Let's use the projection method.
        # A regular hexagon with side S and angle 0 has normals at angles 0, 60, 120, 180, 240, 300.
        # The width in direction theta is 2 * S * |cos(theta - theta_i)| for the relevant facet?
        # Actually, for a regular hexagon of side S, the distance from center to vertex is S.
        # The distance from center to edge midpoint is S * sqrt(3)/2.
        
        # Let's just project points onto the 6 normal directions of the outer hexagon.
        # If the outer hexagon is aligned with axes (angle 0), its normals are at 0, 60, 120, 180, 240, 300 degrees.
        # Let n_k be the unit normal vector.
        # The condition for a point p to be inside is dot(p, n_k) <= S * sqrt(3)/2 for all k?
        # No. The apothem is S * sqrt(3)/2.
        # So max_k |dot(p, n_k)| <= S * sqrt(3)/2.
        # Therefore S >= max_k |dot(p, n_k)| * 2 / sqrt(3).
        
        normals = []
        for k in range(6):
            ang = np.radians(k * 60)
            normals.append(np.array([np.cos(ang), np.sin(ang)]))
        normals = np.array(normals)
        
        # Project all vertices onto all normals
        # verts: (M, 2), normals: (6, 2)
        # projections: (M, 6)
        projections = verts.dot(normals.T)
        max_proj = np.max(np.abs(projections))
        
        S = max_proj * 2.0 / np.sqrt(3.0)
        return S

    def generate_lattice_clusters():
        """
        Generate various rigid configurations of 12 unit hexagons on a hexagonal lattice.
        We use axial coordinates (q, r).
        Distance between (q1, r1) and (q2, r2) is max(|q1-q2|, |r1-r2|, |q1+r1-q2-r2|).
        For non-overlapping unit hexagons, this distance must be >= 1.
        """
        clusters = []
        
        # 1. Compact hexagonal shape (radius 2 is 19 hexagons, radius 1 is 7).
        # We need 12. 
        # A "hexagon of radius 2" has 1 + 6 + 12 = 19.
        # A "hexagon of radius 1" has 1 + 6 = 7.
        # Let's take a central hexagon and 11 neighbors?
        # Or a 2x3 rectangle? Or a 3x4 rectangle?
        
        # Helper to convert axial to cartesian
        def axial_to_cartesian(q, r):
            x = np.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            return [x, y]
            
        # Configuration 1: 3x4 Rectangle (grid-like)
        # q from 0 to 2, r from 0 to 3? 
        # In hex grid, a "rectangle" is tricky. 
        # Let's just use a set of coordinates that form a convex-ish shape.
        
        # Shape 1: "Bar" of 12? Too long.
        # Shape 2: 2 rows of 6?
        coords_2x6 = []
        for r in range(2):
            for q in range(6):
                coords_2x6.append((q, r))
        clusters.append(np.array([axial_to_cartesian(q, r) for q, r in coords_2x6]))
        
        # Shape 3: 3 rows of 4?
        coords_3x4 = []
        for r in range(3):
            for q in range(4):
                coords_3x4.append((q, r))
        clusters.append(np.array([axial_to_cartesian(q, r) for q, r in coords_3x4]))
        
        # Shape 4: Compact cluster (center + 5 neighbors + 6 more)
        # Center (0,0)
        # Neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -> 7 total
        # Add next layer: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0)
        # Let's pick 5 of these 6 to make 12.
        # To keep it symmetric, maybe take a different shape.
        # How about a "triangle" of side 4? 1+3+5+7 = 16. Too many.
        # Side 3: 1+3+5 = 9. Need 3 more.
        
        # Let's just use the two rectangular ones and a few random valid lattice clusters.
        
        # Shape 5: A "snake" or zigzag?
        # Let's generate a few random valid clusters via greedy growth.
        import random
        random.seed(42)
        
        def grow_cluster():
            coords = [(0,0)]
            set_coords = {(0,0)}
            while len(coords) < N:
                # Try to add a neighbor to a random existing cell
                candidate_added = False
                for _ in range(100):
                    base = random.choice(coords)
                    # 6 directions
                    dirs = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
                    d = random.choice(dirs)
                    nq, nr = base[0]+d[0], base[1]+d[1]
                    if (nq, nr) not in set_coords:
                        # Check if it touches any existing cell (it does, by construction)
                        # It is valid if it doesn't overlap. In lattice, distinct coords = no overlap.
                        coords.append((nq, nr))
                        set_coords.add((nq, nr))
                        candidate_added = True
                        break
                if not candidate_added:
                    return None
            return np.array([axial_to_cartesian(q, r) for q, r in coords])
            
        for i in range(20):
            c = grow_cluster()
            if c is not None:
                # Center the cluster at origin to make rotation optimization easier
                centroid = np.mean(c, axis=0)
                c = c - centroid
                clusters.append(c)
                
        # Also add the rectangular ones centered
        for i in range(len(clusters)):
            centroid = np.mean(clusters[i], axis=0)
            clusters[i] = clusters[i] - centroid
            
        return clusters

    def optimize_cluster(cluster, init_angle=0.0):
        """
        Optimize global rotation and translation of a rigid cluster.
        We fix the cluster shape. We rotate it by theta and translate by (tx, ty).
        Objective: Minimize required outer side length.
        """
        def objective(params):
            theta = params[0]
            tx = params[1]
            ty = params[2]
            
            # Rotate cluster
            c = cluster.copy()
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            rotated = c.dot(rot_mat.T)
            
            # Translate
            rotated[:, 0] += tx
            rotated[:, 1] += ty
            
            # Get vertices
            # Note: The hexagons themselves also rotate by theta.
            # The function get_cluster_vertices handles the position and angle of each hex.
            # Here 'rotated' are the centers. The angle of each hex is theta.
            angles = np.full(N, theta)
            verts = get_cluster_vertices(rotated, angles)
            
            return required_outer_side(verts)
            
        # Simple optimization: Grid search + Nelder-Mead
        # Since the function is periodic and relatively smooth, Nelder-Mead is good.
        # Initial guess: theta=0, tx=0, ty=0 (since cluster is centered at origin, this is usually close)
        
        # We can try a few starting angles to avoid local minima if the shape is asymmetric.
        best_val = float('inf')
        best_params = None
        
        # Try 6 starting angles (symmetries of hexagon)
        for start_theta in np.linspace(0, np.pi/3, 6, endpoint=False):
            x0 = np.array([start_theta, 0.0, 0.0])
            
            # Use scipy minimize
            from scipy.optimize import minimize
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
                
        return best_val, best_params

    # Generate clusters
    clusters = generate_lattice_clusters()
    
    best_overall_val = float('inf')
    best_overall_cluster = None
    best_overall_params = None
    
    for i, cluster in enumerate(clusters):
        if time.time() > deadline:
            break
            
        val, params = optimize_cluster(cluster)
        
        if val < best_overall_val:
            best_overall_val = val
            best_overall_cluster = cluster
            best_overall_params = params
            
    if best_overall_cluster is None:
        # Fallback: return a valid but not optimal configuration
        # Just place them in a line
        positions = np.zeros((N, 2))
        for i in range(N):
            positions[i, 0] = i * np.sqrt(3)
            positions[i, 1] = 0
        best_overall_cluster = positions
        best_overall_params = np.array([0.0, 0.0, 0.0])
        best_overall_val = required_outer_side(get_cluster_vertices(positions, np.zeros(N)))
        
    # Reconstruct final positions
    theta = best_overall_params[0]
    tx = best_overall_params[1]
    ty = best_overall_params[2]
    
    c = best_overall_cluster.copy()
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    rotated = c.dot(rot_mat.T)
    rotated[:, 0] += tx
    rotated[:, 1] += ty
    
    inner = []
    for i in range(N):
        inner.append([float(rotated[i, 0]), float(rotated[i, 1]), float(np.degrees(theta))])
        
    # Calculate final side length for return
    verts = get_cluster_vertices(rotated, np.full(N, theta))
    final_side = required_outer_side(verts)
    
    return inner, (0.0, 0.0), float(final_side), 0.0

# EVOLVE-BLOCK-END
