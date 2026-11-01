import numpy as np
import time
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def compute_outer_side_and_verts(positions, angles):
        """
        Compute the exact side length of the smallest regular hexagon (angle 0, centered at origin)
        that contains all unit hexagons at given positions and rotations.
        Returns (side_length, vertices_flat)
        """
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            verts[i] = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
        verts_flat = verts.reshape(-1, 2)
        
        x = verts_flat[:, 0]
        y = verts_flat[:, 1]
        sqrt3 = np.sqrt(3.0)
        
        # Support function projections for a regular hexagon aligned with axes (angle 0)
        # The side length S of the outer hexagon is determined by the max extent in the 6 normal directions.
        # For a hexagon with side S centered at origin, the distance from center to side is S*sqrt(3)/2.
        # The support number h(u) = max(p . u).
        # If we align the outer hexagon such that its sides are perpendicular to the 6 directions of a standard hexagon,
        # the constraint is h(u_k) <= S * sqrt(3)/2.
        # Thus S = (2/sqrt(3)) * max_k h(u_k).
        
        # Directions of normals for a standard hexagon (vertices at 0, 60, ...):
        # Normals are at 30, 90, 150, 210, 270, 330 degrees.
        # cos(30)=sqrt(3)/2, sin(30)=1/2
        # cos(90)=0, sin(90)=1
        # etc.
        
        # Let's just compute the max projection onto the 6 unit normals of a standard hexagon.
        angles_norm = np.radians(np.arange(30, 360, 60))
        normals = np.column_stack([np.cos(angles_norm), np.sin(angles_norm)])
        
        projections = verts_flat.dot(normals.T) # Shape (72, 6)
        max_proj = np.max(projections, axis=0) # Shape (6,)
        S = (2.0 / np.sqrt(3.0)) * np.max(max_proj)
        
        return S, verts_flat

    def objective_rigid(params):
        """
        Objective function for the rigid body cluster.
        params: [cx, cy, theta_deg]
        """
        cx, cy, theta_deg = params
        theta_rad = np.radians(theta_deg)
        
        # Rotate relative positions
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)
        
        # rel_positions is (N, 2)
        # New pos = R * rel + center
        # R = [[cos, -sin], [sin, cos]]
        # x' = x*cos - y*sin + cx
        # y' = x*sin + y*cos + cy
        
        rel = relative_positions  # Global variable set in construct
        x_rel = rel[:, 0]
        y_rel = rel[:, 1]
        
        x_new = x_rel * cos_t - y_rel * sin_t + cx
        y_new = x_rel * sin_t + y_rel * cos_t + cy
        
        positions = np.column_stack([x_new, y_new])
        angles = np.full(N, theta_deg) # All hexagons rotate with the cluster
        
        S, _ = compute_outer_side_and_verts(positions, angles)
        return S

    # 1. Define several candidate cluster topologies (relative positions)
    # All positions are relative to the cluster centroid.
    # We use a honeycomb-like packing.
    # Unit hexagon side = 1.
    # Distance between centers in honeycomb:
    # Nearest neighbors: sqrt(3) apart (flat-to-flat) or 2 apart (vertex-to-vertex)?
    # For regular hexagons of side 1, the distance between centers of adjacent hexagons in a honeycomb is sqrt(3).
    # Let's verify: Width of hexagon is sqrt(3). If they touch flat-to-flat, center distance is sqrt(3).
    # This creates a triangular lattice of centers.
    # Lattice vectors: a1 = (sqrt(3), 0), a2 = (sqrt(3)/2, 1.5)
    # Check distance between a1 and a2: sqrt((sqrt(3)/2)^2 + 1.5^2) = sqrt(3/4 + 9/4) = sqrt(12/4) = sqrt(3). Correct.
    
    sqrt3 = np.sqrt(3.0)
    
    # Pattern 1: 3x4 rectangular block of hexagons (12 hexagons)
    # This is not the most compact, but it's a valid disjoint set.
    # Actually, a 3x4 block in a triangular lattice.
    # Let's generate a set of 12 points from the triangular lattice that form a compact shape.
    # Center at (0,0).
    # Neighbors of (0,0):
    # (sqrt(3), 0), (-sqrt(3), 0)
    # (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5)
    # (sqrt(3)/2, -1.5), (-sqrt(3)/2, -1.5)
    
    # Let's define a few specific cluster shapes manually or via lattice generation.
    
    def generate_cluster_shape(coords_list):
        """
        coords_list: list of (i, j) indices in the triangular lattice.
        Returns array of (N, 2) relative positions centered at centroid.
        """
        pts = []
        for i, j in coords_list:
            x = sqrt3 * (i + j / 2.0)
            y = 1.5 * j
            pts.append([x, y])
        pts = np.array(pts)
        # Center
        centroid = np.mean(pts, axis=0)
        pts -= centroid
        return pts

    # Shape 1: A "flower" of 7 + 5 more
    # Central hex at (0,0). 6 neighbors.
    # Neighbors indices (i,j) in axial coords?
    # Let's just use Cartesian lattice generation.
    # Base vectors: v1 = (sqrt3, 0), v2 = (sqrt3/2, 1.5)
    
    def lattice_point(i, j):
        return np.array([sqrt3 * (i + j/2.0), 1.5 * j])

    # Cluster 1: Compact 12-hexagon shape
    # Try to fit 12 into a roughly circular shape.
    # Layer 0: 1 hex
    # Layer 1: 6 hexes
    # Layer 2: 5 hexes (partial ring)
    # Total: 1 + 6 + 5 = 12.
    
    # Indices for Layer 0: (0,0)
    # Indices for Layer 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Indices for Layer 2: We need 5 more.
    # Next ring neighbors of Layer 1.
    # Let's just pick 5 from the next available lattice points that keep it compact.
    # Points at distance ~ 2*sqrt(3) or sqrt(3)*sqrt(3)=3?
    # Let's use a simple rectangular grid of hexagons: 3 rows, 4 cols.
    # In triangular lattice, a 3x4 block.
    # i in 0..3, j in 0..2?
    # Let's generate a 3x4 grid.
    coords_3x4 = []
    for i in range(4):
        for j in range(3):
            coords_3x4.append((i, j))
    cluster_1 = generate_cluster_shape(coords_3x4)

    # Cluster 2: A more circular 12-hexagon shape
    # Center (0,0)
    # 6 neighbors
    # 5 more from the next ring.
    # Next ring points (distance from center ~ 2*sqrt(3) or 3? No, lattice step is sqrt(3).
    # Distance squared for (i,j) is 3*(i^2 + i*j + j^2).
    # Dist^2=3: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -> 6 points.
    # Dist^2=12: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1) -> 12 points.
    # We have 1 center + 6 inner = 7. We need 5 more from the outer ring.
    # To keep it symmetric-ish, we can't pick 5 from a ring of 12 symmetrically.
    # Maybe 1 center + 6 inner + 5 outer is asymmetric.
    # Alternative: 12 hexagons in a 2x6 strip? No, 3x4 is better.
    # Alternative: 12 hexagons in a hexagonal shape?
    # A hexagon of side 2 has 1+6+12 = 19 hexagons. Too many.
    # A hexagon of side 1 has 7.
    # So 12 is 7 + 5.
    # Let's try a different shape: 4 rows of 3?
    # i in 0..2, j in 0..3.
    coords_3x4_v2 = []
    for i in range(3):
        for j in range(4):
            coords_3x4_v2.append((i, j))
    cluster_2 = generate_cluster_shape(coords_3x4_v2) # Same as cluster 1 essentially, just transposed.

    # Let's try a "diamond" shape or something more compact.
    # How about 1 center, 6 neighbors, and then 5 more placed to fill gaps?
    # Actually, the 3x4 rectangle is quite compact for a rectangular boundary, but the outer hexagon is hexagonal.
    # A more circular cluster might fit better in a hexagon.
    
    # Let's try a cluster based on the 19-hexagon hexagon (side 2), removing the 7 corner hexagons?
    # 19 - 7 = 12.
    # The 19-hexagon hexagon has indices (i,j) such that max(|i|, |j|, |i+j|) <= 1? No.
    # Axial coords for hexagon of side 2:
    # i in -1..1, j in -1..1, |i+j| <= 1? No, that's side 1 (7 hexes).
    # Side 2: max(|i|, |j|, |i+j|) <= 2? No.
    # Let's just stick to the 3x4 grid and maybe a 2x6 grid.
    
    coords_2x6 = []
    for i in range(2):
        for j in range(6):
            coords_2x6.append((i, j))
    cluster_3 = generate_cluster_shape(coords_2x6)

    # And a "flower" of 7 plus 5 random-ish but valid neighbors?
    # Let's just use the 3x4 and 2x6 and maybe a 1x12 line (bad).
    # Let's try a "plus" shape expanded.
    
    clusters = [cluster_1, cluster_2, cluster_3]

    best_side = float('inf')
    best_params = None
    best_cluster_idx = 0

    for c_idx, cluster in enumerate(clusters):
        if time.time() > deadline:
            break
            
        relative_positions = cluster
        
        # 1. Coarse Grid Search for Rotation and Center
        # The cluster is centered at (0,0) in its local frame.
        # We need to find (cx, cy, theta) that minimizes the outer hex side.
        # Due to symmetry of the outer hexagon (6-fold), we only need to search theta in [0, 60).
        # Also, the cluster has some symmetry? Not necessarily.
        # However, the objective function is periodic with 60 deg for theta.
        
        # Initial guess: cx=0, cy=0.
        # We can optimize cx, cy for a fixed theta easily.
        # For a fixed theta, the cluster is a fixed set of points.
        # The outer hex side S(theta, cx, cy) is the max of 6 linear functions of (cx, cy).
        # S = max_k (A_k . [cx, cy] + B_k(theta))
        # This is a convex function of (cx, cy).
        # We can solve this exactly or via a fast local optimizer for each theta.
        
        # Let's sample theta in steps of 1 degree (60 samples).
        theta_samples = np.linspace(0, 60, 60, endpoint=False)
        
        best_theta = 0
        best_cx = 0
        best_cy = 0
        best_S = float('inf')
        
        for theta_deg in theta_samples:
            if time.time() > deadline:
                break
            # For this theta, find optimal cx, cy.
            # We can use a simple gradient descent or even a closed form if we identify active constraints.
            # Given the time budget, let's use a fast local optimization for cx, cy.
            # Or just grid search cx, cy in a small range?
            # The cluster size is ~ 3*sqrt(3) ~ 5.
            # So cx, cy should be in [-3, 3].
            
            # Let's use scipy minimize for (cx, cy) with fixed theta.
            # Actually, we can just define a function that takes (cx, cy) and returns S.
            
            def obj_xy(params_xy):
                cx, cy = params_xy
                return objective_rigid([cx, cy, theta_deg])
            
            # Initial guess 0,0
            try:
                res = minimize(obj_xy, [0, 0], method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
                S_val = res.fun
                if S_val < best_S:
                    best_S = S_val
                    best_theta = theta_deg
                    best_cx, best_cy = res.x
            except:
                continue

        # 2. Refine theta, cx, cy jointly
        # Use the best point from coarse search as initial guess.
        init_params = [best_cx, best_cy, best_theta]
        
        try:
            res_fine = minimize(objective_rigid, init_params, method='Nelder-Mead', 
                                 options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if res_fine.fun < best_S:
                best_S = res_fine.fun
                best_params = res_fine.x
                best_cluster_idx = c_idx
        except:
            pass

        if best_params is None or best_S == float('inf'):
             # Fallback to best from coarse search
             best_params = [best_cx, best_cy, best_theta]
             best_S = objective_rigid(best_params)
             best_cluster_idx = c_idx

    # Ensure we have a result
    if best_params is None:
        # Use first cluster with 0,0,0
        relative_positions = clusters[0]
        best_params = [0, 0, 0]
        best_S = objective_rigid(best_params)

    # Final Calculation
    cx, cy, theta_deg = best_params
    theta_rad = np.radians(theta_deg)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    
    rel = clusters[best_cluster_idx]
    x_rel = rel[:, 0]
    y_rel = rel[:, 1]
    
    x_new = x_rel * cos_t - y_rel * sin_t + cx
    y_new = x_rel * sin_t + y_rel * cos_t + cy
    
    positions = np.column_stack([x_new, y_new])
    angles = np.full(N, theta_deg)
    
    final_side, _ = compute_outer_side_and_verts(positions, angles)
    
    inner = []
    for i in range(N):
        inner.append([float(positions[i, 0]), float(positions[i, 1]), float(theta_deg)])
        
    return inner, (0.0, 0.0), float(final_side), 0.0

# EVOLVE-BLOCK-END
