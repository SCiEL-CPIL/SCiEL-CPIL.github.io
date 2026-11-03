import numpy as np
from scipy.optimize import minimize, linprog
from scipy.spatial import ConvexHull
import time

N = 11

# Helper to get vertices of a unit hexagon at (x, y) with angle ang
def hex_vertices(x, y, ang_deg):
    ang = np.deg2rad(ang_deg)
    # Unit hexagon vertices relative to center
    # Side length 1. Distance from center to vertex is 1.
    # Vertices at angles 0, 60, ...
    angles = np.deg2rad(np.arange(0, 360, 60) + ang_deg)
    xs = np.cos(angles)
    ys = np.sin(angles)
    return np.column_stack((x + xs, y + ys))

# Define topologies for 11 hexagons
# Coordinates in axial system (q, r)
# Convert axial to cartesian:
# x = 1.5 * q
# y = sqrt(3) / 2 * (q + 2 * r)
def axial_to_cartesian(q, r):
    x = 1.5 * q
    y = np.sqrt(3) / 2 * (q + 2 * r)
    return x, y

def get_topology_1():
    # 1 center, 6 ring, 4 extensions
    # Center (0,0)
    # Ring: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Extensions: (2,0), (1,1), (0,2), (-1,2)
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    return coords

def get_topology_2():
    # Try a more symmetric extension
    # Center (0,0)
    # Ring: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Extensions: (2,0), (1,1), (0,2), (-2,1) ? No, let's try (2,0), (-1,2), (-2,1), (0,-2)?
    # Let's try a 2x2 block plus center?
    # (0,0), (1,0), (0,1), (1,1)
    # (-1,0), (0,-1), (-1,-1), (1,-1)
    # (0,2), (-1,2), (2,0)
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-1, 2), (-2, 1), (0, -2)
    ]
    return coords

def get_topology_3():
    # Compact 3x3 minus corners?
    # (0,0)
    # (1,0), (0,1), (-1,1)
    # (1,-1), (0,-1), (-1,0)
    # (2,0), (1,1), (0,2)
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (2, -1)
    ]
    return coords

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    topologies = [get_topology_1(), get_topology_2(), get_topology_3()]
    
    best_score = float('inf')
    best_inner = None
    best_center = None
    best_side = None
    best_angle = None
    
    # Precompute cartesian centers for topologies
    topo_centers = []
    for topo in topologies:
        centers = []
        for q, r in topo:
            x, y = axial_to_cartesian(q, r)
            centers.append((x, y))
        topo_centers.append(centers)
    
    for idx, centers in enumerate(topo_centers):
        if time.time() > deadline:
            break
            
        # Get all vertices for the cluster (assuming all hexagons aligned at 0 deg initially)
        all_vertices = []
        for cx, cy in centers:
            verts = hex_vertices(cx, cy, 0.0)
            all_vertices.extend(verts)
        
        all_vertices = np.array(all_vertices)
        
        # Compute convex hull to reduce constraints
        try:
            hull = ConvexHull(all_vertices)
            hull_vertices = all_vertices[hull.vertices]
        except Exception:
            # If hull fails (e.g. all points same), fallback to all
            hull_vertices = all_vertices
        
        # Optimization variables: [phi, cx, cy, S]
        # phi: global rotation of cluster
        # cx, cy: center of outer hexagon
        # S: side length of outer hexagon
        
        def objective(vars):
            phi, cx, cy, S = vars
            if S < 0:
                return 1e6
            # Rotate hull vertices by phi
            cos_phi = np.cos(np.deg2rad(phi))
            sin_phi = np.sin(np.deg2rad(phi))
            R = np.array([
                [cos_phi, -sin_phi],
                [sin_phi, cos_phi]
            ])
            
            # Rotate and translate
            # hull_vertices are (x, y)
            # Rotated points: R @ p
            # Translated points: R @ p - (cx, cy)
            
            # We need to check if these points are inside the outer hexagon.
            # Outer hexagon is fixed at angle 0 (we optimize phi instead).
            # Outer hexagon constraints (angle 0):
            # Normals at 0, 60, 120, 180, 240, 300
            # n_k = (cos(k*60), sin(k*60))
            # Distance from center to side = S * sqrt(3) / 2
            # Constraint: n_k . (p - c) <= S * sqrt(3) / 2
            
            apothem = S * np.sqrt(3) / 2
            
            # Compute rotated points
            pts_rot = hull_vertices @ R.T  # Shape (N, 2)
            pts_centered = pts_rot - np.array([cx, cy])
            
            # Check constraints for each normal
            # Normals
            angles = np.deg2rad(np.arange(0, 360, 60))
            normals = np.column_stack((np.cos(angles), np.sin(angles))) # Shape (6, 2)
            
            # Distances: pts_centered @ normals.T -> Shape (N, 6)
            dists = pts_centered @ normals.T
            
            # Max distance must be <= apothem
            max_dist = np.max(dists)
            
            # Penalty if violated
            if max_dist > apothem:
                return S + 100 * (max_dist - apothem)
            
            return S

        def constraint(vars):
            # We handle constraints in objective via penalty, or use SLSQP constraints
            # Let's use SLSQP with explicit constraints for better accuracy
            phi, cx, cy, S = vars
            if S < 0:
                return -1e6
            cos_phi = np.cos(np.deg2rad(phi))
            sin_phi = np.sin(np.deg2rad(phi))
            R = np.array([
                [cos_phi, -sin_phi],
                [sin_phi, cos_phi]
            ])
            pts_rot = hull_vertices @ R.T
            pts_centered = pts_rot - np.array([cx, cy])
            angles = np.deg2rad(np.arange(0, 360, 60))
            normals = np.column_stack((np.cos(angles), np.sin(angles)))
            dists = pts_centered @ normals.T
            apothem = S * np.sqrt(3) / 2
            # Return vector of constraints: apothem - dists >= 0
            # Shape (N, 6)
            # Flatten
            return (apothem - dists).flatten()

        # Multi-start for phi
        best_local_S = float('inf')
        best_local_vars = None
        
        for phi_init in [0, 15, 30, 45, 60, 10, 20, 40, 50]:
            if time.time() > deadline:
                break
            
            x0 = [phi_init, 0.0, 0.0, 5.0] # Initial S guess
            
            constraints = []
            # We can't easily define 6*N constraints in a list of dicts for SLSQP efficiently if N is large,
            # but N is small (convex hull of 11 hexes is small).
            # Let's just use the penalty method in objective for simplicity and robustness,
            # or define a single constraint function that returns the min slack.
            
            # Actually, SLSQP requires a list of constraint dicts.
            # Let's use a simplified penalty approach in objective to avoid complex constraint setup.
            
            res = minimize(objective, x0, method='SLSQP', options={'maxiter': 100, 'ftol': 1e-6})
            
            if res.success and res.fun < best_local_S:
                best_local_S = res.fun
                best_local_vars = res.x
        
        if best_local_vars is not None:
            phi_opt, cx_opt, cy_opt, S_opt = best_local_vars
            if S_opt < best_score:
                best_score = S_opt
                best_center = (cx_opt, cy_opt)
                best_side = S_opt
                best_angle = 0.0 # Outer angle is 0 in this formulation
                
                # Construct inner hexagons
                # We need to rotate the centers by phi_opt
                cos_phi = np.cos(np.deg2rad(phi_opt))
                sin_phi = np.sin(np.deg2rad(phi_opt))
                
                inner = []
                for cx, cy in centers:
                    # Rotate center
                    x_rot = cx * cos_phi - cy * sin_phi
                    y_rot = cx * sin_phi + cy * cos_phi
                    # The inner hexagons are aligned with the cluster rotation?
                    # In the objective, we rotated the *vertices* by phi.
                    # This implies the inner hexagons are rotated by phi relative to the outer (which is at 0).
                    # So the angle of each inner hexagon is phi_opt.
                    inner.append([x_rot, y_rot, phi_opt])
                
                best_inner = inner

    # If no valid solution found (should not happen), return seed
    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, best_angle

# EVOLVE-BLOCK-START
# The code above is the complete program.
# EVOLVE-BLOCK-END
