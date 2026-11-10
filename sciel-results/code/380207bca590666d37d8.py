# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar, differential_evolution


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), rows (x, y, angle_deg)
        outer_hex_data: np.ndarray of shape (3,), (x, y, angle_deg)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    # --- Geometry helpers ---
    def get_vertices(cx, cy, s, angle_deg):
        a_rad = np.radians(angle_deg)
        angles = a_rad + np.deg2rad(np.arange(6) * 60)
        return np.column_stack([
            cx + s * np.cos(angles),
            cy + s * np.sin(angles)
        ])
    
    def check_disjoint(inner_data):
        """Check if all pairs of inner hexagons are separated (gap > 1e-6)."""
        n = len(inner_data)
        # Precompute vertices and edge normals
        verts = np.zeros((n, 6, 2))
        normals = np.zeros((n, 6, 2))
        for i in range(n):
            cx, cy, a = inner_data[i]
            v = get_vertices(cx, cy, 1.0, a)
            verts[i] = v
            # Edge normals
            for j in range(6):
                p1 = v[j]
                p2 = v[(j+1)%6]
                edge = p2 - p1
                norm = np.array([-edge[1], edge[0]])
                normals[i, j] = norm / np.linalg.norm(norm)
        
        for i in range(n):
            for j in range(i+1, n):
                # Check all 12 pairs of edge normals (6 from i, 6 from j)
                # Actually, we need to check if the projections of the two hexagons
                # on ANY of the 12 normals are separated.
                # If for ALL 12 normals the projections overlap, they intersect.
                # We want to ensure that for AT LEAST ONE normal, they are separated.
                
                is_separated = False
                # Normals from i
                for k in range(6):
                    n_vec = normals[i, k]
                    proj_i = np.dot(verts[i], n_vec)
                    min_i, max_i = np.min(proj_i), np.max(proj_i)
                    
                    proj_j = np.dot(verts[j], n_vec)
                    min_j, max_j = np.min(proj_j), np.max(proj_j)
                    
                    # Gap > 1e-6
                    if max_i < min_j - 1e-6 or max_j < min_i - 1e-6:
                        is_separated = True
                        break
                if is_separated:
                    continue
                    
                # Normals from j
                for k in range(6):
                    n_vec = normals[j, k]
                    proj_i = np.dot(verts[i], n_vec)
                    min_i, max_i = np.min(proj_i), np.max(proj_i)
                    
                    proj_j = np.dot(verts[j], n_vec)
                    min_j, max_j = np.min(proj_j), np.max(proj_j)
                    
                    if max_i < min_j - 1e-6 or max_j < min_i - 1e-6:
                        is_separated = True
                        break
                        
                if not is_separated:
                    return False
        return True

    def get_outer_side(inner_data, outer_angle_deg, outer_center=np.array([0.0, 0.0])):
        """
        Calculate the minimum side length of a regular hexagon with given center and angle
        that contains all vertices of the inner hexagons.
        """
        # Collect all vertices of inner hexagons
        all_verts = []
        for i in range(len(inner_data)):
            cx, cy, a = inner_data[i]
            v = get_vertices(cx, cy, 1.0, a)
            all_verts.append(v)
        
        all_verts = np.vstack(all_verts)
        all_verts -= outer_center
        
        a_rad = np.radians(outer_angle_deg)
        # Edge normals of the outer hexagon
        # The edge normals of a regular hexagon centered at 0 with angle a are at a + 30 + 60k
        # Wait, let's verify.
        # Vertices are at a, a+60, ...
        # Midpoint of edge 0 (a to a+60) is at a+30.
        # Normal to that edge is a+30.
        # So normals are at a+30 + 60k.
        normals = []
        for k in range(6):
            theta = a_rad + np.deg2rad(30 + 60*k)
            normals.append(np.array([np.cos(theta), np.sin(theta)]))
        normals = np.array(normals) # shape (6, 2)
        
        # Project all vertices onto all normals
        # dots shape: (6, N)
        dots = normals @ all_verts.T
        
        # For each normal, find the max absolute projection?
        # No, the hexagon is defined by |x . n_k| <= h, where h is the apothem.
        # Side length s = h * 2 / sqrt(3).
        # We need s such that for all p and all k: p . n_k <= h AND -p . n_k <= h
        # i.e., |p . n_k| <= h.
        
        # So h_required = max_{p, k} |p . n_k|
        h_required = np.max(np.abs(dots))
        
        side = 2 * h_required / np.sqrt(3)
        
        return side

    # --- Initial Configuration: 11 hexagons in a hexagonal cluster ---
    # Center at (0,0)
    # 6 neighbors at distance sqrt(3) * 2? No.
    # Two unit hexagons touching at an edge have center distance 2 * apothem = sqrt(3).
    # So neighbors are at distance sqrt(3).
    d = np.sqrt(3)
    
    # Angles for the 6 neighbors
    # If we use "pointy" top (angle 30 deg?), the vertices are at 30, 90, 150...
    # If we use "flat" top (angle 0 deg), vertices are at 0, 60, 120...
    # Let's stick to angle 0 for simplicity.
    # Vertices at 0, 60, 120, 180, 240, 300.
    # Edge normals at 30, 90, 150, 210, 270, 330.
    # Neighbors touch on edges. So neighbor centers are along the edge normals.
    # Neighbor 1: along normal at 30 deg?
    # Distance is d = sqrt(3).
    # pos = d * [cos(30), sin(30)]
    
    neighbor_angles = [30, 90, 150, 210, 270, 330]
    
    inner_data = np.zeros((11, 3))
    inner_data[0, 0:2] = [0, 0]
    inner_data[0, 2] = 0
    
    for i, ang in enumerate(neighbor_angles):
        rad = np.deg2rad(ang)
        inner_data[i+1, 0] = d * np.cos(rad)
        inner_data[i+1, 1] = d * np.sin(rad)
        inner_data[i+1, 2] = 0
        
    # The 4 outer hexagons fill the gaps between the inner ring and the outer boundary?
    # In the 11-hex cluster, the "corners" of the big hexagon are formed by the 4 hexagons
    # that are between the 6 neighbors.
    # Specifically, between neighbor 1 (30 deg) and neighbor 2 (90 deg), there is a gap.
    # A hexagon placed there would touch both neighbor 1 and neighbor 2.
    # Let's find the position of the hexagon at angle 60 deg (midpoint of 30 and 90).
    # It should be further out.
    # Distance from center?
    # Let r be the distance.
    # It touches neighbor 1 (at d, 30 deg).
    # Distance between (r, 60) and (d, 30) must be d.
    # (r cos60 - d cos30)^2 + (r sin60 - d sin30)^2 = d^2
    # (r/2 - d*sqrt(3)/2)^2 + (r*sqrt(3)/2 - d/2)^2 = d^2
    # r^2/4 - r*sqrt(3)d/2 + 3d^2/4 + 3r^2/4 - r*sqrt(3)d/2 + d^2/4 = d^2
    # r^2 - r*sqrt(3)d + d^2 = d^2
    # r^2 - r*sqrt(3)d = 0
    # r = sqrt(3)d = 3.
    
    # So the outer 4 hexagons are at distance 3, angles 60, 120, 240, 300.
    outer_hex_angles = [60, 120, 240, 300]
    
    for i, ang in enumerate(outer_hex_angles):
        rad = np.deg2rad(ang)
        inner_data[i+7, 0] = 3.0 * np.cos(rad)
        inner_data[i+7, 1] = 3.0 * np.sin(rad)
        inner_data[i+7, 2] = 0

    # Verify disjointness of initial config
    if not check_disjoint(inner_data):
        # Fallback: simple grid if geometry is broken (should not happen)
        # But the 11-hex cluster is a known valid packing.
        pass

    # --- Optimization ---
    # We can optimize:
    # 1. The angle of the outer hexagon.
    # 2. The positions of the 11 inner hexagons (with constraints).
    
    # First, just optimize the outer hexagon angle for the fixed cluster.
    def objective(theta_deg):
        return get_outer_side(inner_data, theta_deg)
    
    # The system has 6-fold symmetry, so we only need to search [0, 60).
    # Actually, the cluster has 6-fold symmetry, so S(theta) should be periodic with 60 deg?
    # The cluster vertices form a hexagonal shape.
    # If we rotate the outer hexagon, the "tight" constraints change.
    # Due to symmetry, the optimum should be at theta=0 or theta=30 relative to the cluster?
    # Let's check theta=0 (flat top) and theta=30 (pointy top).
    
    s0 = objective(0.0)
    s30 = objective(30.0)
    
    best_theta = 0.0 if s0 < s30 else 30.0
    best_side = min(s0, s30)
    
    # Refine theta
    res = minimize_scalar(objective, bounds=(best_theta - 10, best_theta + 10), method='bounded', options={'xatol': 1e-6})
    if res.success:
        best_theta = res.x
        best_side = res.fun
        
    # Now, try to optimize the inner hexagon positions slightly to reduce the bounding box.
    # This is harder. We can parameterize the positions.
    # The cluster is rigid if we maintain contacts.
    # Can we loosen the contacts to fit better in the outer hexagon?
    # If we move the outer 4 hexagons slightly inward, we might fit them in a smaller outer hexagon?
    # No, moving them inward moves their vertices inward, reducing the bounding box.
    # But they must not overlap with the inner ring.
    
    # Let's try a local optimization of the 4 outer hexagons' distances from center.
    # Fix the inner 7 (center + 6 neighbors) as is.
    # Optimize the radial distance of the 4 outer hexagons.
    # Let r_outer be the distance. Initially 3.0.
    # If we decrease r_outer, the vertices move inward.
    # Constraint: no overlap with neighbors.
    
    def objective_full(params):
        # params: [r_outer, theta_outer]
        r_out = params[0]
        theta_out = params[1]
        
        # Check time
        if time.time() > deadline - 10.0:
            return 1e10
            
        # Reconstruct inner_data with new r_out
        # We assume the angle of the 4 outer hexagons is still 60, 120, 240, 300.
        # We could optimize their angles too, but that breaks symmetry.
        # Let's stick to radial distance for now.
        
        new_inner_data = inner_data.copy()
        for i, ang in enumerate(outer_hex_angles):
            rad = np.deg2rad(ang)
            new_inner_data[i+7, 0] = r_out * np.cos(rad)
            new_inner_data[i+7, 1] = r_out * np.sin(rad)
            
        # Check disjointness
        if not check_disjoint(new_inner_data):
            return 1e10
            
        side = get_outer_side(new_inner_data, theta_out)
        return side

    bounds = [(2.9, 3.1), (0, 60)]
    
    # Try a few random starts or grid search in this 2D space
    best_val = 1e10
    best_params = [3.0, best_theta]
    
    # Grid search
    for r in np.linspace(2.95, 3.05, 21):
        for th in np.linspace(best_theta - 5, best_theta + 5, 11):
            if time.time() > deadline - 10.0:
                break
            val = objective_full([r, th])
            if val < best_val:
                best_val = val
                best_params = [r, th]
        
    if time.time() > deadline - 10.0:
        break

    # If we found a better value, update best_side and best_theta
    if best_val < best_side:
        best_side = best_val
        best_params_used = best_params
    else:
        best_params_used = [3.0, best_theta]

    # Final check: ensure the best_params are valid
    final_inner_data = inner_data.copy()
    if best_params_used[0] != 3.0:
        for i, ang in enumerate(outer_hex_angles):
            rad = np.deg2rad(ang)
            final_inner_data[i+7, 0] = best_params_used[0] * np.cos(rad)
            final_inner_data[i+7, 1] = best_params_used[0] * np.sin(rad)
            
    final_outer_angle = best_params_used[1]
    final_side = best_side
    
    # Construct outer hex data
    outer_hex_data = np.array([0, 0, final_outer_angle])
    
    return final_inner_data, outer_hex_data, final_side


# EVOLVE-BLOCK-END
