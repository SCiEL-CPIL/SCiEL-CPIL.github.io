import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    sqrt3 = math.sqrt(3)

    # Precompute unit hexagon vertices for support function
    # Vertices of a unit hexagon (side 1) centered at origin, one vertex at (1,0)
    # Side length 1 => Circumradius 1
    unit_verts = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])

    def support_hex(n_vec):
        """
        Calculate the support function of a unit hexagon (side 1, circumradius 1)
        in direction n_vec.
        n_vec should be a numpy array of shape (6, 2) for 6 directions, or (2,) for one.
        """
        if n_vec.ndim == 1:
            n_vec = n_vec.reshape(1, -1)
        # Max dot product of vertices with normal
        # unit_verts is (6, 2), n_vec is (M, 2)
        # proj is (M, 6)
        proj = n_vec @ unit_verts.T
        return np.max(proj, axis=1)

    def calc_S(positions, beta_deg):
        """
        Calculate the minimum side length S of an outer hexagon (centered at 0,0)
        with rotation beta_deg that contains the given positions (unit hexagons centered at positions, rotation 0).
        positions: numpy array of shape (N, 2)
        """
        beta_rad = math.radians(beta_deg)
        
        # Normals of the outer hexagon
        # The sides of a regular hexagon are perpendicular to normals at angles 30, 90, 150, 210, 270, 330 + beta
        normal_angles = np.array([30, 90, 150, 210, 270, 330]) * math.pi / 180.0 + beta_rad
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        
        # Support of the unit hexagon (rotation 0) in these normal directions
        h_hex = support_hex(normals) # shape (6,)
        
        # Max projection of the centers onto each normal
        # positions is (N, 2), normals is (6, 2)
        # We want max over N for each of the 6 normals
        # proj_centers is (6, N)
        proj_centers = normals @ positions.T
        max_center_proj = np.max(proj_centers, axis=1) # shape (6,)
        
        # Total extent in each normal direction
        extents = max_center_proj + h_hex
        
        # The outer hexagon must cover these extents.
        # For a regular hexagon of side S, the apothem (distance from center to side) is S * sqrt(3) / 2.
        # The support function in the normal direction is exactly the apothem.
        # So we need S * sqrt(3) / 2 >= max(extents)
        max_extent = np.max(extents)
        
        return max_extent * 2.0 / sqrt3

    def get_topology(type_id):
        """
        Generate 12 disjoint unit hexagon centers.
        Returns numpy array of shape (12, 2) centered at origin.
        """
        centers = []
        
        if type_id == 0:
            # 19-hex cluster minus 6 corners
            # Axial coords (q, r)
            for q in range(-2, 3):
                for r in range(-2, 3):
                    if abs(q) <= 2 and abs(r) <= 2 and abs(q+r) <= 2:
                        # Convert to Cartesian
                        x = q + r/2.0
                        y = r * sqrt3/2.0
                        dist = math.sqrt(x*x + y*y)
                        # Corners are at dist ~ 4.0 (e.g., q=2, r=0 -> x=2, y=0, dist=2? No.
                        # Let's check dist for 19-hex cluster.
                        # (0,0) dist 0
                        # (1,0) dist 1
                        # (2,0) dist 2
                        # (2,-1) x=1.5, y=-0.866, dist = sqrt(2.25+0.75)=sqrt(3)=1.732
                        # (1,1) x=1.5, y=0.866, dist 1.732
                        # (0,2) x=1, y=1.732, dist 2
                        # (-1,2) x=0, y=1.732, dist 1.732
                        # (-2,2) x=-1, y=1.732, dist 2
                        # (-2,1) x=-1.5, y=0.866, dist 1.732
                        # (-2,0) x=-2, y=0, dist 2
                        # (-1,-1) x=-1.5, y=-0.866, dist 1.732
                        # (0,-2) x=-1, y=-1.732, dist 2
                        # (1,-2) x=0, y=-1.732, dist 1.732
                        # (2,-2) x=1, y=-1.732, dist 2
                        # (2,-1) already listed?
                        # Let's just list all 19 and remove the 6 with max dist.
                        centers.append((x, y))
            
            # Sort by distance descending and remove last 6
            centers.sort(key=lambda p: -(p[0]**2 + p[1]**2))
            centers = centers[6:] # Keep the inner 13? No, 19-6=13. We need 12.
            # The 19-hex cluster has 1 center, 6 inner ring, 12 outer ring.
            # Removing 6 corners from outer ring leaves 6 outer + 6 inner + 1 center = 13.
            # We need 12. So remove 7? Or remove 6 corners and 1 side?
            # Let's just take the 12 closest to center from the 19.
            centers.sort(key=lambda p: (p[0]**2 + p[1]**2))
            centers = centers[:12]
            
        elif type_id == 1:
            # 3x4 Parallelogram
            for q in range(3):
                for r in range(4):
                    x = q + r/2.0
                    y = r * sqrt3/2.0
                    centers.append((x, y))
                    
        elif type_id == 2:
            # 2x6 Parallelogram
            for q in range(2):
                for r in range(6):
                    x = q + r/2.0
                    y = r * sqrt3/2.0
                    centers.append((x, y))
                    
        elif type_id == 3:
            # Compact "Cross" or similar
            # Center (0,0)
            # Ring 1: 6
            # Ring 2: 5 (remove one to make 12)
            pts = [(0,0)]
            for i in range(6):
                ang = i * math.pi / 3
                pts.append((math.cos(ang), math.sin(ang) * 2/sqrt3 * sqrt3/2 * 2)) # Just use axial conversion
            # Use axial for simplicity
            axial = [(0,0)]
            ring1 = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
            ring2 = [(2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)]
            # Pick 5 from ring2 to make 1+6+5=12
            # Pick symmetric ones: (2,0), (0,2), (-2,1), (-1,-1), (1,-2)
            selected_ring2 = [(2,0), (0,2), (-2,1), (-1,-1), (1,-2)]
            axial.extend(ring1)
            axial.extend(selected_ring2)
            
            for q, r in axial:
                x = q + r/2.0
                y = r * sqrt3/2.0
                centers.append((x, y))
        else:
            return None
            
        if not centers:
            return None
            
        pos = np.array(centers)
        # Center the cluster
        centroid = np.mean(pos, axis=0)
        pos = pos - centroid
        return pos

    best_S = float('inf')
    best_positions = None
    best_beta = 0.0
    best_type = 0

    topologies_to_try = [0, 1, 2, 3]
    
    for t_id in topologies_to_try:
        if time.time() > deadline - 20:
            break
            
        positions = get_topology(t_id)
        if positions is None:
            continue
            
        # Coarse grid search for beta (outer hexagon rotation)
        # Inner hexagons are fixed at rotation 0.
        # Period is 60 degrees.
        best_local_S = float('inf')
        best_local_beta = 0.0
        
        # Step size 0.5 degrees
        for b in np.arange(0, 60, 0.5):
            S = calc_S(positions, b)
            if S < best_local_S:
                best_local_S = S
                best_local_beta = b
                
        if best_local_S < best_S:
            best_S = best_local_S
            best_positions = positions
            best_beta = best_local_beta
            best_type = t_id
            
    # Refine with Nelder-Mead
    if best_positions is not None and time.time() < deadline - 10:
        # Optimize beta and potentially a small translation (dx, dy)
        # Though centroid centering is usually optimal, allowing small translation can help if the shape is asymmetric.
        
        def objective(params):
            beta, dx, dy = params
            pos_shifted = best_positions + np.array([dx, dy])
            return calc_S(pos_shifted, beta)
            
        init_params = [best_beta, 0.0, 0.0]
        try:
            res = minimize(objective, init_params, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_S:
                best_S = res.fun
                best_beta = res.x[0]
                # Update positions with optimal shift
                best_positions = best_positions + np.array([res.x[1], res.x[2]])
        except:
            pass

    # Final construction
    inner = []
    for i in range(12):
        x, y = best_positions[i]
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), best_S, best_beta

# EVOLVE-BLOCK-END
