"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def hex_vertices(cx, cy, angle, s=1.0):
        """Return the 6 vertices of a regular hexagon."""
        angles = angle + np.arange(6) * np.pi / 3
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def axial_to_cart(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    # Generate candidate sets of 11 hexagons
    # Base: Center + Ring 1
    base_axial = [(0,0)]
    ring1 = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    base_axial += ring1
    
    # Generate Ring 2
    ring2 = []
    # Standard hex ring generation
    # Start at (2,0)
    # Directions: 
    # 1: (1, 0)
    # 2: (0, 1)
    # 3: (-1, 1)
    # 4: (-1, 0)
    # 5: (0, -1)
    # 6: (1, -1)
    # This is getting complicated to hardcode. Let's just use a few known good symmetric sets.
    
    # Set 1: Symmetric "Plus" extension
    # Ring 2 points at axial (2,0), (-2,0), (0,2), (0,-2)
    # Check (0,2): q=0, r=2. x=0, y=3.
    # Check (2,0): q=2, r=0. x=2*sqrt(3), y=0.
    # This set is symmetric.
    set1_extra = [(2,0), (-2,0), (0,2), (0,-2)]
    
    # Set 2: Another symmetric set
    # (2,0), (2,-1), (-2,-1), (-2,0) ? No, that's not symmetric.
    # How about (2,0), (1,-2), (-1,-2), (-2,0)?
    set2_extra = [(2,0), (1,-2), (-1,-2), (-2,0)]
    
    # Set 3: (2,0), (2,-1), (0,-2), (-2,-1)
    set3_extra = [(2,0), (2,-1), (0,-2), (-2,-1)]
    
    candidate_sets = [
        base_axial + set1_extra,
        base_axial + set2_extra,
        base_axial + set3_extra
    ]
    
    # Pre-compute local coordinates for each candidate set (centered at 0,0)
    candidate_local_coords = []
    for coords in candidate_sets:
        pts = np.array([axial_to_cart(q, r) for q, r in coords])
        # Center the set
        centroid = pts.mean(axis=0)
        pts -= centroid
        candidate_local_coords.append(pts)
        
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for idx, local_coords in enumerate(candidate_local_coords):
        if time.time() > deadline:
            break
            
        # We optimize:
        # theta_c: rotation of the cluster
        # theta_cont: rotation of the container
        # We assume the cluster is centered at (0,0) and container at (0,0).
        # If the cluster is not perfectly symmetric, shifting it might help, 
        # but for these symmetric sets, centering is optimal.
        
        def objective(params):
            theta_c = params[0]
            theta_cont = params[1]
            
            # Rotate local coords by theta_c
            cos_c = np.cos(theta_c)
            sin_c = np.sin(theta_c)
            R_c = np.array([[cos_c, -sin_c], [sin_c, cos_c]])
            rotated_coords = local_coords @ R_c.T
            
            # Compute vertices for all 11 hexagons
            # Each hexagon is rotated by theta_c as well (rigid body)
            verts = np.zeros((N, 6, 2))
            for i in range(N):
                cx, cy = rotated_coords[i]
                # Vertex angles relative to hex center are theta_c + k*pi/3
                angles = theta_c + np.arange(6) * np.pi / 3
                vx = cx + np.cos(angles)
                vy = cy + np.sin(angles)
                verts[i] = np.column_stack([vx, vy])
                
            # Transform to container frame (rotate by -theta_cont)
            cos_k = np.cos(-theta_cont)
            sin_k = np.sin(-theta_cont)
            # verts shape (N, 6, 2)
            rx = verts[..., 0] * cos_k - verts[..., 1] * sin_k
            ry = verts[..., 0] * sin_k + verts[..., 1] * cos_k
            
            # Container normals
            # Unit hexagon apothem directions: pi/6, pi/2, 5pi/6, 7pi/6, 3pi/2, 11pi/6
            # Wait, the container is a regular hexagon. Its sides are perpendicular to these normals.
            # The distance from center to side is S * sqrt(3) / 2.
            # So S * sqrt(3) / 2 >= max projection of any vertex onto any normal.
            theta_n = np.pi / 6 + np.arange(6) * np.pi / 3
            nx = np.cos(theta_n)
            ny = np.sin(theta_n)
            
            proj = rx[..., np.newaxis] * nx[np.newaxis, np.newaxis, :] + \
                   ry[..., np.newaxis] * ny[np.newaxis, np.newaxis, :]
            
            max_proj = np.max(np.abs(proj))
            apothem = np.sqrt(3) / 2
            return max_proj / apothem

        # Multi-start optimization
        best_local_S = float('inf')
        best_local_params = None
        
        # Start points for theta_c, theta_cont
        # Symmetry suggests 0 or pi/6 or pi/3
        starts = [
            [0.0, 0.0],
            [0.0, np.pi/6],
            [np.pi/6, 0.0],
            [np.pi/6, np.pi/6],
            [np.pi/3, 0.0],
            [0.0, np.pi/3]
        ]
        
        for start in starts:
            if time.time() > deadline:
                break
            try:
                res = minimize(objective, start, method='Nelder-Mead', 
                               options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_local_S:
                    best_local_S = res.fun
                    best_local_params = res.x
            except Exception:
                pass
        
        if best_local_S < best_S:
            best_S = best_local_S
            # Reconstruct inner hexagons
            theta_c, theta_cont = best_local_params
            cos_c = np.cos(theta_c)
            sin_c = np.sin(theta_c)
            R_c = np.array([[cos_c, -sin_c], [sin_c, cos_c]])
            rotated_coords = local_coords @ R_c.T
            
            inner = []
            for i in range(N):
                cx, cy = rotated_coords[i]
                # Angle in degrees
                inner.append([cx, cy, np.degrees(theta_c)])
            
            best_inner = inner
            best_center = (0.0, 0.0)
            best_angle = np.degrees(theta_cont)
            
    # If no valid result, fallback to a simple lattice
    if best_inner is None:
        # Simple fallback
        axial_coords = [(0,0)]
        axial_coords += [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
        axial_coords += [(2,0), (2,-1), (2,-2), (1,-2)]
        pts = np.array([axial_to_cart(q, r) for q, r in axial_coords])
        centroid = pts.mean(axis=0)
        pts -= centroid
        inner = []
        for i in range(N):
            inner.append([pts[i][0], pts[i][1], 0.0])
        # Estimate S
        # Just return a large S to be safe? Or compute it.
        # Compute S for theta=0
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            cx, cy = pts[i]
            angles = np.arange(6) * np.pi / 3
            verts[i] = np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])
        theta_n = np.pi / 6 + np.arange(6) * np.pi / 3
        nx = np.cos(theta_n)
        ny = np.sin(theta_n)
        proj = verts[..., 0, np.newaxis] * nx[np.newaxis, np.newaxis, :] + \
               verts[..., 1, np.newaxis] * ny[np.newaxis, np.newaxis, :]
        max_proj = np.max(np.abs(proj))
        S = max_proj / (np.sqrt(3) / 2)
        return inner, (0.0, 0.0), S, 0.0

    return best_inner, best_center, best_S, best_angle

# EVOLVE-BLOCK-END
