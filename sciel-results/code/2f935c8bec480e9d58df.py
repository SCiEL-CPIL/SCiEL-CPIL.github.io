import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0 # Be safe

def construct():
    # Check if we have enough time
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Define candidate point sets (axial coordinates)
    # These represent relative positions of the 11 hexagons on a triangular lattice.
    # (0,0) is always present. The lattice is fixed, so relative positions are fixed.
    # We assume all inner hexagons are aligned with the lattice (rotation 0).
    
    candidates = []
    
    # 1. Compact 3x3 block with 2 extra (from seed axial_1 variant, verified compact)
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (2, -1), (0, 2)
    ])

    # 2. Linear-ish chain bent
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (1, 3)
    ])

    # 3. Cross shape
    candidates.append([
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ])

    # 4. 4x4 block minus corners (dense)
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3), (2, 3)
    ])
    # Trim to 11: remove (3,3) if it was there? No, the list above has 13. 
    # Let's make a specific 11-point dense cluster.
    # 3x3 center = 9 points. Add 2 neighbors.
    # Center:
    c3x3 = [
        (0,0), (1,0), (2,0),
        (0,1), (1,1), (2,1),
        (0,2), (1,2), (2,2)
    ]
    # Add (3,0) and (3,1)
    c_dense_1 = c3x3 + [(3,0), (3,1)]
    candidates.append(c_dense_1)
    
    # Add (0,3) and (1,3)
    c_dense_2 = c3x3 + [(0,3), (1,3)]
    candidates.append(c_dense_2)

    # 5. Diamond shape
    candidates.append([
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, 1), (-1, 1),
        (2, 0), (-2, 0),
        (0, 2), (0, -2)
    ])

    # 6. Long strip
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ])
    
    # 7. Asymmetric L
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
        (0, 3), (1, 3)
    ])

    # 8. T-Shape
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (2, 1), (1, 1), (3, 1),
        (2, -1), (1, -1), (3, -1)
    ])
    # Wait, (1,-1) and (3,-1) are not neighbors of (2,0) in a way that creates a solid T?
    # (2,0) neighbors: (1,0),(3,0),(2,1),(1,1),(3,1),(2,-1),(1,-1),(3,-1).
    # So (1,1),(3,1),(2,1) and (1,-1),(3,-1),(2,-1) are all valid neighbors.
    # This forms a valid packing.

    # 9. Hexagonal cluster (1 center, 6 ring, 4 extra)
    ring1 = [
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)
    ]
    # Neighbors of ring1 that are not in ring1:
    # (2,0), (-2,0), (0,2), (2,-2)? No, let's pick 4 compact ones.
    # (2,0) is neighbor of (1,0) and (1,-1).
    # (-2,0) is neighbor of (-1,0) and (-1,1).
    # (0,2) is neighbor of (0,1) and (-1,1).
    # (1,2) is neighbor of (0,1) and (1,1)? No (1,1) is not in ring1? 
    # Ring1 points: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1).
    # (1,1) is NOT in ring1. It is a neighbor of (1,0) and (0,1).
    # So we can add (1,1).
    extras = [(2,0), (-2,0), (0,2), (1,1)]
    c_hex = [(0,0)] + ring1 + extras
    candidates.append(c_hex)

    # 10. Another variation of Hex cluster
    extras2 = [(2,0), (-2,0), (0,2), (0,-2)]
    c_hex2 = [(0,0)] + ring1 + extras2
    candidates.append(c_hex2)

    # Helper: Convert axial to cartesian
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            # Standard pointy-top hex grid axial to pixel
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # Objective Function:
    # Minimize the side length S of the outer regular hexagon.
    # Outer hexagon center (cx, cy), rotation beta.
    # Inner hexagons are fixed at positions pts, rotation 0.
    # Condition: All vertices of inner hexagons must lie inside the outer hexagon.
    # A regular hexagon of side S, center C, rotation beta is the intersection of 6 half-planes.
    # Normal vectors for outer hexagon sides:
    # n_k = (cos(beta + pi/6 + k*pi/3), sin(beta + pi/6 + k*pi/3)) for k=0..5
    # Apothem of outer hexagon A = S * sqrt(3)/2.
    # Constraint for each vertex v of inner hexagons:
    # n_k . (v - C) <= A  for all k, v.
    # To find minimal S for given C, beta:
    # A_req = max_{k, v} (n_k . (v - C))
    # S_req = A_req / (sqrt(3)/2)
    # Note: We must ensure the max is positive? If all projections are negative, the hexagon is large enough to cover them from the center?
    # Actually, the definition of the hexagon is symmetric. The distance from center to boundary in direction n is A.
    # So we need A >= max_{k,v} (n_k . (v-C)) AND A >= max_{k,v} (-n_k . (v-C)).
    # But since n_k spans all 6 directions, max(n_k . x) over k=0..5 covers both positive and negative if we consider the set of normals correctly?
    # The normals are 120 deg apart? No, 60 deg apart.
    # n_0, n_1, ..., n_5.
    # -n_0 is n_3 (rotated 180).
    # So max_{k} |n_k . (v-C)| is the required apothem.
    
    def objective(params):
        cx, cy, beta = params
        C = np.array([cx, cy])
        
        # Normals for outer hexagon
        normals = np.array([
            [math.cos(beta + math.pi/6 + k*math.pi/3), math.sin(beta + math.pi/6 + k*math.pi/3)]
            for k in range(6)
        ])
        
        # Calculate projections of all inner vertices
        # We need vertices of all 11 inner hexagons.
        # To save time, we can pre-compute the set of all vertices relative to their centers.
        # Inner hexagon side 1, apothem APOTHEM.
        # Vertices relative to center:
        # v_j = (cos(j*pi/3), sin(j*pi/3)) for j=0..5? 
        # Wait, regular hexagon vertices are at angles 0, 60, 120...
        # The normal to the side is at 30, 90...
        # If inner hex is aligned with lattice (pointy top), its vertices are at:
        # (1, 0) is not a vertex. 
        # Axial (1,0) is center. 
        # Vertex at (sqrt(3)/2, 1.5)? No.
        # Let's define inner vertices explicitly.
        # Side length 1.
        # Vertices at angles 0, 60, ... relative to center?
        # For a "flat-top" hex (which matches the axial grid orientation usually), 
        # vertices are at 0, 60, 120, 180, 240, 300.
        # x = cos(theta), y = sin(theta).
        # However, our axial_to_cartesian uses pointy-top formula:
        # x = sqrt(3)*(q + r/2), y = 1.5*r.
        # This corresponds to a hexagon with flat sides on top/bottom? 
        # Let's check distance to neighbor (1,0).
        # (0,0) -> (0,0). (1,0) -> (sqrt(3), 0).
        # Distance is sqrt(3).
        # Two touching hexagons of side 1 have center distance 2*apothem = sqrt(3).
        # So the hexagons are "flat-top" in the sense that the flat side is perpendicular to the vector to the neighbor?
        # Vector to (1,0) is along X axis.
        # So the flat side is vertical? No, the normal is along X.
        # So the side is vertical (Y-Z plane in 3D, but 2D so Y-axis).
        # So the hexagon has flat sides facing Left, Right, Top-Left, Top-Right, etc?
        # No, if normal is X, the side is vertical.
        # The vertices are at angles +/-30, +/-90, +/-150?
        # Let's calculate vertices of a hexagon with apothem APOTHEM and normal along X.
        # The vertex in direction 0 is at distance 1 (radius).
        # So vertices are at angles 0, 60, 120, 180, 240, 300.
        # Wait, if normal is along X (angle 0), the side is vertical.
        # The vertices adjacent to that side are at 30 and -30 degrees?
        # Let's verify:
        # Point (cos(30), sin(30)) = (sqrt(3)/2, 1/2).
        # Point (cos(-30), sin(-30)) = (sqrt(3)/2, -1/2).
        # Midpoint is (sqrt(3)/2, 0). This is on the X axis.
        # Distance from center to midpoint is sqrt(3)/2 = APOTHEM.
        # So yes, for a hexagon aligned with the lattice such that a flat side faces the (1,0) neighbor,
        # the vertices are at angles 0, 60, 120, 180, 240, 300.
        
        # Pre-compute local vertices
        angles = np.arange(0, 2*math.pi, math.pi/3)
        local_verts = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # All vertices: center + local_verts for each inner hex
        # pts: (N, 2)
        # local_verts: (6, 2)
        # We can broadcast: (N, 1, 2) + (1, 6, 2) -> (N, 6, 2)
        all_verts = pts_broadcast + local_verts[None, :, :]
        
        # Reshape to (N*6, 2)
        all_verts_flat = all_verts.reshape(-1, 2)
        
        # Subtract center
        all_verts_centered = all_verts_flat - C
        
        # Project onto normals
        # (N*6, 2) @ (2, 6) -> (N*6, 6)
        projs = all_verts_centered @ normals.T
        
        # We need max absolute projection
        max_proj = np.max(np.abs(projs))
        
        # Side length S = max_proj / APOTHEM
        return max_proj / APOTHEM

    # Pre-compute broadcasted points for speed?
    # We will do this inside the loop for each candidate to avoid memory waste if not needed, 
    # but N is small (11), so (11, 6, 2) is tiny.
    
    best_global_S = float('inf')
    best_global_params = None
    best_global_pts = None

    # Sort candidates by quick bounding box estimate
    def quick_estimate(axial_list):
        pts = axial_to_cartesian(axial_list)
        # Use a crude bounding circle or box
        # Center of mass
        c = np.mean(pts, axis=0)
        max_d = np.max(np.linalg.norm(pts - c, axis=1))
        # Radius of circumscribing circle is roughly max_d + 1 (vertex radius)
        # Outer hex apothem must be >= max_d + 1
        # S = A / APOTHEM
        return (max_d + 1.0) / APOTHEM

    estimated_scores = [(quick_estimate(axial), i) for i, axial in enumerate(candidates)]
    estimated_scores.sort()
    
    # Optimization Strategy:
    # 1. Coarse grid for beta (outer rotation).
    # 2. For each beta, optimize (cx, cy) using the fact that for fixed beta,
    #    the optimal center is the center of the minimum bounding box in the hexagonal metric.
    #    Actually, for fixed beta, the objective is convex in (cx, cy).
    #    We can use a simple Nelder-Mead for (cx, cy) for each beta.
    #    Or even better: The optimal cx, cy for fixed beta is such that the max projection
    #    is balanced.
    #    Let's just run Nelder-Mead on (cx, cy, beta) directly from a few starts.
    #    Since the function is smooth (max of linear functions is convex, and we are minimizing a convex function),
    #    Nelder-Mead should work well.
    
    for _, idx in estimated_scores:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        axial = candidates[idx]
        pts = axial_to_cartesian(axial)
        
        # Pre-compute all vertices for this configuration
        angles = np.arange(0, 2*math.pi, math.pi/3)
        local_verts = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_broadcast = pts[:, None, :]
        all_verts = pts_broadcast + local_verts[None, :, :]
        all_verts_flat = all_verts.reshape(-1, 2)
        
        # Objective function closure
        def obj(params):
            cx, cy, beta = params
            C = np.array([cx, cy])
            normals = np.array([
                [math.cos(beta + math.pi/6 + k*math.pi/3), math.sin(beta + math.pi/6 + k*math.pi/3)]
                for k in range(6)
            ])
            all_verts_centered = all_verts_flat - C
            projs = all_verts_centered @ normals.T
            max_proj = np.max(np.abs(projs))
            return max_proj / APOTHEM

        # Initial guess: center of mass, beta=0
        cx0 = np.mean(pts[:, 0])
        cy0 = np.mean(pts[:, 1])
        
        best_local_S = float('inf')
        best_local_params = None
        
        # Try multiple starting betas
        start_betas = [0.0, math.pi/12, math.pi/6, -math.pi/12, -math.pi/6, math.pi/3, -math.pi/3]
        
        for sb in start_betas:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            x0 = [cx0, cy0, sb]
            try:
                res = minimize(obj, x0, method='Nelder-Mead', 
                               options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 5000, 'adaptive': True})
                if res.fun < best_local_S:
                    best_local_S = res.fun
                    best_local_params = res.x
            except Exception:
                pass
        
        # Polish the best local result
        if best_local_params is not None:
            # High precision polish
            try:
                res_polish = minimize(obj, best_local_params, method='Nelder-Mead',
                                      options={'xatol': 1e-14, 'fatol': 1e-14, 'maxiter': 20000, 'adaptive': True})
                if res_polish.fun < best_local_S:
                    best_local_S = res_polish.fun
                    best_local_params = res_polish.x
            except Exception:
                pass
        
        if best_local_S < best_global_S:
            best_global_S = best_local_S
            best_global_params = best_local_params
            best_global_pts = pts

    if best_global_params is None or best_global_S == float('inf'):
        # Fallback: simple centering
        best_global_pts = axial_to_cartesian(candidates[0])
        cx = float(np.mean(best_global_pts[:, 0]))
        cy = float(np.mean(best_global_pts[:, 1]))
        beta = 0.0
        best_global_params = [cx, cy, beta]
        # Calculate S for fallback
        angles = np.arange(0, 2*math.pi, math.pi/3)
        local_verts = np.column_stack([np.cos(angles), np.sin(angles)])
        all_verts = (best_global_pts[:, None, :] + local_verts[None, :, :]).reshape(-1, 2)
        normals = np.array([
            [math.cos(beta + math.pi/6 + k*math.pi/3), math.sin(beta + math.pi/6 + k*math.pi/3)]
            for k in range(6)
        ])
        max_proj = np.max(np.abs((all_verts - np.array([cx, cy])) @ normals.T))
        best_global_S = max_proj / APOTHEM

    cx, cy, beta = best_global_params
    
    # Construct output
    # Inner hexagons: positions from best_global_pts, angle 0 (since we assumed fixed rotation)
    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), 0.0])
        
    center = [float(cx), float(cy)]
    side = float(best_global_S)
    rotation = float(math.degrees(beta))
    
    return inner, center, side, rotation

