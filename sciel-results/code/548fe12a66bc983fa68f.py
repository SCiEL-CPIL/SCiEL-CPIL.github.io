"""
Strategy: Hybrid Rigid-Core with Optimized Periphery.

Key Ideas:
1. Retain the rigid 7-hexagon core (1 center + 6 ring) which is known to be a stable, dense packing.
2. Allow the 4 "gap filler" hexagons to move (translate and rotate) to optimize the overall bounding hexagon size and avoid overlaps more effectively than the rigid lattice assumption.
3. Use a multi-start local optimization (Nelder-Mead) over the positions/angles of the 4 gap fillers and the outer hexagon parameters.
4. This expands the search space significantly compared to the previous rigid cluster rotation-only search, potentially finding a tighter packing.
"""

import time
import numpy as np
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0  # Safety margin before 150s kill

def _get_core_centers():
    """
    Generate centers for the 7-hexagon core (1 center + 6 ring).
    """
    sqrt3 = np.sqrt(3)
    centers = np.array([
        [0.0, 0.0],
        [2*sqrt3, 0.0],
        [2*sqrt3 * np.cos(np.pi/3), 2*sqrt3 * np.sin(np.pi/3)],
        [2*sqrt3 * np.cos(2*np.pi/3), 2*sqrt3 * np.sin(2*np.pi/3)],
        [2*sqrt3 * np.cos(np.pi), 2*sqrt3 * np.sin(np.pi)],
        [2*sqrt3 * np.cos(4*np.pi/3), 2*sqrt3 * np.sin(4*np.pi/3)],
        [2*sqrt3 * np.cos(5*np.pi/3), 2*sqrt3 * np.sin(5*np.pi/3)],
    ])
    return centers

def _get_gap_filler_initial():
    """
    Generate initial positions for the 4 gap fillers based on the standard lattice.
    Returns array of shape (4, 3) -> (x, y, angle_deg)
    """
    sqrt3 = np.sqrt(3)
    # Standard lattice positions for gap fillers (distance 6 from center, angles 30, 150, 210, 330)
    # Note: In the previous code, they were at 6.0 distance.
    # Let's verify the standard hex lattice.
    # Center (0,0). Ring 1 at dist 2*sqrt(3).
    # Gap fillers in the "triangular" gaps of the ring.
    # The vertices of the ring hexagons touch the center hexagon.
    # The gap between two adjacent ring hexagons and the center is a rhombus?
    # Actually, in a perfect hex tiling, there are no gaps.
    # The "gap fillers" in the previous code were placed at distance 6.
    # Let's check the distance between center and gap filler in standard tiling.
    # If we have 1 center, 6 around it. The next shell has 12 hexagons.
    # The "gap fillers" in the previous code (4 of them) were likely the ones that fit in the concave corners of the 7-hexagon cluster.
    # The 7-hexagon cluster has 6 concave corners. The previous code used 4.
    # Let's stick to the previous initialization to maintain continuity, but allow them to move.
    
    angles = [30, 150, 210, 330]
    dist = 6.0
    inits = []
    for a_deg in angles:
        a = np.radians(a_deg)
        x = dist * np.cos(a)
        y = dist * np.sin(a)
        # Angle of the hexagon itself. In standard tiling, they are aligned.
        inits.append([x, y, 0.0])
    return np.array(inits)

def _compute_vertices(centers, angles_deg):
    """
    Compute vertices for a list of hexagons.
    centers: (N, 2)
    angles_deg: (N,)
    Returns: (N*6, 2)
    """
    N = len(centers)
    verts = np.zeros((N*6, 2))
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    unit_verts_x = np.cos(base_angles)
    unit_verts_y = np.sin(base_angles)
    
    for i in range(N):
        cx, cy = centers[i]
        a = np.radians(angles_deg[i])
        ca, sa = np.cos(a), np.sin(a)
        
        # Rotate unit vertices
        vx = ca * unit_verts_x - sa * unit_verts_y
        vy = sa * unit_verts_x + ca * unit_verts_y
        
        verts[i*6:(i+1)*6, 0] = cx + vx
        verts[i*6:(i+1)*6, 1] = cy + vy
        
    return verts

def _min_outer_side_fast(verts):
    """
    Fast approximation of minimum enclosing hexagon side.
    We sample 12 outer rotations and use a simple convex center estimation.
    """
    best_side = np.inf
    sqrt3 = np.sqrt(3)
    
    # Sample outer rotations
    for oa_deg in range(0, 60, 5):
        phi = np.radians(oa_deg)
        ca, sa = np.cos(phi), np.sin(phi)
        
        # Project points onto the 3 axes of the hexagon
        # Axes in global frame:
        # n1 = (1, 0) rotated by phi -> (cos phi, sin phi)
        # n2 = (1/2, sqrt(3)/2) rotated by phi
        # n3 = (-1/2, sqrt(3)/2) rotated by phi
        
        # Actually, the constraints are |n_k . (p - c)| <= S * (sqrt(3)/2) ?
        # For a unit hexagon (side 1), the apothem is sqrt(3)/2.
        # So the distance from center to side is S * sqrt(3)/2.
        # So |n_k . (p - c)| <= S * sqrt(3)/2.
        # S >= 2/sqrt(3) * max_k |n_k . (p - c)|
        
        n1 = np.array([ca, sa])
        n2 = np.array([ca * 0.5 - sa * np.sqrt(3)/2, ca * np.sqrt(3)/2 + sa * 0.5])
        n3 = np.array([-ca * 0.5 - sa * np.sqrt(3)/2, ca * np.sqrt(3)/2 - sa * 0.5])
        
        # To find optimal c, we want to minimize max_k |n_k . p - n_k . c|.
        # This is equivalent to finding the center of the smallest interval in the 3 projected dimensions.
        # For each axis k, let proj_k = n_k . p.
        # We need c such that n_k . c is the midpoint of [min(proj_k), max(proj_k)] for all k?
        # No, c must satisfy all 3.
        # This is a feasibility problem.
        # Approximation: Use the centroid of the vertices as c.
        c = np.mean(verts, axis=0)
        
        # Calculate required S
        dx = verts[:, 0] - c[0]
        dy = verts[:, 1] - c[1]
        
        d1 = np.abs(dx * n1[0] + dy * n1[1])
        d2 = np.abs(dx * n2[0] + dy * n2[1])
        d3 = np.abs(dx * n3[0] + dy * n3[1])
        
        max_d = np.max(np.maximum(d1, np.maximum(d2, d3)))
        side = max_d * 2.0 / np.sqrt(3)
        
        if side < best_side:
            best_side = side
            
    return best_side

def _overlap_penalty(centers, angles_deg):
    """
    Calculate penalty for overlapping hexagons.
    """
    penalty = 0.0
    N = len(centers)
    # Simple distance check: if center distance < 2*sqrt(3) - epsilon, they might overlap.
    # For rotated hexagons, min distance is slightly larger than 2*sqrt(3) if aligned, but can be smaller?
    # Actually, the minimum distance between centers of two disjoint unit hexagons is 2*sqrt(3) if they share an edge.
    # If they are rotated, they can be closer? No, the "width" of a hexagon is 2*sqrt(3).
    # So if distance < 2*sqrt(3), they MUST overlap.
    min_dist = 2 * np.sqrt(3)
    
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                penalty += (min_dist - d) ** 2
                
    return penalty

def construct():
    global START_TIME, DEADLINE
    
    core_centers = _get_core_centers()
    gap_inits = _get_gap_filler_initial()
    
    # Initial full configuration
    centers_init = np.vstack([core_centers, gap_inits[:, :2]])
    angles_init = np.concatenate([[0.0]*7, gap_inits[:, 2]])
    
    best_score = np.inf
    best_params = None
    
    # We will optimize the 4 gap fillers (x, y, angle) and the outer hexagon (cx, cy, phi).
    # Total variables: 4*3 + 3 = 15.
    
    def objective(params):
        if time.time() > DEADLINE - 5.0:
            return np.inf
            
        # Unpack
        gap_params = params[:12].reshape(4, 3)
        outer_params = params[12:]
        cx, cy, phi_deg = outer_params
        
        # Construct full centers
        centers = np.vstack([core_centers, gap_params[:, :2]])
        angles = np.concatenate([[0.0]*7, gap_params[:, 2]])
        
        # Check overlaps
        pen = _overlap_penalty(centers, angles)
        if pen > 1e-3:
            return 1e6 + pen
            
        # Compute vertices
        verts = _compute_vertices(centers, angles)
        
        # Compute min enclosing hexagon side
        # We use the fast method, but we also need to account for the outer hexagon center and rotation.
        # The fast method above assumed we could choose c. Here c and phi are fixed by params.
        # So we just calculate the side length required for this specific c and phi.
        
        phi = np.radians(phi_deg)
        ca, sa = np.cos(phi), np.sin(phi)
        sqrt3 = np.sqrt(3)
        
        n1 = np.array([ca, sa])
        n2 = np.array([ca * 0.5 - sa * np.sqrt(3)/2, ca * np.sqrt(3)/2 + sa * 0.5])
        n3 = np.array([-ca * 0.5 - sa * np.sqrt(3)/2, ca * np.sqrt(3)/2 - sa * 0.5])
        
        dx = verts[:, 0] - cx
        dy = verts[:, 1] - cy
        
        d1 = np.abs(dx * n1[0] + dy * n1[1])
        d2 = np.abs(dx * n2[0] + dy * n2[1])
        d3 = np.abs(dx * n3[0] + dy * n3[1])
        
        max_d = np.max(np.maximum(d1, np.maximum(d2, d3)))
        side = max_d * 2.0 / np.sqrt(3)
        
        return side
    
    # Initial guess for outer params: Centroid of initial config, angle 0
    c0 = np.mean(centers_init, axis=0)
    x0 = np.concatenate([gap_inits.ravel(), [c0[0], c0[1], 0.0]])
    
    # Run optimization
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
        
        if res.fun < best_score and res.fun < 1e5:
            best_score = res.fun
            best_params = res.x
            
    except Exception:
        pass
    
    # If optimization failed or didn't improve, fallback to initial rigid cluster with optimized outer
    if best_params is None or best_score > 5.0: # Threshold check
         # Try a few random restarts or just use the initial config with a better outer search
         x0 = np.concatenate([gap_inits.ravel(), [0.0, 0.0, 0.0]])
         try:
             res = minimize(objective, x0, method='Nelder-Mead', 
                            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
             if res.fun < best_score and res.fun < 1e5:
                 best_score = res.fun
                 best_params = res.x
         except Exception:
             pass

    if best_params is not None:
        gap_params = best_params[:12].reshape(4, 3)
        outer_params = best_params[12:]
        cx, cy, phi_deg = outer_params
        
        centers = np.vstack([core_centers, gap_params[:, :2]])
        angles = np.concatenate([[0.0]*7, gap_params[:, 2]])
        
        inner_list = []
        for i in range(N):
            inner_list.append([float(centers[i][0]), float(centers[i][1]), float(angles[i])])
            
        return inner_list, (float(cx), float(cy)), float(best_score), float(phi_deg)
    else:
        # Fallback: Return the rigid cluster with a decent outer hexagon
        centers = np.vstack([core_centers, gap_inits[:, :2]])
        angles = np.concatenate([[0.0]*7, gap_inits[:, 2]])
        verts = _compute_vertices(centers, angles)
        side = _min_outer_side_fast(verts)
        c0 = np.mean(centers, axis=0)
        
        inner_list = []
        for i in range(N):
            inner_list.append([float(centers[i][0]), float(centers[i][1]), float(angles[i])])
            
        return inner_list, (float(c0[0]), float(c0[1])), float(side), 0.0

if __name__ == "__main__":
    construct()
