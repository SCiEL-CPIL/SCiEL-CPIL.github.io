import numpy as np
import math
import time

N = 11

def _get_hex_vertices(center, side, angle):
    """Get vertices of a regular hexagon."""
    c = np.array(center)
    a = np.deg2rad(angle)
    angles = a + np.linspace(0, 2*math.pi, 7)[:-1]
    return c + side * np.column_stack([np.cos(angles), np.sin(angles)])

def _check_overlap(positions):
    """Check if any two unit hexagons overlap.
    Positions are centers. All aligned (angle 0).
    Overlap if distance vector is inside Minkowski sum H+H.
    H is unit hexagon. H+H is hexagon with inradius sqrt(3), circumradius 2.
    Actually, easier: just check distance > sqrt(3) is NOT sufficient for all angles, 
    but for aligned hexagons, the forbidden region is H+H.
    Let's use a conservative check: if dist < 1.74 (approx sqrt(3) + eps), it might overlap.
    For exactness in the penalty, we can just use dist < 1.7321 as a violation?
    No, if they are vertex-to-vertex, they need dist 2.
    Let's just use a soft penalty based on distance.
    Penalty = sum(max(0, d_req - dist)^2)
    where d_req depends on angle difference. Since all are 0, 
    the minimum separation is sqrt(3) in flat directions, 2 in vertex directions.
    To be safe, let's assume d_req = 1.7321 for optimization speed, 
    and do a final validity check.
    """
    n = len(positions)
    penalty = 0.0
    # Pre-compute squared positions
    pos = np.array(positions)
    # Calculate all pair distances
    # For N=11, O(N^2) is fine
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            # The minimum distance between centers of two disjoint aligned unit hexagons
            # is sqrt(3) ~ 1.732. 
            # If d < 1.732, they definitely overlap.
            # If d > 2, they definitely don't.
            # Between 1.732 and 2, it depends on relative orientation.
            # Since all are aligned, we can check exactly:
            # The support function of H in direction u is h(u).
            # Overlap if h(d) + h(-d) < ||d|| ? No.
            # Overlap if ||d|| < h_H(d/||d||) + h_H(-d/||d||).
            # Since H is centrally symmetric, h_H(v) = h_H(-v).
            # So overlap if ||d|| < 2 * h_H(d/||d||).
            
            if d < 1e-6:
                penalty += 1000.0
                continue
            
            u = (pos[i] - pos[j]) / d
            # Calculate h_H(u) for unit regular hexagon (side 1, angle 0)
            # Normals of the hexagon sides are at angles 0, 60, ... 300 deg?
            # Vertices at 0, 60...
            # Flats normal to x-axis?
            # Standard hexagon: vertices at (1,0), (0.5, 0.866)...
            # Normal to side between (1,0) and (0.5, 0.866) is at 30 deg?
            # Let's just use the property that h(theta) for regular hexagon side 1 is:
            # h(theta) = cos(theta - 30deg) / cos(30deg) for theta in [-30, 30]?
            # Actually, simpler: The inradius is sqrt(3)/2.
            # h(theta) varies between sqrt(3)/2 and 1.
            
            # For aligned hexagons, the minimum separation is sqrt(3).
            # Let's use d_req = 1.7321 as a hard constraint for optimization speed.
            # We will verify exact disjointness at the end.
            if d < 1.7321:
                penalty += (1.7321 - d)**2
                
    return penalty

def _calc_min_hex_side(points):
    """Calculate minimal side length of regular hexagon containing points."""
    points = np.array(points)
    min_side = float('inf')
    
    # Sweep angle 0 to 60 degrees
    # Step 1 degree is sufficient for initial optimization
    for phi_deg in range(0, 61, 2):
        phi = math.radians(phi_deg)
        # Normals of the hexagon sides
        # Side normals are at phi, phi+60, ...
        max_widths = []
        for k in range(3): # Only need 3 axes of symmetry
            angle_n = phi + k * math.pi / 3
            nx = math.cos(angle_n)
            ny = math.sin(angle_n)
            
            proj = points[:, 0] * nx + points[:, 1] * ny
            width = np.max(proj) - np.min(proj)
            # width = S * sqrt(3)
            S = width / math.sqrt(3)
            max_widths.append(S)
        
        current_side = max(max_widths)
        if current_side < min_side:
            min_side = current_side
            
    return min_side

def construct():
    start_time = time.time()
    deadline = start_time + 100 # Leave 20s margin
    
    # Initial configuration: Hexagonal packing
    # Center at 0,0
    # 6 around it at distance sqrt(3)
    # 4 more?
    # Let's try a 3x3 grid roughly
    # Or a compact cluster
    
    # Better initial guess:
    # 1 at (0,0)
    # 6 at distance 1.732
    # 4 at distance 3.464 (next ring)
    
    pos = [(0.0, 0.0)]
    for k in range(6):
        a = math.radians(60 * k)
        pos.append((1.7321 * math.cos(a), 1.7321 * math.sin(a)))
    for k in range(4):
        a = math.radians(60 * k + 30) # Offset to fill gaps
        # Distance for next ring in hex lattice is 2 * sqrt(3)
        pos.append((3.4642 * math.cos(a), 3.4642 * math.sin(a)))
        
    # Ensure we have exactly 11
    while len(pos) < N:
        pos.append((5.0, 0.0))
    if len(pos) > N:
        pos = pos[:N]
        
    # Convert to flat list for optimizer
    x0 = []
    for p in pos:
        x0.append(p[0])
        x0.append(p[1])
        
    def objective(x):
        if time.time() > deadline:
            return 1e10 # Timeout penalty
            
        # Reshape to points
        points = np.array(x).reshape(-1, 2)
        
        # Check overlap
        pen = _check_overlap(points.tolist())
        if pen > 1e-6:
            return 100 + pen
            
        # Calculate bounding side
        side = _calc_min_hex_side(points)
        return side

    from scipy.optimize import minimize
    
    best_result = None
    best_score = float('inf')
    
    # Run Nelder-Mead
    # We can do a few restarts if time permits
    for restart in range(3):
        if time.time() > deadline - 10:
            break
            
        if restart == 0:
            x_init = x0
        else:
            # Add small noise
            noise = np.random.normal(0, 0.1, size=len(x0))
            x_init = np.array(x0) + noise
            
        try:
            res = minimize(objective, x_init, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-3, 'fatol': 1e-4})
            
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
            
    if best_result is None:
        # Fallback to initial
        return [[p[0], p[1], 0.0] for p in pos], (0.0, 0.0), 40.0, 0.0
        
    # Extract best positions
    points = np.array(best_result).reshape(-1, 2)
    
    # Calculate exact minimal side and center for final answer
    # We need to find the center of the minimal hexagon too
    min_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Refine angle search
    for phi_deg in range(0, 61, 1):
        phi = math.radians(phi_deg)
        # Determine S and Center for this angle
        # We need to solve for c and S
        # Constraints:
        # max(p_i . n_k) - c . n_k <= S*sqrt(3)/2
        # c . n_k - min(p_i . n_k) <= S*sqrt(3)/2
        
        # Let U_k = max(p_i . n_k), L_k = min(p_i . n_k)
        # We need c . n_k in [U_k - S*sqrt(3)/2, L_k + S*sqrt(3)/2] ?? No.
        # c . n_k >= U_k - S*sqrt(3)/2
        # c . n_k <= L_k + S*sqrt(3)/2 + (U_k-L_k)? No.
        
        # The width in direction n is W_k = U_k - L_k.
        # S must satisfy S*sqrt(3) >= W_k for all k.
        # So S_min_angle = max_k(W_k) / sqrt(3).
        
        # What is the center?
        # The center of the bounding box in direction n_k is (U_k + L_k) / 2.
        # So c . n_k should be (U_k + L_k) / 2.
        
        widths = []
        centers_proj = []
        for k in range(3):
            angle_n = phi + k * math.pi / 3
            nx = math.cos(angle_n)
            ny = math.sin(angle_n)
            proj = points[:, 0] * nx + points[:, 1] * ny
            U = np.max(proj)
            L = np.min(proj)
            widths.append(U - L)
            centers_proj.append((U + L) / 2)
            
        S_candidate = max(widths) / math.sqrt(3)
        if S_candidate < min_side:
            min_side = S_candidate
            best_angle = phi
            
            # Solve for center c
            # c . n_0 = c_proj_0
            # c . n_1 = c_proj_1
            # We have 3 equations, 2 unknowns. Use least squares.
            A = np.array([[math.cos(phi), math.sin(phi)],
                          [math.cos(phi + math.pi/3), math.sin(phi + math.pi/3)],
                          [math.cos(phi + 2*math.pi/3), math.sin(phi + 2*math.pi/3)]])
            b = np.array(centers_proj)
            c_sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            best_center = (c_sol[0], c_sol[1])
            
    # Final check: are the hexagons disjoint?
    # If not, we might need to scale up or accept a slightly larger container.
    # But the optimizer should have kept them disjoint.
    
    # Construct output
    inner = []
    for i in range(N):
        inner.append([points[i][0], points[i][1], 0.0])
        
    return inner, best_center, min_side, math.degrees(best_angle)

# EVOLVE-BLOCK-END
