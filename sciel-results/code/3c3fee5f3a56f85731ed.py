import math
import numpy as np
import time

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def construct():
    # Deadline guard
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Precompute unit vectors for hexagon normals (relative to angle 0)
    # A regular hexagon with side 1 has apothem sqrt(3)/2.
    # The condition for a point (x,y) to be inside a hexagon centered at (cx,cy) rotated by theta
    # is that its projection onto the 6 normal directions is <= apothem.
    # For non-overlap of two hexagons i and j, the distance between centers projected onto
    # any of the 6 normal directions must be >= 2 * apothem = sqrt(3).
    # Wait, if they are parallel, the Minkowski sum is a hexagon of side 2 (apothem sqrt(3)).
    # So min projection of (p_i - p_j) onto normals must be >= sqrt(3).
    
    normals = np.array([
        [math.cos(math.pi/6 + k*math.pi/3), math.sin(math.pi/6 + k*math.pi/3)]
        for k in range(6)
    ])

    def check_overlap(pts):
        """Returns penalty for overlaps. 0 if no overlap."""
        penalty = 0.0
        n_pts = len(pts)
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                d = pts[i] - pts[j]
                projs = np.abs(d @ normals.T)
                min_proj = np.min(projs)
                if min_proj < math.sqrt(3):
                    penalty += (math.sqrt(3) - min_proj) ** 2
        return penalty

    def calc_S(pts, cx, cy, beta):
        """
        Calculate the minimum side length S of the outer hexagon.
        Outer hexagon center (cx, cy), rotation beta.
        Inner hexagons are assumed to be rotated by the same beta? 
        The task says "each inner hexagon may translate and rotate".
        However, for simplicity and to maintain symmetry, we assume all inner hexagons
        are parallel to the outer hexagon (relative angle 0). 
        If we allow individual rotations, the problem becomes much harder.
        Given the seed program uses a single `angle_deg` for all inner hexagons,
        we stick to that assumption: all inner hexagons share the same orientation.
        
        The outer hexagon must contain all inner hexagons.
        An inner hexagon at (px, py) with orientation beta is contained in the outer hexagon
        centered at (cx, cy) with orientation beta and side S if:
        For all 6 normals n_k of the outer hexagon:
        (px - cx) . n_k + APOTHEM <= S * APOTHEM  (projection of center + apothem of inner <= apothem of outer)
        (cx - px) . n_k + APOTHEM <= S * APOTHEM
        
        Let d_k = (px - cx) . n_k.
        We need S * APOTHEM >= max_k (|d_k| + APOTHEM).
        So S = (max_k (|d_k| + APOTHEM)) / APOTHEM.
        """
        pts_centered = pts - np.array([cx, cy])
        # Rotate points by -beta to align with standard normals?
        # Or rotate normals by beta?
        # Let's rotate the points so that the outer hexagon is aligned with axes (beta=0).
        # Rotation matrix for -beta
        c = math.cos(-beta)
        s = math.sin(-beta)
        R = np.array([[c, -s], [s, c]])
        pts_rot = pts_centered @ R.T
        
        projs = np.abs(pts_rot @ normals.T)
        max_proj = np.max(projs)
        S = (max_proj + APOTHEM) / APOTHEM
        return S

    def total_cost(pts, cx, cy, beta):
        S = calc_S(pts, cx, cy, beta)
        overlap_pen = check_overlap(pts)
        # Weight overlap penalty heavily
        return S + 100.0 * overlap_pen

    # Initialize positions
    # Start with a hexagonal lattice arrangement
    # Center at (0,0)
    pts = np.zeros((N, 2))
    # Hex lattice vectors
    v1 = np.array([math.sqrt(3), 0])
    v2 = np.array([math.sqrt(3)/2, 1.5])
    
    # Generate 11 points in a hex-like shape
    # 1 in center, 6 in first ring, 4 in second ring (partial)
    lattice_pts = []
    lattice_pts.append((0,0))
    # First ring
    for i in range(6):
        ang = i * math.pi / 3
        x = math.sqrt(3) * math.cos(ang)
        y = 1.5 * math.sin(ang) # Wait, hex lattice spacing is sqrt(3) horizontally, 1.5 vertically?
        # Standard hex grid:
        # (0,0)
        # (sqrt(3), 0), (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), (-sqrt(3), 0), (-sqrt(3)/2, -1.5), (sqrt(3)/2, -1.5)
        # Let's use axial to cartesian
        def axial_to_cartesian(q, r):
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            return np.array([x, y])
        
    # Use axial coordinates for initialization
    axial_coords = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2)
    ]
    for q, r in axial_coords:
        pts[len(pts) > 0 and len(lattice_pts) or len(pts)] = axial_to_cartesian(q, r) # This line is messy, let's just fill pts
        
    pts = np.array([axial_to_cartesian(q, r) for q, r in axial_coords])
    
    # Simulated Annealing
    best_S = float('inf')
    best_pts = pts.copy()
    best_cx = 0.0
    best_cy = 0.0
    best_beta = 0.0
    
    # Initial center and beta
    cx, cy = np.mean(pts, axis=0)
    beta = 0.0
    
    current_pts = pts.copy()
    current_cx, current_cy = cx, cy
    current_beta = beta
    
    T = 0.1 # Initial temperature
    T_min = 1e-5
    cooling = 0.995
    
    # Time limit for SA
    sa_end_time = START_TIME + TIME_LIMIT - SAFETY_MARGIN
    
    while time.time() < sa_end_time and T > T_min:
        # Generate move
        # 1. Move a random point
        i = np.random.randint(N)
        dx = np.random.normal(0, 0.1)
        dy = np.random.normal(0, 0.1)
        old_x, old_y = current_pts[i]
        current_pts[i] = [old_x + dx, old_y + dy]
        
        # 2. Move center (less frequent)
        if np.random.random() < 0.1:
            dcx = np.random.normal(0, 0.1)
            dcy = np.random.normal(0, 0.1)
            old_cx, old_cy = current_cx, current_cy
            current_cx += dcx
            current_cy += dcy
        else:
            old_cx, old_cy = current_cx, current_cy
            
        # 3. Rotate (less frequent)
        if np.random.random() < 0.1:
            dbeta = np.random.normal(0, 0.05)
            old_beta = current_beta
            current_beta += dbeta
            # Keep beta in [0, pi/6)
            current_beta = current_beta % (math.pi/6)
        else:
            old_beta = current_beta
            
        # Calculate costs
        old_cost = total_cost(current_pts - np.array([dx, dy]) if i is not None else current_pts, # This is getting complicated to revert
                        old_cx, old_cy, old_beta)
        
        # Better: store old state
        # Re-calculate old cost from saved old state? 
        # Let's just save the whole state before mutation
        
        # Reset to old state for comparison
        current_pts[i] = [old_x, old_y]
        current_cx, current_cy = old_cx, old_cy
        current_beta = old_beta
        
        old_cost = total_cost(current_pts, current_cx, current_cy, current_beta)
        
        # Apply mutation again to get new state
        current_pts[i] = [old_x + dx, old_y + dy]
        if np.random.random() < 0.1: # This random check breaks reproducibility with the one above!
             # We need deterministic mutation logic or save the decision
             pass
        
        # Let's restructure the loop for clarity
        
        # --- Structured SA Step ---
        # 1. Decide mutations
        mut_pts = np.random.random() < 0.8
        mut_c = np.random.random() < 0.2
        mut_b = np.random.random() < 0.2
        
        old_pts = current_pts.copy()
        old_cx, old_cy = current_cx, current_cy
        old_beta = current_beta
        
        if mut_pts:
            i = np.random.randint(N)
            dx = np.random.normal(0, 0.1 * T) # Scale step with T? No, fixed step is better for SA usually, or anneal step size
            dy = np.random.normal(0, 0.1 * T)
            current_pts[i] = [old_pts[i][0] + dx, old_pts[i][1] + dy]
            
        if mut_c:
            dcx = np.random.normal(0, 0.1 * T)
            dcy = np.random.normal(0, 0.1 * T)
            current_cx += dcx
            current_cy += dcy
            
        if mut_b:
            dbeta = np.random.normal(0, 0.05 * T)
            current_beta = (current_beta + dbeta) % (math.pi/6)
            
        new_cost = total_cost(current_pts, current_cx, current_cy, current_beta)
        old_cost = total_cost(old_pts, old_cx, old_cy, old_beta)
        
        delta = new_cost - old_cost
        
        if delta < 0 or np.random.random() < math.exp(-delta / T):
            # Accept
            pass # State is already updated
        else:
            # Reject
            current_pts = old_pts
            current_cx, current_cy = old_cx, old_cy
            current_beta = old_beta
            
        # Track best
        if new_cost < best_S:
            best_S = new_cost
            best_pts = current_pts.copy()
            best_cx = current_cx
            best_cy = current_cy
            best_beta = current_beta
            
        T *= cooling
        
    # Final Polish
    # Use Nelder-Mead on the best configuration found
    def smooth_obj(params):
        n_pts = N
        pts = params[:2*n_pts].reshape(n_pts, 2)
        cx = params[2*n_pts]
        cy = params[2*n_pts+1]
        beta = params[2*n_pts+2]
        beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
        
        S = calc_S(pts, cx, cy, beta)
        overlap_pen = check_overlap(pts)
        return S + 100.0 * overlap_pen

    init_params = np.concatenate([best_pts.ravel(), [best_cx, best_cy, best_beta]])
    
    try:
        from scipy.optimize import minimize
        res = minimize(smooth_obj, init_params, method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res.fun < best_S:
            best_S = res.fun
            n_pts = N
            best_pts = res.x[:2*n_pts].reshape(n_pts, 2)
            best_cx = res.x[2*n_pts]
            best_cy = res.x[2*n_pts+1]
            best_beta = ((res.x[2*n_pts+2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
    except Exception:
        pass

    # Verify and return
    # Ensure no overlap in final result
    if check_overlap(best_pts) > 1e-9:
        # Fallback to lattice
        axial_coords = [
            (0,0),
            (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
            (2,0), (1,1), (0,2), (-1,2)
        ]
        best_pts = np.array([axial_to_cartesian(q, r) for q, r in axial_coords])
        best_cx, best_cy = np.mean(best_pts, axis=0)
        best_beta = 0.0
        best_S = calc_S(best_pts, best_cx, best_cy, best_beta)

    angle_deg = math.degrees(best_beta)
    
    inner = []
    for x, y in best_pts:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(best_cx), float(best_cy)]
    side = float(best_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation

# Helper defined inside construct for scoping, but let's move it out or duplicate
def axial_to_cartesian(q, r):
    x = math.sqrt(3) * (q + r / 2.0)
    y = 1.5 * r
    return np.array([x, y])
