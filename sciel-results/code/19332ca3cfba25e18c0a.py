import math
import numpy as np
import time

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Initial configuration: A compact cluster of 11 hexagons
    # 1 center, 6 ring 1, 4 ring 2 (to make 11)
    # Axial coordinates (q, r)
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    
    # Convert to Cartesian
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append([x, y])
        return np.array(pts)

    # We will simulate multiple restarts with different initial perturbations and cooling schedules
    best_S = float('inf')
    best_pts = None
    best_beta = 0.0
    best_center = None

    # Pre-compute normals for the inner hexagons (for overlap check)
    # Inner hexagons are rotated by 30 deg relative to outer? No, standard orientation.
    # Let's assume inner hexagons have edges horizontal/vertical in their local frame?
    # Standard regular hexagon with flat top has vertices at 0, 60, 120...
    # Apothem is along 30, 90, 150...
    # Let's use the standard "pointy top" or "flat top"?
    # The task says "regular hexagons". The orientation of inner hexagons is free (angle_degrees).
    # However, for packing, it is optimal to align them.
    # Let's fix the inner hexagon orientation to 0 degrees (flat top) and optimize the OUTER hexagon rotation.
    # Wait, the interface allows individual angles for inner hexagons.
    # But for a rigid cluster, they are all the same.
    # Let's fix inner angle to 0 for simplicity in the simulation, and optimize outer beta.
    
    inner_normals = []
    for k in range(6):
        ang = math.pi / 6.0 + k * math.pi / 3.0
        inner_normals.append((math.cos(ang), math.sin(ang)))
    inner_normals = np.array(inner_normals)

    def check_overlap(pts):
        """Check if any two inner hexagons overlap.
        Two unit hexagons (flat top) overlap if the distance between centers is less than sqrt(3)
        in any of the 6 normal directions? No, the Minkowski sum is a hexagon of side 2.
        The condition for non-overlap is that the vector d = p_i - p_j must be outside the Minkowski hexagon.
        The Minkowski hexagon has apothem sqrt(3).
        So, for all 6 normals n, |d . n| >= sqrt(3).
        """
        n_pts = len(pts)
        min_dist_sq = 3.0 # sqrt(3)^2
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                d = pts[i] - pts[j]
                # Check projection on all 6 normals
                for nx, ny in inner_normals:
                    proj = abs(d[0]*nx + d[1]*ny)
                    if proj < math.sqrt(3) - 1e-9:
                        return False
        return True

    def calc_S_and_center(pts, beta):
        """Calculate the minimal enclosing hexagon side length S and center for a given set of points and outer rotation beta.
        The outer hexagon is centered at (cx, cy) with rotation beta.
        The distance from center to each face is D.
        S = (D + APOTHEM) / APOTHEM.
        We want to minimize D.
        The constraints are: for each point p_i and each of the 6 normals n_k of the outer hex,
        (p_i - c) . n_k <= D.
        This is a linear program: min D subject to A c <= b - D?
        Actually:
        Let n_k be the outward normals of the outer hexagon.
        (p_i - c) . n_k <= D  =>  -p_i . n_k + c . n_k <= D
        Also -D - p_i . n_k + c . n_k <= 0? No.
        
        Let's reformulate:
        We want to find c and D such that max_i max_k ( (p_i - c) . n_k ) is minimized.
        Let f(c, D) = 0 if for all i, k: (p_i - c) . n_k <= D.
        
        This is equivalent to:
        min D
        s.t. c . n_k - D <= p_i . n_k  for all i, k
             -c . n_k - D <= -p_i . n_k for all i, k  (symmetric)
        
        Wait, the outer hexagon contains the points.
        The distance from center c to the face with normal n_k is D.
        So the face is at c + D n_k.
        The point p_i must satisfy (p_i - c) . n_k <= D.
        Also (p_i - c) . (-n_k) <= D  =>  -(p_i - c) . n_k <= D.
        So | (p_i - c) . n_k | <= D.
        
        We want to minimize D.
        Variables: cx, cy, D.
        Constraints:
        For each i, k:
        (p_ix - cx) n_kx + (p_iy - cy) n_ky <= D
        -(p_ix - cx) n_kx - (p_iy - cy) n_ky <= D
        
        This is a linear program.
        """
        # Since we are in a tight loop, maybe we can approximate or use a faster method?
        # For 11 points and 6 directions, 11*6*2 = 132 constraints.
        # Variables: cx, cy, D.
        # This is very small. We can use scipy.optimize.linprog.
        
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        # Precompute p_i . n_k
        # p_i is 11x2, n_k is 6x2
        # projections is 11x6
        
        for k in range(6):
            nx, ny = math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0), math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)
            for i in range(len(pts)):
                px, py = pts[i]
                # (px - cx) nx + (py - cy) ny <= D
                # -nx cx - ny cy - D <= - (px nx + py ny)
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                
                # -(px - cx) nx - (py - cy) ny <= D
                # nx cx + ny cy - D <= px nx + py ny
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            from scipy.optimize import linprog
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy)
            else:
                return float('inf'), None
        except Exception:
            return float('inf'), None

    # Force-directed simulation
    def simulate(initial_pts, max_iter=2000, learning_rate=0.01, cooling=0.99):
        pts = initial_pts.copy()
        current_lr = learning_rate
        
        for t in range(max_iter):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
                
            forces = np.zeros_like(pts)
            n_pts = len(pts)
            
            # Repulsive forces (overlap prevention)
            for i in range(n_pts):
                for j in range(i+1, n_pts):
                    d = pts[i] - pts[j]
                    # Check overlap
                    min_proj = float('inf')
                    for nx, ny in inner_normals:
                        proj = abs(d[0]*nx + d[1]*ny)
                        if proj < min_proj:
                            min_proj = proj
                    
                    # The "distance" in the hex metric is min_proj.
                    # We want min_proj >= sqrt(3).
                    # If min_proj < sqrt(3), push apart.
                    # The direction of push is along the normal that gives the min projection.
                    # We need to find which normal gave the min projection.
                    
                    min_proj_val = float('inf')
                    best_normal = None
                    for nx, ny in inner_normals:
                        proj = abs(d[0]*nx + d[1]*ny)
                        if proj < min_proj_val:
                            min_proj_val = proj
                            best_normal = (nx, ny)
                    
                    if min_proj_val < math.sqrt(3):
                        # Overlap
                        # Force is proportional to violation
                        violation = math.sqrt(3) - min_proj_val
                        # Direction: d is from j to i. We want to increase d . n.
                        # If d . n is positive, push i in direction n, j in -n.
                        # If d . n is negative, push i in -n, j in n.
                        # Actually, the constraint is |d . n| >= sqrt(3).
                        # Let's assume the sign is consistent with the normal chosen.
                        # The normal best_normal was chosen based on absolute value.
                        # Let's check the actual signed projection.
                        signed_proj = d[0]*best_normal[0] + d[1]*best_normal[1]
                        if signed_proj < 0:
                            dir_vec = np.array([-best_normal[0], -best_normal[1]])
                        else:
                            dir_vec = np.array([best_normal[0], best_normal[1]])
                        
                        force_mag = violation * 10.0
                        forces[i] += force_mag * dir_vec
                        forces[j] -= force_mag * dir_vec
            
            # Attractive forces (pull to center to minimize size)
            center = np.mean(pts, axis=0)
            for i in range(n_pts):
                d_center = pts[i] - center
                # Pull towards center
                forces[i] -= d_center * 0.1
            
            # Update positions
            pts += forces * current_lr
            current_lr *= cooling
            
            # Optional: check overlap and S periodically to save time
            if t % 10 == 0:
                if check_overlap(pts):
                    S, c = calc_S_and_center(pts, 0.0) # Try beta=0 first? Or optimize beta?
                    # Optimizing beta for every step is too expensive.
                    # We will optimize beta at the end for the best candidates.
                    if S < best_S:
                        best_S = S
                        best_pts = pts.copy()
                        best_center = c
                        
        return best_S, best_pts, best_center

    # Run multiple simulations with different initial conditions
    base_pts = axial_to_cartesian(axial_coords)
    
    # Add some random perturbations
    rng = np.random.default_rng(42)
    
    for restart in range(20):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Perturb initial positions
        perturbation = rng.normal(0, 0.5, size=base_pts.shape)
        init_pts = base_pts + perturbation
        
        # Run simulation
        S, pts, center = simulate(init_pts, max_iter=1000)
        
        if pts is not None:
            # Refine the best beta for this configuration
            best_beta_local = 0.0
            best_S_local = S
            for i in range(12):
                beta = i * (math.pi / 6.0) / 11.0
                S_b, c_b = calc_S_and_center(pts, beta)
                if S_b < best_S_local:
                    best_S_local = S_b
                    best_beta_local = beta
                    best_center = c_b
            
            if best_S_local < best_S:
                best_S = best_S_local
                best_pts = pts
                best_beta = best_beta_local
                best_center = best_center

    # Final polish: Local search on the best configuration found
    if best_pts is not None:
        # Try to relax the best configuration further
        # Use a finer grid for beta
        best_beta_final = best_beta
        best_S_final = best_S
        for i in range(100):
            beta = best_beta - 0.01 + i * 0.0002
            S_b, c_b = calc_S_and_center(best_pts, beta)
            if S_b < best_S_final:
                best_S_final = S_b
                best_beta_final = beta
                best_center = c_b
        
        best_S = best_S_final
        best_beta = best_beta_final

    if best_pts is None:
        # Fallback to initial lattice
        best_pts = base_pts
        best_S, best_center = calc_S_and_center(best_pts, 0.0)
        best_beta = 0.0

    cx, cy = best_center
    angle_deg = math.degrees(best_beta)
    
    # Ensure no overlap in final output
    # If there is overlap, we might need to scale up or return a valid but worse solution
    # For now, trust the simulation. If check_overlap fails, we can try to fix it.
    if not check_overlap(best_pts):
        # Simple fix: scale points from center slightly outward
        scale = 1.0
        while not check_overlap(best_pts) and scale < 1.1:
            scale += 0.001
            best_pts = (best_pts - np.array([cx, cy])) * scale + np.array([cx, cy])
        
        # Recalculate S
        best_S, best_center = calc_S_and_center(best_pts, best_beta)
        cx, cy = best_center

    inner = []
    for x, y in best_pts:
        inner.append([float(x), float(y), 0.0]) # Inner angle 0
        
    center = [float(cx), float(cy)]
    side = float(best_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation

# Dummy call to ensure imports work and time starts
# construct() 
