import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0 (pointy-top, vertex at 0 deg)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0) 
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals) # (6, 2)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def check_disjoint_fast(poly1, poly2):
        """
        Fast check for disjointness using SAT.
        Returns True if disjoint (or touching), False if overlapping.
        """
        def project_polygon(poly, axis):
            proj = poly @ axis
            return np.min(proj), np.max(proj)
            
        axes = []
        for i in range(6):
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            axes.append(normal1)
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            axes.append(normal2)
            
        for axis in axes:
            norm = np.linalg.norm(axis)
            if norm < 1e-9: continue
            axis = axis / norm
            min1, max1 = project_polygon(poly1, axis)
            min2, max2 = project_polygon(poly2, axis)
            if max1 < min2 or max2 < min1:
                return True
                
        return False

    def refine_solution(params, time_limit=10.0):
        """
        Refine a discrete solution using SLSQP with a penalty method for overlaps.
        """
        start = time.time()
        
        def objective(params):
            if time.time() - start > time_limit:
                return 1e6
            positions = params[:2*N].reshape(N, 2)
            angles = params[2*N:]
            
            # Calculate side length
            side_len = calc_min_outer_side(positions, angles, 0.0)
            
            # Calculate penalty for overlaps
            polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    if not check_disjoint_fast(polys[i], polys[j]):
                        penalty += 10.0
                        
            return side_len + penalty

        try:
            res = minimize(
                objective, 
                params, 
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-8, 'eps': 1e-6, 'disp': False}
            )
            
            # Verify validity
            new_positions = res.x[:2*N].reshape(N, 2)
            new_angles = res.x[2*N:]
            polys = [get_hex_verts(new_positions[i][0], new_positions[i][1], 1.0, new_angles[i]) for i in range(N)]
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if not check_disjoint_fast(polys[i], polys[j]):
                        valid = False
                        break
                    if not valid: break
            
            if valid:
                return res.x, calc_min_outer_side(new_positions, new_angles, 0.0)
            else:
                return params, calc_min_outer_side(positions, angles, 0.0)
        except Exception:
            return params, calc_min_outer_side(positions, angles, 0.0)

    # --- Discrete Search Strategy ---
    
    # Define a grid of candidate centers and angles
    # The optimal packing likely has hexagons aligned or rotated by 30/60 degrees.
    # Let's try angles: 0, 30, 60, 90, 120, 150 (or just 0 and 30 for speed first, then expand)
    # To keep it fast, let's start with a moderate grid.
    
    # Estimate max radius needed. For 12 hexagons, area ~ 12 * 2.598 = 31.18.
    # Outer hex area = 2.598 * S^2. S^2 ~ 12. S ~ 3.46.
    # So centers should be within radius ~ 3.
    
    grid_step = 0.25
    max_radius = 3.5
    angles_to_try = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]
    
    # Generate candidate positions
    candidates = []
    for x in np.arange(-max_radius, max_radius + grid_step, grid_step):
        for y in np.arange(-max_radius, max_radius + grid_step, grid_step):
            if x*x + y*y <= max_radius*max_radius:
                for ang in angles_to_try:
                    candidates.append((x, y, ang))
    
    # Sort candidates by distance from origin (prefer central positions first to keep bounding box small?)
    # Actually, for minimization of max extent, we want to pack tightly.
    # A good heuristic is to place the first hexagon at (0,0) and then search for neighbors.
    
    # Since N=12 is small, we can do a backtracking search.
    # However, the state space is large. 
    # Alternative: Use a random restart approach with local search on the grid.
    
    best_score = float('inf')
    best_params = None
    
    # Initial guess: Honeycomb lattice
    sqrt3 = math.sqrt(3)
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * 1.5
            y = i * sqrt3 / 2.0 + j * sqrt3
            lattice_points.append((x, y, 0.0))
            
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_positions = np.array([(p[0], p[1]) for p in lattice_points[:12]])
    init_angles = np.array([p[2] for p in lattice_points[:12]])
    
    x0 = np.concatenate([init_positions.flatten(), init_angles])
    init_score = calc_min_outer_side(init_positions, init_angles, 0.0)
    best_score = init_score
    best_params = x0.copy()
    
    # Refine the initial guess
    if time.time() < deadline - 20:
        refined_params, refined_score = refine_solution(x0, time_limit=10.0)
        if refined_score < best_score:
            best_score = refined_score
            best_params = refined_params

    # Randomized Discrete Local Search
    # We will pick random subsets of positions from the grid and try to optimize them.
    # Or better: Start from a valid configuration (like the lattice) and perform swaps/moves.
    
    # Let's do a simulated annealing style search on the discrete grid + continuous refinement.
    # To keep it simple and fast, we will do multiple random restarts of the continuous optimizer
    # initialized from slightly perturbed discrete lattice configurations.
    
    np.random.seed(42)
    
    # Pre-compute polys for current best to check validity quickly?
    # No, refine_solution does the check.
    
    # Strategy:
    # 1. Generate a set of diverse discrete starting points.
    # 2. Run a short SLSQP refinement on each.
    # 3. Keep the best.
    
    # Generating diverse starts:
    # - Perturb the lattice.
    # - Randomly select 12 points from the grid that are far enough apart.
    
    attempts = 0
    max_attempts = 50 # Limit attempts to ensure time budget
    
    while time.time() < deadline - 15 and attempts < max_attempts:
        attempts += 1
        
        # Create a perturbed start
        if attempts % 2 == 0:
            # Perturb current best
            noise_pos = np.random.normal(0, 0.3, 2*N)
            noise_ang = np.random.normal(0, 15, N) * math.pi / 180
            x_perturbed = best_params.copy()
            x_perturbed[:2*N] += noise_pos
            x_perturbed[2*N:] += noise_ang
        else:
            # Random grid-based start
            # Pick 12 random candidates that are non-overlapping (greedy)
            positions = []
            angles = []
            polys = []
            
            # Shuffle candidates
            cand_indices = list(range(len(candidates)))
            np.random.shuffle(cand_indices)
            
            for idx in cand_indices:
                if len(positions) >= N:
                    break
                x, y, ang = candidates[idx]
                p = get_hex_verts(x, y, 1.0, ang)
                
                # Check against already placed
                valid = True
                for existing_poly in polys:
                    if not check_disjoint_fast(p, existing_poly):
                        valid = False
                        break
                
                if valid:
                    positions.append((x, y))
                    angles.append(ang)
                    polys.append(p)
            
            if len(positions) == N:
                pos_arr = np.array(positions)
                ang_arr = np.array(angles)
                x_perturbed = np.concatenate([pos_arr.flatten(), ang_arr])
            else:
                # If we couldn't find 12 disjoint points on the grid, skip
                continue
        
        # Refine
        refined_params, refined_score = refine_solution(x_perturbed, time_limit=5.0)
        
        if refined_score < best_score:
            best_score = refined_score
            best_params = refined_params

    final_positions = best_params[:2*N].reshape(N, 2)
    final_angles = best_params[2*N:]
    final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0
