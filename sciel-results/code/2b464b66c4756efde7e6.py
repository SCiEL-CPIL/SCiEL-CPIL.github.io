import numpy as np
from scipy.optimize import differential_evolution, minimize, basinhopping
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    N = 12

    # --- Geometry Helpers ---

    def hex_vertices(cx, cy, side, angle_rad):
        """Returns 6x2 array of vertices for a regular hexagon."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_normals(verts):
        """Returns 6x2 array of outward normals for a CCW polygon."""
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / (norms[:, np.newaxis] + 1e-15)
        return normals

    def sat_overlap(v1, v2):
        """
        Returns min overlap depth (negative if disjoint) using SAT.
        """
        normals1 = get_normals(v1)
        normals2 = get_normals(v2)
        normals = np.vstack([normals1, normals2])
        
        p1 = v1 @ normals.T
        p2 = v2 @ normals.T
        
        min1 = p1.min(axis=0)
        max1 = p1.max(axis=0)
        min2 = p2.min(axis=0)
        max2 = p2.max(axis=0)
        
        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        min_overlap = np.min(overlaps)
        
        return min_overlap

    def compute_min_side_fixed_orientation(hexes, angle_deg=0.0, center=(0.0, 0.0)):
        """
        Calculate the minimum side length of a regular hexagon 
        (centered at given center, fixed orientation) that contains all given hexagons.
        """
        angle_rad = np.radians(angle_deg)
        # Normals for a regular hexagon with flat top at angle 0?
        # Vertices at 0, 60, 120...
        # Edges are perpendicular to normals.
        # If vertices are at 0, 60, 120, 180, 240, 300.
        # Edge 0-1 is from 0 to 60. Normal is 30.
        # So normals are at 30, 90, 150, 210, 270, 330.
        normal_angles = angle_rad + np.radians([30, 90, 150, 210, 270, 330])
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj = 0.0
        for h in hexes:
            proj = (h - center) @ normals.T
            max_proj = max(max_proj, np.max(proj))
            
        s = max_proj * 2.0 / np.sqrt(3.0)
        return s

    def optimize_outer_rotation(hexes, center=(0.0, 0.0)):
        """
        Optimize the outer hexagon rotation angle to minimize side length.
        """
        best_angle = 0.0
        best_s = np.inf
        
        # Golden section search
        gr = (np.sqrt(5) - 1) / 2
        a, b = 0.0, 60.0
        
        x1 = b - gr * (b - a)
        x2 = a + gr * (b - a)
        f1 = compute_min_side_fixed_orientation(hexes, x1, center)
        f2 = compute_min_side_fixed_orientation(hexes, x2, center)
        
        for _ in range(30): # Reduced iterations for speed
            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = b - gr * (b - a)
                f1 = compute_min_side_fixed_orientation(hexes, x1, center)
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + gr * (b - a)
                f2 = compute_min_side_fixed_orientation(hexes, x2, center)
                
        candidates = [0.0, 30.0, (a + b) / 2, x1, x2]
        for ang in candidates:
            s = compute_min_side_fixed_orientation(hexes, ang, center)
            if s < best_s:
                best_s = s
                best_angle = ang
        
        return best_s, best_angle

    def get_hexes_from_full_params(params):
        """
        params: 25 values [x1..x12, y1..y12, a1..a12]
        Returns list of 12 hexagons (6x2 arrays).
        """
        hexes = []
        for i in range(N):
            x = params[i]
            y = params[N + i]
            a = params[2*N + i]
            hexes.append(hex_vertices(x, y, 1.0, a))
        return hexes

    def compute_smooth_objective(params, beta=50.0, optimize_rotation=False):
        """
        Smooth objective for global search.
        """
        if time.time() > deadline:
            return 1e10
            
        hexes = get_hexes_from_full_params(params)
        
        # For global search, use fixed rotation (0) for speed
        # For local search, optimize rotation
        if optimize_rotation:
            s_min, _ = optimize_outer_rotation(hexes)
        else:
            s_min = compute_min_side_fixed_orientation(hexes, 0.0)
        
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                # Quick bounding box check
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    penalty += np.log(1.0 + np.exp(beta * overlap)) / beta
                    
        return s_min + penalty

    def compute_objective(params, beta=50.0):
        """
        Objective for local refinement: min_side (optimized rotation) + penalty.
        """
        if time.time() > deadline:
            return 1e10
            
        hexes = get_hexes_from_full_params(params)
        
        s_min, _ = optimize_outer_rotation(hexes)
        
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    penalty += np.log(1.0 + np.exp(beta * overlap)) / beta
                    
        return s_min + penalty

    # --- Initial Guesses ---
    
    def generate_symmetric_init():
        """
        Generate initial params for the symmetric 2-orbit model.
        """
        # Best symmetric params from previous runs or heuristic
        # r1=1.5, th1=0, al1=0, r2=2.5, th2=0, al2=0
        r1, th1, al1 = 1.5, 0.0, 0.0
        r2, th2, al2 = 2.5, 0.0, 0.0
        
        params = np.zeros(3*N)
        for k in range(6):
            angle = np.radians(60 * k)
            
            x1 = r1 * np.cos(th1 + angle)
            y1 = r1 * np.sin(th1 + angle)
            a1 = al1 + angle
            
            x2 = r2 * np.cos(th2 + angle)
            y2 = r2 * np.sin(th2 + angle)
            a2 = al2 + angle
            
            # Hex 2k
            params[2*k] = x1
            params[N + 2*k] = y1
            params[2*N + 2*k] = a1
            
            # Hex 2k+1
            params[2*k+1] = x2
            params[N + 2*k+1] = y2
            params[2*N + 2*k+1] = a2
            
        return params

    def generate_random_init(seed=42):
        rng = np.random.default_rng(seed)
        params = np.zeros(3*N)
        for i in range(N):
            params[i] = rng.uniform(-3, 3)
            params[N+i] = rng.uniform(-3, 3)
            params[2*N+i] = rng.uniform(0, np.pi/3)
        return params

    # --- Optimization ---
    
    bounds = [(-4.0, 4.0)] * N + [(-4.0, 4.0)] * N + [(0.0, np.pi/3)] * N
    
    best_params = None
    best_score = np.inf

    # Phase 1: Differential Evolution
    # Use fixed rotation in objective for speed
    try:
        res_de = differential_evolution(
            lambda p: compute_smooth_objective(p, beta=30.0, optimize_rotation=False),
            bounds,
            maxiter=150,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=False
        )
        if res_de.fun < best_score:
            best_score = res_de.fun
            best_params = res_de.x.copy()
    except Exception as e:
        print(f"DE failed: {e}")

    # Phase 2: Local Refinement with L-BFGS-B
    # Now use optimized rotation in objective
    if best_params is not None:
        try:
            res_local = minimize(
                lambda p: compute_objective(p, beta=50.0),
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res_local.fun < best_score:
                best_score = res_local.fun
                best_params = res_local.x.copy()
        except Exception as e:
            print(f"Local refine failed: {e}")

    # Phase 3: Simulated Annealing Polish
    if best_params is not None and time.time() < deadline - 20:
        def sa_polish(params):
            cur_params = params.copy()
            cur_score = compute_objective(cur_params, beta=50.0)
            best_params_sa = cur_params.copy()
            best_score_sa = cur_score
            
            T = 0.05
            rng = np.random.default_rng(999)
            steps = 0
            max_steps = 2000
            
            while steps < max_steps and time.time() < deadline - 10:
                steps += 1
                # Perturb one or two parameters
                idx1 = rng.integers(0, 3*N)
                idx2 = rng.integers(0, 3*N)
                
                new_params = cur_params.copy()
                new_params[idx1] += rng.normal(0, T)
                new_params[idx2] += rng.normal(0, T)
                
                # Enforce bounds
                for i in range(3*N):
                    new_params[i] = np.clip(new_params[i], bounds[i][0], bounds[i][1])
                
                new_score = compute_objective(new_params, beta=50.0)
                
                delta = new_score - cur_score
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    cur_params = new_params
                    cur_score = new_score
                    
                    if cur_score < best_score_sa:
                        best_score_sa = cur_score
                        best_params_sa = cur_params.copy()
                        
                T = max(0.001, T * 0.999)
                
            return best_params_sa, best_score_sa

        try:
            best_params, best_score = sa_polish(best_params)
        except Exception as e:
            print(f"SA failed: {e}")

    # Phase 4: Final High-Precision Polish
    if best_params is not None and time.time() < deadline - 5:
        try:
            res_polish = minimize(
                lambda p: compute_objective(p, beta=100.0),
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
            )
            if res_polish.fun < best_score:
                best_score = res_polish.fun
                best_params = res_polish.x.copy()
        except Exception as e:
            print(f"Final polish failed: {e}")

    # Fallback
    if best_params is None:
        best_params = generate_symmetric_init()

    # --- Construct Output ---
    
    inner = []
    hexes_list = []
    for i in range(N):
        x = float(best_params[i])
        y = float(best_params[N + i])
        a_rad = float(best_params[2*N + i])
        a_deg = float(np.degrees(a_rad))
        
        inner.append([x, y, a_deg])
        hexes_list.append(hex_vertices(x, y, 1.0, a_rad))
    
    # Optimize outer hexagon rotation for the final layout
    s_final, outer_angle_deg = optimize_outer_rotation(hexes_list)
    
    outer_center = (0.0, 0.0)
    outer_side = float(s_final)

    return inner, outer_center, outer_side, float(outer_angle_deg)
