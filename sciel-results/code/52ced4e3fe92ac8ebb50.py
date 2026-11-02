import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping, linprog
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

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

    def compute_min_side_and_angle(hexes):
        """
        Calculate the minimum side length of a regular hexagon (centered at origin)
        that contains all given hexagons, and the optimal rotation angle.
        """
        # We optimize the angle of the outer hexagon
        # The side length S is determined by the max projection of any vertex onto any of the 6 normals.
        # S = max_k max_i max_j (v_ij . n_k) * 2/sqrt(3)
        
        def calc_side(angle_rad):
            normal_angles = angle_rad + np.radians([30, 90, 150, 210, 270, 330])
            normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
            
            max_proj = 0.0
            # Vectorized check for all hexagons
            for h in hexes:
                proj = h @ normals.T
                max_proj = max(max_proj, np.max(proj))
            
            s = max_proj * 2.0 / np.sqrt(3.0)
            return s

        # Golden section search for optimal angle in [0, 60 deg)
        gr = (np.sqrt(5) - 1) / 2
        a, b = 0.0, np.pi / 3.0
        
        x1 = b - gr * (b - a)
        x2 = a + gr * (b - a)
        f1 = calc_side(x1)
        f2 = calc_side(x2)
        
        for _ in range(30):
            if time.time() > deadline - 1:
                break
            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = b - gr * (b - a)
                f1 = calc_side(x1)
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + gr * (b - a)
                f2 = calc_side(x2)
        
        best_angle = (a + b) / 2
        best_s = calc_side(best_angle)
        
        # Check boundaries just in case
        s_0 = calc_side(0.0)
        if s_0 < best_s:
            best_s = s_0
            best_angle = 0.0
            
        return best_s, best_angle

    def get_hexes_from_params(params):
        """
        params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12]
        Returns list of 12 hexagons (6x2 arrays).
        """
        hexes = []
        for i in range(12):
            x = params[i*3]
            y = params[i*3 + 1]
            a = params[i*3 + 2]
            hexes.append(hex_vertices(x, y, 1.0, a))
        return hexes

    def compute_objective(params):
        """
        Objective: min_side + penalty for overlaps.
        """
        if time.time() > deadline:
            return 1e10
            
        hexes = get_hexes_from_params(params)
        
        s_min, _ = compute_min_side_and_angle(hexes)
        
        penalty = 0.0
        # Check overlaps
        for i in range(12):
            min1 = hexes[i].min(axis=0)
            max1 = hexes[i].max(axis=0)
            for j in range(i + 1, 12):
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)
                
                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue
                    
                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    penalty += overlap * 10.0
                    # Early exit if penalty is huge
                    if penalty > 10.0:
                        return s_min + penalty

        return s_min + penalty

    # --- Optimization ---
    
    # Initial guess: Symmetric 2-orbit solution
    # r1=1.5, r2=2.5 approx? Let's generate a valid starting point
    # We can use the previous best if we had it, but let's create a symmetric one
    r1, th1, al1 = 1.8, 0.0, 0.0
    r2, th2, al2 = 3.0, 0.0, 0.0
    
    init_params = np.zeros(36)
    for k in range(6):
        angle = np.radians(60 * k)
        
        x1 = r1 * np.cos(th1 + angle)
        y1 = r1 * np.sin(th1 + angle)
        a1 = al1 + angle
        
        x2 = r2 * np.cos(th2 + angle)
        y2 = r2 * np.sin(th2 + angle)
        a2 = al2 + angle
        
        idx1 = k * 3
        init_params[idx1] = x1
        init_params[idx1+1] = y1
        init_params[idx1+2] = a1
        
        idx2 = (k + 6) * 3
        init_params[idx2] = x2
        init_params[idx2+1] = y2
        init_params[idx2+2] = a2

    best_params = init_params.copy()
    best_score = compute_objective(best_params)
    
    # Phase 1: Simulated Annealing
    # SA is good for exploring the 36D space without getting stuck in local minima as easily as gradient methods
    T = 0.5
    T_min = 0.001
    cooling_rate = 0.999
    n_steps = 5000
    
    np.random.seed(42)
    
    current_params = best_params.copy()
    current_score = best_score
    
    for step in range(n_steps):
        if time.time() > deadline - 20:
            break
            
        # Perturb
        new_params = current_params.copy()
        # Randomly pick 1 or 2 indices to perturb
        n_perturb = np.random.randint(1, 3)
        for _ in range(n_perturb):
            idx = np.random.randint(0, 36)
            # Step size depends on T
            step_size = 0.1 * (1 + T)
            new_params[idx] += np.random.normal(0, step_size)
            
            # Wrap angles
            if idx % 3 == 2:
                new_params[idx] = new_params[idx] % (np.pi) # 0 to 180 deg is enough for hex symmetry? 0 to 60 is better
                # Actually hexagon has 60 deg symmetry. Let's wrap to 0..pi/3
                new_params[idx] = new_params[idx] % (np.pi/3)
                
        new_score = compute_objective(new_params)
        
        if new_score < current_score:
            current_params = new_params
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_params = current_params.copy()
        else:
            # Accept with probability exp(-(new - current) / T)
            delta = new_score - current_score
            if delta < 0:
                prob = 1.0
            else:
                prob = np.exp(-delta / T)
                
            if np.random.random() < prob:
                current_params = new_params
                current_score = new_score
                
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Phase 2: Local Refinement with L-BFGS-B
    # This should clean up the solution found by SA
    try:
        res_local = minimize(
            compute_objective,
            best_params,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-10}
        )
        if res_local.fun < best_score:
            best_score = res_local.fun
            best_params = res_local.x.copy()
    except Exception:
        pass

    # Phase 3: Try to improve with Nelder-Mead if time allows
    if time.time() < deadline - 10:
        try:
            res_nm = minimize(
                compute_objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_params = res_nm.x.copy()
        except Exception:
            pass

    # Final Polish
    if time.time() < deadline - 5:
        try:
            res_polish = minimize(
                compute_objective,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if res_polish.fun < best_score:
                best_score = res_polish.fun
                best_params = res_polish.x.copy()
        except Exception:
            pass

    # Generate output
    inner = []
    hexes_list = []
    for i in range(12):
        x = best_params[i*3]
        y = best_params[i*3 + 1]
        a = best_params[i*3 + 2]
        
        inner.append([float(x), float(y), float(np.degrees(a))])
        hexes_list.append(hex_vertices(x, y, 1.0, a))
        
    s_final, outer_angle_deg = compute_min_side_and_angle(hexes_list)
    
    outer_center = (0.0, 0.0)
    outer_side = float(s_final)

    return inner, outer_center, outer_side, float(outer_angle_deg)
