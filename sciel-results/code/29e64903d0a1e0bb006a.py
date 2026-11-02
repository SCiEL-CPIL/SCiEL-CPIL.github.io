import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate initial candidate configurations using hexagonal grid patterns
    # These serve as starting points for force-directed relaxation
    def generate_initial_configurations():
        configs = []
        
        # Configuration 1: Linear chain
        chain = [(i, 0) for i in range(N)]
        configs.append(chain)
        
        # Configuration 2: 2xN rectangle
        rect = []
        for r in range(2):
            for c in range(N // 2 + 1):
                if len(rect) < N:
                    rect.append((c, r))
        configs.append(rect)
        
        # Configuration 3: 3xN rectangle
        rect3 = []
        for r in range(3):
            for c in range(N // 3 + 1):
                if len(rect3) < N:
                    rect3.append((c, r))
        configs.append(rect3)
        
        # Configuration 4: Hexagonal cluster (center + rings)
        hex_cluster = [(0, 0)]
        # First ring
        for k in range(6):
            ang = k * math.pi / 3
            q = round(math.cos(ang))
            r = round(math.sin(ang))
            if (q, r) not in hex_cluster:
                hex_cluster.append((q, r))
        # Second ring
        for q in range(-2, 3):
            for r in range(-2, 3):
                if abs(q + r) <= 2 and abs(q) <= 2 and abs(r) <= 2:
                    if (q, r) not in hex_cluster and len(hex_cluster) < N:
                        hex_cluster.append((q, r))
        configs.append(hex_cluster[:N])
        
        # Configuration 5: Diagonal
        diag = [(i, i) for i in range(N)]
        configs.append(diag)
        
        # Configuration 6: Compact 4x3
        compact = []
        for r in range(3):
            for c in range(4):
                if len(compact) < N:
                    compact.append((c, r))
        configs.append(compact)
        
        # Configuration 7: 5x2
        compact2 = []
        for r in range(2):
            for c in range(5):
                if len(compact2) < N:
                    compact2.append((c, r))
        configs.append(compact2)
        
        # Configuration 8: Center + 10 around
        center_plus = [(0, 0)]
        for k in range(6):
            ang = k * math.pi / 3
            q = round(math.cos(ang))
            r = round(math.sin(ang))
            center_plus.append((q, r))
        # Add 4 more
        center_plus.extend([(2, 0), (0, 2), (-2, 0), (0, -2)])
        configs.append(center_plus[:N])
        
        # Configuration 9: Staggered rows
        staggered = []
        for r in range(4):
            offset = 0 if r % 2 == 0 else 1
            for c in range(offset, offset + 3):
                if len(staggered) < N:
                    staggered.append((c, r))
        configs.append(staggered[:N])
        
        # Configuration 10: Random with some structure
        np.random.seed(42)
        random_config = []
        for i in range(N):
            q = np.random.randint(-2, 3)
            r = np.random.randint(-2, 3)
            if (q, r) not in random_config:
                random_config.append((q, r))
        configs.append(random_config)
        
        return configs

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def force_directed_relaxation(pts, iterations=500, dt=0.01, repulsion_strength=1.0, attraction_strength=0.1):
        """
        Force-directed layout to push hexagons apart until they form a stable,
        non-overlapping cluster.
        """
        n = len(pts)
        pts = pts.copy().astype(float)
        
        for it in range(iterations):
            forces = np.zeros_like(pts)
            
            # Repulsive forces between all pairs
            for i in range(n):
                for j in range(i + 1, n):
                    diff = pts[i] - pts[j]
                    dist = np.linalg.norm(diff)
                    if dist < 1e-6:
                        # Jitter if too close
                        diff = np.array([np.random.randn(), np.random.randn()]) * 0.01
                        dist = np.linalg.norm(diff)
                    
                    # Repulsion: stronger when closer
                    if dist < 2.0:  # Only interact within range
                        force_mag = repulsion_strength / (dist * dist)
                        force = force_mag * diff / dist
                        forces[i] += force
                        forces[j] -= force
            
            # Attraction to center (to keep cluster together)
            center = np.mean(pts, axis=0)
            for i in range(n):
                to_center = center - pts[i]
                forces[i] += attraction_strength * to_center
            
            # Update positions
            pts += dt * forces
            pts -= np.mean(pts, axis=0)  # Keep centered
            
            # Check convergence
            if np.max(np.abs(dt * forces)) < 1e-6:
                break
        
        return pts

    def compute_minimal_enclosing_hexagon(pts, beta=0.0):
        """
        Compute the minimal enclosing hexagon for a set of points with given orientation.
        Returns (S, center, D) where S is the side length ratio, center is (cx, cy),
        D is the distance from center to farthest point in the hexagon's normal directions.
        """
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        # For each orientation, find the center that minimizes the maximum projection
        # Use linear programming
        from scipy.optimize import linprog
        
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts:
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def optimize_orientation_and_center(pts):
        """
        Optimize the orientation (beta) and center of the enclosing hexagon.
        """
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        # Coarse grid search over beta
        for i in range(37):
            beta = i * (math.pi / 6) / 36
            S, center, D = compute_minimal_enclosing_hexagon(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine with local optimization
        try:
            def objective(params):
                cx, cy, beta = params
                beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                normals = []
                for k in range(6):
                    ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                    normals.append((math.cos(ang), math.sin(ang)))
                
                max_proj = 0.0
                for nx, ny in normals:
                    for px, py in pts:
                        proj = abs((px - cx) * nx + (py - cy) * ny)
                        if proj > max_proj:
                            max_proj = proj
                
                S = (max_proj + APOTHEM) / APOTHEM
                return S
            
            init = [best_center[0], best_center[1], best_beta]
            res = minimize(objective, init, method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
            
            if res.fun < best_S:
                best_S = res.fun
                best_center = (res.x[0], res.x[1])
                best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass
        
        return best_S, best_beta, best_center

    def local_nlp_refinement(pts, beta_init, center_init):
        """
        Local NLP solver to refine the positions of the hexagons while maintaining
        non-overlap and minimizing the enclosing hexagon size.
        """
        n = len(pts)
        
        def objective(x):
            # x contains: x0, y0, x1, y1, ..., x_{n-1}, y_{n-1}, beta, cx, cy
            pts_arr = x[:2*n].reshape(n, 2)
            beta = x[2*n]
            cx = x[2*n+1]
            cy = x[2*n+2]
            
            # Compute enclosing hexagon size
            normals = []
            for k in range(6):
                ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                normals.append((math.cos(ang), math.sin(ang)))
            
            max_proj = 0.0
            for nx, ny in normals:
                for px, py in pts_arr:
                    proj = abs((px - cx) * nx + (py - cy) * ny)
                    if proj > max_proj:
                        max_proj = proj
            
            S = (max_proj + APOTHEM) / APOTHEM
            
            # Add penalty for overlaps
            penalty = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.linalg.norm(pts_arr[i] - pts_arr[j])
                    if dist < 2.0 - 1e-6:
                        penalty += (2.0 - dist) ** 2 * 1000.0
            
            return S + penalty
        
        # Initial guess
        x0 = np.zeros(2*n + 3)
        for i in range(n):
            x0[2*i] = pts[i][0]
            x0[2*i+1] = pts[i][1]
        x0[2*n] = beta_init
        x0[2*n+1] = center_init[0]
        x0[2*n+2] = center_init[1]
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                         options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.success:
                pts_opt = res.x[:2*n].reshape(n, 2)
                beta_opt = ((res.x[2*n] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                center_opt = (res.x[2*n+1], res.x[2*n+2])
                
                # Recompute S without penalty
                normals = []
                for k in range(6):
                    ang = beta_opt + math.pi / 6.0 + k * math.pi / 3.0
                    normals.append((math.cos(ang), math.sin(ang)))
                
                max_proj = 0.0
                for nx, ny in normals:
                    for px, py in pts_opt:
                        proj = abs((px - center_opt[0]) * nx + (py - center_opt[1]) * ny)
                        if proj > max_proj:
                            max_proj = proj
                
                S_opt = (max_proj + APOTHEM) / APOTHEM
                return S_opt, pts_opt, beta_opt, center_opt
            else:
                return float('inf'), pts, beta_init, center_init
        except Exception:
            return float('inf'), pts, beta_init, center_init

    # Main algorithm
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    configs = generate_initial_configurations()
    
    for axial in configs:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Convert to Cartesian
        pts = axial_to_cartesian(axial)
        
        # Force-directed relaxation
        pts_relaxed = force_directed_relaxation(pts, iterations=300, dt=0.005,
                                                repulsion_strength=0.5, attraction_strength=0.05)
        
        # Optimize orientation and center
        S, beta, center = optimize_orientation_and_center(pts_relaxed)
        
        # Local NLP refinement
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            S_nlp, pts_nlp, beta_nlp, center_nlp = local_nlp_refinement(
                pts_relaxed, beta, center)
            
            if S_nlp < S:
                S = S_nlp
                pts_relaxed = pts_nlp
                beta = beta_nlp
                center = center_nlp
        
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts_relaxed
            best_global_beta = beta
            best_global_center = center

    # Fallback if no valid configuration found
    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(configs[0])
        best_global_center = (float(np.mean(best_global_pts[:, 0])), 
                              float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Verify non-overlap and adjust if needed
    pts_final = best_global_pts.copy()
    
    # Check for overlaps and apply small scaling if necessary
    min_dist = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pts_final[i] - pts_final[j])
            min_dist = min(min_dist, d)
    
    # If any overlap, scale up slightly
    if min_dist < 2.0 - 1e-6:
        scale = 2.0 / min_dist * 1.001
        center_mean = np.mean(pts_final, axis=0)
        pts_final = center_mean + (pts_final - center_mean) * scale
    
    # Recompute final S
    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_final[:, 0] - cx) * nx + (pts_final[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))
    
    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM
    
    inner = []
    for x, y in pts_final:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
