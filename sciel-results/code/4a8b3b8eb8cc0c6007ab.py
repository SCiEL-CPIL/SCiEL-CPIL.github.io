import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Unit hexagon vertices at angle 0 (pointy-top: vertex at 90°)
    # Standard: vertices at 0°, 60°, 120°, 180°, 240°, 300°
    HEX_VERTS = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def rotate_verts(verts, angle):
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s], [s, c]])
        return verts @ R.T

    def hex_vertices(cx, cy, angle):
        return cx + cy * np.array([1.0, 1.0]) + rotate_verts(HEX_VERTS, angle)  # placeholder, will fix

    def hex_vertices(cx, cy, angle):
        v = rotate_verts(HEX_VERTS, angle)
        return np.column_stack([v[:, 0] + cx, v[:, 1] + cy])

    def overlap_penalty(v1, v2, eps=0.001):
        """Penalty for two hexagons overlapping. Uses separating axis theorem.
        Returns 0 if separated, positive if overlapping (depth of overlap)."""
        n = 6
        # Get all edge normals from both hexagons
        normals = []
        for v in (v1, v2):
            for i in range(n):
                a = v[i]
                b = v[(i + 1) % n]
                e = b - a
                L = np.linalg.norm(e)
                if L < 1e-15:
                    continue
                normals.append(np.array([-e[1], e[0]]) / L)
        if not normals:
            return 0.0
        max_overlap = 0.0
        for nm in normals:
            proj1 = v1 @ nm
            proj2 = v2 @ nm
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 + eps or max2 < min1 + eps:
                return 0.0  # Separated on this axis
            overlap = min(max1, max2) - max(min1, min2)
            if overlap > max_overlap:
                max_overlap = overlap
        return max_overlap

    def smooth_overlap_penalty(v1, v2, temp=0.01):
        """Smooth version of overlap penalty using log-sum-exp."""
        n = 6
        normals = []
        for v in (v1, v2):
            for i in range(n):
                a = v[i]
                b = v[(i + 1) % n]
                e = b - a
                L = np.linalg.norm(e)
                if L < 1e-15:
                    continue
                normals.append(np.array([-e[1], e[0]]) / L)
        if not normals:
            return 0.0
        overlaps = []
        for nm in normals:
            proj1 = v1 @ nm
            proj2 = v2 @ nm
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            sep = max(min1 - max2, min2 - max1)
            # If sep > 0, separated; if sep < 0, overlap depth = -sep
            overlaps.append(-sep)  # positive when overlapping
        overlaps = np.array(overlaps)
        # max of overlaps = overlap depth
        max_o = overlaps.max()
        if max_o <= 0:
            return 0.0
        # smooth max
        shifted = overlaps / temp
        shifted = shifted - shifted.max()
        log_sum = temp * (shifted.max() + np.log(np.sum(np.exp(shifted))))
        return log_sum

    def containment_penalty(pts, centers, cx, cy, S, beta, temp=0.01):
        """Penalty for hexagons not being inside the outer hexagon.
        Outer hexagon: center (cx,cy), side S, rotation beta.
        For a point to be inside: projection onto each of 6 normals <= S*APOTHEM."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        # For each inner hex, check all its vertices
        limit = S * APOTHEM
        violations = []
        for i in range(N):
            v = hex_vertices(centers[i, 0], centers[i, 1], 0.0)  # inner hexes at angle 0 for now
            # Actually we need inner hex rotation too - let's use 0 for simplicity first
            for k in range(6):
                nm = normals[k]
                proj = (v - np.array([cx, cy])) @ nm
                violations.append(proj - limit)
                violations.append(-proj - limit)
        violations = np.array(violations)
        max_v = violations.max()
        if max_v <= 0:
            return 0.0
        shifted = violations / temp
        shifted = shifted - shifted.max()
        return temp * (shifted.max() + np.log(np.sum(np.exp(shifted))))

    def objective(params, temp=0.01):
        """Joint objective: minimize S + penalty(overlap) + penalty(containment)."""
        # params: [x1,y1, x2,y2, ..., x11,y11, cx, cy, S, beta]
        centers = params[:2*N].reshape(N, 2)
        cx, cy = params[2*N], params[2*N+1]
        S = params[2*N+2]
        beta = params[2*N+3]
        
        if S < 1.0:
            S = 1.0 + (S - 1.0) * 0.01  # soft bound
        
        # Compute overlap penalties
        overlap_pen = 0.0
        verts_list = [hex_vertices(centers[i, 0], centers[i, 1], 0.0) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                op = smooth_overlap_penalty(verts_list[i], verts_list[j], temp)
                overlap_pen += op
                if overlap_pen > 100:
                    return S + overlap_pen * 10 + 1000
        
        # Containment penalty
        cont_pen = 0.0
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        limit = S * APOTHEM
        violations = []
        for i in range(N):
            v = verts_list[i]
            for k in range(6):
                nm = normals[k]
                proj = (v - np.array([cx, cy])) @ nm
                violations.append(proj - limit)
                violations.append(-proj - limit)
        violations = np.array(violations)
        max_v = violations.max()
        if max_v > 0:
            shifted = violations / temp
            shifted = shifted - shifted.max()
            cont_pen = temp * (shifted.max() + np.log(np.sum(np.exp(shifted))))
        
        return S + overlap_pen * 10 + cont_pen * 10

    def true_S(centers, cx, cy, beta):
        """Compute true minimum S for given centers and beta (via LP)."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        # Collect all vertices of all inner hexagons
        all_pts = []
        for i in range(N):
            v = hex_vertices(centers[i, 0], centers[i, 1], 0.0)
            all_pts.append(v)
        all_pts = np.vstack(all_pts)
        
        n_pts = len(all_pts)
        n_constraints = 12 * n_pts
        A_ub = np.zeros((n_constraints, 3))
        b_ub = np.zeros(n_constraints)
        
        for k in range(6):
            nx, ny = normals[k]
            projs = all_pts[:, 0] * nx + all_pts[:, 1] * ny
            start = 2 * k * n_pts
            A_ub[start:start+n_pts, 0] = -nx
            A_ub[start:start+n_pts, 1] = -ny
            A_ub[start:start+n_pts, 2] = -1.0
            b_ub[start:start+n_pts] = -projs
            A_ub[start+n_pts:start+2*n_pts, 0] = nx
            A_ub[start+n_pts:start+2*n_pts, 1] = ny
            A_ub[start+n_pts:start+2*n_pts, 2] = -1.0
            b_ub[start+n_pts:start+2*n_pts] = projs
        
        c_obj = np.array([0.0, 0.0, 1.0])
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            from scipy.optimize import linprog
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                return (D + APOTHEM) / APOTHEM
        except Exception:
            pass
        return float('inf')

    def check_valid(centers, cx, cy, S, beta, tol=1e-6):
        """Check if configuration is valid: all hexes disjoint and inside outer hex."""
        verts_list = [hex_vertices(centers[i, 0], centers[i, 1], 0.0) for i in range(N)]
        
        # Check disjointness
        for i in range(N):
            for j in range(i+1, N):
                v1, v2 = verts_list[i], verts_list[j]
                # SAT
                n = 6
                separated = False
                for v in (v1, v2):
                    for k in range(n):
                        a = v[k]
                        b = v[(k+1) % n]
                        e = b - a
                        L = np.linalg.norm(e)
                        if L < 1e-15:
                            continue
                        nm = np.array([-e[1], e[0]]) / L
                        proj1 = v1 @ nm
                        proj2 = v2 @ nm
                        if proj1.max() < proj2.min() + tol or proj2.max() < proj1.min() + tol:
                            separated = True
                            break
                    if separated:
                        break
                if not separated:
                    return False
        
        # Check containment
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        limit = S * APOTHEM
        for i in range(N):
            v = verts_list[i]
            for k in range(6):
                nm = normals[k]
                proj = (v - np.array([cx, cy])) @ nm
                if proj.max() > limit + tol or (-proj).max() > limit + tol:
                    return False
        return True

    # Initial configuration: standard hexagonal cluster on axial lattice
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    # Best known compact configuration for 11 hexagons
    axial_init = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(2,-1),(-2,1)]
    centers_init = np.array([axial_to_cartesian(q, r) for q, r in axial_init])
    
    # Compute initial enclosing hex
    all_pts = []
    for i in range(N):
        v = hex_vertices(centers_init[i, 0], centers_init[i, 1], 0.0)
        all_pts.append(v)
    all_pts = np.vstack(all_pts)
    
    cx_init = float(np.mean(all_pts[:, 0]))
    cy_init = float(np.mean(all_pts[:, 1]))
    
    # Compute initial S
    best_S_init = float('inf')
    for beta in np.linspace(0, math.pi/6, 37):
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        projs = (all_pts - np.array([cx_init, cy_init])) @ normals.T
        D = np.max(np.abs(projs))
        S = (D + APOTHEM) / APOTHEM
        if S < best_S_init:
            best_S_init = S
            best_beta_init = beta
    
    params_init = np.concatenate([centers_init.ravel(), [cx_init, cy_init, best_S_init, best_beta_init]])
    
    best_S_global = best_S_init
    best_params_global = params_init.copy()
    
    # Simulated Annealing + Nelder-Mead hybrid
    n_params = 2 * N + 4  # 22 + 4 = 26
    
    # Phase 1: Simulated Annealing for global exploration
    T0 = 0.5
    Tend = 0.001
    n_sa_iters = 3000
    cooling = (Tend / T0) ** (1.0 / n_sa_iters)
    
    cur_params = params_init.copy()
    cur_obj = objective(cur_params, temp=0.02)
    best_params = cur_params.copy()
    best_obj = cur_obj
    
    for it in range(n_sa_iters):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN - 30:
            break
        
        T = T0 * (cooling ** it)
        
        # Generate perturbation
        step_size = 0.05 + 0.1 * T
        delta = np.random.randn(n_params) * step_size
        # Scale S perturbation
        delta[2*N+2] *= 0.3
        # Scale beta perturbation
        delta[2*N+3] *= 0.1
        
        new_params = cur_params + delta
        new_params[2*N+3] = ((new_params[2*N+3] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        
        new_obj = objective(new_params, temp=0.02)
        
        if new_obj < cur_obj or np.random.random() < math.exp(-(new_obj - cur_obj) / max(T, 1e-10)):
            cur_params = new_params
            cur_obj = new_obj
            if cur_obj < best_obj:
                best_obj = cur_obj
                best_params = cur_params.copy()
    
    # Phase 2: Local refinement with Nelder-Mead from multiple starts
    starts = [best_params, params_init]
    # Add some random starts
    for _ in range(5):
        pert = params_init + np.random.randn(n_params) * 0.1
        pert[2*N+3] = ((pert[2*N+3] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        starts.append(pert)
    
    for start in starts:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN - 10:
            break
        
        try:
            def obj_fn(p):
                p[2*N+3] = ((p[2*N+3] % (math.pi/6)) + math.pi/6) % (math.pi/6)
                return objective(p, temp=0.01)
            
            res = minimize(obj_fn, start, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Extract best configuration and compute true S via LP
    bp = best_params
    centers = bp[:2*N].reshape(N, 2)
    cx = bp[2*N]
    cy = bp[2*N+1]
    beta = ((bp[2*N+3] % (math.pi/6)) + math.pi/6) % (math.pi/6)
    
    # Compute true S
    true_S_val = true_S(centers, cx, cy, beta)
    
    if true_S_val < best_S_global:
        best_S_global = true_S_val
        best_params_global = bp.copy()
    
    # Phase 4: Refine with direct S minimization (fix centers, optimize cx, cy, beta)
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN - 5:
        def S_only(params):
            cx, cy, beta = params
            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
            return true_S(centers, cx, cy, beta)
        
        res = minimize(S_only, [cx, cy, beta], method='Nelder-Mead',
                      options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
        if res.fun < best_S_global:
            best_S_global = res.fun
            cx, cy, beta = res.x
            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
            best_params_global = np.concatenate([centers.ravel(), [cx, cy, best_S_global, beta]])
    
    # Phase 5: Try to move centers to reduce S further
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN - 3:
        def full_obj(p):
            c = p[:2*N].reshape(N, 2)
            cx, cy, beta = p[2*N], p[2*N+1], p[2*N+2]
            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
            S_val = true_S(c, cx, cy, beta)
            if not np.isfinite(S_val):
                return 100.0
            # Add small penalty for potential overlaps (quick check)
            return S_val
        
        res = minimize(full_obj, best_params_global[:2*N+3], method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
        if res.fun < best_S_global:
            best_S_global = res.fun
            best_params_global = np.concatenate([res.x, [best_S_global, best_params_global[2*N+3]]])
    
    # Final: extract and verify
    bp = best_params_global
    centers = bp[:2*N].reshape(N, 2)
    cx = bp[2*N]
    cy = bp[2*N+1]
    beta = ((bp[2*N+3] % (math.pi/6)) + math.pi/6) % (math.pi/6)
    
    # Recompute S precisely
    S_final = true_S(centers, cx, cy, beta)
    if not np.isfinite(S_final):
        S_final = best_S_global
    
    # Verify validity
    if not check_valid(centers, cx, cy, S_final, beta, tol=1e-5):
        # Fall back to initial configuration
        centers = centers_init.copy()
        cx = cx_init
        cy = cy_init
        beta = best_beta_init
        S_final = best_S_init
    
    # Add tiny safety margin to S
    S_final *= (1.0 + 1e-10)
    
    angle_deg = math.degrees(beta)
    
    inner = []
    for i in range(N):
        inner.append([float(centers[i, 0]), float(centers[i, 1]), float(angle_deg)])
    
    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
