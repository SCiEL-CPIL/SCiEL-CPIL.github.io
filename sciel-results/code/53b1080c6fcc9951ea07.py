import numpy as np
import time
import warnings
from scipy.optimize import minimize

N = 11
DEADLINE = 145.0  # Safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy)."""
    angle = np.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = angle + np.pi * i / 3.0
        verts[i, 0] = cx + side * np.cos(theta)
        verts[i, 1] = cy + side * np.sin(theta)
    return verts

def _hex_vertices_batch(centers, angles, side=1.0):
    """Vectorized vertex calculation for multiple hexagons.
    centers: (N, 2)
    angles: (N,)
    Returns: (N, 6, 2)
    """
    N = len(centers)
    angles_rad = np.radians(angles)
    # Base angles: 0, pi/3, ...
    base_angles = np.pi * np.arange(6) / 3.0
    # (N, 1) + (1, 6)
    thetas = angles_rad[:, np.newaxis] + base_angles[np.newaxis, :]
    
    verts = np.empty((N, 6, 2))
    verts[:, :, 0] = centers[:, 0:1] + side * np.cos(thetas)
    verts[:, :, 1] = centers[:, 1:2] + side * np.sin(thetas)
    return verts

def _get_valid_fallback():
    """Return a known valid, non-overlapping configuration."""
    # Hexagonal lattice
    d = 2 * np.sqrt(3)
    centers = [(0.0, 0.0)]
    angles = [0.0]
    
    # Ring 1
    for i in range(6):
        angle = np.pi / 3 * i
        centers.append((d * np.cos(angle), d * np.sin(angle)))
        angles.append(0.0)
        
    # Ring 2 (need 4 more)
    d2 = 4 * np.sqrt(3)
    # Pick 4 positions from the second ring that are spread out
    # Second ring positions are at distance 4*sqrt(3) and 2*sqrt(3)*sqrt(3)=6? No.
    # Lattice spacing:
    # (0,0)
    # (d, 0), (d/2, d*sqrt(3)/2)...
    # Let's just use a simple grid or spiral for fallback to be safe.
    # Actually, the lattice above has 7 hexagons.
    # We need 4 more.
    # Place them at (2*d, 0), (-2*d, 0), (d, 2*d*sqrt(3)/2), (-d, -2*d*sqrt(3)/2)?
    # Let's just use a large radius circle for fallback to guarantee validity.
    # Side length will be large, but valid.
    centers = [(0.0, 0.0)]
    angles = [0.0]
    for i in range(N-1):
        angle = 2 * np.pi * i / (N-1)
        # Radius 5.0 is safe for unit hexagons (max width 2*sqrt(3) ~ 3.46)
        # Distance between neighbors on circle of radius R: 2*R*sin(pi/(N-1))
        # 2*5*sin(pi/10) ~ 10*0.309 = 3.09 < 3.46. Overlap!
        # Use Radius 6.0. 2*6*sin(pi/10) ~ 3.7 > 3.46. OK.
        centers.append((6.0 * np.cos(angle), 6.0 * np.sin(angle)))
        angles.append(0.0)
        
    # Optimize outer for this fallback
    side, center, angle = _optimize_outer(centers, angles)
    return centers, angles, center, side, angle

def _optimize_outer(centers, inner_angles):
    """Find optimal outer hexagon for fixed inner positions."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Sweep angle coarsely
    for outer_angle in np.linspace(0, 60, 61):
        side, center = _min_enclosing_side_lp(centers, inner_angles, outer_angle)
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = outer_angle
            
    # Refine around best
    if best_side < float('inf'):
        for outer_angle in np.linspace(best_angle - 5, best_angle + 5, 21):
            side, center = _min_enclosing_side_lp(centers, inner_angles, outer_angle)
            if side < best_side:
                best_side = side
                best_center = center
                best_angle = outer_angle

    return best_side, best_center, best_angle

def _min_enclosing_side_lp(centers, inner_angles, outer_angle_deg):
    """
    Compute minimum side length of a regular hexagon (fixed orientation)
    enclosing all inner hexagons using Linear Programming.
    """
    sqrt3 = np.sqrt(3)
    # Normals of outer hexagon
    angle = np.radians(outer_angle_deg)
    normals = np.empty((6, 2))
    for i in range(6):
        mid_angle = angle + np.pi / 6.0 + np.pi * i / 3.0
        normals[i, 0] = np.cos(mid_angle)
        normals[i, 1] = np.sin(mid_angle)
    
    # Get all vertices
    all_points = np.vstack([_hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i]) 
                            for i in range(len(centers))])
    
    # Projections
    projections = all_points @ normals.T
    A = projections.max(axis=0)
    
    # Variables: x = [cx, cy, r]
    # Minimize r
    c_obj = np.array([0.0, 0.0, 1.0])
    
    # Constraints: n_i . c - r <= -A_i
    A_mat = np.zeros((6, 3))
    b_vec = np.zeros(6)
    
    for i in range(6):
        A_mat[i, 0] = normals[i, 0]
        A_mat[i, 1] = normals[i, 1]
        A_mat[i, 2] = -1.0
        b_vec[i] = -A[i]
        

    # Using linprog is faster
    from scipy.optimize import linprog
    res = linprog(c_obj, A_ub=A_mat, b_ub=b_vec, method='highs')
    
    if res.status != 0:
        return float('inf'), np.array([0.0, 0.0])
        
    cx, cy, r = res.x
    side = 2 * r / sqrt3
    return side, np.array([cx, cy])

def _hex_overlap(c1, a1, c2, a2, tol=1e-9):
    """Check if two unit hexagons overlap using SAT."""
    v1 = _hex_vertices(c1[0], c1[1], 1.0, a1)
    v2 = _hex_vertices(c2[0], c2[1], 1.0, a2)
    
    def get_axes(verts):
        axes = []
        n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            ex = x2 - x1
            ey = y2 - y1
            nx, ny = -ey, ex
            norm = np.sqrt(nx**2 + ny**2)
            if norm > 1e-12:
                axes.append((nx / norm, ny / norm))
        return axes
    
    axes = get_axes(v1) + get_axes(v2)
    for ax, ay in axes:
        proj1 = v1[:, 0] * ax + v1[:, 1] * ay
        proj2 = v2[:, 0] * ax + v2[:, 1] * ay
        if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
            return False
    return True

def _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle_deg, tol=1e-7):
    """Check containment and non-overlap."""
    sqrt3 = np.sqrt(3)
    angle = np.radians(outer_angle_deg)
    normals = np.empty((6, 2))
    for i in range(6):
        mid_angle = angle + np.pi / 6.0 + np.pi * i / 3.0
        normals[i, 0] = np.cos(mid_angle)
        normals[i, 1] = np.sin(mid_angle)
        
    inradius = outer_side * sqrt3 / 2
    
    for i in range(len(centers)):
        verts = _hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i])
        for vx, vy in verts:
            for nx, ny in normals:
                dist = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny
                if dist > inradius + tol:
                    return False
    
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                return False
    
    return True

def _smooth_objective(vars, beta_overlap=10.0, beta_side=10.0):
    """
    vars: [x1, y1, th1, ..., x11, y11, th11, phi_outer]
    Returns: Smooth approximation of (OuterSide + OverlapPenalty)
    """
    # Unpack
    centers = np.empty((N, 2))
    angles = np.empty(N)
    for i in range(N):
        centers[i, 0] = vars[3*i]
        centers[i, 1] = vars[3*i+1]
        angles[i] = vars[3*i+2]
        
    phi_outer = vars[3*N]
    
    # 1. Containment Cost (Smooth Max of Projections)
    # Outer normals
    angle_rad = np.radians(phi_outer)
    normals = np.empty((6, 2))
    for i in range(6):
        mid_angle = angle_rad + np.pi / 6.0 + np.pi * i / 3.0
        normals[i, 0] = np.cos(mid_angle)
        normals[i, 1] = np.sin(mid_angle)
        
    # Vertices of all inner hexagons
    verts = _hex_vertices_batch(centers, angles, 1.0) # (N, 6, 2)
    
    # Projections onto each normal
    # proj[k] = max over all vertices of | (v - C) . n_k |
    # We assume outer center is (0,0) for the objective, as the optimizer will shift centers to minimize this.
    # Actually, we should allow the outer center to move?
    # If we fix outer center to (0,0), the inner hexagons will cluster around it.
    # This is sufficient.
    
    max_proj = np.zeros(6)
    for k in range(6):
        n = normals[k]
        # (v . n) for all vertices
        # verts: (N, 6, 2), n: (2,)
        proj = np.einsum('nij,j->ni', verts, n)
        # Max absolute value
        max_proj[k] = np.max(np.abs(proj))
        
    # Smooth max of max_proj
    # S = (2/sqrt(3)) * (1/beta) * log(sum(exp(beta * max_proj)))
    log_sum_exp = np.log(np.sum(np.exp(beta_side * max_proj)))
    smooth_side = (2.0 / np.sqrt(3.0)) * (log_sum_exp / beta_side)
    
    # 2. Overlap Penalty
    penalty = 0.0
    # To be fast, only check pairs that are "close"
    # But for gradient, we need all pairs?
    # L-BFGS-B uses numerical gradient, so we just need the value.
    # However, 55 pairs * 12 axes is cheap.
    
    for i in range(N):
        for j in range(i + 1, N):
            # Compute min gap between hex i and hex j
            v1 = verts[i]
            v2 = verts[j]
            
            # Get axes (normals to edges)
            # For hex i, normals are at angles th_i + pi/6 + k*pi/3
            # For hex j, normals are at angles th_j + pi/6 + k*pi/3
            
            th1 = np.radians(angles[i])
            th2 = np.radians(angles[j])
            
            axes = []
            for k in range(6):
                a1 = th1 + np.pi/6 + k*np.pi/3
                axes.append((np.cos(a1), np.sin(a1)))
                a2 = th2 + np.pi/6 + k*np.pi/3
                axes.append((np.cos(a2), np.sin(a2)))
                
            min_gap = float('inf')
            for ax, ay in axes:
                proj1 = v1[:, 0] * ax + v1[:, 1] * ay
                proj2 = v2[:, 0] * ax + v2[:, 1] * ay
                gap = min(proj2.min() - proj1.max(), proj1.min() - proj2.max())
                if gap < min_gap:
                    min_gap = gap
                    
            # Penalty for negative gap (overlap)
            # softplus(-gap)
            # if min_gap < 0, penalty is high.
            # P = (1/beta) * log(1 + exp(-beta * min_gap))
            # If min_gap is large positive, P ~ 0.
            # If min_gap is negative, P ~ -min_gap.
            
            # Use a stable softplus
            if min_gap > 0:
                p = 0.0
            else:
                # -beta * min_gap is positive
                x = -beta_overlap * min_gap
                if x > 50:
                    p = -min_gap
                else:
                    p = (1.0 / beta_overlap) * np.log(1.0 + np.exp(x))
            penalty += p * p
            
    return smooth_side + 10.0 * penalty

def _local_optimize(initial_vars, max_iter=1000):
    """Run L-BFGS-B optimization."""
    # Initial value
    f0 = _smooth_objective(initial_vars)
    
    # Bounds: Angles in [0, 360), Positions in [-20, 20]
    bounds = []
    for i in range(N):
        bounds.append((-20, 20)) # x
        bounds.append((-20, 20)) # y
        bounds.append((0, 360)) # angle
    bounds.append((0, 60)) # outer angle
    
    try:
        res = minimize(
            _smooth_objective,
            initial_vars,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-6, 'gtol': 1e-6, 'disp': False}
        )
        return res.x, res.fun
    except Exception as e:
        return initial_vars, f0

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_result = None
    
    # Generate initial configurations
    # 1. Hexagonal Lattice
    d = 2 * np.sqrt(3)
    centers = [(0.0, 0.0)]
    angles = [0.0]
    for i in range(6):
        angle = np.pi / 3 * i
        centers.append((d * np.cos(angle), d * np.sin(angle)))
        angles.append(0.0)
        
    # Add 4 more from second ring
    d2 = 4 * np.sqrt(3)
    # Positions in second ring (axial)
    for i in range(4):
        angle = np.pi / 3 * i
        centers.append((d2 * np.cos(angle), d2 * np.sin(angle)))
        angles.append(0.0)
        
    # Convert to vars
    vars_list = []
    for i in range(N):
        vars_list.append(centers[i][0])
        vars_list.append(centers[i][1])
        vars_list.append(angles[i])
    vars_list.append(0.0) # outer angle
    initial_vars = np.array(vars_list)
    
    # Run optimization
    if time.time() - start_time < DEADLINE - 10:
        opt_vars, opt_val = _local_optimize(initial_vars)
        
        # Extract result
        centers = [(opt_vars[3*i], opt_vars[3*i+1]) for i in range(N)]
        angles = [opt_vars[3*i+2] for i in range(N)]
        outer_angle = opt_vars[3*N]
        
        # Get exact outer side
        side, center, angle = _optimize_outer(centers, angles)
        
        if np.isfinite(side):
            if _check_validity(centers, angles, center, side, angle):
                if side < best_side:
                    best_side = side
                    best_result = (centers, angles, center, side, angle)
    
    # 2. Random Jittered Lattice
    if time.time() - start_time < DEADLINE - 10:
        np.random.seed(42)
        jittered_vars = initial_vars.copy()
        for i in range(N):
            jittered_vars[3*i] += np.random.normal(0, 0.5)
            jittered_vars[3*i+1] += np.random.normal(0, 0.5)
            jittered_vars[3*i+2] += np.random.normal(0, 10.0)
            
        opt_vars, opt_val = _local_optimize(jittered_vars)
        
        centers = [(opt_vars[3*i], opt_vars[3*i+1]) for i in range(N)]
        angles = [opt_vars[3*i+2] for i in range(N)]
        outer_angle = opt_vars[3*N]
        
        side, center, angle = _optimize_outer(centers, angles)
        
        if np.isfinite(side):
            if _check_validity(centers, angles, center, side, angle):
                if side < best_side:
                    best_side = side
                    best_result = (centers, angles, center, side, angle)

    # 3. Random Circle
    if time.time() - start_time < DEADLINE - 10:
        np.random.seed(123)
        random_vars = []
        for i in range(N):
            x = np.random.normal(0, 3.0)
            y = np.random.normal(0, 3.0)
            a = np.random.uniform(0, 360)
            random_vars.extend([x, y, a])
        random_vars.append(0.0)
        
        opt_vars, opt_val = _local_optimize(np.array(random_vars))
        
        centers = [(opt_vars[3*i], opt_vars[3*i+1]) for i in range(N)]
        angles = [opt_vars[3*i+2] for i in range(N)]
        outer_angle = opt_vars[3*N]
        
        side, center, angle = _optimize_outer(centers, angles)
        
        if np.isfinite(side):
            if _check_validity(centers, angles, center, side, angle):
                if side < best_side:
                    best_side = side
                    best_result = (centers, angles, center, side, angle)

    if best_result is None:
        fb_centers, fb_angles, fb_center, fb_side, fb_angle = _get_valid_fallback()
        best_result = (fb_centers, fb_angles, fb_center, fb_side, fb_angle)
        
    centers, inner_angles, outer_center, outer_side, outer_angle = best_result
    
    inner = [[float(centers[i][0]), float(centers[i][1]), float(inner_angles[i])] for i in range(N)]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
