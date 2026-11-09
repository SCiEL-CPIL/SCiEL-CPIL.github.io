import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 11
_DEADLINE = 140.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, theta, side=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with rotation theta."""
    angles = theta + np.pi / 3 * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

def _hex_edge_normals(theta):
    """Return 6 outward normal vectors of a regular hexagon with rotation theta."""
    angles = theta + np.pi / 6 + np.pi / 3 * np.arange(6)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _min_enclosing_side_lp(centers, inner_angles, outer_angle):
    """
    Compute minimum side length of a regular hexagon (fixed orientation)
    enclosing all inner hexagons using Linear Programming.
    Returns (side, center).
    """
    sqrt3 = np.sqrt(3)
    normals = _hex_edge_normals(outer_angle)
    
    # Collect all vertices
    all_points = []
    for i in range(len(centers)):
        v = _hex_vertices(centers[i, 0], centers[i, 1], inner_angles[i], 1.0)
        all_points.append(v)
    all_points = np.vstack(all_points)
    
    # Projections onto each normal direction
    projections = all_points @ normals.T
    A = projections.max(axis=0)  # max projection in each normal direction
    
    # Variables: x = [cx, cy, r] where r is the inradius (distance from center to side)
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
        
    res = linprog(c_obj, A_ub=A_mat, b_ub=b_vec, method='highs')
    
    if res.status != 0:
        return float('inf'), np.array([0.0, 0.0])
        
    cx, cy, r = res.x
    side = 2 * r / sqrt3
    return side, np.array([cx, cy])

def _hex_overlap_sat(c1, a1, c2, a2, tol=1e-9):
    """Check if two unit hexagons overlap using Separating Axis Theorem."""
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

def _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle, tol=1e-7):
    """Check containment and non-overlap."""
    sqrt3 = np.sqrt(3)
    outer_normals = _hex_edge_normals(outer_angle)
    inradius = outer_side * sqrt3 / 2
    
    # Check containment
    for i in range(len(centers)):
        verts = _hex_vertices(centers[i, 0], centers[i, 1], inner_angles[i], 1.0)
        for vx, vy in verts:
            for nx, ny in outer_normals:
                dist = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny
                if dist > inradius + tol:
                    return False
    
    # Check non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_overlap_sat(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                return False
    
    return True

def _optimize_outer(centers, inner_angles):
    """Find optimal outer hexagon for fixed inner positions."""
    best_side = float('inf')
    best_center = np.array([0.0, 0.0])
    best_angle = 0.0
    
    # Sweep angle finely
    for outer_angle in np.linspace(0, np.pi/3, 61):
        side, center = _min_enclosing_side_lp(centers, inner_angles, outer_angle)
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = outer_angle
    
    return best_side, best_center, best_angle

def _objective(v):
    """
    v: [x1, y1, a1, ..., xN, yN, aN, cx, cy, theta]
    """
    hex_vars = v[:3*N]
    outer_vars = v[3*N:]
    
    positions = np.zeros((N, 2))
    angles = np.zeros(N)
    for i in range(N):
        positions[i, 0] = hex_vars[i*3]
        positions[i, 1] = hex_vars[i*3+1]
        angles[i] = hex_vars[i*3+2]
        
    cx = outer_vars[0]
    cy = outer_vars[1]
    theta = outer_vars[2]
    
    outer_center = np.array([cx, cy])
    
    # Compute side length using LP
    side, _ = _min_enclosing_side_lp(positions, angles, theta)
    
    # Overlap penalty using SAT-based check
    pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            if _hex_overlap_sat(positions[i], angles[i], positions[j], angles[j]):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < 2.0:
                    pen += (2.0 - d) ** 2
    
    return side + 10.0 * pen

def _generate_lattice_configs():
    """Generate hexagonal lattice configurations."""
    configs = []
    
    # Hex grid coordinates in axial system
    axial_coords = [(0,0)]
    # Ring 1: 6 hexes
    axial_coords += [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    # Ring 2: first 4 of 12
    axial_coords += [(2,0), (1,1), (0,2), (-1,2)]
    
    selected_axial = axial_coords[:N]
    
    # Convert to Cartesian
    lattice_centers = []
    for q, r in selected_axial:
        x = 1.5 * q
        y = np.sqrt(3) * (r + q/2)
        lattice_centers.append((x, y))
        
    positions = np.array(lattice_centers)
    angles = np.zeros(N)
    cx = np.mean(positions[:, 0])
    cy = np.mean(positions[:, 1])
    theta = 0.0
    
    v = np.concatenate([
        np.array([[x, y, a] for x, y, a in zip(positions[:,0], positions[:,1], angles)]).flatten(),
        [cx, cy, theta]
    ])
    configs.append(v)
    
    # Random perturbations
    np.random.seed(42)
    for _ in range(8):
        v_perturbed = v + np.random.randn(len(v)) * 0.15
        configs.append(v_perturbed)
        
    return configs

def _force_directed_relax(centers, angles, outer_side, outer_center, outer_angle, max_iter=300):
    """
    Force-directed relaxation: push overlapping hexagons apart.
    """
    centers = [list(c) for c in centers]
    angles = list(angles)
    sqrt3 = np.sqrt(3)
    outer_normals = _hex_edge_normals(outer_angle)
    inradius = outer_side * sqrt3 / 2
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
            
        displacements = [[0.0, 0.0] for _ in range(N)]
        
        # Repulsion between overlapping hexagons
        for i in range(N):
            for j in range(i + 1, N):
                if _hex_overlap_sat(centers[i], angles[i], centers[j], angles[j]):
                    dx = centers[j][0] - centers[i][0]
                    dy = centers[j][1] - centers[i][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < 1e-12:
                        dx, dy = 1.0, 0.0
                        dist = 1.0
                    else:
                        dx /= dist
                        dy /= dist
                    
                    # Push apart
                    push = 0.01
                    displacements[i][0] -= dx * push
                    displacements[i][1] -= dy * push
                    displacements[j][0] += dx * push
                    displacements[j][1] += dy * push
        
        # Apply displacements
        for i in range(N):
            centers[i][0] += displacements[i][0]
            centers[i][1] += displacements[i][1]
            
            # Check if outside boundary, pull back
            for nx, ny in outer_normals:
                dist = (centers[i][0] - outer_center[0]) * nx + (centers[i][1] - outer_center[1]) * ny
                if dist > inradius - 1.0:
                    # Pull back toward center
                    centers[i][0] -= nx * 0.01
                    centers[i][1] -= ny * 0.01
    
    return np.array(centers), angles

def construct():
    start_time = time.time()
    best_result = None
    best_side = float('inf')
    
    configs = _generate_lattice_configs()
    
    for v0 in configs:
        if time.time() > _DEADLINE - 10:
            break
            
        try:
            res = minimize(_objective, v0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-8})
            
            if res.fun < best_side:
                v_opt = res.x
                hex_vars = v_opt[:3*N]
                outer_vars = v_opt[3*N:]
                
                positions = np.zeros((N, 2))
                angles = np.zeros(N)
                for i in range(N):
                    positions[i, 0] = hex_vars[i*3]
                    positions[i, 1] = hex_vars[i*3+1]
                    angles[i] = hex_vars[i*3+2]
                    
                cx = outer_vars[0]
                cy = outer_vars[1]
                theta = outer_vars[2]
                
                # Use LP to find optimal outer hexagon
                side, center = _min_enclosing_side_lp(positions, angles, theta)
                
                # Verify validity
                if _check_validity(positions, angles, center, side, theta):
                    if side < best_side:
                        best_side = side
                        inner = [[positions[i, 0], positions[i, 1], np.degrees(angles[i])] for i in range(N)]
                        best_result = (inner, (center[0], center[1]), side, np.degrees(theta))
                        
        except Exception as e:
            pass
    
    # If we have a valid result, try to improve with force-directed relaxation
    if best_result is not None and time.time() < _DEADLINE - 15:
        inner, center, side, angle_deg = best_result
        centers = np.array([[p[0], p[1]] for p in inner])
        angles = np.array([p[2] for p in inner])
        outer_center = np.array(center)
        outer_angle = np.radians(angle_deg)
        
        relaxed_centers, relaxed_angles = _force_directed_relax(
            centers, angles, side, outer_center, outer_angle, max_iter=200
        )
        
        new_side, new_center = _min_enclosing_side_lp(relaxed_centers, relaxed_angles, outer_angle)
        
        if _check_validity(relaxed_centers, relaxed_angles, new_center, new_side, outer_angle):
            if new_side < best_side:
                best_side = new_side
                inner = [[relaxed_centers[i, 0], relaxed_centers[i, 1], np.degrees(relaxed_angles[i])] for i in range(N)]
                best_result = (inner, (new_center[0], new_center[1]), new_side, np.degrees(outer_angle))
    
    if best_result is None:
        # Fallback: simple linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(N)]
        return inner, (0.0, 0.0), 50.0, 0.0
        
    return best_result
