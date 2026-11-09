import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 11
DEADLINE = 135.0  # safety margin before 150s kill
SQRT3 = np.sqrt(3)

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy)."""
    angle = np.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = angle + np.pi * i / 3.0
        verts[i, 0] = cx + side * np.cos(theta)
        verts[i, 1] = cy + side * np.sin(theta)
    return verts

def _hex_edge_normals(angle_deg):
    """Return 6 outward normal vectors of a regular hexagon."""
    angle = np.radians(angle_deg)
    normals = np.empty((6, 2))
    for i in range(6):
        mid_angle = angle + np.pi / 6.0 + np.pi * i / 3.0
        normals[i, 0] = np.cos(mid_angle)
        normals[i, 1] = np.sin(mid_angle)
    return normals

def _min_enclosing_side_lp(centers, inner_angles, outer_angle_deg):
    """
    Compute minimum side length of a regular hexagon (fixed orientation)
    enclosing all inner hexagons using Linear Programming.
    Returns (side, center).
    """
    normals = _hex_edge_normals(outer_angle_deg)
    
    # Collect all vertices
    all_points = np.vstack([_hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i]) 
                            for i in range(len(centers))])
    
    # A_i = max projection in direction n_i
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
        
    # Use scipy linprog
    from scipy.optimize import linprog
    res = linprog(c_obj, A_ub=A_mat, b_ub=b_vec, method='highs')
    
    if res.status != 0:
        return float('inf'), np.array([0.0, 0.0])
        
    cx, cy, r = res.x
    side = 2 * r / SQRT3
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

def _hex_min_distance(c1, a1, c2, a2):
    """
    Compute the minimum distance between two hexagons (positive if separated,
    negative if overlapping). Uses SAT to find minimum separation.
    """
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
    min_sep = float('inf')
    
    for ax, ay in axes:
        proj1 = v1[:, 0] * ax + v1[:, 1] * ay
        proj2 = v2[:, 0] * ax + v2[:, 1] * ay
        sep = min(proj1.max(), proj2.max()) - max(proj1.min(), proj2.min())
        min_sep = min(min_sep, sep)
    
    return -min_sep  # negative if overlapping

def _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle_deg, tol=1e-7):
    """Check containment and non-overlap."""
    outer_normals = _hex_edge_normals(outer_angle_deg)
    inradius = outer_side * SQRT3 / 2
    
    for i in range(len(centers)):
        verts = _hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i])
        for vx, vy in verts:
            for nx, ny in outer_normals:
                dist = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny
                if dist > inradius + tol:
                    return False
    
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                return False
    
    return True

def _optimize_outer(centers, inner_angles):
    """Find optimal outer hexagon for fixed inner positions."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Sweep angle finely
    for outer_angle in np.linspace(0, 60, 121):
        side, center = _min_enclosing_side_lp(centers, inner_angles, outer_angle)
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = outer_angle
    
    return best_side, best_center, best_angle

def _try_arrangement(centers, inner_angles=None):
    """Try an arrangement and return (side, result) or (inf, None)."""
    if inner_angles is None:
        inner_angles = [0.0] * N
    
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                return float('inf'), None
    
    # Find optimal outer hexagon
    side, center, angle = _optimize_outer(centers, inner_angles)
    
    # Verify validity
    if _check_validity(centers, inner_angles, center, side, angle):
        return side, (centers, inner_angles, center, side, angle)
    
    return float('inf'), None

def _get_valid_fallback():
    """Return a known valid, non-overlapping configuration."""
    centers = []
    angles = [0.0] * N
    # Use a sparse arrangement
    for i in range(N):
        angle = 2 * np.pi * i / N
        r = 3.0 + 0.5 * (i % 3)
        centers.append((r * np.cos(angle), r * np.sin(angle)))
    
    side, center, angle = _optimize_outer(centers, angles)
    return centers, angles, center, side, angle

def _contact_graph_optimize(centers, angles, max_time=60.0):
    """
    Optimize using a contact-graph based approach with SLSQP.
    The cost function is the outer hexagon side length, with penalties for overlaps.
    """
    start_time = time.time()
    
    # Flatten parameters: x0, y0, a0, x1, y1, a1, ...
    x0 = np.array([c[0] for c in centers] + [c[1] for c in centers] + angles)
    
    def cost(params):
        if time.time() - start_time > max_time:
            return float('inf')
        
        centers_local = []
        angles_local = []
        for i in range(N):
            cx = params[3*i]
            cy = params[3*i+1]
            ang = params[3*i+2]
            centers_local.append((cx, cy))
            angles_local.append(ang)
        
        # Compute outer hexagon
        side, outer_center, outer_angle = _optimize_outer(centers_local, angles_local)
        if not np.isfinite(side):
            return 1e6
        
        # Penalty for overlaps
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist = _hex_min_distance(centers_local[i], angles_local[i], 
                                        centers_local[j], angles_local[j])
                if dist < 0:
                    penalty += dist * dist
        
        # Penalty for containment violations
        outer_normals = _hex_edge_normals(outer_angle)
        inradius = side * SQRT3 / 2
        for i in range(N):
            verts = _hex_vertices(centers_local[i][0], centers_local[i][1], 1.0, angles_local[i])
            for vx, vy in verts:
                for nx, ny in outer_normals:
                    d = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny
                    if d > inradius:
                        penalty += (d - inradius) * 10
        
        return side + 100 * penalty
    
    # Initial cost
    current_cost = cost(x0)
    
    # Use SLSQP
    result = minimize(cost, x0, method='SLSQP', options={'maxiter': 200, 'ftol': 1e-6})
    
    if result.success and np.isfinite(result.fun):
        best_centers = []
        best_angles = []
        for i in range(N):
            best_centers.append((result.x[3*i], result.x[3*i+1]))
            best_angles.append(result.x[3*i+2])
        
        side, center, angle = _optimize_outer(best_centers, best_angles)
        if _check_validity(best_centers, best_angles, center, side, angle):
            return best_centers, best_angles, side, center, angle
    
    return None

def _simulate_annealing(initial_centers, initial_angles, max_time=60.0):
    """
    Simulated Annealing with better perturbation strategy.
    """
    start_time = time.time()
    centers = [list(c) for c in initial_centers]
    angles = list(initial_angles)
    
    # Initial cost
    current_side, current_center, current_angle = _optimize_outer(centers, angles)
    if not np.isfinite(current_side):
        return initial_centers, initial_angles, float('inf'), None, None
        
    best_side = current_side
    best_centers = [list(c) for c in centers]
    best_angles = list(angles)
    best_center = current_center
    best_angle = current_angle
    
    T = 0.3
    T_min = 0.005
    cooling_rate = 0.9995
    
    np.random.seed(42)
    
    while time.time() - start_time < max_time:
        if T < T_min:
            break
            
        # Generate new state
        new_centers = [list(c) for c in centers]
        new_angles = list(angles)
        
        # Random perturbation - move 1-2 hexagons
        n_moves = np.random.randint(1, 3)
        for _ in range(n_moves):
            idx = np.random.randint(N)
            scale = 0.05 + 0.1 * T / 0.3
            delta_x = np.random.normal(0, scale)
            delta_y = np.random.normal(0, scale)
            new_centers[idx][0] += delta_x
            new_centers[idx][1] += delta_y
        
        # Sometimes rotate
        if np.random.rand() < 0.3:
            rot_idx = np.random.randint(N)
            new_angles[rot_idx] += np.random.normal(0, 3.0)
        
        # Check validity (overlaps)
        valid = True
        for i in range(N):
            for j in range(i + 1, N):
                if _hex_overlap(new_centers[i], new_angles[i], new_centers[j], new_angles[j]):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            T *= cooling_rate
            continue
            
        # Calculate new cost
        new_side, new_center, new_angle = _optimize_outer(new_centers, new_angles)
        
        if not np.isfinite(new_side):
            T *= cooling_rate
            continue
            
        delta = new_side - current_side
        
        if delta < 0:
            centers = new_centers
            angles = new_angles
            current_side = new_side
            current_center = new_center
            current_angle = new_angle
            
            if current_side < best_side:
                best_side = current_side
                best_centers = [list(c) for c in centers]
                best_angles = list(angles)
                best_center = current_center
                best_angle = current_angle
        else:
            prob = np.exp(-delta / T)
            if np.random.rand() < prob:
                centers = new_centers
                angles = new_angles
                current_side = new_side
                current_center = new_center
                current_angle = new_angle
                
        T *= cooling_rate
        
    return best_centers, best_angles, best_side, best_center, best_angle

def _generate_diverse_starts():
    """Generate diverse initial configurations."""
    starts = []
    
    # 1. Hexagonal lattice with center
    d1 = 2 * SQRT3
    first_ring = [(d1 * np.cos(np.pi/3*i), d1 * np.sin(np.pi/3*i)) for i in range(6)]
    
    # Second ring positions
    d2 = 4 * SQRT3
    second_ring = []
    for i in range(6):
        a = np.pi/3*i
        second_ring.append((d2 * np.cos(a), d2 * np.sin(a)))
        a2 = np.pi/3*i + np.pi/6
        second_ring.append((d2 * np.cos(a2), d2 * np.sin(a2)))
    
    # Various subsets of second ring
    subsets = [
        [0, 2, 4, 6],
        [1, 3, 5, 7],
        [0, 3, 6, 9],
        [0, 2, 5, 7],
        [1, 4, 7, 10],
        [0, 1, 8, 9],
    ]
    
    for subset in subsets:
        centers = [(0.0, 0.0)] + first_ring + [second_ring[i] for i in subset]
        angles = [0.0] * N
        starts.append((centers, angles))
    
    # 2. Ring without center
    ring_centers = first_ring + [second_ring[i] for i in [0, 2, 4, 6, 8, 10]]
    angles = [0.0] * N
    starts.append((ring_centers, angles))
    
    # 3. Linear-ish arrangement
    centers = []
    for i in range(N):
        x = (i - N//2) * 2.2
        y = 0.3 * (i % 3 - 1)
        centers.append((x, y))
    angles = [0.0] * N
    starts.append((centers, angles))
    
    # 4. Random valid-ish start
    np.random.seed(123)
    centers = []
    for i in range(N):
        angle = 2 * np.pi * i / N + np.random.normal(0, 0.2)
        r = 2.5 + np.random.normal(0, 0.5)
        centers.append((r * np.cos(angle), r * np.sin(angle)))
    angles = [np.random.normal(0, 10) for _ in range(N)]
    starts.append((centers, angles))
    
    return starts

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_result = None
    
    # Generate diverse starts
    starts = _generate_diverse_starts()
    
    # Try each start with SA
    for centers, angles in starts:
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # Quick check if start is valid
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if _hex_overlap(centers[i], angles[i], centers[j], angles[j]):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Run SA from this start
        sa_centers, sa_angles, sa_side, sa_center, sa_angle = _simulate_annealing(
            centers, angles, max_time=DEADLINE - time.time() - 15
        )
        
        if _check_validity(sa_centers, sa_angles, sa_center, sa_side, sa_angle):
            if sa_side < best_side:
                best_side = sa_side
                best_result = (sa_centers, sa_angles, sa_center, sa_side, sa_angle)
        
        # Also try contact graph optimization from best found so far
        if best_result is not None and time.time() - start_time < DEADLINE - 10:
            cg_result = _contact_graph_optimize(
                best_result[0], best_result[1], 
                max_time=DEADLINE - time.time() - 5
            )
            if cg_result is not None:
                cg_centers, cg_angles, cg_side, cg_center, cg_angle = cg_result
                if cg_side < best_side:
                    best_side = cg_side
                    best_result = (cg_centers, cg_angles, cg_center, cg_side, cg_angle)
    
    # Final fallback
    if best_result is None:
        fb_centers, fb_angles, fb_center, fb_side, fb_angle = _get_valid_fallback()
        best_result = (fb_centers, fb_angles, fb_center, fb_side, fb_angle)
        
    centers, inner_angles, outer_center, outer_side, outer_angle = best_result
    
    # Final verification
    if not _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle):
        fb_centers, fb_angles, fb_center, fb_side, fb_angle = _get_valid_fallback()
        centers, inner_angles, outer_center, outer_side, outer_angle = fb_centers, fb_angles, fb_center, fb_side, fb_angle

    inner = [[float(centers[i][0]), float(centers[i][1]), float(inner_angles[i])] for i in range(N)]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
