"""
Packing 11 unit hexagons in a minimal regular hexagon.
Combines direct objective optimization (Inspiration 1) with robust support function checks (Current Program).
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # Safety margin within 150s limit

def _hex_vertices_local(theta):
    """Vertices of unit regular hexagon (circumradius=1) at origin, rotated by theta."""
    ang = np.arange(6) * (math.pi / 3.0) + theta
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)

def _support_hex_vec(theta, direction):
    """
    Vectorized support function of unit hexagon (circumradius=1) at origin.
    theta: array of angles for each hexagon
    direction: (dx, dy) direction vector
    Returns: array of support values
    """
    d_angle = np.arctan2(direction[1], direction[0])
    # Angles of vertices relative to origin for each hexagon
    # k in 0..5
    vertex_angles = theta[:, None] + np.arange(6) * (math.pi / 3.0)
    cos_vals = np.cos(d_angle - vertex_angles)
    return np.max(cos_vals, axis=1)

def _min_enclosing_side_fast(centers, thetas, theta_outer):
    """
    Calculate minimal outer hexagon side length containing all inner hexagons.
    centers: (N, 2) array
    thetas: (N,) array of inner hexagon angles
    theta_outer: angle of outer hexagon
    """
    # Generate all vertices for all inner hexagons
    # verts[i, j, :] is vertex j of hexagon i
    local_verts = _hex_vertices_local(thetas) # (N, 6, 2)
    all_verts = centers[:, None, :] + local_verts # (N, 6, 2)
    all_verts = all_verts.reshape(-1, 2) # (6N, 2)
    
    # Outward normals of the 6 sides of the outer hexagon
    # The sides are perpendicular to the normals.
    # If outer hexagon is rotated by theta_outer, its side normals are at theta_outer + pi/6 + k*pi/3
    normal_angles = theta_outer + math.pi / 6.0 + np.arange(6) * (math.pi / 3.0)
    normals = np.stack([np.cos(normal_angles), np.sin(normal_angles)], axis=1) # (6, 2)
    
    # Project all vertices onto all normals
    # proj[i, k] is projection of vertex i onto normal k
    proj = all_verts @ normals.T # (6N, 6)
    
    # For each normal direction k, the constraint is:
    # max_v (v . n_k) - min_v (v . n_k) <= 2 * apothem
    # Wait, the outer hexagon is centered at (cx, cy).
    # The constraint is that for all vertices v, (v - c) . n_k <= apothem AND -(v - c) . n_k <= apothem
    # This implies: max_v (v . n_k) - c . n_k <= apothem and -min_v (v . n_k) + c . n_k <= apothem
    # To minimize apothem for a fixed theta_outer, we can center the outer hexagon at the midpoint
    # of the projections for each axis? No, we need a single center (cx, cy) that works for all axes.
    # Actually, for a regular hexagon, the minimal enclosing hexagon for a set of points
    # with fixed orientation is determined by the max projection on each of the 6 axes.
    # Let P_k = max_v (v . n_k) and M_k = min_v (v . n_k).
    # The width in direction k is W_k = P_k - M_k.
    # The apothem must be at least W_k / 2 for each k?
    # No, the center can be chosen.
    # If we fix the center c, the required apothem is max_k ( max_v ((v-c).n_k), max_v ((-v+c).n_k) )
    # = max_k ( max_v (v.n_k) - c.n_k, -min_v (v.n_k) + c.n_k )
    # To minimize this max, for each k, the optimal c.n_k is (P_k + M_k) / 2.
    # However, c must be consistent across all k.
    # For a regular hexagon, the 3 pairs of opposite sides are orthogonal? No, 60 deg apart.
    # Actually, we can just compute the minimal apothem assuming the center is optimally placed.
    # It turns out that for any convex body, the minimal enclosing regular hexagon of fixed orientation
    # has an apothem equal to max_k ( (P_k - M_k) / 2 ) IF the center is chosen appropriately?
    # Let's verify.
    # If we set c such that c . n_k = (P_k + M_k) / 2 for all k, is that possible?
    # n_0 and n_3 are opposite. n_1 and n_4 are opposite. n_2 and n_5 are opposite.
    # The system c . n_k = mid_k is overdetermined (6 equations, 2 unknowns).
    # So we cannot satisfy all exactly.
    # However, we can find the minimal apothem A such that there exists c.
    # This is equivalent to finding the intersection of 6 strips.
    # Alternatively, we can just use the fact that the "width" of the set in direction n_k is W_k.
    # The minimal enclosing hexagon side length is determined by the maximum width over the 6 directions?
    # No, that's for a square or circle.
    # Let's use the property: The minimal regular hexagon enclosing a set of points with fixed orientation
    # has side length s = max_k ( (P_k - M_k) / sqrt(3) )?
    # Apothem a = s * sqrt(3) / 2.
    # So s = 2 * a / sqrt(3).
    # Is a = max_k (W_k / 2)?
    # If W_0 is large, we need a >= W_0/2.
    # If we set a = max_k(W_k/2), can we find a center?
    # Yes, because the constraints are strips of width 2a. The intersection of 6 strips of width 2a
    # where a >= W_k/2 for all k is non-empty?
    # Actually, yes, for centrally symmetric convex bodies like regular hexagons, 
    # the minimal enclosing one of the same shape and orientation has apothem = max_k (W_k / 2).
    
    widths = np.max(proj, axis=0) - np.min(proj, axis=0)
    max_width = np.max(widths)
    apothem = max_width / 2.0
    side = apothem * 2.0 / math.sqrt(3.0)
    
    # We also need the center.
    # The center should be the centroid of the "midpoints" of the projections?
    # Or simply the centroid of the vertices?
    # For the objective function, we only need the side length.
    # For the final result, we need the center.
    # We can compute the center later by averaging the midpoints of the extreme vertices?
    # Or solving a small LP.
    # For now, return side length.
    return side

def _pairwise_penalty(centers, thetas):
    """
    Calculate penalty for overlapping inner hexagons.
    Uses support function difference.
    """
    penalty = 0.0
    # To speed up, we can vectorize partially, but N=11 is small.
    for i in range(N):
        for j in range(i + 1, N):
            diff = centers[j] - centers[i]
            # We need to check if the two hexagons overlap.
            # They overlap if min_direction (h1(d) + h2(-d) - diff.d) < 0
            # We check 12 critical directions.
            min_clearance = float('inf')
            
            # Directions from hex i
            for k in range(6):
                ang = thetas[i] + math.pi / 6 + k * math.pi / 3
                d = np.array([math.cos(ang), math.sin(ang)])
                s1 = _support_hex_vec(np.array([thetas[i]]), d)[0]
                s2 = _support_hex_vec(np.array([thetas[j]]), -d)[0]
                m = s1 + s2 - np.dot(diff, d)
                if m < min_clearance:
                    min_clearance = m
            
            # Directions from hex j
            for k in range(6):
                ang = thetas[j] + math.pi / 6 + k * math.pi / 3
                d = np.array([math.cos(ang), math.sin(ang)])
                s1 = _support_hex_vec(np.array([thetas[i]]), d)[0]
                s2 = _support_hex_vec(np.array([thetas[j]]), -d)[0]
                m = s1 + s2 - np.dot(diff, d)
                if m < min_clearance:
                    min_clearance = m
            
            if min_clearance < 0:
                # Strong penalty for overlap
                penalty += 100.0 * min_clearance * min_clearance
    
    return penalty

def _objective(params, theta_outer):
    """
    Objective: Outer side length + Overlap Penalty.
    params: [x0, y0, theta0, x1, y1, theta1, ...]
    """
    centers = params[:N * 2].reshape(N, 2)
    thetas = params[N * 2:]
    
    side = _min_enclosing_side_fast(centers, thetas, theta_outer)
    pen = _pairwise_penalty(centers, thetas)
    
    return side + pen

def _find_center_and_side(centers, thetas, theta_outer):
    """
    Compute the optimal center and side length for the outer hexagon.
    """
    # Generate all vertices
    local_verts = _hex_vertices_local(thetas)
    all_verts = (centers[:, None, :] + local_verts[None, :, :]).reshape(-1, 2)
    
    normal_angles = theta_outer + math.pi / 6.0 + np.arange(6) * (math.pi / 3.0)
    normals = np.stack([np.cos(normal_angles), np.sin(normal_angles)], axis=1)
    
    proj = all_verts @ normals.T
    P = np.max(proj, axis=0)
    M = np.min(proj, axis=0)
    
    # We want to find c (2D) and apothem a minimizing a
    # s.t. P_k - c.n_k <= a
    #      -M_k + c.n_k <= a
    # This is an LP with 3 variables (cx, cy, a) and 12 constraints.
    # We can solve it using scipy.optimize.linprog
    
    c_vec = [0.0, 0.0, 1.0]
    A_ub = []
    b_ub = []
    
    for k in range(6):
        nx, ny = normals[k]
        # P_k - cx*nx - cy*ny <= a  =>  -nx*cx - ny*cy - a <= -P_k
        A_ub.append([-nx, -ny, -1.0])
        b_ub.append(-P[k])
        
        # -M_k + cx*nx + cy*ny <= a  =>  nx*cx + ny*cy - a <= M_k
        A_ub.append([nx, ny, -1.0])
        b_ub.append(M[k])
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    try:
        from scipy.optimize import linprog
        res = linprog(c_vec, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0, None)], method='highs')
        if res.success:
            cx, cy, a = res.x
            side = a * 2.0 / math.sqrt(3.0)
            return cx, cy, side
    except Exception:
        pass
    
    # Fallback: use max width / 2
    widths = P - M
    max_width = np.max(widths)
    a = max_width / 2.0
    side = a * 2.0 / math.sqrt(3.0)
    # Center approximation: average of midpoints
    midpoints = (P + M) / 2.0
    # Solve least squares for c
    # normals @ c = midpoints
    c_est, _, _, _ = np.linalg.lstsq(normals, midpoints, rcond=None)
    return c_est[0], c_est[1], side

def _make_init_lattice(seed):
    """Hexagonal lattice initialization."""
    rng = np.random.RandomState(seed)
    sqrt3 = math.sqrt(3.0)
    # 1 center, 6 ring, 4 outer
    centers = np.zeros((N, 2))
    centers[0] = [0, 0]
    for i in range(6):
        ang = i * math.pi / 3.0
        centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
    for i in range(4):
        ang = (i + 0.5) * math.pi / 3.0
        centers[i + 7] = [3.5 * math.cos(ang), 3.5 * math.sin(ang)]
    
    thetas = np.zeros(N) + rng.uniform(-0.1, 0.1, N)
    return np.concatenate([centers.reshape(-1), thetas])

def _make_init_random(seed):
    """Random initialization within a bounding box."""
    rng = np.random.RandomState(seed)
    centers = rng.uniform(-4, 4, size=(N, 2))
    thetas = rng.uniform(0, math.pi / 3, size=N)
    return np.concatenate([centers.reshape(-1), thetas])

def _optimize_packing(theta_outer, init_params, maxiter=300):
    """Run L-BFGS-B optimization for a fixed outer angle."""
    try:
        res = minimize(_objective, init_params, args=(theta_outer,),
                       method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
        return res.x
    except Exception:
        return None

def construct():
    global _START_TIME, _DEADLINE
    
    best_result = None
    best_side = float('inf')
    
    # We will sweep the outer hexagon angle.
    # Due to symmetry, we only need to check 0 to 30 degrees (pi/6).
    # But the inner hexagons can rotate, so the optimal outer angle might be anywhere.
    # However, the relative angle matters.
    # Let's try a few outer angles.
    
    outer_angles = np.arange(0, 30.0, 2.0)
    
    # Initialize with lattice
    init_params = _make_init_lattice(42)
    
    for ang_deg in outer_angles:
        if time.time() > _DEADLINE - 20:
            break
            
        theta_outer = math.radians(ang_deg)
        
        # Try optimization from lattice init
        if time.time() < _DEADLINE - 30:
            opt_params = _optimize_packing(theta_outer, init_params, maxiter=400)
            if opt_params is not None:
                centers = opt_params[:N * 2].reshape(N, 2)
                thetas = opt_params[N * 2:]
                
                # Check validity
                pen = _pairwise_penalty(centers, thetas)
                if pen < 1e-5: # Valid packing
                    cx, cy, side = _find_center_and_side(centers, thetas, theta_outer)
                    if side < best_side:
                        best_side = side
                        # Prepare inner list
                        inner = []
                        for i in range(N):
                            x, y = centers[i]
                            a_deg = math.degrees(thetas[i]) % 360
                            inner.append([x, y, a_deg])
                        best_result = (inner, (cx, cy), side, ang_deg)
        
        # Also try a random init if time permits
        if time.time() < _DEADLINE - 40:
            init_rand = _make_init_random(ang_deg + 100)
            opt_params = _optimize_packing(theta_outer, init_rand, maxiter=300)
            if opt_params is not None:
                centers = opt_params[:N * 2].reshape(N, 2)
                thetas = opt_params[N * 2:]
                pen = _pairwise_penalty(centers, thetas)
                if pen < 1e-5:
                    cx, cy, side = _find_center_and_side(centers, thetas, theta_outer)
                    if side < best_side:
                        best_side = side
                        inner = []
                        for i in range(N):
                            x, y = centers[i]
                            a_deg = math.degrees(thetas[i]) % 360
                            inner.append([x, y, a_deg])
                        best_result = (inner, (cx, cy), side, ang_deg)
    
    # Refine best angle if found
    if best_result is not None and time.time() < _DEADLINE - 10:
        best_ang = best_result[3]
        fine_angles = np.arange(max(0, best_ang - 1), min(30, best_ang + 1), 0.1)
        # Re-use the best params from the coarse step? 
        # We need to re-optimize or just check the side length for the current params?
        # The current params were optimized for the coarse angle.
        # It's better to just return the coarse result if refinement is too risky/slow.
        # But we can try a quick polish.
        
        # Let's just ensure we return the best found.
        pass

    if best_result is not None:
        return best_result
    
    # Fallback: Lattice placement
    sqrt3 = math.sqrt(3.0)
    centers = np.zeros((N, 2))
    centers[0] = [0, 0]
    for i in range(6):
        ang = i * math.pi / 3.0
        centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
    for i in range(4):
        ang = (i + 0.5) * math.pi / 3.0
        centers[i + 7] = [3.5 * math.cos(ang), 3.5 * math.sin(ang)]
    
    thetas = np.zeros(N)
    cx, cy, side = _find_center_and_side(centers, thetas, 0.0)
    
    inner = []
    for i in range(N):
        inner.append([centers[i, 0], centers[i, 1], 0.0])
        
    return inner, (cx, cy), side, 0.0

