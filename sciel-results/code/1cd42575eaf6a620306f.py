import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12
SQRT3 = math.sqrt(3.0)

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts[i, 0] = cx + side * math.cos(theta)
        verts[i, 1] = cy + side * math.sin(theta)
    return verts

def _outer_face_normals(angle_deg=0.0):
    """Return 6 unit face normals of outer hexagon (perpendicular to edges)."""
    a = math.radians(angle_deg)
    normals = np.empty((6, 2))
    for i in range(6):
        theta = a + math.pi / 6.0 + i * math.pi / 3.0
        normals[i, 0] = math.cos(theta)
        normals[i, 1] = math.sin(theta)
    return normals

def _max_projection(verts_list, normals):
    """Maximum projection of all vertices onto the 6 face normals."""
    max_proj = 0.0
    for verts in verts_list:
        projs = verts @ normals.T  # shape (6, 6)
        max_proj = max(max_proj, float(np.max(np.abs(projs))))
    return max_proj

def _min_outer_side(verts_list, outer_angle_deg=0.0):
    """Minimum outer side length to contain all hexagons."""
    normals = _outer_face_normals(outer_angle_deg)
    max_proj = _max_projection(verts_list, normals)
    # inradius = S * sqrt(3)/2, so S = 2*max_proj/sqrt(3)
    return 2.0 * max_proj / SQRT3

def _sat_overlap_depth(v1, v2, tol=1e-9):
    """Maximum overlap depth between two convex polygons. 0 if no overlap."""
    min_overlap = float('inf')
    for poly in (v1, v2):
        n = len(poly)
        for i in range(n):
            j = (i + 1) % n
            edge = poly[j] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            p1 = v1 @ normal
            p2 = v2 @ normal
            min1, max1 = p1.min(), p1.max()
            min2, max2 = p2.min(), p2.max()
            if max1 < min2 + tol or max2 < min1 + tol:
                return 0.0
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < min_overlap:
                min_overlap = overlap
    return min_overlap if min_overlap < float('inf') else 0.0

def _objective(params, S_target=None):
    """
    Objective: minimize outer side + overlap penalty.
    If S_target is given, we check if all hexagons fit in outer hex of side S_target.
    """
    positions = np.array(params).reshape(N, 3)
    xs = positions[:, 0]
    ys = positions[:, 1]
    thetas = positions[:, 2]
    
    verts_list = []
    for i in range(N):
        v = _hex_vertices(xs[i], ys[i], math.degrees(thetas[i]), 1.0)
        verts_list.append(v)
    
    # Overlap penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(verts_list[i], verts_list[j])
            if depth > 0:
                penalty += depth * depth
    
    if S_target is not None:
        # Check containment in outer hex of side S_target
        inradius = S_target * SQRT3 / 2.0
        normals = _outer_face_normals(0.0)
        containment_violation = 0.0
        for verts in verts_list:
            projs = verts @ normals.T
            # Each projection must be <= inradius in absolute value
            viol = np.max(np.abs(projs)) - inradius
            if viol > 0:
                containment_violation += viol * viol
        return 1000.0 * penalty + 1000.0 * containment_violation
    else:
        S = _min_outer_side(verts_list, 0.0)
        return S + 50.0 * penalty

def _generate_honeycomb_guess():
    """Generate a honeycomb-like initial guess."""
    guess = np.zeros((N, 3))
    # Center hexagon
    guess[0] = [0, 0, 0]
    # Ring of 6 at distance sqrt(3)
    for i in range(6):
        angle = i * math.pi / 3.0
        guess[i+1, 0] = SQRT3 * math.cos(angle)
        guess[i+1, 1] = SQRT3 * math.sin(angle)
        guess[i+1, 2] = 0.0
    # 5 more in outer ring, pick 5 of 12 positions at distance 2*sqrt(3)
    for i in range(5):
        angle = i * math.pi / 3.0
        guess[i+7, 0] = 2*SQRT3 * math.cos(angle)
        guess[i+7, 1] = 2*SQRT3 * math.sin(angle)
        guess[i+7, 2] = 0.0
    return guess

def _generate_grid_guess():
    """Generate a grid-like initial guess."""
    guess = np.zeros((N, 3))
    idx = 0
    # 4 columns, 3 rows in hex lattice
    for col in range(4):
        for row in range(3):
            x = col * 1.5 + (row % 2) * 0.75
            y = row * SQRT3 / 2.0
            guess[idx, 0] = x
            guess[idx, 1] = y
            guess[idx, 2] = 0.0
            idx += 1
    # Center
    guess[:, 0] -= np.mean(guess[:, 0])
    guess[:, 1] -= np.mean(guess[:, 1])
    return guess

def _generate_compact_guess():
    """Generate a compact cluster guess with some rotation."""
    guess = np.zeros((N, 3))
    # Place in a tighter configuration
    positions = [
        (0, 0, 0),
        (SQRT3, 0, 0), (-SQRT3, 0, 0),
        (SQRT3/2, 1.5, 0), (SQRT3/2, -1.5, 0),
        (-SQRT3/2, 1.5, 0), (-SQRT3/2, -1.5, 0),
        (2*SQRT3, 0, 0), (0, 3, 0), (0, -3, 0),
        (SQRT3, 3, 0), (-SQRT3, -3, 0)
    ]
    for i, (x, y, a) in enumerate(positions):
        guess[i] = [x, y, a]
    guess[:, 0] -= np.mean(guess[:, 0])
    guess[:, 1] -= np.mean(guess[:, 1])
    return guess

def _local_optimize(guess, deadline, max_iter=300):
    """Run local optimization from a starting guess."""
    bounds = [(-8, 8), (-8, 8), (-math.pi, math.pi)] * N
    try:
        res = minimize(
            _objective,
            guess.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1000}
        )
        if time.time() > deadline:
            return None
        return res.x
    except Exception:
        return None

def _refine_with_bisection(params, deadline):
    """Given a valid packing, try to tighten the outer hexagon via bisection on S."""
    # First, compute the current S
    positions = np.array(params).reshape(N, 3)
    verts_list = []
    for i in range(N):
        v = _hex_vertices(positions[i, 0], positions[i, 1], math.degrees(positions[i, 2]), 1.0)
        verts_list.append(v)
    
    # Check validity
    for i in range(N):
        for j in range(i + 1, N):
            if _sat_overlap_depth(verts_list[i], verts_list[j]) > 1e-6:
                return params, _min_outer_side(verts_list, 0.0)
    
    S_current = _min_outer_side(verts_list, 0.0)
    
    # Try to optimize positions to reduce S further
    best_params = params
    best_S = S_current
    
    for attempt in range(5):
        if time.time() > deadline:
            break
        try:
            res = minimize(
                lambda p: _objective(p),
                best_params,
                method='L-BFGS-B',
                bounds=[(-8, 8), (-8, 8), (-math.pi, math.pi)] * N,
                options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-11, 'maxfun': 500}
            )
            new_params = res.x
            new_positions = new_params.reshape(N, 3)
            new_verts = []
            for i in range(N):
                v = _hex_vertices(new_positions[i, 0], new_positions[i, 1], math.degrees(new_positions[i, 2]), 1.0)
                new_verts.append(v)
            
            # Check validity
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    if _sat_overlap_depth(new_verts[i], new_verts[j]) > 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                new_S = _min_outer_side(new_verts, 0.0)
                if new_S < best_S:
                    best_S = new_S
                    best_params = new_params
        except Exception:
            break
    
    return best_params, best_S

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit
    
    best_S = float('inf')
    best_params = None
    
    # Generate initial guesses
    guesses = [
        _generate_honeycomb_guess(),
        _generate_grid_guess(),
        _generate_compact_guess(),
    ]
    
    # Add some randomized guesses
    np.random.seed(42)
    for _ in range(3):
        g = np.random.uniform(-3, 3, (N, 3))
        g[:, 2] = np.random.uniform(-math.pi, math.pi, N)
        guesses.append(g)
    
    for guess in guesses:
        if time.time() > deadline:
            break
        
        result = _local_optimize(guess, deadline, max_iter=200)
        if result is None:
            continue
        
        # Refine
        refined_params, S = _refine_with_bisection(result, deadline)
        
        if S < best_S:
            best_S = S
            best_params = refined_params
    
    # Final refinement with more iterations if time permits
    if best_params is not None and time.time() < deadline - 10:
        refined_params, S = _refine_with_bisection(best_params, deadline)
        if S < best_S:
            best_S = S
            best_params = refined_params
    
    # Fallback if no valid solution
    if best_params is None:
        inner = []
        for i in range(N):
            inner.append([i * 2.5, 0.0, 0.0])
        return inner, (0.0, 0.0), 25.0, 0.0
    
    # Format output
    inner = []
    for i in range(N):
        x, y, theta_rad = best_params[i]
        theta_deg = math.degrees(theta_rad) % 360.0
        inner.append([float(x), float(y), float(theta_deg)])
    
    return inner, (0.0, 0.0), float(best_S), 0.0

# EVOLVE-BLOCK-END
