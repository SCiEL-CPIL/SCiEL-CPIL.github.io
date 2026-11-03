"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # safety margin before the 150s kill

# EVOLVE-BLOCK-START

def _hex_vertices(theta):
    """
    Returns vertices of a regular hexagon with side length 1, circumradius 1,
    centered at origin, rotated by theta.
    """
    return np.array([[math.cos(theta + k * math.pi / 3.0),
                      math.sin(theta + k * math.pi / 3.0)] for k in range(6)])

def _sat_overlap(A, B):
    """
    Fast SAT check for two convex polygons A and B (arrays of (x,y)).
    Returns True if they overlap.
    """
    # Check axes from A
    for i in range(6):
        p1 = A[i]
        p2 = A[(i + 1) % 6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        projA = A @ normal
        projB = B @ normal
        if projA.max() < projB.min() or projB.max() < projA.min():
            return False
    # Check axes from B
    for i in range(6):
        p1 = B[i]
        p2 = B[(i + 1) % 6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        projA = A @ normal
        projB = B @ normal
        if projA.max() < projB.min() or projB.max() < projA.min():
            return False
    return True

def _min_dist_hex_hex(A, B):
    """
    Approximate minimum distance between two hexagons using vertex-to-edge and edge-to-edge checks.
    Returns negative value if overlapping.
    """
    min_dist = np.inf
    # Check all vertices of A against edges of B
    for v in A:
        for i in range(6):
            p1 = B[i]
            p2 = B[(i + 1) % 6]
            edge = p2 - p1
            t = np.dot(v - p1, edge) / np.dot(edge, edge)
            t = max(0, min(1, t))
            proj = p1 + t * edge
            d = np.linalg.norm(v - proj)
            # If vertex is "inside" the edge strip, we need to check if it's inside the polygon
            # For convex polygons, if min dist is 0, they touch. If < 0, they overlap.
            # A simple heuristic: if distance is very small, check SAT for precise overlap status
            if d < min_dist:
                min_dist = d
    
    # Check all vertices of B against edges of A
    for v in B:
        for i in range(6):
            p1 = A[i]
            p2 = A[(i + 1) % 6]
            edge = p2 - p1
            t = np.dot(v - p1, edge) / np.dot(edge, edge)
            t = max(0, min(1, t))
            proj = p1 + t * edge
            d = np.linalg.norm(v - proj)
            if d < min_dist:
                min_dist = d
                
    # If min_dist is 0, they might be touching or overlapping.
    # We need a signed distance.
    # If they overlap, return a negative value.
    # Use SAT to determine overlap and magnitude of penetration
    if min_dist < 1e-9:
        if _sat_overlap(A, B):
            # Estimate penetration depth
            # Find the axis with minimum separation
            min_sep = np.inf
            for poly in (A, B):
                for i in range(6):
                    p1 = poly[i]
                    p2 = poly[(i + 1) % 6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    nl = np.linalg.norm(normal)
                    if nl < 1e-9: continue
                    normal = normal / nl
                    projA = A @ normal
                    projB = B @ normal
                    sep = min(projA.max() - projB.min(), projB.max() - projA.min())
                    if sep < min_sep:
                        min_sep = sep
            # min_sep is negative if overlapping
            return min_sep
    return min_dist

def _energy(params, outer_theta, outer_center, side):
    """
    Calculate total energy of the configuration.
    params: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11]
    """
    centers = params[0:N*3:3]
    ys = params[1:N*3:3]
    angles = params[2:N*3:3]
    
    # Reshape to (N, 2) and (N,)
    centers = np.column_stack((centers, ys))
    
    verts = []
    for i in range(N):
        verts.append(_hex_vertices(angles[i]) + centers[i])
        
    energy = 0.0
    
    # 1. Pairwise collision energy
    # Use a smooth penalty: if d < 0 (overlap), penalty is exp(k * d) or d^2
    # Let's use a soft potential: 0.5 * max(0, -d)^2 + small repulsive term
    for i in range(N):
        for j in range(i + 1, N):
            d = _min_dist_hex_hex(verts[i], verts[j])
            if d < 0:
                # Quadratic penalty for overlap
                energy += 10.0 * d * d
            else:
                # Weak repulsion to keep them separated
                energy += 0.01 * np.exp(-5.0 * d)
                
    # 2. Containment energy
    # For each inner hexagon, check if it fits in the outer hexagon
    # Outer hexagon is defined by center, side, theta
    # We can check the distance of each inner vertex to the outer hexagon boundary
    # If a vertex is outside, add penalty
    
    # Outer hexagon vertices
    outer_verts = outer_center + side * _hex_vertices(outer_theta)
    
    for i in range(N):
        for v in verts[i]:
            # Check if v is inside outer hexagon
            # Use point-in-polygon or distance to boundary
            # Simpler: project v onto the 6 normal directions of the outer hexagon
            # The width of the outer hexagon in normal direction n is side * sqrt(3)
            # The center of the outer hexagon projects to center_proj
            # The max allowed projection is center_proj + 0.5 * width
            
            # Normals of outer hexagon
            norms = np.array([[math.cos(outer_theta + math.pi / 6.0 + k * math.pi / 3.0),
                               math.sin(outer_theta + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])
            
            for k in range(6):
                n = norms[k]
                # Projection of v
                proj_v = np.dot(v, n)
                # Projection of outer center
                proj_c = np.dot(outer_center, n)
                # Max allowed projection
                max_proj = proj_c + 0.5 * side * math.sqrt(3.0)
                
                excess = proj_v - max_proj
                if excess > 0:
                    energy += 10.0 * excess * excess
                    
    return energy

def _grad_energy(params, outer_theta, outer_center, side):
    """
    Numerical gradient of the energy function.
    """
    eps = 1e-5
    grad = np.zeros_like(params)
    base_energy = _energy(params, outer_theta, outer_center, side)
    
    for i in range(len(params)):
        orig = params[i]
        params[i] = orig + eps
        e_plus = _energy(params, outer_theta, outer_center, side)
        params[i] = orig - eps
        e_minus = _energy(params, outer_theta, outer_center, side)
        params[i] = orig
        grad[i] = (e_plus - e_minus) / (2 * eps)
        
    return grad

def _optimize_config(params, outer_theta, outer_center, side, time_limit=5.0):
    """
    Run L-BFGS-B optimization on the inner hexagon positions and angles.
    """
    start_time = time.time()
    
    # Define the objective function
    def obj(p):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        return _energy(p, outer_theta, outer_center, side)
        
    # We use numerical gradient because analytic gradient is complex
    # L-BFGS-B with numerical gradient is robust
    
    try:
        res = minimize(obj, params, method='L-BFGS-B', 
                       options={'maxiter': 500, 'maxfun': 1000, 'maxls': 20})
        return res.x
    except TimeoutError:
        return params
    except Exception:
        return params

def _is_valid(centers, angles, outer_center, side, outer_theta, tol=1e-6):
    """
    Check if the configuration is valid (no overlaps, all inside).
    """
    verts = [_hex_vertices(angles[i]) + centers[i] for i in range(N)]
    
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            if _sat_overlap(verts[i], verts[j]):
                return False
                
    # Check containment
    norms = np.array([[math.cos(outer_theta + math.pi / 6.0 + k * math.pi / 3.0),
                       math.sin(outer_theta + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])
    
    for i in range(N):
        for v in verts[i]:
            for k in range(6):
                n = norms[k]
                proj_v = np.dot(v, n)
                proj_c = np.dot(outer_center, n)
                max_proj = proj_c + 0.5 * side * math.sqrt(3.0)
                if proj_v > max_proj + tol:
                    return False
    return True

def _bounding_side_fast(centers, angles, outer_theta):
    """
    Calculate the minimum side length of a regular hexagon (orientation outer_theta)
    that contains all inner hexagons.
    """
    norms = np.array([[math.cos(outer_theta + math.pi / 6.0 + k * math.pi / 3.0),
                       math.sin(outer_theta + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])
    
    min_proj = np.inf
    max_proj = -np.inf
    
    for i in range(N):
        verts = _hex_vertices(angles[i]) + centers[i]
        projs = verts @ norms.T
        min_proj = np.minimum(min_proj, projs.min(axis=0))
        max_proj = np.maximum(max_proj, projs.max(axis=0))
        
    widths = max_proj - min_proj
    return float(np.max(widths) / math.sqrt(3.0))

def _refine_outer(inner, outer_center, side, outer_theta):
    """
    Refine the outer hexagon parameters to minimize side length.
    inner: list of [x, y, angle_deg]
    """
    centers = np.array([[r[0], r[1]] for r in inner])
    angles = np.array([math.radians(r[2]) for r in inner])
    
    best_side = side
    best_center = np.array(outer_center)
    best_theta = outer_theta
    
    # Try a few angles
    for ang in np.linspace(0, math.pi/6, 7):
        s = _bounding_side_fast(centers, angles, ang)
        if s < best_side:
            best_side = s
            best_theta = ang
            # Re-center
            # The bounding box center in projection space should map to the outer center
            norms = np.array([[math.cos(ang + math.pi / 6.0 + k * math.pi / 3.0),
                               math.sin(ang + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])
            min_proj = np.inf
            max_proj = -np.inf
            for i in range(N):
                verts = _hex_vertices(angles[i]) + centers[i]
                projs = verts @ norms.T
                min_proj = np.minimum(min_proj, projs.min(axis=0))
                max_proj = np.maximum(max_proj, projs.max(axis=0))
            
            mid_proj = 0.5 * (min_proj + max_proj)
            # Solve for center
            # mid_proj = center @ norms.T
            # center = mid_proj @ pinv(norms.T)
            try:
                new_center = mid_proj @ np.linalg.pinv(norms.T)
                best_center = new_center
            except:
                pass
                
    return best_side, best_center, best_theta

def construct():
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)
    
    best_side = float('inf')
    best_inner = None
    best_outer = (0.0, 0.0)
    best_theta = 0.0
    
    # Initializer: Triangular lattice
    d = 2.0 * math.sqrt(3.0)
    a1 = np.array([d, 0.0])
    a2 = np.array([d / 2.0, d * math.sqrt(3.0) / 2.0])
    pts = []
    for i in range(-1, 3):
        for j in range(-1, 2):
            pts.append(i * a1 + j * a2)
    pts = np.array(pts)
    c = pts.mean(axis=0)
    pts = pts - c
    
    initial_inner = [[float(p[0]), float(p[1]), 0.0] for p in pts]
    initial_side, initial_center, initial_theta = _refine_outer(initial_inner, (0,0), 10.0, 0.0)
    
    # Simulated Annealing loop
    T = 0.5
    current_inner = initial_inner
    current_side = initial_side
    current_center = np.array(initial_center)
    current_theta = initial_theta
    
    # Convert to params array for optimization
    params = np.array([x for row in current_inner for x in row])
    
    # We will alternate between SA perturbations and local gradient descent
    
    max_iterations = 50
    iter_count = 0
    
    while time.time() - start_time < DEADLINE and iter_count < max_iterations:
        iter_count += 1
        
        # 1. Local Gradient Descent to settle
        # Limit time per optimization
        time_left = DEADLINE - (time.time() - start_time)
        opt_time = min(10.0, time_left * 0.5)
        
        if opt_time > 1.0:
            optimized_params = _optimize_config(params, current_theta, current_center, current_side, time_limit=opt_time)
            params = optimized_params
            
            # Extract new inner
            new_inner = [[float(params[i*3]), float(params[i*3+1]), float(math.degrees(params[i*3+2]))] for i in range(N)]
            
            # Check validity and update side
            if _is_valid(np.array([[r[0], r[1]] for r in new_inner]), 
                         np.array([math.radians(r[2]) for r in new_inner]), 
                         current_center, current_side, current_theta):
                
                s, c, t = _refine_outer(new_inner, current_center, current_side, current_theta)
                
                if s < best_side:
                    best_side = s
                    best_inner = new_inner
                    best_outer = (float(c[0]), float(c[1]))
                    best_theta = t
                    
                current_inner = new_inner
                current_side = s
                current_center = c
                current_theta = t
                params = np.array([x for row in current_inner for x in row])
                
        # 2. SA Perturbation
        # Pick a random hexagon to move
        i = random.randrange(N)
        dx = random.gauss(0, T)
        dy = random.gauss(0, T)
        da = random.gauss(0, T * 0.5)
        
        old_x, old_y, old_a = params[i*3], params[i*3+1], params[i*3+2]
        
        params[i*3] += dx
        params[i*3+1] += dy
        params[i*3+2] += da
        
        # Calculate new energy (approximate side length)
        new_inner = [[float(params[j*3]), float(params[j*3+1]), float(math.degrees(params[j*3+2]))] for j in range(N)]
        
        # Quick check: if it's valid, calculate side
        # If invalid, it's a high energy state
        # We accept based on Metropolis criterion
        # But we need a cost function. Let's use the side length if valid, else inf + penalty
        # For simplicity, we'll just perturb and let the gradient descent fix it.
        # This is a "stochastic local search" approach.
        
        # Cool down
        T *= 0.95
        
    # Final polish
    if best_inner is not None:
        # One last gradient descent with more time
        time_left = DEADLINE - (time.time() - start_time)
        if time_left > 2.0:
            params = np.array([x for row in best_inner for x in row])
            optimized_params = _optimize_config(params, best_theta, np.array(best_outer), best_side, time_limit=time_left * 0.8)
            new_inner = [[float(optimized_params[i*3]), float(optimized_params[i*3+1]), float(math.degrees(optimized_params[i*3+2]))] for i in range(N)]
            
            if _is_valid(np.array([[r[0], r[1]] for r in new_inner]), 
                         np.array([math.radians(r[2]) for r in new_inner]), 
                         np.array(best_outer), best_side, best_theta):
                s, c, t = _refine_outer(new_inner, best_outer, best_side, best_theta)
                if s < best_side:
                    best_side = s
                    best_inner = new_inner
                    best_outer = (float(c[0]), float(c[1]))
                    best_theta = t

    if best_inner is None:
        # Fallback to seed
        return initial_inner, (0.0, 0.0), 10.0, 0.0

    return best_inner, best_outer, best_side, math.degrees(best_theta)

# EVOLVE-BLOCK-END
