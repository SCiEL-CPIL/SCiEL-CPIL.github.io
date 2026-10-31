"""
Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle.
New structural approach:
- Direct Cartesian parameterization with analytic gradients
- Simulated annealing with adaptive step size for global exploration
- COBYLA for constrained local refinement
- Multiple diverse initializations (symmetric, random, structured)
"""

import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

N = 11
H = np.sqrt(3) / 2
TOTAL_AREA = 0.5 * 1.0 * H  # Area of unit equilateral triangle

# Triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])
V = np.array([A, B, C])  # (3, 2)

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)
_I = tri_idx[:, 0]
_J = tri_idx[:, 1]
_K = tri_idx[:, 2]

def cross2d(a, b):
    """2D cross product magnitude."""
    return a[0]*b[1] - a[1]*b[0]

def point_in_triangle(p, verts):
    """Check if point p is inside or on triangle with vertices verts."""
    # Barycentric coordinate method
    x, y = p
    x1, y1 = verts[0]
    x2, y2 = verts[1]
    x3, y3 = verts[2]
    
    denom = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
    if abs(denom) < 1e-15:
        return False
    
    a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / denom
    b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / denom
    c = 1 - a - b
    
    return a >= -1e-9 and b >= -1e-9 and c >= -1e-9

def clamp_to_triangle(pts):
    """
    Clamp points to be inside the triangle.
    For each point, if it's outside, project it to the nearest edge.
    """
    pts = pts.copy()
    for i in range(len(pts)):
        p = pts[i]
        if not point_in_triangle(p, V):
            # Find the nearest point on the triangle boundary
            # Check each edge
            best_dist = np.inf
            best_point = p
            for e in range(3):
                p1 = V[e]
                p2 = V[(e+1) % 3]
                # Project p onto line segment p1-p2
                d = p2 - p1
                t = np.dot(p - p1, d) / np.dot(d, d)
                t = np.clip(t, 0, 1)
                proj = p1 + t * d
                dist = np.linalg.norm(p - proj)
                if dist < best_dist:
                    best_dist = dist
                    best_point = proj
            # Move slightly inside
            centroid = np.mean(V, axis=0)
            direction = centroid - best_point
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                direction = direction / norm
            else:
                direction = np.array([0.0, 0.0])
            pts[i] = best_point + 1e-6 * direction
    return pts

def compute_areas(pts):
    """
    Compute areas of all triangles formed by N points.
    pts: (N, 2) array of Cartesian coordinates.
    Returns array of shape (165,) with areas divided by TOTAL_AREA.
    """
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    
    # Area = 0.5 * |cross(pj - pi, pk - pi)|
    v1 = pj - pi
    v2 = pk - pi
    cross_vals = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross_vals) / TOTAL_AREA
    return areas

def min_area(pts):
    """Compute the minimum triangle area (normalized)."""
    return np.min(compute_areas(pts))

def objective_and_grad(pts_flat):
    """
    Objective: negative soft-min of areas.
    Returns (value, gradient).
    """
    pts = pts_flat.reshape(N, 2)
    pts = clamp_to_triangle(pts)
    areas = compute_areas(pts)
    
    # Soft min with beta
    beta = 80.0
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        # Fallback to hard min
        min_idx = np.argmin(areas)
        return -areas[min_idx], None
    
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_val = -log_sum / beta
    value = -soft_min_val
    
    # Gradient of soft min
    # d(soft_min)/d(area_i) = -exp(-beta * area_i) / sum(exp(-beta * area_j))
    exp_scaled = np.exp(scaled - max_scaled)
    weights = exp_scaled / np.sum(exp_scaled)
    d_softmin_dareas = -weights
    
    # Gradient of area_i w.r.t. points
    # area_i = 0.5 * |cross(pj - pi, pk - pi)| / TOTAL_AREA
    # For the specific triangle (i, j, k):
    # area = 0.5 * |cross(pj - pi, pk - pi)| / TOTAL_AREA
    # d(area)/d(pi) = -0.5 * cross(pk - pi, e) / TOTAL_AREA ... 
    # Actually, let's compute it more carefully.
    # cross(pj - pi, pk - pi) = (pj_x - pi_x)(pk_y - pi_y) - (pj_y - pi_y)(pk_x - pi_x)
    # d/d(pi_x) = -(pk_y - pi_y) + (pj_y - pi_y)
    # d/d(pi_y) = (pk_x - pi_x) - (pj_x - pi_x)
    # d/d(pj_x) = (pk_y - pi_y)
    # d/d(pj_y) = -(pk_x - pi_x)
    # d/d(pk_x) = -(pj_y - pi_y)
    # d/d(pk_y) = (pj_x - pi_x)
    
    grad = np.zeros((N, 2))
    for t in range(len(tri_indices)):
        w = d_softmin_dareas[t]
        if abs(w) < 1e-15:
            continue
        i, j, k = tri_indices[t]
        pi = pts[i]
        pj = pts[j]
        pk = pts[k]
        
        # cross = (pj_x - pi_x)(pk_y - pi_y) - (pj_y - pi_y)(pk_x - pi_x)
        # sign of cross determines the absolute value derivative
        cross_val = (pj[0] - pi[0])*(pk[1] - pi[1]) - (pj[1] - pi[1])*(pk[0] - pi[0])
        sign = 1.0 if cross_val >= 0 else -1.0
        
        factor = 0.5 * sign / TOTAL_AREA * w
        
        # d(area)/d(pi)
        d_pi_x = -(pk[1] - pi[1]) + (pj[1] - pi[1])
        d_pi_y = (pk[0] - pi[0]) - (pj[0] - pi[0])
        grad[i, 0] += factor * d_pi_x
        grad[i, 1] += factor * d_pi_y
        
        # d(area)/d(pj)
        d_pj_x = (pk[1] - pi[1])
        d_pj_y = -(pk[0] - pi[0])
        grad[j, 0] += factor * d_pj_x
        grad[j, 1] += factor * d_pj_y
        
        # d(area)/d(pk)
        d_pk_x = -(pj[1] - pi[1])
        d_pk_y = (pj[0] - pi[0])
        grad[k, 0] += factor * d_pk_x
        grad[k, 1] += factor * d_pk_y
    
    return value, grad.flatten()

def hard_objective(pts_flat):
    """Hard min objective for COBYLA."""
    pts = pts_flat.reshape(N, 2)
    pts = clamp_to_triangle(pts)
    return -min_area(pts)

def generate_symmetric_init():
    """
    Generate a symmetric initial configuration.
    For 11 points, we can use a hexagonal-like pattern with symmetry.
    """
    pts = np.zeros((N, 2))
    # Center point
    pts[0] = np.array([0.5, H/3])  # Centroid
    
    # 6 points around center at radius r
    r1 = 0.2
    for i in range(6):
        angle = i * np.pi / 3
        pts[i+1] = np.array([0.5, H/3]) + r1 * np.array([np.cos(angle), np.sin(angle)])
    
    # 4 more points near edges
    # Two near bottom edge
    pts[7] = np.array([0.25, 0.05])
    pts[8] = np.array([0.75, 0.05])
    # Two near top
    pts[9] = np.array([0.25, 2*H/3 - 0.05])
    pts[10] = np.array([0.75, 2*H/3 - 0.05])
    
    return clamp_to_triangle(pts)

def generate_random_init(seed=None):
    """Generate random valid points inside the triangle."""
    rng = np.random.default_rng(seed)
    pts = np.zeros((N, 2))
    for i in range(N):
        # Random point in triangle using barycentric coords
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        pts[i] = u * A + v * B + (1 - u - v) * C
    return pts

def generate_structured_init(seed=None):
    """Generate a structured grid-like initialization."""
    rng = np.random.default_rng(seed)
    pts = []
    
    # Try a triangular grid
    for row in range(4):
        for col in range(4):
            if row + col < 4:
                # Barycentric coordinates
                u = col / 3.0
                v = row / 3.0
                if u + v <= 1.0:
                    p = u * A + v * B + (1 - u - v) * C
                    pts.append(p)
    
    # If we don't have exactly N, adjust
    while len(pts) < N:
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        p = u * A + v * B + (1 - u - v) * C
        pts.append(p)
    
    pts = np.array(pts[:N])
    return clamp_to_triangle(pts)

def simulated_annealing(pts, time_budget=20.0):
    """
    Simulated annealing to explore the configuration space.
    """
    start_time = time.time()
    current_pts = pts.copy()
    current_val = min_area(current_pts)
    best_pts = current_pts.copy()
    best_val = current_val
    
    T = 0.1
    T_min = 1e-6
    alpha = 0.995
    step_scale = 0.1
    rng = np.random.default_rng(12345)
    
    while T > T_min and (time.time() - start_time) < time_budget:
        # Generate perturbation
        noise = rng.normal(0, step_scale, (N, 2))
        new_pts = clamp_to_triangle(current_pts + noise)
        new_val = min_area(new_pts)
        
        # Accept if better or with probability exp((new - current) / T)
        if new_val > current_val:
            current_pts = new_pts
            current_val = new_val
            if new_val > best_val:
                best_val = new_val
                best_pts = new_pts.copy()
        else:
            delta = new_val - current_val
            if delta > 0:
                accept_prob = 1.0
            else:
                accept_prob = np.exp(delta / T)
            if rng.random() < accept_prob:
                current_pts = new_pts
                current_val = new_val
        
        T *= alpha
        step_scale *= 0.999
    
    return best_pts, best_val

def local_refine_cobyla(pts, time_budget=10.0):
    """
    Local refinement using COBYLA (derivative-free, handles constraints).
    """
    start_time = time.time()
    
    # Define constraints: points must be inside triangle
    # We'll use penalty in the objective instead
    
    def obj_with_penalty(x):
        pts = x.reshape(N, 2)
        pts = clamp_to_triangle(pts)
        val = min_area(pts)
        # Penalty for being outside (should be 0 after clamping)
        penalty = 0.0
        for i in range(N):
            if not point_in_triangle(pts[i], V):
                penalty += 10.0
        return -val + penalty
    
    bounds = [(-0.5, 1.5)] * (N * 2)
    
    try:
        result = minimize(
            obj_with_penalty,
            pts.flatten(),
            method='COBYLA',
            options={'maxiter': 500, 'rhobeg': 0.01, 'catol': 1e-10},
        )
        if (time.time() - start_time) < time_budget:
            refined_pts = clamp_to_triangle(result.x.reshape(N, 2))
            refined_val = min_area(refined_pts)
            if refined_val > min_area(pts):
                return refined_pts
    except Exception:
        pass
    
    return pts

def local_refine_nelder_mead(pts, time_budget=10.0):
    """
    Local refinement using Nelder-Mead.
    """
    start_time = time.time()
    
    try:
        result = minimize(
            hard_objective,
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
        )
        if (time.time() - start_time) < time_budget:
            refined_pts = clamp_to_triangle(result.x.reshape(N, 2))
            refined_val = min_area(refined_pts)
            if refined_val > min_area(pts):
                return refined_pts
    except Exception:
        pass
    
    return pts

def construct():
    """
    Place 11 points on or inside a unit-area equilateral triangle to maximize
    the smallest area of a triangle formed by three points.
    Returns an array of shape (11, 2) in Cartesian coordinates.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    best_pts = None
    best_val = -np.inf
    
    def update_best(pts):
        nonlocal best_pts, best_val
        pts = clamp_to_triangle(pts)
        val = min_area(pts)
        if val > best_val:
            best_val = val
            best_pts = pts.copy()
            return True
        return False
    
    # Phase 1: Try multiple diverse initializations
    initializations = []
    
    # Symmetric init
    initializations.append(generate_symmetric_init())
    
    # Structured inits
    for seed in [1, 2, 3]:
        initializations.append(generate_structured_init(seed=seed))
    
    # Random inits
    for seed in range(10, 25):
        initializations.append(generate_random_init(seed=seed))
    
    # Phase 2: For each initialization, run SA + local refinement
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 40:
            break
        
        # Quick evaluation
        init_val = min_area(init_pts)
        update_best(init_pts)
        
        # Simulated annealing
        sa_time = min(15.0, deadline - time.time() - 30)
        if sa_time > 2.0:
            sa_pts, sa_val = simulated_annealing(init_pts, time_budget=sa_time)
            update_best(sa_pts)
            current_pts = sa_pts
        else:
            current_pts = init_pts
        
        # Local refinement with Nelder-Mead
        nm_time = min(8.0, deadline - time.time() - 20)
        if nm_time > 2.0:
            nm_pts = local_refine_nelder_mead(current_pts, time_budget=nm_time)
            update_best(nm_pts)
            current_pts = nm_pts
        
        # Local refinement with COBYLA
        co_time = min(5.0, deadline - time.time() - 10)
        if co_time > 1.0:
            co_pts = local_refine_cobyla(current_pts, time_budget=co_time)
            update_best(co_pts)
    
    # Phase 3: Perturb best and re-optimize
    if best_pts is not None:
        for i in range(6):
            if time.time() > deadline - 15:
                break
            rng = np.random.default_rng(100 + i)
            perturbed = clamp_to_triangle(best_pts + rng.normal(0, 0.02, (N, 2)))
            update_best(perturbed)
            
            # Quick SA
            sa_time = min(8.0, deadline - time.time() - 10)
            if sa_time > 2.0:
                sa_pts, _ = simulated_annealing(perturbed, time_budget=sa_time)
                update_best(sa_pts)
            
            # NM polish
            nm_time = min(5.0, deadline - time.time() - 5)
            if nm_time > 1.0:
                nm_pts = local_refine_nelder_mead(sa_pts if sa_time > 2.0 else perturbed, time_budget=nm_time)
                update_best(nm_pts)
    
    # Phase 4: Final global polish if time remains
    if best_pts is not None and time.time() < deadline - 10:
        final_time = deadline - time.time() - 5
        if final_time > 5.0:
            final_pts = local_refine_nelder_mead(best_pts, time_budget=final_time)
            update_best(final_pts)
    
    # Return the best configuration
    if best_pts is not None:
        return best_pts
    else:
        # Fallback
        return generate_random_init(seed=0)

# EVOLVE-BLOCK-END
