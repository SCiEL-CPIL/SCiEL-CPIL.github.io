import numpy as np
import time
from itertools import combinations

N = 11
SQRT3 = np.sqrt(3.0)
H = SQRT3 / 2.0

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Precompute all triangle index combinations
TRI_IDX = np.array(list(combinations(range(N), 3)))
M = len(TRI_IDX)  # C(11,3) = 165

def min_triangle_area(points):
    """Compute the minimum area over all C(n,3) triangles using vectorization."""
    pi = points[TRI_IDX[:, 0]]
    pj = points[TRI_IDX[:, 1]]
    pk = points[TRI_IDX[:, 2]]
    # Cross product: (pj - pi) x (pk - pi)
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pk[:, 0] - pi[:, 0]) * (pj[:, 1] - pi[:, 1])
    areas = 0.5 * np.abs(cross)
    return float(np.min(areas))

def is_inside_triangle(p, eps=1e-9):
    x, y = p
    return (y >= -eps) and (y <= H + eps) and (x >= y / SQRT3 - eps) and (x <= 1 - y / SQRT3 + eps)

def project_to_triangle(p):
    x, y = p
    if y < 0:
        y = 0.0
    if y > H:
        y = H
    x_min = y / SQRT3
    x_max = 1.0 - y / SQRT3
    if x < x_min:
        x = x_min
    if x > x_max:
        x = x_max
    return np.array([x, y])

def project_all_to_triangle(points):
    """Project all points to be inside the triangle."""
    for i in range(len(points)):
        points[i] = project_to_triangle(points[i])
    return points

def symmetric_initial():
    """Symmetric initial configuration: 3 vertices, 3 edge midpoints, 1 center, 4 more."""
    pts = []
    # 3 vertices
    pts.append([0.0, 0.0])
    pts.append([1.0, 0.0])
    pts.append([0.5, H])
    # 3 edge midpoints
    pts.append([0.5, 0.0])
    pts.append([0.25, H / 2.0])
    pts.append([0.75, H / 2.0])
    # Center
    pts.append([0.5, H / 3.0])
    # 4 more points: symmetric arrangement
    # Two points on the lower third
    pts.append([0.3, H / 6.0])
    pts.append([0.7, H / 6.0])
    # Two points slightly above center
    pts.append([0.35, H / 2.5])
    pts.append([0.65, H / 2.5])
    return np.array(pts)

def random_initial(seed):
    """Random points inside the triangle."""
    rng = np.random.RandomState(seed)
    pts = []
    while len(pts) < N:
        u, v = rng.random(2)
        if u + v < 1.0:
            x = u + 0.5 * v
            y = H * v
            pts.append([x, y])
    return np.array(pts)

def force_directed_optimize(points, deadline, max_iter=3000):
    """
    Force-directed approach: treat points as nodes in a graph.
    - Attractive springs pull points toward a target separation based on current best area.
    - Repulsive forces push points apart to avoid collisions.
    - Constraint forces keep points inside the triangle.
    Uses a symplectic-like (velocity Verlet) integrator.
    """
    points = points.copy().astype(float)
    velocities = np.zeros_like(points)
    mass = 1.0
    dt = 0.01
    damping = 0.95
    
    best_val = min_triangle_area(points)
    best_points = points.copy()
    
    # Target separation: derived from current min area
    # For a triangle with area A, the "inradius-like" separation is roughly sqrt(A)
    # We'll use a target distance that we adjust
    target_dist = 0.15  # initial guess
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(points)
        
        # 1. Repulsive forces between all pairs (Coulomb-like)
        for i in range(N):
            for j in range(i + 1, N):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-8:
                    diff = np.array([1e-8, 0.0])
                    dist = 1e-8
                # Repulsion: strong at short range
                rep = 0.01 / (dist * dist)
                forces[i] += rep * diff / dist
                forces[j] -= rep * diff / dist
        
        # 2. Attractive forces: pull points toward a target separation
        # We want all pairwise distances to be at least some value
        # This is a soft constraint
        for i in range(N):
            for j in range(i + 1, N):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist < target_dist:
                    # Attract away from each other (like a spring that resists compression)
                    spring = 0.5 * (target_dist - dist)
                    forces[i] += spring * diff / (dist + 1e-8)
                    forces[j] -= spring * diff / (dist + 1e-8)
        
        # 3. Constraint forces: keep points inside the triangle
        for i in range(N):
            x, y = points[i]
            # Bottom edge: y >= 0
            if y < 0.05:
                forces[i, 1] += 10.0 * (0.05 - y)
            # Left edge: x >= y / sqrt(3)
            x_min = y / SQRT3
            if x < x_min + 0.05:
                forces[i, 0] += 10.0 * (x_min + 0.05 - x)
            # Right edge: x <= 1 - y / sqrt(3)
            x_max = 1.0 - y / SQRT3
            if x > x_max - 0.05:
                forces[i, 0] -= 10.0 * (x - (x_max - 0.05))
            # Top edge: y <= H
            if y > H - 0.05:
                forces[i, 1] -= 10.0 * (y - (H - 0.05))
        
        # Velocity Verlet integration
        # Update velocities
        velocities = damping * velocities + (forces / mass) * dt
        # Update positions
        points = points + velocities * dt
        
        # Project to triangle
        points = project_all_to_triangle(points)
        
        # Evaluate
        val = min_triangle_area(points)
        if val > best_val:
            best_val = val
            best_points = points.copy()
            # Adapt target distance based on current best
            # The minimum triangle area A relates to separation d roughly as A ~ d^2
            # So d ~ sqrt(A)
            target_dist = max(0.05, min(0.3, np.sqrt(best_val) * 1.2))
        
        # Dampen velocities slightly to stabilize
        velocities *= 0.99
    
    return best_points, best_val

def nelder_mead_optimize(points, deadline, max_iter=2000):
    """Nelder-Mead style optimization on the full point set."""
    from scipy.optimize import minimize
    
    points = points.copy().astype(float)
    
    def objective(flat):
        pts = flat.reshape(N, 2)
        pts = project_all_to_triangle(pts)
        return -min_triangle_area(pts)
    
    flat = points.flatten()
    
    best_val = min_triangle_area(points)
    best_points = points.copy()
    
    # Use Nelder-Mead with adaptive settings
    try:
        result = minimize(
            objective, flat, method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8,
                     'adaptive': True, 'initial_simplex': 0.01}
        )
        if result.x is not None:
            candidate = result.x.reshape(N, 2)
            candidate = project_all_to_triangle(candidate)
            val = min_triangle_area(candidate)
            if val > best_val:
                best_val = val
                best_points = candidate
    except Exception:
        pass
    
    return best_points, best_val

def micro_optimize(points, deadline, max_iter=500):
    """Fine-grained coordinate descent for final polishing."""
    points = points.copy()
    best_val = min_triangle_area(points)
    
    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for ss in step_sizes:
        if time.time() > deadline:
            break
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                for dx in np.linspace(-ss, ss, 5):
                    for dy in np.linspace(-ss, ss, 5):
                        new_p = points[i] + np.array([dx, dy])
                        new_p = project_to_triangle(new_p)
                        if not is_inside_triangle(new_p, eps=1e-7):
                            continue
                        new_points = points.copy()
                        new_points[i] = new_p
                        val = min_triangle_area(new_points)
                        if val > best_val + 1e-10:
                            best_val = val
                            points = new_points
                            improved = True
    
    return points, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_value = -1
    
    # Multiple initializations
    inits = [symmetric_initial()]
    for seed in [42, 123, 456, 789, 101]:
        inits.append(random_initial(seed))
    
    for init_pts in inits:
        if time.time() > deadline:
            break
        
        # Phase 1: Force-directed optimization
        fd_pts, fd_val = force_directed_optimize(init_pts, deadline, max_iter=2000)
        if fd_val > best_value:
            best_value = fd_val
            best_points = fd_pts.copy()
        
        # Phase 2: Nelder-Mead polishing
        if time.time() < deadline - 15:
            nm_pts, nm_val = nelder_mead_optimize(fd_pts, deadline, max_iter=1000)
            if nm_val > best_value:
                best_value = nm_val
                best_points = nm_pts.copy()
        
        # Phase 3: Micro optimization
        if time.time() < deadline - 5:
            mo_pts, mo_val = micro_optimize(best_points, deadline, max_iter=200)
            if mo_val > best_value:
                best_value = mo_val
                best_points = mo_pts.copy()
    
    # Final micro optimization pass
    if best_points is not None and time.time() < deadline - 3:
        final_pts, final_val = micro_optimize(best_points, deadline, max_iter=100)
        if final_val > best_value:
            best_value = final_val
            best_points = final_pts.copy()
    
    if best_points is None:
        best_points = symmetric_initial()
    
    # Ensure all points are inside
    for i in range(len(best_points)):
        best_points[i] = project_to_triangle(best_points[i])
    
    return best_points
