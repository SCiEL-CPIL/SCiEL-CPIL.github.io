import numpy as np
import time
from itertools import combinations

N = 11
SQRT3 = np.sqrt(3.0)
H = SQRT3 / 2.0  # height of the equilateral triangle with side 1

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Precompute triangle indices
TRI_IDX = np.array(list(combinations(range(N), 3)))
M = len(TRI_IDX)  # C(11,3) = 165

def min_triangle_area(points):
    """Compute the minimum area over all C(n,3) triangles using vectorization."""
    pi = points[TRI_IDX[:, 0]]
    pj = points[TRI_IDX[:, 1]]
    pk = points[TRI_IDX[:, 2]]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return float(np.min(areas))

def is_inside_triangle(p, eps=1e-9):
    """Check if point p is inside the equilateral triangle."""
    x, y = p
    return (y >= -eps) and (y <= H + eps) and (x >= y/SQRT3 - eps) and (x <= 1 - y/SQRT3 + eps)

def project_to_triangle(p):
    """Project a point onto the triangle if it's outside."""
    x, y = p
    if y < 0:
        y = 0
    if y > H:
        y = H
    x_min = y / SQRT3
    x_max = 1 - y / SQRT3
    if x < x_min:
        x = x_min
    if x > x_max:
        x = x_max
    return np.array([x, y])

def force_directed_optimize(points, deadline, max_steps=20000, dt=0.002):
    """
    Force-directed graph approach:
    - Each point is a node
    - Attractive forces pull points apart to maintain minimum area
    - Repulsive forces prevent collision
    - Use velocity Verlet (symplectic) integrator
    """
    points = points.copy().astype(np.float64)
    velocities = np.zeros_like(points)
    
    best_points = points.copy()
    best_val = min_triangle_area(points)
    
    # Target separation: derived from current best area
    # For equilateral triangle, area = (sqrt(3)/4) * s^2
    # So s = sqrt(4 * area / sqrt(3))
    target_sep = np.sqrt(4.0 * best_val / SQRT3)
    
    # Force parameters
    k_repulse = 5.0   # repulsion strength
    k_attract = 1.0   # attraction to target distance
    damping = 0.95     # velocity damping
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Compute forces
        forces = np.zeros_like(points)
        
        # Pairwise interactions
        for i in range(N):
            for j in range(i + 1, N):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                
                if dist < 1e-10:
                    # Add small random perturbation
                    diff = np.random.randn(2) * 1e-6
                    dist = np.linalg.norm(diff)
                
                direction = diff / dist
                
                # Repulsive force (strong when close)
                if dist < target_sep * 0.5:
                    f_repulse = k_repulse * (target_sep * 0.5 - dist) ** 2
                else:
                    f_repulse = 0.0
                
                # Attractive force to maintain target separation
                if dist < target_sep:
                    f_attract = k_attract * (target_sep - dist)
                else:
                    f_attract = -k_attract * 0.1 * (dist - target_sep)
                
                f_total = f_repulse + f_attract
                forces[i] += f_total * direction
                forces[j] -= f_total * direction
        
        # Boundary forces: push points away from edges
        for i in range(N):
            x, y = points[i]
            
            # Bottom edge (y = 0)
            if y < 0.05:
                forces[i, 1] += k_repulse * (0.05 - y)
            
            # Top edge (y = H)
            if y > H - 0.05:
                forces[i, 1] -= k_repulse * (y - (H - 0.05))
            
            # Left edge (x = y/sqrt(3))
            x_min = y / SQRT3
            if x < x_min + 0.05:
                forces[i, 0] += k_repulse * (x_min + 0.05 - x)
            
            # Right edge (x = 1 - y/sqrt(3))
            x_max = 1 - y / SQRT3
            if x > x_max - 0.05:
                forces[i, 0] -= k_repulse * (x - (x_max - 0.05))
        
        # Velocity Verlet integration
        # Update positions
        points += velocities * dt + 0.5 * forces * dt * dt
        
        # Compute new forces (reuse forces for simplicity, proper Verlet would recompute)
        new_forces = forces  # Simplified: use same forces
        
        # Update velocities
        velocities = (velocities + 0.5 * (forces + new_forces) * dt) * damping
        
        # Project to triangle
        for i in range(N):
            points[i] = project_to_triangle(points[i])
        
        # Evaluate
        val = min_triangle_area(points)
        if val > best_val:
            best_val = val
            best_points = points.copy()
            # Update target separation
            target_sep = np.sqrt(4.0 * best_val / SQRT3)
    
    return best_points, best_val

def nelder_mead_polish(points, deadline, max_iter=500, tol=1e-8):
    """Nelder-Mead simplex optimization for final polish."""
    points = points.copy()
    best_val = min_triangle_area(points)
    
    # Create simplex
    simplex = [points.copy()]
    step = 0.01
    for i in range(N * 2):
        p = points.copy()
        p[i // 2, i % 2] += step
        p = np.array([project_to_triangle(p[j]) for j in range(N)])
        simplex.append(p)
    
    simplex = np.array(simplex)
    values = np.array([min_triangle_area(p) for p in simplex])
    
    # Nelder-Mead parameters
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho = 0.5     # contraction
    sigma = 0.5   # shrink
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Sort by value
        order = np.argsort(values)
        simplex = simplex[order]
        values = values[order]
        
        # Check convergence
        if np.std(values) < tol:
            break
        
        # Centroid of all but worst point
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        worst = simplex[-1]
        reflected = centroid + alpha * (centroid - worst)
        reflected = np.array([project_to_triangle(reflected[i]) for i in range(N)])
        ref_val = min_triangle_area(reflected)
        
        if values[0] <= ref_val < values[-2]:
            simplex[-1] = reflected
            values[-1] = ref_val
        elif ref_val < values[0]:
            # Expansion
            expanded = centroid + gamma * (reflected - centroid)
            expanded = np.array([project_to_triangle(expanded[i]) for i in range(N)])
            exp_val = min_triangle_area(expanded)
            if exp_val < ref_val:
                simplex[-1] = expanded
                values[-1] = exp_val
            else:
                simplex[-1] = reflected
                values[-1] = ref_val
        else:
            # Contraction
            contracted = centroid + rho * (worst - centroid)
            contracted = np.array([project_to_triangle(contracted[i]) for i in range(N)])
            con_val = min_triangle_area(contracted)
            if con_val < values[-1]:
                simplex[-1] = contracted
                values[-1] = con_val
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    simplex[i] = np.array([project_to_triangle(simplex[i][j]) for j in range(N)])
                    values[i] = min_triangle_area(simplex[i])
        
        # Update best
        best_idx = np.argmin(values)
        if values[best_idx] > best_val:
            best_val = values[best_idx]
            best_points = simplex[best_idx].copy()
    
    return best_points, best_val

def generate_structured_initial():
    """Generate a structured initial guess with good symmetry."""
    points = []
    # Vertices
    points.append([0.0, 0.0])
    points.append([1.0, 0.0])
    points.append([0.5, H])
    
    # Edge midpoints
    points.append([0.5, 0.0])
    points.append([0.25, H/2])
    points.append([0.75, H/2])
    
    # Center
    points.append([0.5, H/3])
    
    # Additional points
    points.append([0.25, 0.0])
    points.append([0.75, 0.0])
    points.append([0.35, H/4])
    points.append([0.65, H/4])
    
    return np.array(points[:N])

def generate_random_points(seed):
    """Generate random points inside the triangle using rejection sampling."""
    rng = np.random.RandomState(seed)
    points = []
    while len(points) < N:
        u, v = rng.random(2)
        if u + v < 1:
            a = 1 - u - v
            b = u
            c = v
            x = b * 1.0 + c * 0.5
            y = c * H
            points.append([x, y])
    return np.array(points)

def local_optimize(points, max_iter=300, step_size=0.1, deadline=None):
    """Local optimization using coordinate descent / hill climbing."""
    points = points.copy()
    best_val = min_triangle_area(points)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        improved = False
        for i in range(N):
            best_local = points[i].copy()
            best_local_val = best_val
            
            for _ in range(25):
                if deadline is not None and time.time() > deadline:
                    break
                    
                dx = np.random.uniform(-step_size, step_size, 2)
                new_p = best_local + dx
                new_p = project_to_triangle(new_p)
                
                if not is_inside_triangle(new_p, eps=1e-6):
                    continue
                    
                new_points = points.copy()
                new_points[i] = new_p
                
                val = min_triangle_area(new_points)
                if val > best_local_val + 1e-10:
                    best_local_val = val
                    best_local = new_p
                    improved = True
            
            points[i] = best_local
        
        current_val = min_triangle_area(points)
        if current_val > best_val:
            best_val = current_val
        
        if not improved:
            step_size *= 0.8
            if step_size < 1e-6:
                break
    
    return points, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_points = None
    best_value = -1
    
    # Strategy 1: Force-directed optimization from structured initial
    init_points = generate_structured_initial()
    fd_points, fd_val = force_directed_optimize(init_points, deadline, max_steps=15000, dt=0.003)
    
    if fd_val > best_value:
        best_value = fd_val
        best_points = fd_points.copy()
    
    # Strategy 2: Local optimization from structured initial
    if time.time() < deadline - 20:
        lo_points, lo_val = local_optimize(init_points, max_iter=200, step_size=0.1, deadline=deadline)
        if lo_val > best_value:
            best_value = lo_val
            best_points = lo_points.copy()
    
    # Strategy 3: Force-directed from random initializations
    for seed in range(5):
        if time.time() > deadline - 15:
            break
        
        random_init = generate_random_points(seed * 42 + 7)
        fd_points, fd_val = force_directed_optimize(random_init, deadline, max_steps=10000, dt=0.003)
        
        if fd_val > best_value:
            best_value = fd_val
            best_points = fd_points.copy()
    
    # Final polish with Nelder-Mead
    if best_points is not None and time.time() < deadline - 10:
        nm_points, nm_val = nelder_mead_polish(best_points, deadline, max_iter=300, tol=1e-8)
        if nm_val > best_value:
            best_value = nm_val
            best_points = nm_points.copy()
    
    # Verify all points are inside the triangle
    if best_points is not None:
        for i, p in enumerate(best_points):
            if not is_inside_triangle(p, eps=1e-6):
                best_points[i] = project_to_triangle(p)
    
    if best_points is None:
        best_points = generate_structured_initial()
    
    return best_points
