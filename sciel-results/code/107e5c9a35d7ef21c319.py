"""
Packing 12 unit regular hexagons into a regular hexagon.
Strategy:
1. Fast SAT overlap check with penetration depth.
2. Simulated Annealing to minimize the bounding hexagon side length.
3. Discrete optimization of the outer hexagon orientation.
4. Final verification and center/orientation refinement.
"""
import time
import math
import random
import numpy as np

N = 12
DEADLINE = 145.0  # Safety margin before the 150s kill

# EVOLVE-BLOCK-START

def _hex_vertices(theta, center=(0.0, 0.0), side=1.0):
    """
    Returns vertices of a regular hexagon with given side length,
    centered at 'center', rotated by 'theta'.
    """
    cx, cy = center
    # Vertices of a unit hexagon (side 1) centered at origin, rotated by theta
    # The distance from center to vertex is 'side' for a regular hexagon.
    # The problem states "side length 1". So R=1.
    verts = np.array([
        [math.cos(theta + k * math.pi / 3.0), math.sin(theta + k * math.pi / 3.0)] 
        for k in range(6)
    ])
    return verts + np.array([cx, cy])

def _sat_overlap_depth(A, B):
    """
    Checks overlap between two convex polygons A and B (arrays of (x,y)).
    Returns (overlap_depth, axis) if overlapping, else (0.0, None).
    overlap_depth is the minimum translation distance to separate them.
    """
    min_pen = np.inf
    min_axis = None
    
    # Check axes from A
    for i in range(len(A)):
        p1 = A[i]
        p2 = A[(i + 1) % len(A)]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        nl = math.hypot(normal[0], normal[1])
        if nl < 1e-9: continue
        normal = normal / nl
        
        projA = A @ normal
        projB = B @ normal
        
        maxA = projA.max()
        minA = projA.min()
        maxB = projB.max()
        minB = projB.min()
        
        # If separated
        if maxA < minB or maxB < minA:
            return 0.0, None
        
        # Calculate penetration depth in this direction
        depth = min(maxA, maxB) - max(minA, minB)
        if depth < min_pen:
            min_pen = depth
            # Determine direction: from A to B or B to A?
            cA = A.mean(axis=0) @ normal
            cB = B.mean(axis=0) @ normal
            if cB > cA:
                min_axis = normal
            else:
                min_axis = -normal

    # Check axes from B
    for i in range(len(B)):
        p1 = B[i]
        p2 = B[(i + 1) % len(B)]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        nl = math.hypot(normal[0], normal[1])
        if nl < 1e-9: continue
        normal = normal / nl
        
        projA = A @ normal
        projB = B @ normal
        
        maxA = projA.max()
        minA = projA.min()
        maxB = projB.max()
        minB = projB.min()
        
        if maxA < minB or maxB < minA:
            return 0.0, None
        
        depth = min(maxA, maxB) - max(minA, minB)
        if depth < min_pen:
            min_pen = depth
            cA = A.mean(axis=0) @ normal
            cB = B.mean(axis=0) @ normal
            if cB > cA:
                min_axis = normal
            else:
                min_axis = -normal

    if min_pen == np.inf:
        return 0.0, None
    return min_pen, min_axis

def _is_valid(centers, angles, tol=1e-9):
    """
    Checks if the configuration has no overlaps.
    """
    verts = [_hex_vertices(angles[i], centers[i]) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            depth, _ = _sat_overlap_depth(verts[i], verts[j])
            if depth > tol:
                return False
    return True

def _bounding_side(centers, angles, outer_theta):
    """
    Calculates the minimum side length of a regular hexagon (orientation outer_theta)
    that contains all inner hexagons.
    """
    # Normals for the outer hexagon sides
    # The normal to the side between vertex k and k+1 is at angle theta + (k+0.5)*pi/3.
    # Which is theta + pi/6 + k*pi/3.
    norms = np.array([
        [math.cos(outer_theta + math.pi / 6.0 + k * math.pi / 3.0),
         math.sin(outer_theta + math.pi / 6.0 + k * math.pi / 3.0)] 
        for k in range(6)
    ])
    
    min_proj = np.inf * np.ones(6)
    max_proj = -np.inf * np.ones(6)
    
    for i in range(N):
        verts = _hex_vertices(angles[i], centers[i])
        projs = verts @ norms.T
        min_proj = np.minimum(min_proj, projs.min(axis=0))
        max_proj = np.maximum(max_proj, projs.max(axis=0))
        
    widths = max_proj - min_proj
    # Width w = s * sqrt(3)
    return float(np.max(widths) / math.sqrt(3.0))

def _physics_relax(centers, angles, steps=200, dt=0.01):
    """
    Simple physics simulation to resolve overlaps and compact the pack.
    Returns updated centers and angles.
    """
    centers = np.array(centers, dtype=float)
    angles = np.array(angles, dtype=float)
    vel = np.zeros((N, 2))
    omega = np.zeros(N)
    
    for s in range(steps):
        # Forces
        for i in range(N):
            # Attraction to center
            vel[i] += -0.05 * centers[i]
            # Damping
            vel[i] *= 0.9
            omega[i] *= 0.9
            
        # Collision resolution
        verts = [_hex_vertices(angles[i], centers[i]) for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                depth, axis = _sat_overlap_depth(verts[i], verts[j])
                if depth > 0 and axis is not None:
                    # Force is proportional to depth
                    f = 5.0 * depth
                    vel[i] -= f * axis
                    vel[j] += f * axis
                    # Small torque to help escape local minima
                    omega[i] += random.uniform(-0.1, 0.1) * depth
                    omega[j] += random.uniform(-0.1, 0.1) * depth

        # Integrate
        centers += vel * dt
        angles += omega * dt

    return centers, angles

def _simulated_annealing(initial_centers, initial_angles, outer_theta, T_start=0.5, T_end=0.01, alpha=0.999):
    """
    Simulated Annealing to minimize the bounding side length.
    """
    centers = np.array(initial_centers, dtype=float)
    angles = np.array(initial_angles, dtype=float)
    
    # Ensure initial state is valid
    if not _is_valid(centers, angles):
        c, a = _physics_relax(centers, angles, steps=100)
        centers, angles = c, a
        if not _is_valid(centers, angles):
            return None, None, float('inf')

    current_side = _bounding_side(centers, angles, outer_theta)
    best_side = current_side
    best_centers = centers.copy()
    best_angles = angles.copy()
    
    T = T_start
    step_size = 0.5
    
    while T > T_end:
        # Pick a random hexagon to move
        i = random.randrange(N)
        
        # Random perturbation
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, step_size * 0.5)
        
        old_cx, old_cy, old_a = centers[i][0], centers[i][1], angles[i]
        
        centers[i][0] += dx
        centers[i][1] += dy
        angles[i] += da
        
        # Check validity
        if not _is_valid(centers, angles):
            # Revert
            centers[i][0] = old_cx
            centers[i][1] = old_cy
            angles[i] = old_a
            T *= alpha
            continue
        
        new_side = _bounding_side(centers, angles, outer_theta)
        
        # Acceptance criterion
        if new_side < current_side or random.random() < math.exp(-(new_side - current_side) / T):
            current_side = new_side
            if new_side < best_side:
                best_side = new_side
                best_centers = centers.copy()
                best_angles = angles.copy()
        else:
            # Revert
            centers[i][0] = old_cx
            centers[i][1] = old_cy
            angles[i] = old_a
            
        T *= alpha
        # Reduce step size slightly
        step_size *= 0.9995
        
    return best_centers, best_angles, best_side

def construct():
    start_time = time.time()
    random.seed(42)
    
    best_side = float('inf')
    best_params = None
    best_outer_center = (0.0, 0.0)
    best_outer_theta = 0.0

    # Initializer: Hexagonal lattice
    def init_lattice():
        d = 2.0 * math.sqrt(3.0) # Distance between centers of touching hexagons
        a1 = np.array([d, 0.0])
        a2 = np.array([d / 2.0, d * math.sqrt(3.0) / 2.0])
        pts = []
        # 4x3 block is 12 points
        for i in range(-1, 3):
            for j in range(-1, 2):
                p = i * a1 + j * a2
                pts.append(p)
        pts = np.array(pts)
        c = pts.mean(axis=0)
        pts = pts - c
        centers = pts.tolist()
        angles = [0.0] * N
        return centers, angles

    def init_random():
        centers = []
        angles = []
        for _ in range(N):
            centers.append([random.uniform(-2, 2), random.uniform(-2, 2)])
            angles.append(random.uniform(0, math.pi/3))
        return centers, angles

    # Optimization Loop
    initializers = [init_lattice, init_random, init_random, init_random]
    outer_thetas = [0.0, math.pi/6, math.pi/3]
    
    for init_func in initializers:
        if time.time() - start_time > DEADLINE - 20:
            break
            
        centers, angles = init_func()
        
        # 1. Physics Relaxation to get a valid starting point
        c_rel, a_rel = _physics_relax(centers, angles, steps=100)
        
        # 2. Simulated Annealing for each outer theta
        for outer_theta in outer_thetas:
            if time.time() - start_time > DEADLINE - 10:
                break
                
            try:
                c_opt, a_opt, side_opt = _simulated_annealing(c_rel, a_rel, outer_theta)
                
                if c_opt is not None:
                    # Check validity again (SA should guarantee this, but just in case)
                    if _is_valid(c_opt, a_opt):
                        side = _bounding_side(c_opt, a_opt, outer_theta)
                        if side < best_side:
                            best_side = side
                            # Store params
                            best_params = np.zeros(3 * N)
                            for i in range(N):
                                best_params[3*i] = c_opt[i][0]
                                best_params[3*i+1] = c_opt[i][1]
                                best_params[3*i+2] = a_opt[i]
                            best_outer_theta = outer_theta
                            
                            # Calculate optimal center for this configuration
                            # The center of the bounding box in normal space
                            norms = np.array([
                                [math.cos(outer_theta + math.pi / 6.0 + k * math.pi / 3.0),
                                 math.sin(outer_theta + math.pi / 6.0 + k * math.pi / 3.0)] 
                                for k in range(6)
                            ])
                            min_proj = np.inf * np.ones(6)
                            max_proj = -np.inf * np.ones(6)
                            for i in range(N):
                                verts = _hex_vertices(a_opt[i], c_opt[i])
                                projs = verts @ norms.T
                                min_proj = np.minimum(min_proj, projs.min(axis=0))
                                max_proj = np.maximum(max_proj, projs.max(axis=0))
                            
                            # The center of the bounding hexagon is the midpoint of the projections
                            # For a regular hexagon, the center is the average of the 6 support points?
                            # Let's use the centroid of the vertices as a robust estimate.
                            all_verts = []
                            for i in range(N):
                                all_verts.append(_hex_vertices(a_opt[i], c_opt[i]))
                            all_verts = np.vstack(all_verts)
                            centroid = all_verts.mean(axis=0)
                            best_outer_center = tuple(float(x) for x in centroid)
                            
            except Exception as e:
                pass

    if best_params is None:
        # Fallback
        centers, angles = init_lattice()
        best_params = np.zeros(3 * N)
        for i in range(N):
            best_params[3*i] = centers[i][0]
            best_params[3*i+1] = centers[i][1]
            best_params[3*i+2] = angles[i]
        best_side = _bounding_side(np.array(centers), np.array(angles), 0.0)
        best_outer_theta = 0.0
        best_outer_center = (0.0, 0.0)

    # Finalize
    inner = []
    for i in range(N):
        x = float(best_params[3*i])
        y = float(best_params[3*i+1])
        a_deg = float(math.degrees(best_params[3*i+2]))
        inner.append([x, y, a_deg])
        
    return inner, best_outer_center, best_side, float(math.degrees(best_outer_theta))

# EVOLVE-BLOCK-END
