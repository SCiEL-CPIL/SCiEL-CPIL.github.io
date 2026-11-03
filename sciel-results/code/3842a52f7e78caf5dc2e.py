"""Simulated annealing packing of 12 unit hexagons in a regular hexagon."""
import time
import math
import random
import numpy as np

N = 12
DEADLINE = 135.0  # safety margin before the 150s kill

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _hex_vertices(theta):
    # Regular hexagon, side 1, circumradius 1, rotated by theta
    return np.array([[math.cos(theta + k * math.pi / 3.0),
                      math.sin(theta + k * math.pi / 3.0)] for k in range(6)])

def _norms(theta):
    # 6 edge-normal directions of a regular hexagon oriented by theta
    return np.array([[math.cos(theta + math.pi / 6.0 + k * math.pi / 3.0),
                      math.sin(theta + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])

def _bounding_side_fast(pts, theta):
    # side length of smallest regular hexagon (orientation theta, centered at origin)
    # containing the set of points pts.
    nrm = _norms(theta)
    P = pts @ nrm.T
    req = P.max(axis=0)
    return 2.0 * float(np.max(req)) / math.sqrt(3.0)

def _compute_center(pts, theta):
    # Compute optimal center for bounding hexagon with orientation theta
    nrm = _norms(theta)
    P = pts @ nrm.T
    # For each face k, we need c[k] such that P[:,k] - c[k] <= s*sqrt(3)/2
    # and c[k] - P[:,k] <= s*sqrt(3)/2 (symmetric)
    # Optimal: c[k] = (P[:,k].max() + P[:,k].min()) / 2
    c = (P.max(axis=0) + P.min(axis=0)) / 2.0
    # Transform back to world coords
    # c_world = nrm.T @ c
    c_world = nrm.T @ c
    return c_world

def _is_valid(centers, angles, outer_side, outer_theta, outer_center):
    """Check if all hexagons are non-overlapping and contained in outer hexagon."""
    EPS = 1e-9
    verts = [_hex_vertices(angles[k]) + centers[k] for k in range(N)]
    
    # Check pairwise overlaps via SAT
    for i in range(N):
        for j in range(i + 1, N):
            A = verts[i]
            B = verts[j]
            separated = False
            for poly in (A, B):
                for k in range(6):
                    p1 = poly[k]
                    p2 = poly[(k + 1) % 6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    projA = A @ normal
                    projB = B @ normal
                    if projA.max() < projB.min() - EPS or projB.max() < projA.min() - EPS:
                        separated = True
                        break
                if separated:
                    break
            if not separated:
                return False
    
    # Check containment in outer hexagon
    nrm = _norms(outer_theta)
    c = np.array(outer_center) @ nrm.T
    for k in range(N):
        P = verts[k] @ nrm.T
        for face in range(6):
            if (P[:, face] - c[face]).max() > outer_side * math.sqrt(3.0) / 2.0 + EPS:
                return False
    
    return True

# ---------------------------------------------------------------------------
# Seed generators
# ---------------------------------------------------------------------------
def _seed_lattice():
    """Triangular lattice seed, centered at origin."""
    d = 2.0 * math.sqrt(3.0) + 0.001
    a1 = np.array([d, 0.0])
    a2 = np.array([d / 2.0, d * math.sqrt(3.0) / 2.0])
    # 4x3 grid pattern
    pts = []
    for i in range(-1, 3):
        for j in range(-1, 2):
            pts.append(i * a1 + j * a2)
    pts = np.array(pts)
    c = pts.mean(axis=0)
    pts = pts - c
    return pts

def _seed_hex_cluster():
    """Hexagonal cluster: center + 6 neighbors + 5 more."""
    d = 2.0 * math.sqrt(3.0) + 0.001
    a1 = np.array([d, 0.0])
    a2 = np.array([d / 2.0, d * math.sqrt(3.0) / 2.0])
    coords = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
              (2, 0), (-2, 0), (0, 2), (0, -2), (2, -2)]
    pts = []
    for (i, j) in coords:
        pts.append(i * a1 + j * a2)
    pts = np.array(pts)
    c = pts.mean(axis=0)
    pts = pts - c
    return pts

def _seed_random():
    """Random positions in a disk of radius 5."""
    pts = []
    for _ in range(N):
        r = random.uniform(0, 4.0)
        theta = random.uniform(0, 2 * math.pi)
        pts.append([r * math.cos(theta), r * math.sin(theta)])
    pts = np.array(pts)
    c = pts.mean(axis=0)
    pts = pts - c
    return pts

# ---------------------------------------------------------------------------
# Simulated annealing
# ---------------------------------------------------------------------------
def _physics_relax(centers, angles, outer_side, outer_theta, outer_center, steps=500):
    """Simple physics-based relaxation to resolve overlaps."""
    centers = centers.copy()
    angles = angles.copy()
    vel = np.zeros((N, 2))
    
    for s in range(steps):
        if time.time() > DEADLINE:
            break
        verts = [_hex_vertices(angles[k]) + centers[k] for k in range(N)]
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                A = verts[i]
                B = verts[j]
                # Check overlap
                overlapping = True
                min_sep = float('inf')
                best_axis = None
                for poly in (A, B):
                    for k in range(6):
                        p1 = poly[k]
                        p2 = poly[(k + 1) % 6]
                        edge = p2 - p1
                        normal = np.array([-edge[1], edge[0]])
                        nl = math.hypot(normal[0], normal[1])
                        if nl < 1e-9:
                            continue
                        normal = normal / nl
                        projA = A @ normal
                        projB = B @ normal
                        sep = min(projA.max() - projB.min(), projB.max() - projA.min())
                        if sep >= -1e-9:
                            overlapping = False
                            break
                        if sep < min_sep:
                            min_sep = sep
                            best_axis = normal
                    if not overlapping:
                        break
                if overlapping and best_axis is not None:
                    f = 0.3 * (-min_sep)
                    vel[i] += f * best_axis
                    vel[j] -= f * best_axis
        
        # Containment force
        nrm = _norms(outer_theta)
        c = np.array(outer_center) @ nrm.T
        for i in range(N):
            P = verts[i] @ nrm.T
            for face in range(6):
                exceed = (P[:, face] - c[face]).max() - outer_side * math.sqrt(3.0) / 2.0
                if exceed > 0:
                    vel[i] -= 0.5 * exceed * nrm[face]
        
        # Mild attraction to center
        for i in range(N):
            vel[i] += -0.01 * centers[i]
        
        # Integrate
        vel *= 0.9
        centers += vel * 0.05
    
    return centers, angles

def _simulated_annealing(centers, angles, outer_theta, max_iters=20000):
    """Simulated annealing to minimize bounding hexagon side length."""
    best_side = float('inf')
    best_centers = centers.copy()
    best_angles = angles.copy()
    
    # Initial evaluation
    all_pts = np.vstack([_hex_vertices(angles[k]) + centers[k] for k in range(N)])
    side = _bounding_side_fast(all_pts, outer_theta)
    if side < best_side:
        best_side = side
        best_centers = centers.copy()
        best_angles = angles.copy()
    
    T = 0.5  # initial temperature
    T_min = 0.001
    cooling = 0.9995
    
    for it in range(max_iters):
        if time.time() > DEADLINE:
            break
        
        # Propose move
        i = random.randint(0, N - 1)
        move_type = random.random()
        
        if move_type < 0.7:
            # Move center
            delta = np.array([random.gauss(0, T), random.gauss(0, T)])
            new_centers = centers.copy()
            new_centers[i] += delta
            new_angles = angles.copy()
        else:
            # Rotate
            new_centers = centers.copy()
            new_angles = angles.copy()
            new_angles[i] += random.gauss(0, T * 0.5)
        
        # Evaluate
        all_pts = np.vstack([_hex_vertices(new_angles[k]) + new_centers[k] for k in range(N)])
        new_side = _bounding_side_fast(all_pts, outer_theta)
        
        # Check validity (non-overlapping)
        if not _is_valid(new_centers, new_angles, new_side, outer_theta, (0.0, 0.0)):
            # Reject invalid moves
            T *= cooling
            continue
        
        # Accept/reject
        if new_side < side:
            side = new_side
            centers = new_centers
            angles = new_angles
            if side < best_side:
                best_side = side
                best_centers = centers.copy()
                best_angles = angles.copy()
        else:
            delta_E = new_side - side
            if random.random() < math.exp(-delta_E / T):
                side = new_side
                centers = new_centers
                angles = new_angles
        
        T = max(T * cooling, T_min)
    
    return best_centers, best_angles, best_side

# ---------------------------------------------------------------------------
# Outer angle/center refinement
# ---------------------------------------------------------------------------
def _refine_outer(centers, angles):
    """Sweep outer angle and compute optimal center; return best (side, center, angle)."""
    all_pts = np.vstack([_hex_vertices(angles[k]) + centers[k] for k in range(N)])
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for ang in np.linspace(0.0, math.pi / 3.0, 181):
        nrm = _norms(ang)
        P = all_pts @ nrm.T
        # Optimal center in normal space
        c_normal = (P.max(axis=0) + P.min(axis=0)) / 2.0
        c_world = nrm.T @ c_normal
        side = 2.0 * float(np.max(P.max(axis=0) - c_normal)) / math.sqrt(3.0)
        if side < best_side - 1e-12:
            best_side = side
            best_center = (float(c_world[0]), float(c_world[1]))
            best_angle = float(ang)
    
    return best_side, best_center, best_angle

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def construct():
    start = time.time()
    random.seed(42)
    np.random.seed(42)

    best_side = float('inf')
    best_centers = None
    best_angles = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    def _consider(centers, angles, side, center, angle):
        nonlocal best_side, best_centers, best_angles, best_center, best_angle
        if side < best_side:
            best_side = side
            best_centers = centers.copy()
            best_angles = angles.copy()
            best_center = center
            best_angle = angle

    # Try multiple seeds and outer angles
    seed_funcs = [_seed_lattice, _seed_hex_cluster, _seed_random]
    outer_angles = [0.0, math.pi / 6.0, math.pi / 3.0 * 0.5, math.pi / 3.0 * 0.25]

    for seed_func in seed_funcs:
        for oa in outer_angles:
            if time.time() > DEADLINE:
                break
            
            centers = seed_func()
            angles = np.zeros(N)
            
            # Physics relaxation to resolve initial overlaps
            centers, angles = _physics_relax(centers, angles, 8.0, oa, (0.0, 0.0), steps=300)
            
            # Simulated annealing
            bc, ba, bs = _simulated_annealing(centers, angles, oa, max_iters=15000)
            
            # Refine outer
            side, center, angle = _refine_outer(bc, ba)
            
            _consider(bc, ba, side, center, angle)
    
    # Final fine sweep on best configuration
    if best_centers is not None:
        side, center, angle = _refine_outer(best_centers, best_angles)
        _consider(best_centers, best_angles, side, center, angle)

    # Fallback
    if best_centers is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Format output
    inner = []
    for k in range(N):
        inner.append([float(best_centers[k][0]), float(best_centers[k][1]), float(math.degrees(best_angles[k]))])
    
    return inner, best_center, best_side, math.degrees(best_angle)
# EVOLVE-BLOCK-END
