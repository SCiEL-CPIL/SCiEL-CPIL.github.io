"""Combined: hexagonal lattice representation + SAT validation + SA refinement."""
import time
import math
import random
import numpy as np

N = 12
DEADLINE = 145.0  # Safety margin before the 150s kill

# EVOLVE-BLOCK-START

def _hex_vertices(theta):
    """Vertices of unit hexagon (circumradius 1) centered at origin, rotated by theta."""
    return np.array([[math.cos(theta + k * math.pi / 3.0),
                      math.sin(theta + k * math.pi / 3.0)] for k in range(6)])

def _sat_overlap(A, B):
    """SAT check for two convex hexagons. Returns True if they overlap."""
    for i in range(6):
        p1 = A[i]
        p2 = A[(i + 1) % 6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        projA = A @ normal
        projB = B @ normal
        if projA.max() < projB.min() or projB.max() < projA.min():
            return False
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

def _bounding_side(centers, angles, outer_theta):
    """Min side length of regular hexagon (orientation outer_theta) containing all inner hexagons."""
    norms = np.array([[math.cos(outer_theta + math.pi / 6.0 + k * math.pi / 3.0),
                       math.sin(outer_theta + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])
    min_proj = np.full(6, np.inf)
    max_proj = np.full(6, -np.inf)
    for i in range(N):
        verts = _hex_vertices(angles[i]) + centers[i]
        projs = verts @ norms.T
        min_proj = np.minimum(min_proj, projs.min(axis=0))
        max_proj = np.maximum(max_proj, projs.max(axis=0))
    widths = max_proj - min_proj
    return float(np.max(widths) / math.sqrt(3.0))

def _is_valid(centers, angles):
    """Check no overlaps among inner hexagons."""
    verts = [_hex_vertices(angles[i]) + centers[i] for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if _sat_overlap(verts[i], verts[j]):
                return False
    return True

def _lattice_positions(rot, shifts, scale=1.0):
    """
    Generate positions on a triangular lattice.
    rot: lattice rotation angle
    shifts: list of (i, j) integer lattice coordinates for 12 hexagons
    scale: lattice spacing multiplier (base spacing = 2*sqrt(3) for non-overlapping unit hexagons)
    Returns (centers, angles) where all hexagons share the same orientation = rot.
    """
    base_spacing = 2.0 * math.sqrt(3.0) * scale
    a1 = np.array([base_spacing, 0.0])
    a2 = np.array([base_spacing / 2.0, base_spacing * math.sqrt(3.0) / 2.0])
    # Rotate the lattice
    c, s = math.cos(rot), math.sin(rot)
    R = np.array([[c, -s], [s, c]])
    a1 = R @ a1
    a2 = R @ a2
    
    centers = []
    for (i, j) in shifts:
        p = i * a1 + j * a2
        centers.append(p)
    centers = np.array(centers)
    # Center the configuration
    centroid = centers.mean(axis=0)
    centers = centers - centroid
    angles = [rot] * N
    return centers, angles

def _lattice_positions_mixed(rot, shifts, scale=1.0, individual_rotations=None):
    """
    Lattice positions with optional individual rotation offsets.
    individual_rotations: list of N rotation offsets (in radians)
    """
    if individual_rotations is None:
        individual_rotations = [0.0] * N
    base_spacing = 2.0 * math.sqrt(3.0) * scale
    a1 = np.array([base_spacing, 0.0])
    a2 = np.array([base_spacing / 2.0, base_spacing * math.sqrt(3.0) / 2.0])
    c, s = math.cos(rot), math.sin(rot)
    R = np.array([[c, -s], [s, c]])
    a1 = R @ a1
    a2 = R @ a2
    
    centers = []
    for (i, j) in shifts:
        p = i * a1 + j * a2
        centers.append(p)
    centers = np.array(centers)
    centroid = centers.mean(axis=0)
    centers = centers - centroid
    angles = [rot + individual_rotations[i] for i in range(N)]
    return centers, angles

def _sa_refine(centers, angles, outer_theta, T_start=0.3, T_end=0.005, alpha=0.9995, max_steps=5000):
    """
    Simulated Annealing to minimize bounding side length while maintaining validity.
    """
    centers = np.array(centers, dtype=float)
    angles = np.array(angles, dtype=float)
    
    current_side = _bounding_side(centers, angles, outer_theta)
    best_side = current_side
    best_centers = centers.copy()
    best_angles = angles.copy()
    
    T = T_start
    step_size = 0.4
    steps = 0
    
    while T > T_end and steps < max_steps:
        i = random.randrange(N)
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, step_size * 0.3)
        
        old_cx, old_cy, old_a = centers[i][0], centers[i][1], angles[i]
        centers[i][0] += dx
        centers[i][1] += dy
        angles[i] += da
        
        if not _is_valid(centers, angles):
            centers[i][0] = old_cx
            centers[i][1] = old_cy
            angles[i] = old_a
        else:
            new_side = _bounding_side(centers, angles, outer_theta)
            if new_side < current_side or random.random() < math.exp(-(new_side - current_side) / T):
                current_side = new_side
                if new_side < best_side:
                    best_side = new_side
                    best_centers = centers.copy()
                    best_angles = angles.copy()
            else:
                centers[i][0] = old_cx
                centers[i][1] = old_cy
                angles[i] = old_a
        
        T *= alpha
        step_size *= 0.9998
        steps += 1
    
    return best_centers, best_angles, best_side

def _nelder_mead_pack(params, outer_theta):
    """
    Nelder-Mead style local optimization on continuous parameters.
    params: flat array of [x0,y0,a0, x1,y1,a1, ...] (3*N values)
    Returns optimized params and the side length.
    """
    def objective(p):
        centers = p[:2*N].reshape(N, 2)
        angles = p[2*N:]
        if not _is_valid(centers, angles):
            # Penalty for overlap
            verts = [_hex_vertices(angles[i]) + centers[i] for i in range(N)]
            penalty = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    if _sat_overlap(verts[i], verts[j]):
                        penalty += 1.0
            return 100.0 + penalty + _bounding_side(centers, angles, outer_theta)
        return _bounding_side(centers, angles, outer_theta)
    
    # Simple Nelder-Mead implementation
    n = len(params)
    simplex = [np.array(params, dtype=float)]
    for k in range(n):
        p = np.array(params, dtype=float)
        p[k] += 0.1 if p[k] == 0 else 0.1 * p[k]
        simplex.append(p)
    
    for _ in range(300):
        vals = [(objective(v), v) for v in simplex]
        vals.sort(key=lambda x: x[0])
        simplex = [v for _, v in vals]
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        
        # Reflection
        alpha_r = 1.0
        xr = centroid + alpha_r * (centroid - worst)
        fr = objective(xr)
        
        if fr < vals[0][0]:
            # Expansion
            gamma = 2.0
            xe = centroid + gamma * (centroid - worst)
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
            else:
                simplex[-1] = xr
        elif fr < vals[-2][0]:
            simplex[-1] = xr
        else:
            # Contraction
            rho = 0.5
            xc = centroid + rho * (worst - centroid)
            fc = objective(xc)
            if fc < vals[-1][0]:
                simplex[-1] = xc
            else:
                # Shrink
                for k in range(1, n + 1):
                    simplex[k] = simplex[0] + 0.5 * (simplex[k] - simplex[0])
    
    vals = [(objective(v), v) for v in simplex]
    vals.sort(key=lambda x: x[0])
    return vals[0][1], vals[0][0]

def _generate_lattice_shifts():
    """
    Generate candidate lattice shift patterns for 12 hexagons.
    Returns list of shift patterns, each a list of 12 (i,j) tuples.
    """
    patterns = []
    
    # Pattern 1: 4x3 rectangular block (centered)
    shifts1 = [(i, j) for i in range(-1, 3) for j in range(-1, 2)]
    patterns.append(shifts1)
    
    # Pattern 2: 3x4 rectangular block
    shifts2 = [(i, j) for i in range(-2, 2) for j in range(-1, 3)]
    patterns.append(shifts2)
    
    # Pattern 3: Hexagonal cluster (center + 6 ring + 5 more)
    # Center
    # Ring 1: 6 neighbors
    # Ring 2: pick 5 from 12
    shifts3 = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    shifts3.extend(ring1)
    # Ring 2 candidates
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2), (0, 2), (1, 1)]
    # Pick 5 that form a compact shape
    shifts3.extend(ring2[0:5])
    patterns.append(shifts3)
    
    # Pattern 4: Two rows of 6
    shifts4 = [(i, 0) for i in range(-3, 3)] + [(i, 1) for i in range(-3, 3)]
    patterns.append(shifts4)
    
    # Pattern 5: 6x2 with offset
    shifts5 = [(i, 0) for i in range(-3, 3)] + [(i + 0.5, 1) for i in range(-3, 3)]
    # Convert to integers by using a different lattice basis
    # Actually for triangular lattice, let's use proper integer coords
    shifts5 = [(i, 0) for i in range(-3, 3)] + [(i, 1) for i in range(-3, 3)]
    patterns.append(shifts5)
    
    # Pattern 6: Cross/plus shape
    shifts6 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1), 
               (2, 0), (-2, 0), (0, 2), (0, -2), (1, -2)]
    patterns.append(shifts6)
    
    # Pattern 7: Compact 3x4 with one missing corner
    shifts7 = [(i, j) for i in range(-2, 2) for j in range(-1, 3) if not (i == 1 and j == 2)]
    shifts7.append((2, 1))
    patterns.append(shifts7)
    
    # Pattern 8: Two offset rows
    shifts8 = [(i, 0) for i in range(-2, 4)] + [(i, 1) for i in range(-3, 3)]
    patterns.append(shifts8)
    
    return patterns

def _compute_optimal_center(centers, angles, outer_theta):
    """Compute the center of the tightest bounding regular hexagon."""
    norms = np.array([[math.cos(outer_theta + math.pi / 6.0 + k * math.pi / 3.0),
                       math.sin(outer_theta + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])
    min_proj = np.full(6, np.inf)
    max_proj = np.full(6, -np.inf)
    for i in range(N):
        verts = _hex_vertices(angles[i]) + centers[i]
        projs = verts @ norms.T
        min_proj = np.minimum(min_proj, projs.min(axis=0))
        max_proj = np.maximum(max_proj, projs.max(axis=0))
    
    # The center in each normal direction is the midpoint
    # For a regular hexagon, these should all give the same point
    # Use least-squares to find the best center
    midpoints = (min_proj + max_proj) / 2.0
    # Solve norms.T @ c = midpoints (overdetermined, use least squares)
    c, _, _, _ = np.linalg.lstsq(norms.T, midpoints, rcond=None)
    return tuple(float(x) for x in c)

def construct():
    start_time = time.time()
    random.seed(42)
    
    best_side = float('inf')
    best_inner = None
    best_outer = (0.0, 0.0)
    best_theta = 0.0
    
    def elapsed():
        return time.time() - start_time
    
    def update_best(centers, angles, outer_theta):
        nonlocal best_side, best_inner, best_outer, best_theta
        s = _bounding_side(centers, angles, outer_theta)
        if s < best_side:
            best_side = s
            best_inner = [[float(centers[i][0]), float(centers[i][1]), float(math.degrees(angles[i]))] for i in range(N)]
            best_outer = _compute_optimal_center(centers, angles, outer_theta)
            best_theta = outer_theta
    
    # Get lattice shift patterns
    patterns = _generate_lattice_shifts()
    
    # For each pattern, optimize lattice rotation and try a few outer orientations
    for pat_idx, shifts in enumerate(patterns):
        if elapsed() > DEADLINE - 30:
            break
        
        # Try a range of lattice rotations
        for rot in np.linspace(0, math.pi / 3, 13):
            if elapsed() > DEADLINE - 30:
                break
            
            # Get initial positions from lattice
            centers, angles = _lattice_positions(rot, shifts)
            
            # Check validity (should be valid for scale=1.0 since base spacing prevents overlap)
            if not _is_valid(centers, angles):
                continue
            
            # Try a few outer orientations
            for outer_theta in np.linspace(0, math.pi / 3, 7):
                if elapsed() > DEADLINE - 20:
                    break
                
                current_side = _bounding_side(centers, angles, outer_theta)
                if current_side < best_side:
                    update_best(centers, angles, outer_theta)
                
                # SA refinement
                if elapsed() < DEADLINE - 15:
                    c_ref, a_ref, s_ref = _sa_refine(centers, angles, outer_theta, 
                                                      T_start=0.2, T_end=0.005, 
                                                      alpha=0.9997, max_steps=2000)
                    if s_ref < best_side:
                        update_best(c_ref, a_ref, outer_theta)
                    
                    # Nelder-Mead refinement on the SA result
                    if elapsed() < DEADLINE - 10:
                        params = np.concatenate([c_ref.flatten(), a_ref])
                        opt_params, opt_val = _nelder_mead_pack(params, outer_theta)
                        if opt_val < best_side:
                            opt_centers = opt_params[:2*N].reshape(N, 2)
                            opt_angles = opt_params[2*N:]
                            if _is_valid(opt_centers, opt_angles):
                                update_best(opt_centers, opt_angles, outer_theta)
    
    # If we haven't found anything valid, fallback to a simple grid
    if best_inner is None:
        shifts = [(i, j) for i in range(-1, 3) for j in range(-1, 2)]
        centers, angles = _lattice_positions(0.0, shifts)
        best_inner = [[float(centers[i][0]), float(centers[i][1]), 0.0] for i in range(N)]
        best_outer = (0.0, 0.0)
        best_side = _bounding_side(centers, angles, 0.0)
        best_theta = 0.0
    
    return best_inner, best_outer, best_side, math.degrees(best_theta)

# EVOLVE-BLOCK-END
