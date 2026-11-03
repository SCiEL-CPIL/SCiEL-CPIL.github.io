"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

N = 12
DEADLINE = 145.0  # Safety margin before the 150s kill

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

def _bounding_side_fast(centers, angles, outer_theta):
    """
    Calculates the minimum side length of a regular hexagon (orientation outer_theta)
    that contains all inner hexagons defined by centers and angles.
    """
    # Normals for the outer hexagon
    norms = np.array([[math.cos(outer_theta + math.pi / 6.0 + k * math.pi / 3.0),
                       math.sin(outer_theta + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])
    
    min_proj = np.inf
    max_proj = -np.inf
    
    for i in range(N):
        verts = _hex_vertices(angles[i]) + centers[i]
        # Project vertices onto the 6 normal directions
        projs = verts @ norms.T
        min_proj = np.minimum(min_proj, projs.min(axis=0))
        max_proj = np.maximum(max_proj, projs.max(axis=0))
        
    widths = max_proj - min_proj
    # Side length s relates to width w by w = s * sqrt(3)
    return float(np.max(widths) / math.sqrt(3.0))

def _is_valid(centers, angles, tol=1e-9):
    """
    Checks if the configuration has no overlaps.
    """
    verts = [_hex_vertices(angles[i]) + centers[i] for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if _sat_overlap(verts[i], verts[j]):
                return False
    return True

def _compute_optimal_center(centers, angles, outer_theta):
    """
    Compute the optimal center for the outer hexagon given the configuration.
    The center c minimizes the maximum distance to the sides, which is equivalent to
    centering the bounding box.
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
    
    # The center in the normal basis is the midpoint of the projections
    # c_proj = (min_proj + max_proj) / 2
    # We need to map this back to Cartesian.
    # Since the normals are not orthogonal, we solve for c such that n_k . c = c_proj_k
    # This is an overdetermined system. We use least squares.
    c_proj = (min_proj + max_proj) / 2.0
    
    # A system: norms @ c = c_proj
    # norms is 6x2. c is 2x1.
    c, residuals, rank, sv = np.linalg.lstsq(norms, c_proj, rcond=None)
    
    return tuple(float(x) for x in c)

def _get_lattice_configs():
    """
    Generates a few candidate configurations of 12 hexagons on a triangular lattice.
    The lattice spacing is d = sqrt(3) + epsilon.
    """
    d = math.sqrt(3.0) + 0.01 # Slight gap to ensure disjointness
    e1 = np.array([d, 0.0])
    e2 = np.array([d / 2.0, d * math.sqrt(3.0) / 2.0])
    
    def gen(uvs):
        pts = []
        for u, v in uvs:
            pts.append(u * e1 + v * e2)
        return np.array(pts)
    
    configs = []
    
    # Config 1: 3x4 parallelogram
    uvs_1 = [(u, v) for u in range(3) for v in range(4)]
    configs.append(gen(uvs_1))
    
    # Config 2: Hexagonal cluster (radius 1 is 7, add 5 to make 12)
    # Center (0,0), neighbors (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    hex_7 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    # Add 5 more to extend one side, e.g., u=2
    extra_5 = [(2,0), (2,-1), (2,-2), (1,-2), (0,-2)]
    uvs_2 = hex_7 + extra_5
    configs.append(gen(uvs_2))
    
    # Config 3: A more compact "blob"
    # Try a 2x6 rectangle? Or a rhombus?
    uvs_3 = [(u, v) for u in range(2) for v in range(6)]
    configs.append(gen(uvs_3))
    
    # Config 4: Random lattice points, then we will let SA fix them
    # Just use a random set of 12 unique lattice points
    random.seed(42)
    uvs_4 = set()
    while len(uvs_4) < 12:
        u = random.randint(-2, 2)
        v = random.randint(-2, 2)
        uvs_4.add((u, v))
    configs.append(gen(list(uvs_4)))
    
    return configs

def _optimize_cluster(lattice_pts, outer_theta, time_limit=20.0):
    """
    Optimizes the positions and rotations of hexagons initialized at lattice_pts.
    Allows small deviations from the lattice to find a tighter packing.
    """
    start_time = time.time()
    
    # Initial centers and angles
    centers = lattice_pts.copy()
    # Center the cluster
    centers -= centers.mean(axis=0)
    angles = np.zeros(N)
    
    # Check validity
    if not _is_valid(centers, angles):
        # If invalid, spread them out slightly
        scale = 1.1
        centers *= scale
        if not _is_valid(centers, angles):
            return None, None, float('inf')
            
    current_side = _bounding_side_fast(centers, angles, outer_theta)
    best_side = current_side
    best_centers = centers.copy()
    best_angles = angles.copy()
    
    T = 0.5
    T_end = 0.01
    alpha = 0.99
    step_size = 0.1
    
    while T > T_end:
        if time.time() - start_time > time_limit:
            break
            
        # Pick a random hexagon
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
            centers[i][0] = old_cx
            centers[i][1] = old_cy
            angles[i] = old_a
            T *= alpha
            continue
        
        new_side = _bounding_side_fast(centers, angles, outer_theta)
        
        # Acceptance
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
        step_size *= 0.995
        
    return best_centers, best_angles, best_side

def construct():
    start_time = time.time()
    random.seed(42)
    
    best_side = float('inf')
    best_inner = None
    best_outer = (0.0, 0.0)
    best_theta = 0.0
    
    lattice_configs = _get_lattice_configs()
    
    # Try different outer hexagon orientations
    # The optimal orientation is likely aligned with the lattice or rotated by 30 deg
    outer_thetas = np.linspace(0, math.pi/3, 7)
    
    for config in lattice_configs:
        if time.time() - start_time > DEADLINE - 20:
            break
            
        for outer_theta in outer_thetas:
            if time.time() - start_time > DEADLINE - 10:
                break
                
            # Optimize the cluster for this outer orientation
            c_opt, a_opt, side_opt = _optimize_cluster(config, outer_theta, time_limit=15.0)
            
            if c_opt is not None:
                # Calculate the optimal center for the outer hexagon
                center_opt = _compute_optimal_center(c_opt, a_opt, outer_theta)
                
                # Verify the side length with the new center?
                # The side length is determined by the width, which is independent of the center translation.
                # So side_opt is correct.
                
                if side_opt < best_side:
                    best_side = side_opt
                    best_inner = [[float(c_opt[i][0]), float(c_opt[i][1]), float(math.degrees(a_opt[i]))] for i in range(N)]
                    best_outer = center_opt
                    best_theta = outer_theta

    # Fallback if no valid configuration found
    if best_inner is None:
        # Use the first lattice config as is, with identity rotation
        config = lattice_configs[0]
        centers = config - config.mean(axis=0)
        angles = np.zeros(N)
        best_inner = [[float(c[0]), float(c[1]), 0.0] for c in centers]
        best_side = _bounding_side_fast(centers, angles, 0.0)
        best_outer = (0.0, 0.0)
        best_theta = 0.0

    return best_inner, best_outer, best_side, math.degrees(best_theta)

# EVOLVE-BLOCK-END
