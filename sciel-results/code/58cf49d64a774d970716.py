"""Combined 11-hexagon packing: Topology-based optimization with 1D angle search
and local positional refinement."""
import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = math.sqrt(3) / 2.0

# Precomputed constants
NORMALS = np.array([[math.cos(math.radians(30 + i * 60)),
                     math.sin(math.radians(30 + i * 60))] for i in range(6)])
HEX_ANG = np.arange(6) * np.pi / 3.0

# EVOLVE-BLOCK-START

def hex_vertices_arr(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as a (6,2) numpy array."""
    thetas = angle_rad + HEX_ANG
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0

    ox, oy = outer_center
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])

    max_proj = -np.inf
    for (ix, iy, iangle) in inner:
        verts = hex_vertices_arr(ix, iy, 1.0, math.radians(iangle))
        local = (verts - np.array([ox, oy])) @ R.T
        projs = local @ NORMALS.T
        cur = projs.max()
        if cur > max_proj:
            max_proj = cur

    return 2.0 * max_proj / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap using SAT."""
    axes = []
    for poly in (v1, v2):
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            normal = (-edge[1], edge[0])
            norm = math.sqrt(normal[0]**2 + normal[1]**2)
            if norm > 1e-9:
                axes.append((normal[0] / norm, normal[1] / norm))
    
    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    polys = []
    for (x, y, a) in inner:
        polys.append(hex_vertices_arr(x, y, 1.0, math.radians(a)))
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap_check(polys[i], polys[j], eps):
                return False
    return True

def compute_overlap_penalty(inner, min_dist=SQRT3 - 0.005):
    """Compute a smooth penalty for overlaps based on center distances."""
    n = len(inner)
    if n < 2:
        return 0.0

    xs = np.array([inner[i][0] for i in range(n)])
    ys = np.array([inner[i][1] for i in range(n)])

    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists_sq = dx * dx + dy * dy

    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dists_sq[iu])

    diff = np.maximum(0.0, min_dist - dists)
    return float(np.sum(diff * diff))

def optimize_outer_1d(inner):
    """
    Optimize the outer hexagon rotation angle, assuming the center is the centroid.
    Returns (best_angle, best_side).
    """
    cx = sum(x for x, y, a in inner) / len(inner)
    cy = sum(y for x, y, a in inner) / len(inner)
    
    def objective(angle_deg):
        return compute_required_side(inner, (cx, cy), angle_deg)
    
    # Global search: 0 to 30 degrees (symmetry period is 60, but 30 is enough for uniqueness in many cases, 
    # actually 60 degrees is the full symmetry period for a regular hexagon container.
    # However, the inner shape might have lower symmetry. Let's search 0-60.
    
    best_val = float('inf')
    best_angle = 0.0
    
    # Grid search first to find good regions
    for a in range(0, 60, 5):
        val = objective(a)
        if val < best_val:
            best_val = val
            best_angle = a
            
    # Refine around the best grid point
    # Also check nearby points to handle local minima
    search_ranges = []
    # Refine the best found
    search_ranges.append((best_angle - 5, best_angle + 5))
    # Check all grid points just in case
    for a in range(0, 60, 5):
        search_ranges.append((a - 5, a + 5))
        
    for start, end in search_ranges:
        try:
            res = minimize_scalar(objective, bounds=(start, end), method='bounded', 
                                  options={'xatol': 1e-10})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_angle = res.x
        except Exception:
            pass
            
    return best_angle, best_val, (cx, cy)

def local_search(inner, time_limit):
    """
    Perform a local search to refine inner positions, keeping them disjoint.
    Returns the best inner list found.
    """
    start_time = time.time()
    current_inner = [list(h) for h in inner]
    cx, cy = sum(h[0] for h in current_inner)/N, sum(h[1] for h in current_inner)/N
    
    # Initial side length with optimized outer
    _, initial_side, _ = optimize_outer_1d(current_inner)
    best_side = initial_side
    best_inner = [list(h) for h in current_inner]
    
    T = 0.05
    T_min = 0.001
    cooling = 0.9995
    
    def get_side_and_center(inner_list):
        angle, side, center = optimize_outer_1d(inner_list)
        return side, center, angle

    while time.time() < start_time + time_limit:
        # Perturb a random hexagon
        idx = np.random.randint(0, N)
        new_inner = [list(h) for h in current_inner]
        
        # Small random move
        delta_x = np.random.normal(0, 0.05)
        delta_y = np.random.normal(0, 0.05)
        new_inner[idx][0] += delta_x
        new_inner[idx][1] += delta_y
        
        # Check disjointness
        if not check_all_disjoint(new_inner, eps=1e-5):
            continue
            
        # Evaluate
        new_side, _, _ = get_side_and_center(new_inner)
        
        # Accept if better or with probability exp(-delta/T)
        if new_side < best_side:
            best_side = new_side
            best_inner = [list(h) for h in new_inner]
            current_inner = [list(h) for h in new_inner]
        else:
            delta = new_side - best_side
            if np.random.random() < math.exp(-delta / max(T, T_min)):
                current_inner = [list(h) for h in new_inner]
                
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_inner, best_side

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Candidate 1: Symmetric Cross (Center + 6 Ring + 4 Diagonal)
    # Axial coords: (q, r)
    # Center: (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 Diagonals: (2,0), (0,2), (-2,1), (-1,-1) -- Wait, (-2,1) and (-1,-1) are not symmetric to (2,0) and (0,2) in a simple way.
    # Let's use the "3-4-3" like cluster or the "Cross" that fits in a hexagon better.
    # A known good packing for 11 is a 3-4-3-1? No.
    # Let's try the "Hexagon of side 2" minus 8 corners.
    # Side 2 hexagon has 19 hexes.
    # If we remove the 8 hexes at the corners of the side-2 hexagon?
    # The side-2 hexagon corners are at axial distances 2 from center in 6 directions.
    # Those are (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2).
    # Removing these 6 leaves 13.
    # Remove 2 more?
    
    # Let's stick to the candidates from the previous version which were tested, but optimize them better.
    candidates = []
    
    # Candidate 1: Symmetric Cross
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append(axial_1)

    # Candidate 2: Cluster (Center + 6 + 4 adjacent)
    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    candidates.append(axial_2)

    best_global_side = float('inf')
    best_global_inner = None
    best_global_center = None
    best_global_angle = 0.0

    for idx, axial_coords in enumerate(candidates):
        if time.time() > deadline - 30:
            break

        if len(axial_coords) > N:
            axial_coords = axial_coords[:N]

        # Convert axial to cartesian
        # x = 1.5 * q
        # y = sqrt(3) * (r + q/2)
        centers = []
        for q, r in axial_coords:
            x = 1.5 * q
            y = SQRT3 * (r + q / 2.0)
            centers.append((x, y))

        initial_inner = [[x, y, 0.0] for x, y in centers]

        # Phase 1: Optimize outer container for this fixed topology
        angle, side, center = optimize_outer_1d(initial_inner)
        
        if side < best_global_side:
            best_global_side = side
            best_global_inner = initial_inner
            best_global_center = center
            best_global_angle = angle

        # Phase 2: Local search to improve inner positions
        if time.time() < deadline - 10:
            time_for_ls = min(10.0, deadline - time.time() - 5)
            improved_inner, improved_side = local_search(initial_inner, time_for_ls)
            
            # Re-optimize outer for the improved inner
            angle2, side2, center2 = optimize_outer_1d(improved_inner)
            
            if side2 < best_global_side:
                best_global_side = side2
                best_global_inner = improved_inner
                best_global_center = center2
                best_global_angle = angle2

    # Fallback if no valid result found (should not happen)
    if best_global_inner is None:
        best_global_inner = [[0,0,0]] * N
        best_global_center = (0,0)
        best_global_side = 10.0
        best_global_angle = 0.0

    return best_global_inner, best_global_center, best_global_side, best_global_angle

# EVOLVE-BLOCK-END
