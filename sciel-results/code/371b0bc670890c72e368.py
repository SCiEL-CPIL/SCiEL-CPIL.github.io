import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

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
    # Normals for a regular hexagon pointing outwards from the center
    # Vertices are at 0, 60, ... degrees. Normals are at 30, 90, ... degrees
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate point to align outer hexagon with axes (if needed) 
            # Actually, we can just project onto the rotated normals directly
            # Let's stick to projecting onto the normals defined in the rotated frame
            # If outer hex is rotated by alpha, its normals are at 30+alpha + i*60
            # The code above calculates normals for 0 rotation. 
            # We need to rotate the point OR the normal. 
            # Let's rotate the point by -outer_angle to align the container with axes.
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    # Distance from center to side is (side * SQRT3) / 2
    # max_projection is the distance from center to the furthest vertex in the aligned frame
    # So max_projection = (S * SQRT3) / 2  => S = 2 * max_projection / SQRT3
    return 2.0 * max_projection / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap using SAT."""
    axes = []
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
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
    for i in range(n):
        vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
        for j in range(i + 1, n):
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True

def compute_overlap_penalty(inner, min_dist=SQRT3 - 0.01):
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
    
    # Penalty if distance is less than min_dist
    # min_dist for hexagons of side 1 is SQRT3 (distance between centers in lattice)
    # If they are rotated, the minimum distance can be smaller, but SQRT3 is a safe heuristic for penalty
    # Actually, for hexagons, the minimum center distance to avoid overlap is SQRT3 if aligned.
    # If rotated 30 deg, it's 1.5.
    # Using SQRT3 - small epsilon as a threshold is standard for lattice-based penalties.
    diff = np.maximum(0.0, min_dist - dists)
    return float(np.sum(diff * diff))

def generate_seeds():
    """Generate multiple diverse initial configurations for N=11 hexagons."""
    seeds = []
    
    # Seed 1: Standard 3-4-4 lattice (rows of 3, 4, 4)
    # Axial coordinates for a hexagonal packing
    # Row 0 (top): 3 hexagons
    # Row 1 (mid): 4 hexagons
    # Row 2 (bot): 4 hexagons
    # This is a standard dense packing.
    
    # Let's use axial coordinates (q, r)
    # Center at (0,0)
    # We want to pack 11.
    # A common shape is a central hex, surrounded by 6, and 4 more.
    
    # Configuration A: "Cluster"
    coords_A = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    
    # Configuration B: "Strip" - 4-3-4
    coords_B = [
        # Row -1 (y = -SQRT3/2 * 1.5 ? No, let's just use axial)
        # q, r
        (-1, 1), (0, 1), (1, 1), (2, 1),  # 4 hexes
        (-1, 0), (0, 0), (1, 0),           # 3 hexes
        (-2, -1), (-1, -1), (0, -1), (1, -1) # 4 hexes? 
        # Let's just take the first 11 from a larger grid
    ]
    
    # Simpler: Just generate a 5x5 grid of axial coords and take 11 closest to center
    all_coords = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            all_coords.append((q, r))
    
    # Sort by distance from center (0,0) in axial metric
    def axial_dist(c):
        return abs(c[0]) + abs(c[1]) + abs(c[0] + c[1])
        
    all_coords.sort(key=axial_dist)
    
    # Take 11 closest
    coords_C = all_coords[:11]
    
    # Add some random perturbations to break symmetry and create diversity
    import random
    random.seed(42)
    
    for base_coords in [coords_A, coords_C]:
        for _ in range(5):
            perturbed = []
            for q, r in base_coords:
                # Convert to Cartesian
                x = 1.5 * q
                y = SQRT3 * (r + q / 2.0)
                # Perturb
                x += random.uniform(-0.1, 0.1)
                y += random.uniform(-0.1, 0.1)
                angle = random.uniform(0, 360)
                perturbed.append([x, y, angle])
            seeds.append(perturbed)
            
    # Add a "ring" configuration
    ring_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2) # Same as A actually, let's vary it
    ]
    # Let's try a different set: 2-3-3-3
    # This might be harder to generate manually, so we stick to the sorted grid approach with different seeds
    
    return seeds

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin for 150s limit
    
    seeds = generate_seeds()
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0, 0)
    best_outer_angle = 0.0
    
    # We will optimize the positions of the inner hexagons.
    # Due to symmetry, we can fix the outer hexagon to center (0,0) and angle 0, 
    # and just move the inner hexagons. 
    # Wait, the objective is to minimize the outer side length.
    # If we fix outer center to (0,0) and angle 0, the "side length" is determined by the max projection of inner hexagons.
    # So we just need to minimize the max projection of the inner hexagons from the origin.
    # This simplifies the problem: minimize f(x) = max_i max_v proj(v_i)
    # subject to no overlaps.
    
    def objective(params):
        # params: [x0, y0, a0, x1, y1, a1, ...]
        inner = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            inner.append([x, y, a])
        
        # Compute side length assuming outer center (0,0) and angle 0
        side = compute_required_side(inner, (0, 0), 0.0)
        
        # Compute overlap penalty
        penalty = compute_overlap_penalty(inner)
        
        # We want to minimize side. If there is overlap, we add a large penalty.
        # If penalty is non-zero, the configuration is invalid.
        return side + 1000.0 * penalty
    
    # Multi-start optimization
    for seed_idx, seed in enumerate(seeds):
        if time.time() > deadline - 10:
            break
            
        # Prepare initial parameters
        x0 = []
        for x, y, a in seed:
            x0.extend([x, y, a])
            
        try:
            # Use SLSQP or L-BFGS-B
            # L-BFGS-B is faster but doesn't handle constraints well directly, we use penalty.
            # SLSQP can be slower but more robust for this size.
            # Let's try L-BFGS-B with a tight tolerance.
            
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < best_side:
                # Verify validity
                inner_res = []
                for i in range(N):
                    inner_res.append([res.x[i*3], res.x[i*3+1], res.x[i*3+2]])
                
                if check_all_disjoint(inner_res, eps=1e-5):
                    # Calculate exact side
                    side_exact = compute_required_side(inner_res, (0, 0), 0.0)
                    if side_exact < best_side:
                        best_side = side_exact
                        best_inner = inner_res
                        best_outer_center = (0, 0)
                        best_outer_angle = 0.0
        except Exception:
            pass
    
    # If we didn't find anything valid, fallback to a known valid configuration
    if best_inner is None:
        # Use the first seed as is, ensuring it's valid by spreading them out slightly
        # But seeds are usually valid if generated from lattice.
        # Let's just return the best found or a default
        if len(seeds) > 0:
             best_inner = seeds[0]
             best_side = compute_required_side(best_inner, (0,0), 0.0)
        else:
             # Fallback to a simple grid
             best_inner = [[i*1.8, 0, 0] for i in range(N)]
             best_side = 100.0

    # Final polish: Optimize the outer center and angle as well, with inner fixed
    # This can slightly reduce the side length if the bounding box is not centered
    def objective_outer(params):
        cx, cy, angle = params
        return compute_required_side(best_inner, (cx, cy), angle)
    
    # Initial guess for outer center is the centroid of inner hexagons
    cx_avg = sum(p[0] for p in best_inner) / N
    cy_avg = sum(p[1] for p in best_inner) / N
    
    # Try a few angles
    best_outer_val = best_side
    best_outer_params = [cx_avg, cy_avg, 0.0]
    
    for init_angle in [0, 30, 60]:
        if time.time() > deadline - 2:
            break
        try:
            res_outer = minimize(objective_outer, [cx_avg, cy_avg, float(init_angle)], 
                                 method='Nelder-Mead', 
                                 options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res_outer.fun < best_outer_val:
                best_outer_val = res_outer.fun
                best_outer_params = res_outer.x
        except Exception:
            pass
    
    best_outer_center = (best_outer_params[0], best_outer_params[1])
    best_outer_angle = best_outer_params[2]
    best_side = best_outer_val
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
