import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    theta = np.radians(angle_deg)
    angles = theta + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def polygons_overlap(poly1, poly2, tol=1e-9):
    """Check if two convex polygons overlap using SAT (Separating Axis Theorem)."""
    polys = [poly1, poly2]
    for poly in polys:
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            nx, ny = -dy, dx
            proj1 = np.dot(poly1, [nx, ny])
            proj2 = np.dot(poly2, [nx, ny])
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 - tol or max2 < min1 - tol:
                return False
    return True

def compute_min_outer_side(inner, outer_center, outer_angle_deg):
    """Compute the minimum outer hexagon side length needed to contain all inner hexagons."""
    theta = np.radians(outer_angle_deg)
    normals = np.column_stack([
        np.cos(theta + (np.arange(6) + 0.5) * np.pi / 3.0),
        np.sin(theta + (np.arange(6) + 0.5) * np.pi / 3.0)
    ])
    
    max_s = 0.0
    for row in inner:
        ix, iy, iangle = row
        verts = hex_vertices(ix, iy, 1.0, iangle)
        for v in verts:
            rel = v - np.array(outer_center)
            dists = np.dot(normals, rel)
            required_s = np.max(np.maximum(dists, 0)) * 2.0 / np.sqrt(3.0)
            max_s = max(max_s, required_s)
    return max_s

def check_no_overlaps(inner, tol=1e-6):
    """Check that no two inner hexagons overlap."""
    for i in range(len(inner)):
        vi = hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        for j in range(i+1, len(inner)):
            vj = hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            if polygons_overlap(vi, vj, tol):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s limit
    
    sqrt3 = np.sqrt(3.0)
    
    # Use proper hexagonal lattice packing
    # For 12 hexagons, use a compact hexagonal cluster
    # Hexagonal numbers: 1, 7, 19... so 12 is between 7 and 19
    # Use a 3-row arrangement: 4, 4, 4 with proper offsets
    
    # Proper hex grid: adjacent hexagons touch when center distance = 2*apothem = sqrt(3)
    # For flat-top hexagons (angle=0), width = 2*side = 2, height = sqrt(3)*side = sqrt(3)
    # Horizontal spacing between centers in same row: 2*side = 2 (for flat-top)
    # Actually for pointy-top (angle=30), width = sqrt(3), height = 2
    
    # Let's use pointy-top hexagons (angle = 30 degrees)
    # Width = sqrt(3), Height = 2
    # Horizontal spacing: sqrt(3)
    # Vertical spacing: 1.5 (3/4 of height)
    # Offset alternate rows by sqrt(3)/2
    
    angle_deg = 30.0  # pointy-top
    
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 4 hexagons
    
    inner_init = []
    # Row 0: y = -1.5
    for col in range(4):
        x = (col - 1.5) * sqrt3
        y = -1.5
        inner_init.append([x, y, angle_deg])
    # Row 1: y = 0, offset by sqrt3/2
    for col in range(4):
        x = (col - 1.5) * sqrt3 + sqrt3/2
        y = 0.0
        inner_init.append([x, y, angle_deg])
    # Row 2: y = 1.5
    for col in range(4):
        x = (col - 1.5) * sqrt3
        y = 1.5
        inner_init.append([x, y, angle_deg])
    
    # Center the arrangement
    xs = [p[0] for p in inner_init]
    ys = [p[1] for p in inner_init]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for i in range(N):
        inner_init[i][0] -= cx
        inner_init[i][1] -= cy
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0  # flat-top outer hexagon
    
    # Verify initial configuration has no overlaps
    if not check_no_overlaps(inner_init):
        # Fallback: spread out more
        for i in range(N):
            inner_init[i][0] *= 1.2
            inner_init[i][1] *= 1.2
    
    # Initial parameters
    x0 = []
    for i in range(N):
        x0.append(inner_init[i][0])
        x0.append(inner_init[i][1])
        x0.append(inner_init[i][2])
    x0 = np.array(x0)
    
    best_params = x0.copy()
    
    def objective(params):
        inner = []
        for i in range(N):
            inner.append([params[3*i], params[3*i+1], params[3*i+2]])
        
        side = compute_min_outer_side(inner, outer_center, outer_angle)
        
        # Strong overlap penalty
        overlap_penalty = 0.0
        for i in range(N):
            vi = hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            for j in range(i+1, N):
                vj = hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
                if polygons_overlap(vi, vj, tol=1e-8):
                    overlap_penalty += 1000.0
        
        return side + overlap_penalty
    
    best_cost = objective(x0)
    
    # Optimize with Nelder-Mead
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
    
    if result.fun < best_cost and check_no_overlaps([[result.x[3*i], result.x[3*i+1], result.x[3*i+2]] for i in range(N)]):
        best_params = result.x
        best_cost = result.fun
    
    # Try a few restarts if time allows
    elapsed = time.time() - start_time
    if elapsed < deadline - 15:
        for trial in range(5):
            if time.time() - start_time > deadline - 15:
                break
            np.random.seed(trial * 1000 + 42)
            perturbed = best_params + np.random.normal(0, 0.3, size=len(best_params))
            result2 = minimize(objective, perturbed, method='Nelder-Mead',
                              options={'maxiter': 1500, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            inner_test = [[result2.x[3*i], result2.x[3*i+1], result2.x[3*i+2]] for i in range(N)]
            if result2.fun < best_cost and check_no_overlaps(inner_test):
                best_params = result2.x
                best_cost = result2.fun
    
    # Extract final inner positions
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    # Final verification: if overlaps exist, use the initial valid configuration
    if not check_no_overlaps(inner):
        inner = [[float(x), float(y), float(a)] for x, y, a in inner_init]
    
    final_side = compute_min_outer_side(inner, outer_center, outer_angle)
    
    return inner, outer_center, final_side, outer_angle

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
