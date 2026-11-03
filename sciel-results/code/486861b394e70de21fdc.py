"""Rewritten program: SA + Nelder-Mead polish for 12 unit hexagons in a regular hexagon."""
import time
import math
import random
import numpy as np

N = 12

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 130.0  # seconds, safety margin before 150s kill

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    verts = []
    for i in range(6):
        ang = math.radians(angle_deg + 60.0 * i)
        x = cx + side * math.cos(ang)
        y = cy + side * math.sin(ang)
        verts.append((x, y))
    return verts

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px, py) is inside regular hexagon centered at (cx, cy)."""
    # Transform to local frame
    dx = px - cx
    dy = py - cy
    ang = -math.radians(angle_deg)
    lx = dx * math.cos(ang) - dy * math.sin(ang)
    ly = dx * math.sin(ang) + dy * math.cos(ang)
    # Regular hexagon with flat top: |lx| <= side, |ly| <= side*sqrt(3)/2, and |lx| + |ly|/sqrt(3) <= side
    # Actually for a regular hexagon with vertices at angles 0, 60, 120, 180, 240, 300:
    # The hexagon is the intersection of 6 half-planes.
    # For unit hexagon (side=1), the apothem is sqrt(3)/2.
    # In local coords, the hexagon is: |x| <= 1, |y| <= sqrt(3)/2, |x| + |y|/sqrt(3) <= 1
    # Wait, let me think more carefully.
    # A regular hexagon with side length s has vertices at (s*cos(k*pi/3), s*sin(k*pi/3)) for k=0..5.
    # The hexagon can be described as: for each of the 6 edges, the point must be on the correct side.
    # Edge k connects vertex k and vertex k+1. The inward normal points toward center.
    # For vertex at angle theta_k = k*pi/3, the edge midpoint is at distance apothem = s*sqrt(3)/2 from center,
    # at angle theta_k + pi/6.
    # The constraint is: dot(point - center, inward_normal_k) <= apothem
    # where inward_normal_k = (cos(theta_k + pi/6), sin(theta_k + pi/6))
    
    apothem = side * math.sqrt(3) / 2.0
    for i in range(6):
        edge_angle = math.radians(angle_deg + 60.0 * i + 30.0)
        nx = math.cos(edge_angle)
        ny = math.sin(edge_angle)
        if lx * nx + ly * ny > apothem:
            return False
    return True

def hexagons_disjoint(hex1, hex2):
    """Check if two unit hexagons are disjoint (no overlap).
    hex1, hex2: (cx, cy, angle_deg)
    """
    cx1, cy1, a1 = hex1
    cx2, cy2, a2 = hex2
    
    # Quick distance check: if centers are far enough, they're disjoint
    dx = cx1 - cx2
    dy = cy1 - cy2
    dist = math.sqrt(dx*dx + dy*dy)
    # Max distance between any two points in a unit hexagon is 2 (diameter)
    # So if dist > 2, they're definitely disjoint
    if dist > 2.0 + 1e-9:
        return True
    
    # Check if any vertex of hex1 is inside hex2, or any vertex of hex2 is inside hex1
    verts1 = hex_vertices(cx1, cy1, 1.0, a1)
    verts2 = hex_vertices(cx2, cy2, 1.0, a2)
    
    for v in verts1:
        if point_in_hexagon(v[0], v[1], cx2, cy2, 1.0, a2):
            return False
    for v in verts2:
        if point_in_hexagon(v[0], v[1], cx1, cy1, 1.0, a1):
            return False
    
    # Also need to check edge intersections, but for convex polygons,
    # if no vertex is inside the other, they may still intersect via edges.
    # For safety, let's also check if edges intersect.
    # Actually, for convex polygons, if no vertex of one is inside the other,
    # they are disjoint OR one edge crosses another. Let's check edge intersections.
    
    def segments_intersect(p1, p2, p3, p4):
        def ccw(a, b, c):
            return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    for i in range(6):
        for j in range(6):
            if segments_intersect(verts1[i], verts1[(i+1)%6], verts2[j], verts2[(j+1)%6]):
                return False
    
    return True

def all_disjoint(inner):
    """Check if all inner hexagons are pairwise disjoint."""
    for i in range(N):
        for j in range(i+1, N):
            if not hexagons_disjoint(inner[i], inner[j]):
                return False
    return True

def min_outer_side(inner):
    """Find the minimum side length of a regular hexagon (any orientation, centered at origin) 
    that contains all inner hexagons. Returns (side, center_x, center_y, angle_deg)."""
    # We'll search over outer hexagon angle and center to minimize side length.
    # For a given outer hexagon orientation and center, the minimum side length 
    # is determined by the maximum "distance" from center to any vertex of any inner hex,
    # projected appropriately.
    
    # For a regular hexagon with orientation theta, the support function in direction phi is:
    # h(phi) = side * max_i cos(phi - (theta + i*pi/3)) ... actually it's the apothem-based formula.
    # The hexagon contains point p if for all 6 edge normals n_k, dot(p - center, n_k) <= apothem = side*sqrt(3)/2
    # So side >= (2/sqrt(3)) * max_k dot(p - center, n_k)
    
    # To minimize side, we need to minimize max over all vertices p of (2/sqrt(3)) * max_k dot(p - center, n_k(theta))
    
    # Let's collect all vertices of all inner hexagons
    all_verts = []
    for (cx, cy, ang) in inner:
        verts = hex_vertices(cx, cy, 1.0, ang)
        all_verts.extend(verts)
    
    if not all_verts:
        return 1.0, 0.0, 0.0, 0.0
    
    # Search over outer angle theta
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Coarse search over theta
    for theta_deg in range(0, 180, 10):
        theta = math.radians(theta_deg)
        # Edge normals
        normals = []
        for k in range(6):
            ang = theta + math.pi/6 + k * math.pi/3
            normals.append((math.cos(ang), math.sin(ang)))
        
        # For a given center (cx, cy), the required side is:
        # side = (2/sqrt(3)) * max over all vertices p of max over normals n of dot(p - center, n)
        # We want to minimize this over center.
        # This is a convex optimization in center (min of max of linear functions).
        # Use a simple approach: try a few candidate centers.
        
        # The optimal center is roughly the centroid of all vertices
        cx_mean = sum(p[0] for p in all_verts) / len(all_verts)
        cy_mean = sum(p[1] for p in all_verts) / len(all_verts)
        
        # Try centroid and a few offsets
        candidates = [(cx_mean, cy_mean)]
        for dx in [-0.5, 0, 0.5]:
            for dy in [-0.5, 0, 0.5]:
                candidates.append((cx_mean + dx, cy_mean + dy))
        
        for (cx, cy) in candidates:
            max_val = 0.0
            for p in all_verts:
                for (nx, ny) in normals:
                    val = (p[0] - cx) * nx + (p[1] - cy) * ny
                    if val > max_val:
                        max_val = val
            side = (2.0 / math.sqrt(3.0)) * max_val
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = theta_deg
    
    # Refine with finer theta search around best
    for theta_deg in [best_angle - 5, best_angle - 2, best_angle, best_angle + 2, best_angle + 5]:
        theta = math.radians(theta_deg)
        normals = []
        for k in range(6):
            ang = theta + math.pi/6 + k * math.pi/3
            normals.append((math.cos(ang), math.sin(ang)))
        
        cx_mean = sum(p[0] for p in all_verts) / len(all_verts)
        cy_mean = sum(p[1] for p in all_verts) / len(all_verts)
        
        candidates = [(cx_mean, cy_mean)]
        for dx in [-0.2, -0.1, 0, 0.1, 0.2]:
            for dy in [-0.2, -0.1, 0, 0.1, 0.2]:
                candidates.append((cx_mean + dx, cy_mean + dy))
        
        for (cx, cy) in candidates:
            max_val = 0.0
            for p in all_verts:
                for (nx, ny) in normals:
                    val = (p[0] - cx) * nx + (p[1] - cy) * ny
                    if val > max_val:
                        max_val = val
            side = (2.0 / math.sqrt(3.0)) * max_val
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = theta_deg
    
    return best_side, best_center[0], best_center[1], best_angle

def fitness(inner):
    """Lower is better. Returns infinity if not all disjoint."""
    if not all_disjoint(inner):
        return float('inf')
    side, cx, cy, ang = min_outer_side(inner)
    return side

# EVOLVE-BLOCK-START
def construct():
    global START_TIME
    START_TIME = time.time()
    
    random.seed(42)
    np.random.seed(42)
    
    # Initial population: hexagonal packing
    def initial_config():
        # Place hexagons in a honeycomb-like pattern
        inner = []
        # 3 rows of 4? Or 4 rows of 3?
        # Let's try a compact arrangement
        # For 12 hexagons, a 3x4 grid in honeycomb
        dx = 1.5  # horizontal spacing for flat-top hexagons
        dy = math.sqrt(3)  # vertical spacing
        
        positions = []
        for row in range(3):
            for col in range(4):
                x = col * dx + (row % 2) * dx/2
                y = row * dy
                positions.append((x, y))
        
        # Center them
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)
        
        inner = []
        for (x, y) in positions:
            inner.append([x - cx, y - cy, 0.0])
        
        return inner
    
    best_inner = initial_config()
    best_score = fitness(best_inner)
    if math.isinf(best_score):
        # Try a different initial config
        best_inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        best_score = fitness(best_inner)
    
    current_inner = [list(x) for x in best_inner]
    current_score = best_score
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    max_iters = 50000
    
    for it in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        # Generate perturbation
        new_inner = [list(x) for x in current_inner]
        
        # Perturb 1-3 random hexagons
        n_perturb = random.randint(1, 3)
        for _ in range(n_perturb):
            idx = random.randint(0, N - 1)
            new_inner[idx][0] += random.gauss(0, 0.3)
            new_inner[idx][1] += random.gauss(0, 0.3)
            new_inner[idx][2] += random.gauss(0, 15.0)
            new_inner[idx][2] %= 60.0  # Hexagon has 60-degree symmetry
        
        new_score = fitness(new_inner)
        
        # Acceptance criterion
        if new_score < current_score:
            current_inner = new_inner
            current_score = new_score
            if current_score < best_score:
                best_inner = [list(x) for x in current_inner]
                best_score = current_score
        elif random.random() < math.exp(-(new_score - current_score) / T):
            current_inner = new_inner
            current_score = new_score
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Nelder-Mead polish on best solution
    if time.time() - START_TIME < TIME_LIMIT - 20:
        best_inner = nelder_mead_polish(best_inner, best_score)
    
    side, cx, cy, ang = min_outer_side(best_inner)
    return best_inner, (cx, cy), side, ang

def nelder_mead_polish(inner, score):
    """Polish using Nelder-Mead on the flattened parameters."""
    # Flatten: 3 params per hexagon (x, y, angle)
    x0 = []
    for h in inner:
        x0.extend([h[0], h[1], h[2]])
    x0 = np.array(x0, dtype=float)
    
    def objective(x):
        if time.time() - START_TIME > TIME_LIMIT - 15:
            return float('inf')
        new_inner = []
        for i in range(N):
            new_inner.append([x[3*i], x[3*i+1], x[3*i+2] % 60.0])
        return fitness(new_inner)
    
    # Use scipy's minimize if available, otherwise manual Nelder-Mead
    try:
        from scipy.optimize import minimize
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
        if result.fun < score:
            best_inner = []
            x = result.x
            for i in range(N):
                best_inner.append([x[3*i], x[3*i+1], x[3*i+2] % 60.0])
            return best_inner
    except:
        pass
    
    return inner
# EVOLVE-BLOCK-END
