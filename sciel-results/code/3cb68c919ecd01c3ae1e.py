"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math
import itertools

def get_hex_vertices(cx, cy, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon."""
    a = math.radians(angle_deg)
    angles = a + math.pi / 6 + np.arange(6) * math.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def sat_overlap_check(h1, h2, tol=1e-9):
    """Check if two convex polygons overlap using SAT."""
    edges = []
    for poly in [h1, h2]:
        for i in range(6):
            j = (i + 1) % 6
            edges.append(poly[j] - poly[i])
    
    for edge in edges:
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal = normal / norm
        
        proj1 = h1 @ normal
        proj2 = h2 @ normal
        
        if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
            return False
    
    return True

def check_all_disjoint(positions, angles, tol=1e-9):
    """Check that all hexagons are pairwise disjoint."""
    n = len(positions)
    verts_list = []
    for i in range(n):
        verts_list.append(get_hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i]))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = math.hypot(positions[i][0] - positions[j][0], positions[i][1] - positions[j][1])
            if dist > 2.0 + tol:
                continue
            if sat_overlap_check(verts_list[i], verts_list[j], tol):
                return False
    return True

def compute_required_side(ocx, ocy, oor_deg, positions, angles):
    """Compute the minimum outer hexagon side length to contain all inner hexagons."""
    max_val = 0.0
    oor = math.radians(oor_deg)
    
    for i in range(len(positions)):
        verts = get_hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
        for vx, vy in verts:
            dx, dy = vx - ocx, vy - ocy
            cos_a, sin_a = math.cos(-oor), math.sin(-oor)
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            
            for k in range(6):
                theta = math.pi / 6 + k * math.pi / 3
                val = lx * math.cos(theta) + ly * math.sin(theta)
                if val > max_val:
                    max_val = val
    
    return max_val / (math.sqrt(3) / 2)

def compute_required_side_fast(ocx, ocy, oor_deg, positions, angles):
    """Vectorized computation of required side length."""
    oor = math.radians(oor_deg)
    cos_o, sin_o = math.cos(-oor), math.sin(-oor)
    
    max_val = 0.0
    for i in range(len(positions)):
        verts = get_hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
        dx = verts[:, 0] - ocx
        dy = verts[:, 1] - ocy
        lx = dx * cos_o - dy * sin_o
        ly = dx * sin_o + dy * cos_o
        
        for k in range(6):
            theta = math.pi / 6 + k * math.pi / 3
            vals = lx * math.cos(theta) + ly * math.sin(theta)
            if vals.max() > max_val:
                max_val = vals.max()
    
    return max_val / (math.sqrt(3) / 2)

def check_inside(positions, angles, oc, side, oor_deg, tol=1e-9):
    """Check if all inner hexagons are inside the outer hexagon."""
    ocx, ocy = oc
    oor = math.radians(oor_deg)
    cos_a, sin_a = math.cos(-oor), math.sin(-oor)
    
    for i in range(len(positions)):
        verts = get_hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
        dx = verts[:, 0] - ocx
        dy = verts[:, 1] - ocy
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        
        for k in range(6):
            theta = math.pi / 6 + k * math.pi / 3
            if (lx * math.cos(theta) + ly * math.sin(theta)).max() > side * math.sqrt(3) / 2 + tol:
                return False
    return True

def hex_grid_positions(n):
    """Generate hexagonal lattice positions."""
    positions = [(0.0, 0.0)]
    sqrt3 = math.sqrt(3)
    ring = 1
    while len(positions) < n:
        for q in range(-ring, ring + 1):
            for r in range(-ring, ring + 1):
                if max(abs(q), abs(r), abs(q + r)) == ring:
                    x = 1.5 * q
                    y = sqrt3 * (r + q / 2.0)
                    if len(positions) < n:
                        positions.append((x, y))
        ring += 1
    return positions[:n]

def generate_polyhexes(n):
    """Generate all polyhexes of size n using BFS/DFS."""
    if n == 1:
        return [((0, 0),)]
    
    # Use a simple enumeration approach
    # Start with a single hexagon and add neighbors
    results = set()
    
    def add_hex(cell, current, count):
        if count == n:
            # Normalize: translate so min q, then min r is at origin
            min_q = min(q for q, r in current)
            min_r = min(r for q, r in current if q == min_q)
            normalized = frozenset((q - min_q, r - min_r) for q, r in current)
            results.add(normalized)
            return
        
        # Try adding each of 6 neighbors
        q, r = cell
        for dq, dr in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
            new_cell = (q + dq, r + dr)
            if new_cell not in current:
                add_hex(new_cell, current | {new_cell}, count + 1)
    
    add_hex((0, 0), {(0, 0)}, 1)
    return [set(poly) for poly in results]

def polyhex_to_positions(polyhex):
    """Convert a polyhex (set of axial coords) to Cartesian positions."""
    sqrt3 = math.sqrt(3)
    positions = []
    for q, r in sorted(polyhex):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        positions.append((x, y))
    return positions

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    sqrt3 = math.sqrt(3)
    
    # Strategy 1: Use hex grid arrangement
    positions_grid = hex_grid_positions(N)
    angles_grid = [0.0] * N
    
    # Strategy 2: Try polyhex arrangements
    polyhex_positions_list = []
    try:
        polyhexes = generate_polyhexes(N)
        for poly in polyhexes[:50]:  # Limit to first 50 for time
            pos = polyhex_to_positions(poly)
            if check_all_disjoint(pos, [0.0] * N):
                polyhex_positions_list.append(pos)
    except Exception:
        pass
    
    # Combine all candidate arrangements
    all_candidates = [positions_grid]
    all_candidates.extend(polyhex_positions_list)
    
    best_side = 1e10
    best_oc = (0.0, 0.0)
    best_oor = 0.0
    best_positions = positions_grid
    best_angles = angles_grid
    
    for positions in all_candidates:
        if time.time() > deadline - 15.0:
            break
        
        angles = [0.0] * N
        
        # Optimize outer hexagon for this arrangement
        def objective(x):
            if time.time() > deadline - 3.0:
                return 1e6
            ocx, ocy, oor = x
            return compute_required_side_fast(ocx, ocy, oor, positions, angles)
        
        best_local_side = 1e10
        best_local_oc = (0.0, 0.0)
        best_local_oor = 0.0
        
        # Try multiple initial rotations
        for oor_init in [0.0, 30.0, 60.0, 15.0, 45.0, 75.0]:
            if time.time() > deadline - 5.0:
                break
            
            x0 = [0.0, 0.0, oor_init]
            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
                if res.fun < best_local_side:
                    ocx, ocy, oor = res.x
                    exact_side = compute_required_side(ocx, ocy, oor, positions, angles)
                    if check_inside(positions, angles, (ocx, ocy), exact_side, oor):
                        best_local_side = exact_side
                        best_local_oc = (ocx, ocy)
                        best_local_oor = oor
            except Exception:
                pass
        
        if best_local_side < best_side:
            best_side = best_local_side
            best_oc = best_local_oc
            best_oor = best_local_oor
            best_positions = positions
            best_angles = angles
    
    # Polish the best result
    if time.time() < deadline - 10.0:
        def objective_polish(x):
            if time.time() > deadline - 2.0:
                return 1e6
            ocx, ocy, oor = x
            return compute_required_side_fast(ocx, ocy, oor, best_positions, best_angles)
        
        x0 = [best_oc[0], best_oc[1], best_oor]
        try:
            res = minimize(objective_polish, x0, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-11, 'fatol': 1e-13})
            if res.fun < best_side:
                ocx, ocy, oor = res.x
                exact_side = compute_required_side(ocx, ocy, oor, best_positions, best_angles)
                if check_inside(best_positions, best_angles, (ocx, ocy), exact_side, oor) and \
                   check_all_disjoint(best_positions, best_angles):
                    best_side = exact_side
                    best_oc = (ocx, ocy)
                    best_oor = oor
        except Exception:
            pass
    
    # Add tiny epsilon for numerical safety
    best_side *= (1.0 + 1e-8)
    
    inner = [[best_positions[i][0], best_positions[i][1], best_angles[i]] for i in range(N)]
    return inner, best_oc, best_side, best_oor
# EVOLVE-BLOCK-END
