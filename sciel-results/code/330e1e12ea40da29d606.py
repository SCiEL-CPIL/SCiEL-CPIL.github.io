"""
Hexagon packing: 12 unit hexagons in a regular hexagon.
Strategy: Enumerate hexagonal-lattice cluster topologies, then optimize
global rotation/translation of each rigid cluster.
"""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 12
DEADLINE = 148.0  # Safety margin from 150s limit

# Hexagonal lattice basis vectors (distance sqrt(3) between centers)
LATTICE_A = np.array([np.sqrt(3), 0.0])
LATTICE_B = np.array([np.sqrt(3)/2, 1.5])

def hexagon_vertices(cx, cy, side, angle_deg):
    """Vertices of regular hexagon centered at (cx,cy)."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + np.radians(60 * i)
        verts.append([cx + side * np.cos(theta), cy + side * np.sin(theta)])
    return np.array(verts)

def required_side(centers, angles, outer_center, outer_angle):
    """Min outer side length to contain all hexagons."""
    oa = np.radians(outer_angle)
    cos_a, sin_a = np.cos(oa), np.sin(oa)
    max_proj = 0.0
    for i in range(len(centers)):
        verts = hexagon_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
        for v in verts:
            rel = np.array([v[0] - outer_center[0], v[1] - outer_center[1]])
            # Rotate to outer hexagon's frame
            rx = rel[0] * cos_a + rel[1] * sin_a
            ry = -rel[0] * sin_a + rel[1] * cos_a
            for k in range(6):
                theta = np.radians(60 * k)
                d = abs(rx * np.cos(theta) + ry * np.sin(theta))
                if d > max_proj:
                    max_proj = d
    return max_proj / (np.sqrt(3) / 2)

def sat_separation(c1, a1, c2, a2):
    """Min separation between two unit hexagons using SAT."""
    v1 = hexagon_vertices(c1[0], c1[1], 1.0, a1)
    v2 = hexagon_vertices(c2[0], c2[1], 1.0, a2)
    
    axes = []
    for v in [v1, v2]:
        for i in range(6):
            e = v[(i+1)%6] - v[i]
            ln = np.linalg.norm(e)
            if ln > 1e-12:
                axes.append(np.array([-e[1], e[0]]) / ln)
    
    min_sep = float('inf')
    for ax in axes:
        p1 = np.dot(v1, ax)
        p2 = np.dot(v2, ax)
        sep = max(p1.min() - p2.max(), p2.min() - p1.max())
        if sep < min_sep:
            min_sep = sep
    return min_sep

def check_valid(centers, angles, side, outer_center, outer_angle):
    """Check no overlaps and all inside outer hexagon."""
    actual_side = required_side(centers, angles, outer_center, outer_angle)
    if actual_side > side + 1e-8:
        return False
    for i in range(N):
        for j in range(i+1, N):
            if sat_separation(centers[i], angles[i], centers[j], angles[j]) < -1e-8:
                return False
    return True

def generate_lattice_clusters():
    """Generate distinct connected polyhexes of size 12 on hexagonal lattice."""
    clusters = []
    seen = set()
    
    # BFS/DFS to grow clusters
    def normalize(cells):
        """Translate cluster so min coords are at origin, return canonical form."""
        cells = sorted([(int(round(x)), int(round(y))) for x, y in cells])
        min_x = min(c[0] for c in cells)
        min_y = min(c[1] for c in cells)
        norm = tuple(sorted((c[0]-min_x, c[1]-min_y) for c in cells))
        return norm
    
    def is_connected(cells):
        """Check if cells form a connected polyhex."""
        cell_set = set(cells)
        if not cell_set:
            return False
        visited = set()
        stack = [cells[0]]
        while stack:
            c = stack.pop()
            if c in visited:
                continue
            visited.add(c)
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]:
                nb = (c[0]+dx, c[1]+dy)
                if nb in cell_set and nb not in visited:
                    stack.append(nb)
        return len(visited) == len(cell_set)
    
    # Start with single cell, grow
    from collections import deque
    
    def grow(cells, depth):
        if len(cells) == N:
            if is_connected(cells):
                key = normalize(cells)
                if key not in seen:
                    seen.add(key)
                    clusters.append(list(cells))
            return
        if depth > 100:  # Safety
            return
        # Try adding adjacent cells
        for c in cells:
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]:
                nc = (c[0]+dx, c[1]+dy)
                if nc not in cells:
                    grow(cells + [nc], depth + 1)
    
    # This is too slow for full enumeration. Use known polyhex shapes instead.
    # Generate a few well-known topologies manually.
    
    # Topology 1: 3x4 rectangle (in lattice coords)
    rect = [(i, j) for i in range(3) for j in range(4)]
    clusters.append(rect)
    
    # Topology 2: 2x6 rectangle
    rect2 = [(i, j) for i in range(2) for j in range(6)]
    clusters.append(rect2)
    
    # Topology 3: Hexagonal cluster (center + 6 + 5 more)
    # Center at (0,0), ring of 6, then 5 more
    hex_cluster = [(0,0)]
    # 6 neighbors
    hex_cluster += [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]
    # Add 5 more to make 12
    hex_cluster += [(2,0),(2,-1),(1,1),(0,2),(-1,2)]
    clusters.append(hex_cluster)
    
    # Topology 4: Elongated hexagon
    elong = [(i, 0) for i in range(6)] + [(i, 1) for i in range(6)]
    clusters.append(elong)
    
    # Topology 5: L-shape
    l_shape = [(i, 0) for i in range(6)] + [(0, j) for j in range(1, 7)]
    clusters.append(l_shape)
    
    # Topology 6: T-shape
    t_shape = [(i, 0) for i in range(6)] + [(2, j) for j in range(1, 7)]
    clusters.append(t_shape)
    
    # Topology 7: Compact blob
    blob = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),
            (0,2),(1,2),(2,2),(1,3),(2,3)]
    clusters.append(blob)
    
    # Topology 8: Diagonal strip
    diag = [(i, i) for i in range(6)] + [(i, i+1) for i in range(6)]
    clusters.append(diag)
    
    # Topology 9: 4x3 with one missing corner
    four_three = [(i, j) for i in range(4) for j in range(3)]
    four_three = four_three[:11] + [four_three[11]]  # just use all 12
    # Actually 4x3 = 12, already covered by rect
    # Let's do a different one: remove corner, add elsewhere
    alt = [(i, j) for i in range(4) for j in range(3) if not (i==0 and j==0)] + [(4, 0)]
    clusters.append(alt)
    
    # Topology 10: Star-like
    star = [(0,0)]
    star += [(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1)]
    star += [(2,0),(2,-1),(3,-1),(3,0),(1,1)]
    clusters.append(star)
    
    return clusters

def cluster_to_positions(cells):
    """Convert lattice cell coords to actual center positions."""
    positions = []
    for (i, j) in cells:
        x = i * LATTICE_A[0] + j * LATTICE_B[0]
        y = i * LATTICE_A[1] + j * LATTICE_B[1]
        positions.append([x, y])
    return positions

def optimize_cluster(cells, time_limit=5.0):
    """Optimize rotation and translation of a rigid cluster."""
    start = time.time()
    positions = cluster_to_positions(cells)
    
    # Center the cluster
    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])
    centered = [[p[0]-cx, p[1]-cy] for p in positions]
    
    def objective(params):
        if time.time() - start > time_limit:
            return 1e10
        tx, ty, angle_deg = params
        angle = np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # Rotate and translate positions
        rotated = []
        for px, py in centered:
            rx = px * cos_a - py * sin_a
            ry = px * sin_a + py * cos_a
            rotated.append([rx + tx, ry + ty])
        
        # All inner hexagons have same angle as cluster rotation
        angles = [angle_deg] * N
        
        side = required_side(rotated, angles, [0,0], 0.0)
        
        # Check overlaps
        min_sep = float('inf')
        for i in range(N):
            for j in range(i+1, N):
                s = sat_separation(rotated[i], angle_deg, rotated[j], angle_deg)
                if s < min_sep:
                    min_sep = s
        
        # Penalty for overlaps
        penalty = max(0, -min_sep) ** 2 * 100
        return side + penalty
    
    # Try multiple initial angles
    best_result = None
    best_score = float('inf')
    
    for init_angle in [0, 15, 30, 45, 60]:
        if time.time() - start > time_limit - 1:
            break
        x0 = [0, 0, float(init_angle)]
        res = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if res.fun < best_score:
            best_score = res.fun
            best_result = res.x
    
    if best_result is None:
        return None
    
    tx, ty, angle_deg = best_result
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    
    final_positions = []
    for px, py in centered:
        rx = px * cos_a - py * sin_a
        ry = px * sin_a + py * cos_a
        final_positions.append([rx + tx, ry + ty])
    
    angles = [angle_deg] * N
    side = required_side(final_positions, angles, [0,0], 0.0)
    
    return final_positions, angles, side

def fine_tune(positions, angles, side, time_limit=10.0):
    """Fine-tune individual hexagon positions and angles."""
    start = time.time()
    
    def objective(params):
        if time.time() - start > time_limit:
            return 1e10
        pos = []
        ang = []
        for i in range(N):
            pos.append([params[3*i], params[3*i+1]])
            ang.append(params[3*i+2])
        
        s = required_side(pos, ang, [0,0], 0.0)
        
        min_sep = float('inf')
        for i in range(N):
            for j in range(i+1, N):
                sep = sat_separation(pos[i], ang[i], pos[j], ang[j])
                if sep < min_sep:
                    min_sep = sep
        
        penalty = max(0, -min_sep) ** 2 * 100
        return s + penalty
    
    params0 = []
    for i in range(N):
        params0.extend([positions[i][0], positions[i][1], angles[i]])
    
    res = minimize(objective, params0, method='L-BFGS-B',
                  options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
    
    if res.fun >= side:
        return positions, angles, side
    
    new_pos = []
    new_ang = []
    for i in range(N):
        new_pos.append([res.x[3*i], res.x[3*i+1]])
        new_ang.append(res.x[3*i+2])
    
    new_side = required_side(new_pos, new_ang, [0,0], 0.0)
    return new_pos, new_ang, new_side

def construct():
    start_time = time.time()
    
    best_positions = None
    best_angles = None
    best_side = float('inf')
    
    clusters = generate_lattice_clusters()
    
    for idx, cells in enumerate(clusters):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break
        
        try:
            result = optimize_cluster(cells, time_limit=8.0)
            if result is None:
                continue
            
            pos, ang, side = result
            
            if side < best_side:
                best_side = side
                best_positions = pos
                best_angles = ang
        except Exception:
            continue
    
    # Fine-tune the best result
    if best_positions is not None and time.time() - start_time < DEADLINE - 15:
        try:
            fp, fa, fs = fine_tune(best_positions, best_angles, best_side,
                                   time_limit=15.0)
            if fs < best_side:
                best_positions = fp
                best_angles = fa
                best_side = fs
        except Exception:
            pass
    
    # Final validation
    if best_positions is not None:
        valid = check_valid(best_positions, best_angles, best_side, [0,0], 0.0)
        if valid:
            inner = [[p[0], p[1], a] for p, a in zip(best_positions, best_angles)]
            return inner, [0.0, 0.0], best_side, 0.0
    
    # Fallback: circular arrangement
    inner = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        x = 3.0 * np.cos(theta)
        y = 3.0 * np.sin(theta)
        inner.append([x, y, 0.0])
    
    side = required_side(
        [[p[0], p[1]] for p in inner],
        [a for _, _, a in inner],
        [0,0], 0.0
    )
    return inner, [0.0, 0.0], side, 0.0

