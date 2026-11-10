"""Packing 11 unit hexagons inside a regular hexagon using GA with cluster approach."""
import numpy as np
import random
import math

N = 11

def _hexagon_vertices(angle_deg=0.0, center=(0.0, 0.0)):
    """Get vertices of a unit regular hexagon with given rotation and center."""
    cx, cy = center
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        verts.append((cx + math.cos(theta), cy + math.sin(theta)))
    return np.array(verts)

def _point_in_unit_hex_local(rx, ry):
    """Check if point (rx, ry) is inside a unit hexagon centered at origin with vertex at angle 0."""
    inradius = math.sqrt(3) / 2
    for i in range(6):
        theta = math.radians(30 + i * 60)
        nx, ny = math.cos(theta), math.sin(theta)
        if rx * nx + ry * ny > inradius + 1e-9:
            return False
    return True

def _seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between two line segments."""
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p3 = np.array(p3, dtype=float)
    p4 = np.array(p4, dtype=float)
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(p1 - p3)
    elif a < 1e-12:
        t = f / e
        t = max(0, min(1, t))
        return np.linalg.norm(p1 - (p3 + t * d2))
    elif e < 1e-12:
        t = -np.dot(d1, r) / a
        t = max(0, min(1, t))
        return np.linalg.norm((p1 + t * d1) - p3)
    c = np.dot(d1, r)
    b = np.dot(d1, d2)
    denom = a * e - b * b
    if abs(denom) > 1e-12:
        s = (b * f - c * e) / denom
        t = (a * f - c * b) / denom
    else:
        s = 0.0
        t = f / e
    s = max(0, min(1, s))
    t = max(0, min(1, t))
    ps = p1 + s * d1
    pt = p3 + t * d2
    return np.linalg.norm(ps - pt)

def _hex_overlap_dist(h1_center, h1_angle, h2_center, h2_angle):
    """Return minimum distance between two unit hexagons (negative if overlapping)."""
    v1 = _hexagon_vertices(h1_angle, h1_center)
    v2 = _hexagon_vertices(h2_angle, h2_center)
    min_dist = float('inf')
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        for j in range(6):
            p3 = v2[j]
            p4 = v2[(j + 1) % 6]
            d = _seg_seg_dist(p1, p2, p3, p4)
            if d < min_dist:
                min_dist = d
    # Check containment
    cx1, cy1 = h1_center
    a2 = math.radians(h2_angle)
    dx, dy = cx1 - h2_center[0], cy1 - h2_center[1]
    rx = dx * math.cos(-a2) - dy * math.sin(-a2)
    ry = dx * math.sin(-a2) + dy * math.cos(-a2)
    if _point_in_unit_hex_local(rx, ry):
        return -min_dist
    cx2, cy2 = h2_center
    a1 = math.radians(h1_angle)
    dx, dy = cx2 - h1_center[0], cy2 - h1_center[1]
    rx = dx * math.cos(-a1) - dy * math.sin(-a1)
    ry = dx * math.sin(-a1) + dy * math.cos(-a1)
    if _point_in_unit_hex_local(rx, ry):
        return -min_dist
    return min_dist

def _compute_outer_side(inner_positions, inner_angles, outer_center=(0.0, 0.0), outer_angle=0.0):
    """Compute minimum outer hexagon circumradius (side length) to contain all inner hexagons."""
    cx, cy = outer_center
    a = math.radians(outer_angle)
    cos_a = math.cos(-a)
    sin_a = math.sin(-a)
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    max_required_R = 0.0
    for idx in range(len(inner_positions)):
        h_center = inner_positions[idx]
        h_angle = inner_angles[idx]
        verts = _hexagon_vertices(h_angle, h_center)
        for v in verts:
            dx, dy = v[0] - cx, v[1] - cy
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            for nx, ny in normals:
                proj = rx * nx + ry * ny
                if proj > 0:
                    R_needed = proj / (math.sqrt(3) / 2)
                    if R_needed > max_required_R:
                        max_required_R = R_needed
    return max_required_R

def _fitness(inner_positions, inner_angles):
    """Return outer side length, or inf if overlapping."""
    for i in range(N):
        for j in range(i + 1, N):
            d = _hex_overlap_dist(inner_positions[i], inner_angles[i],
                                  inner_positions[j], inner_angles[j])
            if d < -1e-6:
                return float('inf')
    return _compute_outer_side(inner_positions, inner_angles)

# EVOLVE-BLOCK-START
def construct():
    random.seed(42)
    np.random.seed(42)
    
    def make_initial():
        positions = [(0.0, 0.0)]
        angles = [0.0]
        for i in range(6):
            theta = i * math.pi / 3
            dist = math.sqrt(3)
            positions.append((dist * math.cos(theta), dist * math.sin(theta)))
            angles.append(0.0)
        for i in range(4):
            theta = (30 + i * 90) * math.pi / 180
            dist = 2 * math.sqrt(3)
            positions.append((dist * math.cos(theta), dist * math.sin(theta)))
            angles.append(0.0)
        for i in range(N):
            positions[i] = (positions[i][0] + random.gauss(0, 0.15),
                           positions[i][1] + random.gauss(0, 0.15))
            angles[i] = random.gauss(0, 5)
        return positions, angles
    
    pop_size = 50
    generations = 60
    population = [make_initial() for _ in range(pop_size)]
    
    best_fitness = float('inf')
    best_individual = None
    
    for gen in range(generations):
        scored = []
        for i in range(pop_size):
            pos, ang = population[i]
            f = _fitness(pos, ang)
            scored.append((f, pos, ang))
            if f < best_fitness:
                best_fitness = f
                best_individual = (pos, ang)
        
        scored.sort(key=lambda x: x[0])
        next_pop = [scored[i][1:] for i in range(min(5, pop_size))]
        
        while len(next_pop) < pop_size:
            p1 = population[random.randint(0, pop_size - 1)]
            p2 = population[random.randint(0, pop_size - 1)]
            child_pos = []
            child_ang = []
            for i in range(N):
                if random.random() < 0.5:
                    child_pos.append(p1[0][i])
                    child_ang.append(p1[1][i])
                else:
                    child_pos.append(p2[0][i])
                    child_ang.append(p2[1][i])
            for i in range(N):
                if random.random() < 0.35:
                    child_pos[i] = (child_pos[i][0] + random.gauss(0, 0.25),
                                   child_pos[i][1] + random.gauss(0, 0.25))
                if random.random() < 0.35:
                    child_ang[i] += random.gauss(0, 8)
            next_pop.append((child_pos, child_ang))
        population = next_pop
    
    pos, ang = best_individual
    outer_side = _fitness(pos, ang)
    
    if outer_side > 8 or outer_side == float('inf'):
        positions = [(0.0, 0.0)]
        angles = [0.0]
        for i in range(6):
            theta = i * math.pi / 3
            dist = math.sqrt(3)
            positions.append((dist * math.cos(theta), dist * math.sin(theta)))
            angles.append(0.0)
        for i in range(4):
            theta = (30 + i * 90) * math.pi / 180
            dist = 2 * math.sqrt(3)
            positions.append((dist * math.cos(theta), dist * math.sin(theta)))
            angles.append(0.0)
        pos, ang = positions, angles
        outer_side = _fitness(pos, ang)
    
    inner = [[pos[i][0], pos[i][1], ang[i]] for i in range(N)]
    return inner, (0.0, 0.0), outer_side, 0.0
# EVOLVE-BLOCK-END
