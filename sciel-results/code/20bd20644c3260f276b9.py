"""Hexagon packing: 11 unit hexagons in a regular hexagon.
Uses a contact-graph-based approach with force-directed relaxation
and simulated annealing to minimize the outer hexagon side length.
"""
import numpy as np
import time
import random
import math
from itertools import combinations

N = 11
DEADLINE = 145.0
START_TIME = time.time()


def _remaining_time():
    return DEADLINE - (time.time() - START_TIME)


def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with side length 'side'."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + k * np.pi / 3
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)


def _point_in_hex(px, py, cx, cy, side, angle_deg, tol=1e-9):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    dx = px - cx
    dy = py - cy
    angle_rad = np.radians(angle_deg)
    ca = np.cos(-angle_rad)
    sa = np.sin(-angle_rad)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca

    inradius = side * np.sqrt(3) / 2
    for k in range(6):
        a = np.radians(30 + k * 60)
        nx = np.cos(a)
        ny = np.sin(a)
        if lx * nx + ly * ny > inradius + tol:
            return False
    return True


def _hexagons_disjoint(verts1, verts2, tol=1e-7):
    """Check if two convex hexagons are disjoint using SAT."""
    n1 = len(verts1)
    n2 = len(verts2)

    def get_normals(verts):
        normals = []
        n = len(verts)
        for i in range(n):
            j = (i + 1) % n
            edge = verts[j] - verts[i]
            nx = -edge[1]
            ny = edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                normals.append([nx/norm, ny/norm])
        return normals

    normals1 = get_normals(verts1)
    normals2 = get_normals(verts2)
    all_normals = normals1 + normals2

    for nx, ny in all_normals:
        proj1 = np.array([v[0]*nx + v[1]*ny for v in verts1])
        proj2 = np.array([v[0]*nx + v[1]*ny for v in verts2])

        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)

        if max1 < min2 - tol or max2 < min1 - tol:
            return True

    return False


def _min_enclosing_side(inner_positions, outer_center, outer_angle):
    """Find minimum side length of outer hexagon to contain all inner hexagons."""
    cx, cy = outer_center
    best_side = 0.0

    for (ix, iy, iangle) in inner_positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        for v in iverts:
            dx = v[0] - cx
            dy = v[1] - cy
            angle_rad = np.radians(outer_angle)
            ca = np.cos(-angle_rad)
            sa = np.sin(-angle_rad)
            lx = dx * ca - dy * sa
            ly = dx * sa + dy * ca

            r = np.sqrt(lx*lx + ly*ly)
            if r < 1e-12:
                continue

            theta = np.arctan2(ly, lx)
            best_cos = 0.0
            for k in range(6):
                a = np.radians(30 + k * 60)
                d = abs(((theta - a) % (2*np.pi) + np.pi) % (2*np.pi))
                d = min(d, 2*np.pi - d)
                c = np.cos(d)
                if c > best_cos:
                    best_cos = c

            if best_cos < 1e-12:
                best_side = max(best_side, 1000.0)
            else:
                required_side = 2.0 * r * best_cos / np.sqrt(3)
                best_side = max(best_side, required_side)

    return best_side


def _check_validity(inner_positions, outer_center, outer_side, outer_angle, tol=1e-6):
    """Check if configuration is valid: all inner hexagons inside outer, no overlaps."""
    cx, cy = outer_center

    for (ix, iy, iangle) in inner_positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        for v in iverts:
            if not _point_in_hex(v[0], v[1], cx, cy, outer_side, outer_angle, tol=tol):
                return False

    all_verts = [_hex_vertices(ix, iy, 1.0, iangle) for (ix, iy, iangle) in inner_positions]
    for i in range(N):
        for j in range(i+1, N):
            if not _hexagons_disjoint(all_verts[i], all_verts[j], tol=tol):
                return False

    return True


def _hex_overlap_distance(verts1, verts2):
    """Estimate the minimum distance needed to separate two overlapping hexagons."""
    # Use SAT to find the minimum penetration depth
    def get_normals(verts):
        normals = []
        n = len(verts)
        for i in range(n):
            j = (i + 1) % n
            edge = verts[j] - verts[i]
            nx = -edge[1]
            ny = edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                normals.append([nx/norm, ny/norm])
        return normals

    normals1 = get_normals(verts1)
    normals2 = get_normals(verts2)
    all_normals = normals1 + normals2

    min_penetration = 0.0
    for nx, ny in all_normals:
        proj1 = np.array([v[0]*nx + v[1]*ny for v in verts1])
        proj2 = np.array([v[0]*nx + v[1]*ny for v in verts2])

        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)

        # If separated, no penetration on this axis
        if max1 < min2 or max2 < min1:
            continue

        # Overlapping: penetration is the overlap amount
        penetration = min(max1, max2) - max(min1, min2)
        if penetration > min_penetration:
            min_penetration = penetration

    return min_penetration


def _force_directed_relax(positions, outer_center, outer_angle, max_iters=200, lr=0.05):
    """Force-directed relaxation to reduce overlaps and minimize bounding hexagon."""
    positions = [list(p) for p in positions]
    best_side = _min_enclosing_side(positions, outer_center, outer_angle)
    best_positions = [list(p) for p in positions]

    for it in range(max_iters):
        if _remaining_time() < 15.0:
            break

        # Compute forces
        forces = np.zeros((N, 2))

        # Repulsion between overlapping hexagons
        all_verts = [_hex_vertices(p[0], p[1], 1.0, p[2]) for p in positions]
        for i in range(N):
            for j in range(i+1, N):
                if not _hexagons_disjoint(all_verts[i], all_verts[j], tol=1e-4):
                    # Compute repulsion force
                    ci = np.mean(all_verts[i], axis=0)
                    cj = np.mean(all_verts[j], axis=0)
                    diff = ci - cj
                    dist = np.linalg.norm(diff)
                    if dist > 1e-8:
                        repulsion = 1.0 / (dist + 0.1)
                        forces[i] += repulsion * diff / dist
                        forces[j] -= repulsion * diff / dist

        # Attraction toward center to reduce bounding hexagon
        cx, cy = outer_center
        for i in range(N):
            px, py = positions[i][0], positions[i][1]
            dx = px - cx
            dy = py - cy
            dist = np.sqrt(dx*dx + dy*dy)
            if dist > 1e-8:
                attraction = 0.01
                forces[i] -= attraction * np.array([dx, dy]) / dist

        # Apply forces
        for i in range(N):
            positions[i][0] += lr * forces[i][0]
            positions[i][1] += lr * forces[i][1]

        # Check improvement
        new_side = _min_enclosing_side(positions, outer_center, outer_angle)
        if new_side < best_side:
            best_side = new_side
            best_positions = [list(p) for p in positions]

        # Decay learning rate
        lr *= 0.98

    return best_positions, best_side


def _simulate_annealing(positions, outer_center, outer_angle, max_iters=500, T_start=1.0, T_end=0.01):
    """Simulated annealing to optimize positions."""
    positions = [list(p) for p in positions]
    best_side = _min_enclosing_side(positions, outer_center, outer_angle)
    best_positions = [list(p) for p in positions]
    current_side = best_side

    T = T_start
    cool_rate = (T_end / T_start) ** (1.0 / max_iters)

    for it in range(max_iters):
        if _remaining_time() < 10.0:
            break

        # Generate a random perturbation
        i = random.randint(0, N-1)
        dx = random.gauss(0, 0.1)
        dy = random.gauss(0, 0.1)
        da = random.gauss(0, 5.0)

        new_positions = [list(p) for p in positions]
        new_positions[i][0] += dx
        new_positions[i][1] += dy
        new_positions[i][2] = (new_positions[i][2] + da) % 360

        new_side = _min_enclosing_side(new_positions, outer_center, outer_angle)
        delta = new_side - current_side

        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or random.random() < math.exp(-delta / T):
            positions = new_positions
            current_side = new_side

            if current_side < best_side:
                best_side = current_side
                best_positions = [list(p) for p in positions]

        T *= cool_rate

    return best_positions, best_side


def _generate_lattice_config():
    """Generate a hexagonal lattice initial configuration."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    dist = 2.0 * np.sqrt(3)
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([dist * np.cos(a), dist * np.sin(a), 0.0])
    dist2 = dist * np.sqrt(3)
    for a_deg in [30, 90, 150, 210]:
        a = np.radians(a_deg)
        positions.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
    return positions


def _generate_random_config():
    """Generate a random initial configuration."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    for i in range(1, N):
        r = random.uniform(1.0, 4.0)
        a = random.uniform(0, 2*np.pi)
        angle = random.uniform(0, 360)
        positions.append([r * np.cos(a), r * np.sin(a), angle])
    return positions


def _generate_ring_config():
    """Generate a ring configuration with center and two rings."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    # Inner ring: 6 hexagons
    dist1 = 2.0 * np.sqrt(3)
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([dist1 * np.cos(a), dist1 * np.sin(a), 0.0])
    # Outer ring: 4 hexagons
    dist2 = 3.0 * np.sqrt(3)
    for a_deg in [30, 90, 150, 210]:
        a = np.radians(a_deg)
        positions.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
    return positions


def construct():
    """Main entry point."""
    random.seed(42)
    np.random.seed(42)

    best_result = None
    best_score = float('inf')

    outer_angles = [0.0, 30.0, 15.0, 45.0, 10.0, 20.0, 60.0]

    for oa in outer_angles:
        if _remaining_time() < 20.0:
            break

        initial_configs = [
            _generate_lattice_config(),
            _generate_ring_config(),
            _generate_random_config(),
            _generate_random_config(),
        ]

        for init_pos in initial_configs:
            if _remaining_time() < 15.0:
                break

            positions = [list(p) for p in init_pos]

            # Force-directed relaxation
            positions, side = _force_directed_relax(positions, (0.0, 0.0), oa, max_iters=150, lr=0.05)

            # Simulated annealing
            positions, side = _simulate_annealing(positions, (0.0, 0.0), oa, max_iters=300, T_start=0.5, T_end=0.01)

            # Verify validity
            if _check_validity(positions, (0.0, 0.0), side, oa, tol=1e-5):
                if side < best_score:
                    best_score = side
                    best_result = (positions, (0.0, 0.0), side, oa)

            # Try with adjusted outer center
            for dcx in [-0.2, -0.1, 0.0, 0.1, 0.2]:
                for dcy in [-0.2, -0.1, 0.0, 0.1, 0.2]:
                    if _remaining_time() < 8.0:
                        break
                    oc = (dcx, dcy)
                    side2 = _min_enclosing_side(positions, oc, oa)
                    if _check_validity(positions, oc, side2, oa, tol=1e-5):
                        if side2 < best_score:
                            best_score = side2
                            best_result = (positions, oc, side2, oa)

    if best_result is not None:
        return best_result

    # Fallback: known valid configuration
    positions = []
    positions.append([0.0, 0.0, 0.0])
    dist = 3.5
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([dist * np.cos(a), dist * np.sin(a), 0.0])
    dist2 = 6.0
    for k in range(4):
        a = np.radians(30 + 60 * k)
        positions.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])

    side = _min_enclosing_side(positions, (0.0, 0.0), 0.0)
    return positions, (0.0, 0.0), side, 0.0
