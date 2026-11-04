import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 135.0
    random.seed(42)
    np.random.seed(42)

    def hex_vertices(cx, cy, side, angle_rad):
        """Return 6 vertices of a regular hexagon."""
        verts = np.zeros((6, 2))
        for i in range(6):
            theta = angle_rad + i * math.pi / 3
            verts[i, 0] = cx + side * math.cos(theta)
            verts[i, 1] = cy + side * math.sin(theta)
        return verts

    def polygon_overlap(v1, v2, eps=1e-9):
        """Check if two convex polygons overlap (interior intersection)."""
        # Check if any vertex of v1 is inside v2 or vice versa
        # Also check edge intersections
        def point_in_polygon(px, py, poly):
            n = len(poly)
            inside = False
            j = n - 1
            for i in range(n):
                xi, yi = poly[i]
                xj, yj = poly[j]
                if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                    inside = not inside
                j = i
            return inside

        def segments_intersect(p1, p2, p3, p4):
            def cross(o, a, b):
                return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
            d1 = cross(p3, p4, p1)
            d2 = cross(p3, p4, p2)
            d3 = cross(p1, p2, p3)
            d4 = cross(p1, p2, p4)
            if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
               ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
                return True
            return False

        # Check vertices of v1 inside v2
        for i in range(6):
            if point_in_polygon(v1[i,0], v1[i,1], v2):
                return True
        # Check vertices of v2 inside v1
        for i in range(6):
            if point_in_polygon(v2[i,0], v2[i,1], v1):
                return True
        # Check edge intersections
        for i in range(6):
            for j in range(6):
                if segments_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                    return True
        return False

    def max_separation(v1, v2):
        """Compute how much two hexagons need to be separated (0 if disjoint)."""
        # If they don't overlap, return 0
        # Estimate overlap depth via minimum translation vector
        # Simplified: check if any vertex of one is inside the other
        def point_in_polygon(px, py, poly):
            n = len(poly)
            inside = False
            j = n - 1
            for i in range(n):
                xi, yi = poly[i]
                xj, yj = poly[j]
                if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                    inside = not inside
                j = i
            return inside

        # Quick check: if no vertex is inside, assume disjoint
        overlap = False
        for i in range(6):
            if point_in_polygon(v1[i,0], v1[i,1], v2):
                overlap = True
                break
        if not overlap:
            for i in range(6):
                if point_in_polygon(v2[i,0], v2[i,1], v1):
                    overlap = True
                    break
        if not overlap:
            return 0.0
        return 0.01  # Small penalty for overlap

    def compute_objective(x):
        """Compute outer hexagon side length for a given configuration vector.
        x = [x0,y0,a0, x1,y1,a1, ..., x11,y11,a11, ocx, ocy, otheta]
        Total length: 12*3 + 3 = 39
        """
        if time.time() > deadline - 5:
            return 1e10

        centers = []
        angles = []
        for i in range(12):
            centers.append((x[3*i], x[3*i+1]))
            angles.append(x[3*i+2])

        ocx = x[36]
        ocy = x[37]
        otheta = x[38]

        # Compute max projection for each of 6 outer hexagon face normals
        max_proj = 0.0
        for k in range(6):
            normal_angle = otheta + math.pi/6 + k * math.pi/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for i in range(12):
                cx, cy = centers[i]
                a = angles[i]
                for vi in range(6):
                    vert_angle = a + vi * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    proj = (vx - ocx) * nx + (vy - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
        side = 2.0 * max_proj / math.sqrt(3)

        # Penalty for overlapping hexagons
        penalty = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                v1 = hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
                v2 = hex_vertices(centers[j][0], centers[j][1], 1.0, angles[j])
                sep = max_separation(v1, v2)
                penalty += sep * 100.0

        return side + penalty

    def generate_lattice_initial(scale=1.0):
        """Generate a hexagonal lattice initial configuration."""
        eps = 0.01
        d = (math.sqrt(3) + eps) * scale
        u = np.array([d, 0.0])
        v = np.array([d/2.0, d*math.sqrt(3)/2.0])

        centers = [np.array([0.0, 0.0])]
        ring1_vectors = [u, v, v-u, -u, -v, u-v]
        for vec in ring1_vectors:
            centers.append(vec.copy())

        ring2_candidates = []
        basis = [u, v, v-u, -u, -v, u-v]
        seen = set()
        for i in range(6):
            p1 = ring1_vectors[i]
            for b in basis:
                p2 = p1 + b
                key = (round(p2[0], 6), round(p2[1], 6))
                if key in seen:
                    continue
                if np.allclose(p2, np.array([0.0, 0.0])):
                    continue
                is_r1 = False
                for p1j in ring1_vectors:
                    if np.allclose(p2, p1j):
                        is_r1 = True
                        break
                if not is_r1:
                    ring2_candidates.append(p2.copy())
                    seen.add(key)

        ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
        for i in range(5):
            centers.append(ring2_candidates[i].copy())

        assert len(centers) == 12
        x = np.zeros(39)
        for i in range(12):
            x[3*i] = centers[i][0]
            x[3*i+1] = centers[i][1]
            x[3*i+2] = math.pi/6  # 30 degrees
        x[36] = 0.0
        x[37] = 0.0
        x[38] = 0.0
        return x

    def perturb(x, sigma=0.1):
        """Add random perturbation."""
        xp = x.copy()
        for i in range(12):
            xp[3*i] += random.gauss(0, sigma)
            xp[3*i+1] += random.gauss(0, sigma)
            xp[3*i+2] += random.gauss(0, sigma * 0.3)
        xp[36] += random.gauss(0, sigma * 0.5)
        xp[37] += random.gauss(0, sigma * 0.5)
        xp[38] += random.gauss(0, sigma * 0.3)
        return xp

    # Multi-start SLSQP optimization
    best_x = None
    best_val = float('inf')

    # Start 1: clean lattice
    x0 = generate_lattice_initial(1.0)
    val0 = compute_objective(x0)
    if val0 < best_val:
        best_val = val0
        best_x = x0.copy()

    # SLSQP from lattice
    try:
        res = minimize(compute_objective, x0, method='SLSQP',
                       options={'maxiter': 50, 'ftol': 1e-8})
        if res.success or res.fun < best_val:
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
    except Exception:
        pass

    # Multi-start with perturbations
    n_starts = 8
    for s in range(n_starts):
        if time.time() > deadline - 20:
            break

        # Different perturbation scales
        sigma = 0.05 + s * 0.05
        x_init = perturb(generate_lattice_initial(1.0), sigma)

        # Also try slightly different lattice scales
        if s % 3 == 1:
            x_init = perturb(generate_lattice_initial(0.98), sigma)
        elif s % 3 == 2:
            x_init = perturb(generate_lattice_initial(1.02), sigma)

        val_init = compute_objective(x_init)
        if val_init < best_val:
            best_val = val_init
            best_x = x_init.copy()

        try:
            res = minimize(compute_objective, x_init, method='SLSQP',
                           options={'maxiter': 80, 'ftol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

    # Final polish: fine-tune outer hexagon parameters with grid search
    if best_x is not None:
        centers = [(best_x[3*i], best_x[3*i+1]) for i in range(12)]
        angles = [best_x[3*i+2] for i in range(12)]

        best_outer_side = best_val
        best_ocx, best_ocy, best_otheta = best_x[36], best_x[37], best_x[38]

        # Grid search on outer parameters
        for otheta_deg in np.linspace(math.degrees(best_otheta) - 15, math.degrees(best_otheta) + 15, 31):
            otheta_r = math.radians(otheta_deg)
            for ocx in np.linspace(best_ocx - 0.3, best_ocx + 0.3, 7):
                for ocy in np.linspace(best_ocy - 0.3, best_ocy + 0.3, 7):
                    max_proj = 0.0
                    for k in range(6):
                        normal_angle = otheta_r + math.pi/6 + k * math.pi/3
                        nx = math.cos(normal_angle)
                        ny = math.sin(normal_angle)
                        for i in range(12):
                            cx, cy = centers[i]
                            a = angles[i]
                            for vi in range(6):
                                vert_angle = a + vi * math.pi / 3
                                vx = cx + math.cos(vert_angle)
                                vy = cy + math.sin(vert_angle)
                                proj = (vx - ocx) * nx + (vy - ocy) * ny
                                if proj > max_proj:
                                    max_proj = proj
                    side = 2.0 * max_proj / math.sqrt(3)
                    if side < best_outer_side - 1e-12:
                        best_outer_side = side
                        best_ocx, best_ocy, best_otheta = ocx, ocy, otheta_r

        inner = [[centers[i][0], centers[i][1], math.degrees(angles[i])] for i in range(12)]
        outer_center = (best_ocx, best_ocy)
        outer_side = best_outer_side
        outer_angle = math.degrees(best_otheta)
        return inner, outer_center, outer_side, outer_angle

    # Fallback: return lattice configuration
    x0 = generate_lattice_initial(1.0)
    centers = [(x0[3*i], x0[3*i+1]) for i in range(12)]
    angles = [x0[3*i+2] for i in range(12)]
    inner = [[centers[i][0], centers[i][1], math.degrees(angles[i])] for i in range(12)]
    max_proj = 0.0
    for k in range(6):
        normal_angle = math.pi/6 + k * math.pi/3
        nx = math.cos(normal_angle)
        ny = math.sin(normal_angle)
        for i in range(12):
            cx, cy = centers[i]
            a = angles[i]
            for vi in range(6):
                vert_angle = a + vi * math.pi / 3
                vx = cx + math.cos(vert_angle)
                vy = cy + math.sin(vert_angle)
                proj = vx * nx + vy * ny
                if proj > max_proj:
                    max_proj = proj
    outer_side = 2.0 * max_proj / math.sqrt(3)
    return inner, (0.0, 0.0), outer_side, 0.0
# EVOLVE-BLOCK-END
