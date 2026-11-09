"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    SQRT3 = math.sqrt(3)
    R = 1.0  # circumradius of unit hexagon
    INR = SQRT3 / 2.0  # inradius

    def hex_verts(cx, cy, ang_deg):
        a = math.radians(ang_deg)
        return [(cx + R * math.cos(a + i * math.pi / 3),
                 cy + R * math.sin(a + i * math.pi / 3)) for i in range(6)]

    def hex_normals(ang_deg):
        a = math.radians(ang_deg)
        return [(math.cos(a + i * math.pi / 3 + math.pi / 6),
                 math.sin(a + i * math.pi / 3 + math.pi / 6)) for i in range(6)]

    def is_disjoint(v1, v2, n1, n2):
        for nx, ny in n1 + n2:
            p1 = [vx * nx + vy * ny for vx, vy in v1]
            p2 = [vx * nx + vy * ny for vx, vy in v2]
            if max(p1) < min(p2) or max(p2) < min(p1):
                return True
        return False

    def outer_side(hex_list, oa_deg):
        oa = math.radians(oa_deg)
        max_p = 0.0
        for verts in hex_list:
            for vx, vy in verts:
                for i in range(6):
                    nx = math.cos(oa + i * math.pi / 3 + math.pi / 6)
                    ny = math.sin(oa + i * math.pi / 3 + math.pi / 6)
                    p = abs(vx * nx + vy * ny)
                    if p > max_p:
                        max_p = p
        return 2.0 * max_p / SQRT3

    def axial_to_cart(q, r, s=1.0):
        return (SQRT3 * s * (q + r / 2.0), 1.5 * s * r)

    # Diverse initial topologies
    topologies = [
        # Honeycomb-ish
        [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)],
        # Ring + center
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)],
        # Line + offsets
        [(0,0),(1,0),(-1,0),(0,1),(1,1),(-1,1),(0,-1),(1,-1),(-1,-1),(2,0),(-2,0)],
    ]

    def init_params(coords):
        params = []
        for q, r in coords:
            x, y = axial_to_cart(q, r)
            params.extend([x, y, 0.0])
        return np.array(params)

    def random_init():
        params = []
        for _ in range(N):
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            a = random.uniform(0, 60)
            params.extend([x, y, a])
        return np.array(params)

    def relax(params, max_iter=5000):
        """Physics-based repulsion relaxation."""
        p = params.copy()
        dt = 0.05
        damping = 0.92
        # Adaptive dt
        for it in range(max_iter):
            if time.time() > deadline - 20.0:
                break
            # Centroid shift
            cx_avg = np.mean(p[0::3])
            cy_avg = np.mean(p[1::3])
            # Shift to center
            for i in range(N):
                p[i*3] -= cx_avg
                p[i*3+1] -= cy_avg

            forces = np.zeros_like(p)
            verts = []
            norms = []
            for i in range(N):
                cx, cy, a = p[i*3], p[i*3+1], p[i*3+2]
                verts.append(hex_verts(cx, cy, a))
                norms.append(hex_normals(a))

            # Pairwise repulsion
            for i in range(N):
                for j in range(i+1, N):
                    if not is_disjoint(verts[i], verts[j], norms[i], norms[j]):
                        # Compute penetration vector (simple: center-to-center)
                        dx = p[i*3] - p[j*3]
                        dy = p[i*3+1] - p[j*3+1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < 1e-9:
                            dx = random.uniform(-0.1, 0.1)
                            dy = random.uniform(-0.1, 0.1)
                            dist = math.sqrt(dx*dx + dy*dy)
                        # Penetration depth
                        pen = 2 * R - dist
                        if pen > 0:
                            fx = pen * dx / dist
                            fy = pen * dy / dist
                            forces[i*3] += fx
                            forces[i*3+1] += fy
                            forces[j*3] -= fx
                            forces[j*3+1] -= fy

            # Wall constraint: keep inside a current bounding hexagon
            # Estimate current outer side
            hex_list = [hex_verts(p[i*3], p[i*3+1], p[i*3+2]) for i in range(N)]
            side = outer_side(hex_list, 0.0)
            # Target: slightly smaller than current to drive compression
            target = side * 0.999
            # For each vertex of each hexagon, if it exceeds the target hexagon, push back
            for i in range(N):
                cx, cy, a = p[i*3], p[i*3+1], p[i*3+2]
                for vx, vy in hex_verts(cx, cy, a):
                    # Check against 6 walls of target hexagon
                    for k in range(6):
                        nx = math.cos(k * math.pi / 3 + math.pi / 6)
                        ny = math.sin(k * math.pi / 3 + math.pi / 6)
                        proj = vx * nx + vy * ny
                        limit = target * SQRT3 / 2.0
                        if proj > limit:
                            # Push center back
                            forces[i*3] -= (proj - limit) * nx * 0.5
                            forces[i*3+1] -= (proj - limit) * ny * 0.5
                        elif proj < -limit:
                            forces[i*3] += (-limit - proj) * nx * 0.5
                            forces[i*3+1] += (-limit - proj) * ny * 0.5

            # Update
            for i in range(N):
                p[i*3] += dt * forces[i*3] * damping
                p[i*3+1] += dt * forces[i*3+1] * damping
                # Angle: small random jitter to escape local minima
                if it % 50 == 0:
                    p[i*3+2] += random.gauss(0, 1.0)
                p[i*3+2] = p[i*3+2] % 60

        return p

    def refine(params, max_iter=2000):
        """Finer relaxation with smaller dt."""
        p = params.copy()
        dt = 0.01
        damping = 0.95
        for it in range(max_iter):
            if time.time() > deadline - 15.0:
                break
            cx_avg = np.mean(p[0::3])
            cy_avg = np.mean(p[1::3])
            for i in range(N):
                p[i*3] -= cx_avg
                p[i*3+1] -= cy_avg

            forces = np.zeros_like(p)
            verts = []
            norms = []
            for i in range(N):
                cx, cy, a = p[i*3], p[i*3+1], p[i*3+2]
                verts.append(hex_verts(cx, cy, a))
                norms.append(hex_normals(a))

            for i in range(N):
                for j in range(i+1, N):
                    if not is_disjoint(verts[i], verts[j], norms[i], norms[j]):
                        dx = p[i*3] - p[j*3]
                        dy = p[i*3+1] - p[j*3+1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < 1e-9:
                            dx = random.uniform(-0.01, 0.01)
                            dy = random.uniform(-0.01, 0.01)
                            dist = math.sqrt(dx*dx + dy*dy)
                        pen = 2 * R - dist
                        if pen > 0:
                            fx = pen * dx / dist
                            fy = pen * dy / dist
                            forces[i*3] += fx
                            forces[i*3+1] += fy
                            forces[j*3] -= fx
                            forces[j*3+1] -= fy

            hex_list = [hex_verts(p[i*3], p[i*3+1], p[i*3+2]) for i in range(N)]
            side = outer_side(hex_list, 0.0)
            target = side * 0.9999
            for i in range(N):
                cx, cy, a = p[i*3], p[i*3+1], p[i*3+2]
                for vx, vy in hex_verts(cx, cy, a):
                    for k in range(6):
                        nx = math.cos(k * math.pi / 3 + math.pi / 6)
                        ny = math.sin(k * math.pi / 3 + math.pi / 6)
                        proj = vx * nx + vy * ny
                        limit = target * SQRT3 / 2.0
                        if proj > limit:
                            forces[i*3] -= (proj - limit) * nx * 0.3
                            forces[i*3+1] -= (proj - limit) * ny * 0.3
                        elif proj < -limit:
                            forces[i*3] += (-limit - proj) * nx * 0.3
                            forces[i*3+1] += (-limit - proj) * ny * 0.3

            for i in range(N):
                p[i*3] += dt * forces[i*3] * damping
                p[i*3+1] += dt * forces[i*3+1] * damping
                if it % 100 == 0:
                    p[i*3+2] += random.gauss(0, 0.3)
                p[i*3+2] = p[i*3+2] % 60

        return p

    best_params = None
    best_side = float('inf')

    # Multi-start: topologies + random
    inits = [init_params(t) for t in topologies]
    for _ in range(5):
        inits.append(random_init())

    for init in inits:
        if time.time() > deadline - 30.0:
            break

        # Coarse relaxation
        p1 = relax(init, max_iter=3000)
        # Fine relaxation
        p2 = refine(p1, max_iter=2000)

        # Validate
        cx_avg = np.mean(p2[0::3])
        cy_avg = np.mean(p2[1::3])
        for i in range(N):
            p2[i*3] -= cx_avg
            p2[i*3+1] -= cy_avg

        hex_list = [hex_verts(p2[i*3], p2[i*3+1], p2[i*3+2]) for i in range(N)]
        norms_list = [hex_normals(p2[i*3+2]) for i in range(N)]

        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if not is_disjoint(hex_list[i], hex_list[j], norms_list[i], norms_list[j]):
                    valid = False
                    break
            if not valid:
                break

        if valid:
            # Sweep outer rotation
            best_oa = 0.0
            best_s = float('inf')
            for oa in range(0, 600, 2):
                s = outer_side(hex_list, oa * 0.1)
                if s < best_s:
                    best_s = s
                    best_oa = oa * 0.1
            # Fine sweep
            for oa in np.arange(best_oa - 2, best_oa + 2, 0.05):
                s = outer_side(hex_list, oa)
                if s < best_s:
                    best_s = s
                    best_oa = oa

            if best_s < best_side:
                best_side = best_s
                best_params = p2.copy()
                best_params_oa = best_oa

    if best_params is None:
        # Fallback: use first topology as-is
        p = init_params(topologies[0])
        cx_avg = np.mean(p[0::3])
        cy_avg = np.mean(p[1::3])
        for i in range(N):
            p[i*3] -= cx_avg
            p[i*3+1] -= cy_avg
        hex_list = [hex_verts(p[i*3], p[i*3+1], p[i*3+2]) for i in range(N)]
        best_side = outer_side(hex_list, 0.0)
        best_params = p
        best_params_oa = 0.0

    inner_list = []
    for i in range(N):
        inner_list.append([best_params[i*3], best_params[i*3+1], best_params[i*3+2]])

    return (inner_list, (0.0, 0.0), best_side, best_params_oa)

# EVOLVE-BLOCK-END
