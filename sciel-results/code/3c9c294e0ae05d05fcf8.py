import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import Delaunay

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def hex_vertices(cx, cy, s=1.0):
        """Return 6 vertices of unit hexagon centered at (cx, cy), flat-top orientation."""
        a = np.radians(0)
        pts = np.array([[cx + s * np.cos(a + np.radians(60 * k)),
                         cy + s * np.sin(a + np.radians(60 * k))] for k in range(6)])
        return pts

    def required_side(positions):
        """Exact minimum outer side (angle 0, centered at origin) to contain all unit hexagons."""
        all_v = np.vstack([hex_vertices(*p) for p in positions])
        x, y = all_v[:, 0], all_v[:, 1]
        s3 = np.sqrt(3.0)
        vals = np.array([
            x + y / s3,
            2 * y / s3,
            -x + y / s3,
            -x - y / s3,
            -2 * y / s3,
            x - y / s3
        ])
        return np.max(vals)

    def sat_disjoint(v1, v2, tol=1e-7):
        """SAT: return True if two convex polygons are disjoint."""
        for poly in (v1, v2):
            for i in range(6):
                j = (i + 1) % 6
                ex, ey = poly[j] - poly[i]
                nx, ny = -ey, ex
                nl = np.hypot(nx, ny)
                if nl < 1e-12:
                    continue
                nx, ny = nx / nl, ny / nl
                p1 = v1 @ np.array([nx, ny])
                p2 = v2 @ np.array([nx, ny])
                if p1.max() < p2.min() - tol or p2.max() < p1.min() - tol:
                    return True
        return False

    def all_disjoint(positions, tol=1e-6):
        """Check all pairs are disjoint."""
        verts = [hex_vertices(*p) for p in positions]
        for i in range(N):
            for j in range(i + 1, N):
                if not sat_disjoint(verts[i], verts[j], tol):
                    return False
        return True

    def min_pairwise_sep(positions):
        """Approximate minimum separation between any pair of hexagons using vertex-to-vertex distances."""
        verts = [hex_vertices(*p) for p in positions]
        min_sep = float('inf')
        for i in range(N):
            for j in range(i + 1, N):
                # Minimum distance between two convex polygons approximated by min vertex-to-vertex
                d = np.min([np.linalg.norm(a - b) for a in verts[i] for b in verts[j]])
                # For hexagons, actual separation is roughly d - 2*apothem_overlap
                # But for penalty, we just need to detect overlap: if any vertex of one is inside the other
                # Simpler: check if SAT says they overlap
                if not sat_disjoint(verts[i], verts[j], 1e-7):
                    min_sep = min(min_sep, -1.0)  # mark as overlapping
        return min_sep

    # Better: smooth penalty using signed distance approximation
    def penalty(positions, power=2):
        """Penalty for overlaps. 0 if all disjoint."""
        verts = [hex_vertices(*p) for p in positions]
        pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                if not sat_disjoint(verts[i], verts[j], 1e-7):
                    # Estimate overlap depth via min vertex-to-opposite-edge
                    # Use simpler: count as penalty 1.0 per overlapping pair
                    pen += 1.0
        return pen ** power

    def objective(params, penalty_weight=100.0):
        """Total objective = outer_side + penalty_weight * overlap_penalty."""
        positions = params.reshape(N, 2)
        side = required_side(positions)
        pen = penalty(positions)
        return side + penalty_weight * pen

    def objective_grad(params, penalty_weight=100.0):
        """Objective with numerical gradient."""
        positions = params.reshape(N, 2)
        side = required_side(positions)
        pen = penalty(positions)
        val = side + penalty_weight * pen
        # Numerical gradient for side
        eps = 1e-6
        grad = np.zeros(2 * N)
        for k in range(2 * N):
            p2 = params.copy()
            p2[k] += eps
            positions2 = p2.reshape(N, 2)
            side2 = required_side(positions2)
            pen2 = penalty(positions2)
            grad[k] = (side2 + penalty_weight * pen2 - val) / eps
        return val, grad

    # Initial configurations
    def init_hex_lattice():
        """Hexagonal close-packing: 7 hexes in center + 5 on ring."""
        s3 = np.sqrt(3)
        pts = []
        # Center
        pts.append([0, 0])
        # Ring 1 (6 hexes)
        for k in range(6):
            a = np.radians(60 * k)
            pts.append([s3 * np.cos(a), 1.5 * np.sin(a)])
        # 5 more on outer ring
        for k in range(5):
            a = np.radians(60 * k + 30)
            r = 2 * 1.5 * np.sqrt(3)  # approximate
            pts.append([r * np.cos(a), r * np.sin(a)])
        return np.array(pts[:N])

    def init_circular():
        pts = np.zeros((N, 2))
        for i in range(N):
            a = 2 * np.pi * i / N
            r = 2.5
            pts[i] = [r * np.cos(a), r * np.sin(a)]
        return pts

    def init_random(seed=42):
        np.random.seed(seed)
        return np.random.uniform(-4, 4, (N, 2))

    inits = [init_hex_lattice(), init_circular(), init_random(1), init_random(7), init_random(99)]

    best_pos = None
    best_side = float('inf')

    # Strategy 1: Differential evolution for global search
    def de_objective(params):
        positions = params.reshape(N, 2)
        side = required_side(positions)
        if not all_disjoint(positions, 1e-5):
            return side + 50.0
        return side

    if time.time() < deadline - 60:
        try:
            bounds = [(-5, 5)] * (2 * N)
            result = differential_evolution(
                de_objective, bounds,
                maxiter=200, popsize=15, tol=1e-6,
                seed=42, init='sobol',
                update_interval=10, mutation=(0.5, 1.0), recombination=0.9
            )
            pos = result.x.reshape(N, 2)
            if all_disjoint(pos, 1e-5):
                s = required_side(pos)
                if s < best_side:
                    best_side = s
                    best_pos = pos
        except Exception:
            pass

    # Strategy 2: L-BFGS-B local refinement from multiple starts
    for init in inits:
        if time.time() > deadline - 30:
            break

        # Ensure initial is valid
        pos0 = init.copy()
        for scale in [1.0, 1.2, 1.5, 2.0]:
            if all_disjoint(pos0, 1e-5):
                break
            pos0 = init * scale
        else:
            continue

        for pw in [50.0, 20.0, 10.0]:
            if time.time() > deadline - 30:
                break

            params0 = pos0.flatten()
            try:
                res = minimize(
                    objective_grad, params0, args=(pw,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-10, 'maxls': 50}
                )
                pos = res.x.reshape(N, 2)
                if all_disjoint(pos, 1e-6):
                    s = required_side(pos)
                    if s < best_side:
                        best_side = s
                        best_pos = pos
            except Exception:
                continue

    # Strategy 3: Local SA from best found
    if best_pos is not None and time.time() < deadline - 20:
        cur = best_pos.copy()
        cur_side = best_side
        T = 0.5
        for it in range(20000):
            if time.time() > deadline - 5:
                break
            idx = np.random.randint(N)
            dx, dy = np.random.normal(0, 0.3, 2)
            new_pos = cur.copy()
            new_pos[idx] += [dx, dy]
            if not all_disjoint(new_pos, 1e-6):
                continue
            ns = required_side(new_pos)
            d = ns - cur_side
            if d < 0 or np.random.random() < np.exp(-d / max(T, 1e-8)):
                cur = new_pos
                cur_side = ns
                if cur_side < best_side:
                    best_side = cur_side
                    best_pos = cur.copy()
            T *= 0.9999

    # Fallback
    if best_pos is None:
        # Spread linearly
        best_pos = np.array([[i * 3.0, 0.0] for i in range(N)], dtype=float)
        best_side = required_side(best_pos)

    # Final check
    if not all_disjoint(best_pos, 1e-5):
        # Last resort: very spread out
        best_pos = np.array([[i * 4.0, 0.0] for i in range(N)], dtype=float)
        best_side = required_side(best_pos)

    final_side = required_side(best_pos)

    inner = []
    for i in range(N):
        inner.append([float(best_pos[i, 0]), float(best_pos[i, 1]), 0.0])

    return inner, (0.0, 0.0), float(final_side), 0.0

# EVOLVE-BLOCK-END
