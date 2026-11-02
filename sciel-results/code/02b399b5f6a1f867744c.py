import math
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def hex_vertices(cx, cy, angle_deg):
        """Return 6 vertices of a unit regular hexagon centered at (cx,cy) with rotation."""
        ang = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = ang + k * math.pi / 3.0
            verts.append((cx + math.cos(theta), cy + math.sin(theta)))
        return np.array(verts)

    def support_function(points, normals):
        """Max of dot products of points with normals, and min (negative of max of -dot)."""
        # points: (n, 2), normals: (6, 2)
        projs = points @ normals.T  # (n, 6)
        max_proj = np.max(projs, axis=0)  # (6,)
        min_proj = np.min(projs, axis=0)  # (6,)
        return np.maximum(max_proj, -min_proj)  # (6,)

    def enclosing_hex_side(params):
        """
        params: [x1,y1,a1, x2,y2,a2, ..., xn,yn,an, cx, cy, outer_angle]
        Returns the side length of the minimal enclosing regular hexagon.
        """
        n = len(params) // 3
        centers = np.array(params[:2*n]).reshape(n, 2)
        angles_deg = params[2*n:3*n]
        cx, cy = params[3*n], params[3*n+1]
        outer_angle = params[3*n+2]

        # All vertices of all inner hexagons
        all_verts = []
        for i in range(n):
            verts = hex_vertices(centers[i,0], centers[i,1], angles_deg[i])
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)  # (6n, 2)

        # Normals for the outer hexagon
        normals = []
        for k in range(6):
            ang = outer_angle + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        normals = np.array(normals)  # (6, 2)

        # Support function in each normal direction
        sup = support_function(all_verts, normals)  # (6,)
        D = np.max(sup)  # Max distance from center to any vertex in any normal dir

        # Side length of enclosing hexagon with apothem D + APOTHEM
        # Wait: the apothem of the enclosing hex must be >= D + APOTHEM?
        # Actually, the inner hexagons have their own extent. The enclosing hex's apothem
        # must be at least the max support of the union of inner hexagons.
        # The side length S relates to apothem A by A = S * sqrt(3)/2, so S = 2A/sqrt(3)
        A_required = D  # The apothem of enclosing hex must be >= D
        S = 2.0 * A_required / math.sqrt(3.0)
        return S

    def objective(params):
        """Objective: minimize enclosing hex side, with penalty for overlaps."""
        n = len(params) // 3
        centers = np.array(params[:2*n]).reshape(n, 2)
        angles_deg = params[2*n:3*n]

        # Check pairwise overlaps using separation axis theorem for hexagons
        penalty = 0.0
        for i in range(n):
            vi = hex_vertices(centers[i,0], centers[i,1], angles_deg[i])
            for j in range(i+1, n):
                vj = hex_vertices(centers[j,0], centers[j,1], angles_deg[j])
                # SAT: check all edge normals of both hexagons
                overlap = True
                for verts in [vi, vj]:
                    for k in range(6):
                        p1 = verts[k]
                        p2 = verts[(k+1) % 6]
                        edge = p2 - p1
                        normal = np.array([-edge[1], edge[0]])
                        norm_len = np.linalg.norm(normal)
                        if norm_len < 1e-12:
                            continue
                        normal = normal / norm_len
                        proj_i = vi @ normal
                        proj_j = vj @ normal
                        min_i, max_i = np.min(proj_i), np.max(proj_i)
                        min_j, max_j = np.min(proj_j), np.max(proj_j)
                        if max_i < min_j or max_j < min_i:
                            overlap = False
                            break
                    if not overlap:
                        break
                if overlap:
                    # Compute minimum translation distance (approximate)
                    # Use the narrowest gap
                    min_gap = float('inf')
                    for verts in [vi, vj]:
                        for k in range(6):
                            p1 = verts[k]
                            p2 = verts[(k+1) % 6]
                            edge = p2 - p1
                            normal = np.array([-edge[1], edge[0]])
                            norm_len = np.linalg.norm(normal)
                            if norm_len < 1e-12:
                                continue
                            normal = normal / norm_len
                            proj_i = vi @ normal
                            proj_j = vj @ normal
                            min_i, max_i = np.min(proj_i), np.max(proj_i)
                            min_j, max_j = np.min(proj_j), np.max(proj_j)
                            gap = max(min_j - max_i, min_i - max_j, 0.0)
                            if gap < min_gap:
                                min_gap = gap
                    penalty += max(0.0, 1.0 - min_gap) ** 2

        S = enclosing_hex_side(params)
        return S + 100.0 * penalty

    # Initial guess: hexagonal packing
    def initial_guess():
        params = []
        # Center hexagon
        params.extend([0.0, 0.0, 0.0])
        # 6 surrounding hexagons
        for k in range(6):
            ang = k * math.pi / 3.0
            x = 2.0 * math.cos(ang)
            y = 2.0 * math.sin(ang)
            params.extend([x, y, 0.0])
        # 4 more in the next ring (partial)
        # Place at (3, 0), (1.5, 2.6), (-1.5, 2.6), (-3, 0)
        extra = [(3.0, 0.0), (1.5, 2.6), (-1.5, 2.6), (-3.0, 0.0)]
        for x, y in extra:
            params.extend([x, y, 0.0])
        # Outer hexagon center and angle
        params.extend([0.0, 0.0, 0.0])
        return np.array(params)

    x0 = initial_guess()

    # Differential evolution for global search
    bounds = [(-5, 5), (-5, 5), (-180, 180)] * N + [(-5, 5), (-5, 5), (-30, 30)]

    best_result = None
    best_val = float('inf')

    # Try a few random restarts with differential evolution
    np.random.seed(42)
    for restart in range(3):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break

        if restart == 0:
            x_init = x0
        else:
            x_init = x0 + np.random.uniform(-1, 1, size=x0.shape)

        try:
            result = differential_evolution(
                objective, bounds, x0=x_init,
                maxiter=200, popsize=15, tol=1e-8,
                seed=42 + restart, polish=False,
                init='latinhypercube'
            )
            if result.fun < best_val:
                best_val = result.fun
                best_result = result.x
        except Exception:
            pass

    if best_result is None:
        best_result = x0
        best_val = objective(best_result)

    # Refine with L-BFGS-B
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            refined = minimize(
                objective, best_result, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12}
            )
            if refined.fun < best_val:
                best_val = refined.fun
                best_result = refined.x
    except Exception:
        pass

    # Extract final parameters
    n = len(best_result) // 3
    centers = np.array(best_result[:2*n]).reshape(n, 2)
    angles_deg = best_result[2*n:3*n]
    cx, cy = best_result[3*n], best_result[3*n+1]
    outer_angle = best_result[3*n+2]

    # Compute actual S
    S = enclosing_hex_side(best_result)

    # Build inner list
    inner = []
    for i in range(n):
        inner.append([float(centers[i,0]), float(centers[i,1]), float(angles_deg[i])])

    center = [float(cx), float(cy)]
    side = float(S)
    rotation = float(math.degrees(outer_angle) % 60.0)

    return inner, center, side, rotation
