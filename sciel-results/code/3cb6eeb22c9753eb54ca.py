import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0

# 6 vertex offsets for a unit regular hexagon (side 1, centered at origin, flat-top orientation)
# Vertices at angles 0, 60, 120, 180, 240, 300 degrees
V = np.array([
    [1.0, 0.0],
    [0.5, math.sqrt(3) / 2.0],
    [-0.5, math.sqrt(3) / 2.0],
    [-1.0, 0.0],
    [-0.5, -math.sqrt(3) / 2.0],
    [0.5, -math.sqrt(3) / 2.0],
])

# 6 outward unit normals for the inner hexagons (perpendicular to edges, pointing outward)
# Edge normals at angles 30, 90, 150, 210, 270, 330 degrees
INNER_NORMALS = np.array([
    [math.cos(math.pi / 6), math.sin(math.pi / 6)],
    [math.cos(math.pi / 2), math.sin(math.pi / 2)],
    [math.cos(5 * math.pi / 6), math.sin(5 * math.pi / 6)],
    [math.cos(7 * math.pi / 6), math.sin(7 * math.pi / 6)],
    [math.cos(3 * math.pi / 2), math.sin(3 * math.pi / 2)],
    [math.cos(11 * math.pi / 6), math.sin(11 * math.pi / 6)],
])

# 6 outward unit normals for the outer hexagon (perpendicular to its edges)
# Outer hexagon side S, apothem S*sqrt(3)/2. Normals at angles 0, 60, 120, 180, 240, 300
OUTER_NORMALS = np.array([
    [math.cos(0), math.sin(0)],
    [math.cos(math.pi / 3), math.sin(math.pi / 3)],
    [math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3)],
    [math.cos(math.pi), math.sin(math.pi)],
    [math.cos(4 * math.pi / 3), math.sin(4 * math.pi / 3)],
    [math.cos(5 * math.pi / 3), math.sin(5 * math.pi / 3)],
])

def hexagon_vertices(cx, cy):
    """Return 6 vertices of unit hexagon centered at (cx, cy)."""
    return V + np.array([cx, cy])

def soft_max(vals, temp):
    """Numerically stable soft maximum."""
    m = np.max(vals)
    if m > 1e6:
        return m
    shifted = (vals - m) / temp
    shifted = np.clip(shifted, -50, 50)
    return m + temp * np.log(np.sum(np.exp(shifted)))

def objective(params, temp=0.05):
    """
    params: [x1..x11, y1..y11, S]
    Returns soft outer side length + penalty for overlaps.
    """
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return 1e10

    positions = params[:2 * N].reshape(N, 2)
    S = params[2 * N]
    if S < 0.5:
        return 1e10

    # --- Outer containment: each inner vertex must be inside outer hexagon ---
    # Outer hexagon centered at origin, side S, apothem a_out = S * sqrt(3)/2
    a_out = S * APOTHEM

    # For each inner hexagon vertex, project onto each outer normal
    # Constraint: |dot(v, n)| <= a_out for all vertices v, normals n
    # We want min S such that max over all (vertex, normal) of |dot(v,n)| / APOTHEM <= S

    # Collect all inner vertices: (N*6, 2)
    all_vertices = np.array([hexagon_vertices(px, py) for px, py in positions])  # (66, 2)

    # Project all vertices onto all 6 outer normals: (66, 6)
    projs = all_vertices @ OUTER_NORMALS.T
    abs_projs = np.abs(projs).ravel()  # (396,)

    # S must be >= max(abs_projs) / APOTHEM
    # Soft max to make differentiable
    S_needed = soft_max(abs_projs, temp) / APOTHEM

    # Objective: we want to minimize S, but S is a parameter.
    # Use a penalty: (S - S_needed)^2 if S < S_needed, else small value
    # Actually, better: minimize S_needed directly, and report S = S_needed
    # But we also need to handle overlaps.

    # --- Inner overlap: for each pair of inner hexagons, enforce separation ---
    # Two convex hexagons are disjoint iff there exists a separating axis.
    # For two regular hexagons with same orientation, the separating axes are the 6 edge normals.
    # For pair (i, j), the minimum distance along normal n is:
    #   min over k of dot(v_ik, n)  vs  max over k of dot(v_jk, n)
    # They overlap on axis n if min_i > max_j AND min_j > max_i (i.e., projections overlap)
    # Actually: they are separated on axis n if max_i < min_j OR max_j < min_i
    # They overlap if for ALL 6 axes, the projections overlap.

    # Simpler: use the fact that for two unit hexagons with same orientation,
    # the Minkowski difference is a hexagon with side 2.
    # The centers must be at distance >= 2 * APOTHEM = sqrt(3) in the hexagonal metric.
    # Actually, the minimum distance between centers for non-overlapping unit hexagons (same orientation)
    # is sqrt(3) in the direction of edge normals, and 2 in the direction of vertex normals.
    # More precisely: the support function of the Minkowski difference.

    # For two identical oriented regular hexagons, the separating axes are the 6 edge normals.
    # Along normal n_k, the half-width of each hexagon is APOTHEM.
    # So the centers must satisfy: |dot(c_i - c_j, n_k)| >= 2 * APOTHEM = sqrt(3) for some k?
    # No: they are separated if there exists k such that |dot(c_i - c_j, n_k)| >= 2 * APOTHEM.
    # They overlap if for ALL k, |dot(c_i - c_j, n_k)| < 2 * APOTHEM.

    # Wait, that's not right either. Let me think again.
    # Hexagon A centered at c_A, Hexagon B centered at c_B.
    # A = {c_A + v : v in H}, B = {c_B + v : v in H} where H is unit hexagon.
    # A and B are disjoint iff (c_A - c_B) is not in H - H = 2H (Minkowski sum of H and -H).
    # 2H is a hexagon with side 2, apothem 2*APOTHEM = sqrt(3).
    # So (c_A - c_B) must be outside the hexagon 2H.
    # The hexagon 2H has normals INNER_NORMALS and apothem sqrt(3).
    # So (c_A - c_B) is inside 2H iff for all k: |dot(c_A - c_B, INNER_NORMALS[k])| <= sqrt(3).
    # They overlap iff for all k: |dot(c_A - c_B, INNER_NORMALS[k])| < sqrt(3).
    # They are separated iff there exists k: |dot(c_A - c_B, INNER_NORMALS[k])| >= sqrt(3).

    # Penalty: for each pair, compute the minimum over k of (sqrt(3) - |dot(c_i - c_j, n_k)|)
    # If this min is positive, they overlap. We want to push it to <= 0.
    # Soft penalty: sum over pairs of soft_max(0, max_k (sqrt(3) - |dot|))^2

    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            diff = positions[i] - positions[j]
            # Project onto 6 inner normals
            proj_vals = np.abs(diff @ INNER_NORMALS.T)  # (6,)
            # If all proj_vals < sqrt(3), they overlap
            # Overlap depth = max_k (sqrt(3) - proj_vals[k])
            # If max_k (sqrt(3) - proj_vals[k]) > 0, they overlap
            margin = np.sqrt(3) - proj_vals  # (6,)
            # If any margin > 0, they are separated on that axis
            # They overlap iff all margin < 0, i.e., max(margin) < 0
            # Overlap amount = -min(margin) if min(margin) < 0
            # Actually: they overlap iff min(margin) < 0
            # The "violation" is how much they overlap = -min(margin) when min < 0
            min_margin = np.min(margin)
            if min_margin < 0:
                overlap_penalty += (-min_margin) ** 2

    # Total objective: S_needed + lambda * overlap_penalty
    # We want S_needed to be small AND overlap_penalty to be 0
    # Use a weighted sum
    lam = 10.0
    total = S_needed + lam * overlap_penalty

    return total

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: simple grid
        positions = []
        for i in range(N):
            positions.append([i * 1.8, 0.0])
        S = 10.0
        inner = [[float(x), float(y), 0.0] for x, y in positions]
        return inner, [0.0, 0.0], float(S), 0.0

    best_S = float('inf')
    best_positions = None

    # Multiple random restarts
    num_restarts = 8
    rng = np.random.default_rng(42)

    for restart in range(num_restarts):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break

        # Initialize positions in a rough hexagonal cluster
        # Use axial coordinates for initialization
        axial_coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (1, 1), (-1, 2), (-2, 1), (2, -1)
        ]
        init_positions = []
        for q, r in axial_coords:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            init_positions.append([x, y])
        init_positions = np.array(init_positions)

        # Add small random perturbation
        if restart > 0:
            init_positions = init_positions + rng.normal(0, 0.1, size=(N, 2))

        # Initial S estimate
        max_proj = 0
        for px, py in init_positions:
            verts = hexagon_vertices(px, py)
            for v in verts:
                for n in OUTER_NORMALS:
                    p = abs(v @ n)
                    if p > max_proj:
                        max_proj = p
        init_S = max_proj / APOTHEM + 0.5

        x0 = np.concatenate([init_positions.ravel(), [init_S]])

        # Optimize
        try:
            # Use L-BFGS-B with a decreasing temperature
            temps = [0.1, 0.05, 0.02, 0.01]
            current_params = x0.copy()

            for temp in temps:
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break

                def obj(p, t=temp):
                    return objective(p, t)

                res = minimize(obj, current_params, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})

                if res.fun < 1e10:
                    current_params = res.x

            # Extract final S and positions
            final_positions = current_params[:2 * N].reshape(N, 2)
            final_S = current_params[2 * N]

            # Verify: compute actual S_needed from final positions
            all_vertices = np.array([hexagon_vertices(px, py) for px, py in final_positions])
            projs = all_vertices @ OUTER_NORMALS.T
            abs_projs = np.abs(projs).ravel()
            S_needed = np.max(abs_projs) / APOTHEM

            # Check overlaps
            overlap = False
            for i in range(N):
                for j in range(i + 1, N):
                    diff = final_positions[i] - final_positions[j]
                    proj_vals = np.abs(diff @ INNER_NORMALS.T)
                    if np.all(proj_vals < np.sqrt(3) - 1e-9):
                        overlap = True
                        break
                if overlap:
                    break

            if not overlap:
                # Valid packing: S = S_needed
                if S_needed < best_S:
                    best_S = S_needed
                    best_positions = final_positions.copy()
            else:
                # Invalid: penalize
                if S_needed < best_S and not overlap:
                    pass

        except Exception:
            pass

    # If no valid packing found, use a fallback
    if best_positions is None:
        # Use the axial cluster with a large enough S
        axial_coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (1, 1), (-1, 2), (-2, 1), (2, -1)
        ]
        best_positions = []
        for q, r in axial_coords:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            best_positions.append([x, y])
        best_positions = np.array(best_positions)

        all_vertices = np.array([hexagon_vertices(px, py) for px, py in best_positions])
        projs = all_vertices @ OUTER_NORMALS.T
        abs_projs = np.abs(projs).ravel()
        best_S = np.max(abs_projs) / APOTHEM

    # Center the packing at origin
    centroid = np.mean(best_positions, axis=0)
    best_positions = best_positions - centroid

    # Recompute S after centering
    all_vertices = np.array([hexagon_vertices(px, py) for px, py in best_positions])
    projs = all_vertices @ OUTER_NORMALS.T
    abs_projs = np.abs(projs).ravel()
    best_S = np.max(abs_projs) / APOTHEM

    # Add tiny safety margin
    best_S += 1e-9

    inner = [[float(x), float(y), 0.0] for x, y in best_positions]
    center = [0.0, 0.0]
    side = float(best_S)
    rotation = 0.0

    return inner, center, side, rotation
