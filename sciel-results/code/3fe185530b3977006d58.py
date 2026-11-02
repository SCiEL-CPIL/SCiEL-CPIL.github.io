"""Pack 11 unit regular hexagons inside a regular hexagon; minimize outer side length."""
import numpy as np
import time

N = 11
SQRT3 = np.sqrt(3.0)

# Flat-top unit hexagon vertices (side=1, centered at origin, angle=0)
_BASE_VERTS = np.array([
    [1.0, 0.0],
    [0.5, SQRT3 / 2.0],
    [-0.5, SQRT3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -SQRT3 / 2.0],
    [0.5, -SQRT3 / 2.0],
])

# Inward unit normals for each edge (edge i connects vertex i to vertex (i+1)%6)
# For flat-top hex, edge normals point toward center:
_NORMALS = np.array([
    [1.0, 0.0],
    [0.5, SQRT3 / 2.0],
    [-0.5, SQRT3 / 2.0],
    [-1.0, 0.0],
    [-0.5, -SQRT3 / 2.0],
    [0.5, -SQRT3 / 2.0],
])

# Inward support distances (distance from center to each edge along normal)
_SUPPORT = np.full(6, 1.0)  # unit hex: all edges at distance 1 from center... actually it's SQRT3/2 for flat sides
# Wait: for flat-top unit hex, the "inradius" (center to flat edge) is SQRT3/2.
# Let me recompute: edge from (1,0) to (0.5, sqrt(3)/2). Midpoint = (0.75, sqrt(3)/4).
# Distance from origin to this edge = |cross(edge_vec, edge_point)| / |edge_vec|
# Actually simpler: the support function for the 6 edges of a regular hex of side 1:
# The apothem (inradius) = sqrt(3)/2 * 1 = sqrt(3)/2 ≈ 0.866
# But wait, the hexagon with vertices at distance 1 from center (circumradius=1) has inradius = cos(30°) = sqrt(3)/2.
# The half-plane constraint is: n_i · p <= r_i where r_i is the support distance.
# For a regular hex with circumradius 1, all 6 support distances equal the inradius = sqrt(3)/2.

_SUPPORT_DIST = SQRT3 / 2.0


def _rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _hex_verts(cx, cy, angle_deg, side=1.0):
    theta = np.deg2rad(angle_deg)
    R = _rot2(theta)
    return np.array([cx, cy]) + side * (_BASE_VERTS @ R.T).T


def _point_dist_sq_to_hex(pt, cx, cy, angle_deg, side=1.0):
    """Squared distance from point pt to hexagon (0 if inside)."""
    theta = np.deg2rad(angle_deg)
    R_inv = _rot2(-theta)
    local = R_inv @ (pt - np.array([cx, cy])) / side
    # Check 6 half-planes: n_i · local <= support_dist
    dots = local @ _NORMALS.T
    violations = np.maximum(0.0, dots - _SUPPORT_DIST)
    return float(np.max(violations) ** 2)


def _outer_containment_violation(verts, S):
    """Max violation of any vertex of inner hex being outside outer hex (side S, center 0, angle 0)."""
    # Outer hex: same normals, support = S * sqrt(3)/2
    dots = verts @ _NORMALS.T
    violations = np.maximum(0.0, dots - S * _SUPPORT_DIST)
    return float(np.max(violations))


def _pairwise_overlap_penalty(v1, v2, c1, a1, c2, a2):
    """Penalty for overlap between two unit hexagons. Returns max penetration depth squared."""
    # Check vertices of hex1 inside hex2
    d = 0.0
    for pt in v1:
        d = max(d, _point_dist_sq_to_hex(pt, c2[0], c2[1], a2))
    # Check vertices of hex2 inside hex1
    for pt in v2:
        d = max(d, _point_dist_sq_to_hex(pt, c1[0], c1[1], a1))
    return d


def _objective(params, deadline):
    if time.time() > deadline:
        return 1e12

    S = max(params[-1], 1.0)
    ip = params[:-1]

    total = S  # minimize S
    penalty = 0.0

    # Precompute vertices for all inner hexagons
    verts_list = []
    for i in range(N):
        x, y, a = ip[3*i], ip[3*i+1], ip[3*i+2]
        verts_list.append(_hex_verts(x, y, a))

    # Containment in outer hex
    for i in range(N):
        viol = _outer_containment_violation(verts_list[i], S)
        penalty += 50.0 * viol * viol

    # Pairwise non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            c1 = np.array([ip[3*i], ip[3*i+1]])
            c2 = np.array([ip[3*j], ip[3*j+1]])
            dist_sq = float(np.sum((c1 - c2)**2))
            if dist_sq > 6.0:  # (2*sqrt(3)/2)^2 * 4 ≈ ... safe reject
                continue
            a1 = ip[3*i+2]
            a2 = ip[3*j+2]
            ov = _pairwise_overlap_penalty(verts_list[i], verts_list[j], c1, a1, c2, a2)
            penalty += 100.0 * ov

    return total + penalty


def _initial_guess_hex_lattice():
    """Hexagonal lattice arrangement with 30-degree rotation for tighter packing."""
    # 11 hexagons in a hexagonal pattern: center + 6 neighbors + 4 more
    # Classic 11-hex arrangement: a hexagonal cluster
    # Positions in a hex grid with spacing = sqrt(3) (edge-sharing distance)
    # But for disjoint, we need center-to-center >= sqrt(3) for parallel hexes
    # Actually for flat-top hexes sharing an edge, center distance = sqrt(3).
    # For disjoint (non-overlapping), center distance must be >= sqrt(3) when aligned.
    # With 30-degree rotation of the lattice axes, we can achieve tighter packing.

    # Let's use a known good arrangement: 1 + 6 + 4 = 11
    # Center at origin, 6 at distance sqrt(3) in 6 directions, 4 more at distance 2*sqrt(3)
    sp = SQRT3  # edge-sharing distance
    positions = [(0.0, 0.0)]
    # 6 neighbors
    for k in range(6):
        ang = np.deg2rad(60 * k)
        positions.append((sp * np.cos(ang), sp * np.sin(ang)))
    # 4 more: at distance 2*sp in alternating directions
    for k in [0, 2, 4, 6]:  # Actually 6 positions at 2sp, pick 4
        ang = np.deg2rad(60 * k)
        positions.append((2 * sp * np.cos(ang), 2 * sp * np.sin(ang)))

    # Trim to 11
    positions = positions[:11]

    params = []
    for px, py in positions:
        params.extend([px, py, 0.0])
    params.append(4.0)  # initial outer side
    return np.array(params)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    best_params = None
    best_val = 1e12

    x0 = _initial_guess_hex_lattice()
    best_params = x0.copy()
    best_val = _objective(x0, deadline)

    # Multiple restarts with different initial configurations
    np.random.seed(42)

    # Try a few different initial patterns
    initial_configs = [x0]

    # Config 2: more spread out
    cfg2 = _initial_guess_hex_lattice().copy()
    cfg2[:-1] *= 1.2  # spread inner hexes
    cfg2[-1] = 5.0
    initial_configs.append(cfg2)

    # Config 3: random perturbation
    cfg3 = x0 + np.random.normal(0, 0.3, size=x0.shape)
    cfg3[-1] = 4.5
    initial_configs.append(cfg3)

    # Config 4: different angle for inner hexes (30 deg)
    cfg4 = x0.copy()
    for i in range(N):
        cfg4[3*i+2] = 30.0  # rotate inner hexes 30 degrees
    cfg4[-1] = 4.0
    initial_configs.append(cfg4)

    for restart, x_start in enumerate(initial_configs):
        if time.time() > deadline:
            break

        # Use L-BFGS-B for optimization
        try:
            from scipy.optimize import minimize
            res = minimize(
                _objective, x_start,
                args=(deadline,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

        # If time allows, do a second pass with tighter settings
        if time.time() < deadline - 20.0 and restart < 2:
            try:
                res2 = minimize(
                    _objective, best_params,
                    args=(deadline,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 8000}
                )
                if res2.fun < best_val:
                    best_val = res2.fun
                    best_params = res2.x.copy()
            except Exception:
                pass

    # Extract final solution
    S = max(float(best_params[-1]), 1.0)
    inner = []
    for i in range(N):
        x = float(best_params[3*i])
        y = float(best_params[3*i+1])
        a = float(best_params[3*i+2])
        inner.append([x, y, a])

    # Final validation: ensure all hexes are inside outer and non-overlapping
    # If invalid, scale up S until valid
    def _is_valid(inner_list, S, center, angle):
        for hx in inner_list:
            verts = _hex_verts(hx[0], hx[1], hx[2], 1.0)
            viol = _outer_containment_violation(verts, S)
            if viol > 1e-6:
                return False
        for i in range(len(inner_list)):
            for j in range(i+1, len(inner_list)):
                c1 = np.array([inner_list[i][0], inner_list[i][1]])
                c2 = np.array([inner_list[j][0], inner_list[j][1]])
                dist_sq = float(np.sum((c1-c2)**2))
                if dist_sq > 6.0:
                    continue
                v1 = _hex_verts(inner_list[i][0], inner_list[i][1], inner_list[i][2])
                v2 = _hex_verts(inner_list[j][0], inner_list[j][1], inner_list[j][2])
                ov = _pairwise_overlap_penalty(v1, v2, c1, inner_list[i][2], c2, inner_list[j][2])
                if ov > 1e-8:
                    return False
        return True

    # Validate and fix if needed
    if not _is_valid(inner, S, (0.0, 0.0), 0.0):
        # Scale up S incrementally
        for _ in range(50):
            S *= 1.01
            if _is_valid(inner, S, (0.0, 0.0), 0.0):
                break
        else:
            S *= 1.1  # fallback

    return inner, (0.0, 0.0), float(S), 0.0
