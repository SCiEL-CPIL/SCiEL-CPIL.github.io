# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = _T0 + 560.0  # safety margin before the 630s kill
_EPS = 1e-6

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    return np.stack([cx + s * np.cos(th), cy + s * np.sin(th)], axis=1)

def _hex_edge_normals(cx, cy, s, a_deg):
    v = _hex_vertices(cx, cy, s, a_deg)
    nrm = []
    for i in range(6):
        p1 = v[i]
        p2 = v[(i + 1) % 6]
        e = p2 - p1
        L = np.hypot(e[0], e[1])
        if L < 1e-12:
            nrm.append(np.array([0.0, 0.0]))
        else:
            nrm.append(np.array([-e[1], e[0]]) / L)
    return np.array(nrm)  # (6,2)

def _separated_fast(A, B):
    # A,B: (6,2) vertex arrays of two unit hexes.
    # 12 normals: outward of A (6) and outward of B (6).
    # For each normal, project both sets; separated iff gap > EPS on some normal.
    # Outward normal for edge i (from v_i to v_{i+1}) is (-(e_y), e_x)/|e|.
    def outward_normals(V):
        e = np.roll(V, -1, axis=0) - V
        L = np.linalg.norm(e, axis=1)
        n = np.stack([-e[:, 1], e[:, 0]], axis=1) / L[:, None]
        return n
    nA = outward_normals(A)  # (6,2)
    nB = outward_normals(B)  # (6,2)
    N = np.vstack([nA, nB])  # (12,2)
    # Project A and B onto each normal: (12,6) each
    PA = A @ N.T  # (6,12) -> we want (12,6): transpose
    PA = PA.T
    PB = B @ N.T
    PB = PB.T
    minA = PA.min(axis=1)
    maxA = PA.max(axis=1)
    minB = PB.min(axis=1)
    maxB = PB.max(axis=1)
    # gap = min(0, maxA - minB) or min(0, maxB - minA) ... separated if intervals apart
    gap1 = maxA - minB  # if > 0, A's max is beyond B's min -> possibly apart on + side
    gap2 = maxB - minA
    # We need: exists normal where (maxA < minB - EPS) or (maxB < minA - EPS)
    sep = (maxA < minB - _EPS) | (maxB < minA - _EPS)
    return bool(sep.any())

def _all_pairs_separated(hexes):
    # hexes: list of (cx,cy,angle_deg). Use unit side.
    V = [_hex_vertices(cx, cy, 1.0, a) for (cx, cy, a) in hexes]
    m = len(V)
    for i in range(m):
        for j in range(i + 1, m):
            if not _separated_fast(V[i], V[j]):
                return False
    return True

def _outer_vertex_order(ox, oy, S, a_deg):
    # vertices i=0..5: (ox + S cos(a+60i), oy + S sin(a+60i))
    return _hex_vertices(ox, oy, S, a_deg)

def _inside_outer(hexes, ox, oy, S, a_deg):
    # For each inner vertex p, for each outer edge p1->p2, cross(p2-p1, p-p1) >= -EPS
    Vout = _outer_vertex_order(ox, oy, S, a_deg)
    for (cx, cy, ang) in hexes:
        Vin = _hex_vertices(cx, cy, 1.0, ang)
        for p in Vin:
            for i in range(6):
                p1 = Vout[i]
                p2 = Vout[(i + 1) % 6]
                e = p2 - p1
                d = p - p1
                c = e[0] * d[1] - e[1] * d[0]
                if c < -_EPS:
                    return False
    return True

def _validity(hexes, ox, oy, S, a_deg):
    if not np.all(np.isfinite(hexes)) or not np.all(np.isfinite([ox, oy, S, a_deg])):
        return False
    if S < np.sqrt(11.0) - 1e-12:
        return False
    if not _all_pairs_separated(hexes):
        return False
    if not _inside_outer(hexes, ox, oy, S, a_deg):
        return False
    return True

def _min_outer_side(hexes, ox, oy, a_deg):
    # Find minimal S such that all inner vertices are inside outer hex with given orientation/center.
    # For a convex polygon (outer hexagon), a point is inside iff for each edge, cross >= 0.
    # We need S such that for all inner points p and all edges (defined by orientation a and center (ox,oy)):
    #   cross(v_{i+1} - v_i, p - v_i) >= 0
    # v_i = (ox, oy) + S * u_i where u_i = (cos(a+60i), sin(a+60i)).
    # This is linear in S for each (p, i). We can compute the minimal S that satisfies all.
    # Let's derive: edge from v_i to v_{i+1}.
    #   v_i = c + S*u_i, v_{i+1} = c + S*u_{i+1}
    #   e = S*(u_{i+1}-u_i), d = p - c - S*u_i
    #   cross(e, d) = S * cross(u_{i+1}-u_i, p - c - S*u_i)
    #   = S * [ cross(u_{i+1}-u_i, p-c) - S*cross(u_{i+1}-u_i, u_i) ]
    # cross(u_{i+1}-u_i, u_i) = cross(u_{i+1}, u_i) - cross(u_i, u_i) = cross(u_{i+1}, u_i)
    # For unit hex, u_{i+1} is u_i rotated by 60 deg. cross(u_{i+1}, u_i) = -cross(u_i, u_{i+1}) = -sin(60) = -sqrt(3)/2.
    # So cross(e,d) = S * [ cross(u_{i+1}-u_i, p-c) - S * (-sqrt(3)/2) ]
    #   = S * cross(u_{i+1}-u_i, p-c) + S^2 * (sqrt(3)/2)
    # We need this >= 0 for all i,p.
    # Let A_ip = cross(u_{i+1}-u_i, p-c). Then condition: S*A_ip + S^2*(sqrt(3)/2) >= 0.
    # Since S^2*(sqrt(3)/2) >= 0 always, if A_ip >= 0, condition holds for all S>=0.
    # If A_ip < 0, we need S^2*(sqrt(3)/2) >= -S*A_ip  =>  S*(sqrt(3)/2) >= -A_ip  =>  S >= -2*A_ip/sqrt(3).
    # So S_min = max over all (i,p) with A_ip < 0 of (-2*A_ip/sqrt(3)).
    # Also S >= sqrt(11).
    c = np.array([ox, oy])
    a = np.deg2rad(a_deg)
    u = np.stack([np.cos(a + np.deg2rad(60.0 * np.arange(6))),
                  np.sin(a + np.deg2rad(60.0 * np.arange(6)))], axis=1)
    S_min = np.sqrt(11.0)
    for (cx, cy, ang) in hexes:
        Vin = _hex_vertices(cx, cy, 1.0, ang)
        for p in Vin:
            pc = p - c
            for i in range(6):
                ui = u[i]
                uj = u[(i + 1) % 6]
                e_u = uj - ui
                A = e_u[0] * pc[1] - e_u[1] * pc[0]
                if A < 0:
                    S_req = -2.0 * A / np.sqrt(3.0)
                    if S_req > S_min:
                        S_min = S_req
    return float(S_min)

def _score(hexes, ox, oy, a_deg):
    S = _min_outer_side(hexes, ox, oy, a_deg)
    if S < np.sqrt(11.0) - 1e-9:
        S = np.sqrt(11.0)
    if not _validity(hexes, ox, oy, S, a_deg):
        return -1.0, None
    return 1.0 / S, S

def _hex_grid_positions(n_ring=2):
    # Generate axial coords within hex distance n_ring from origin.
    pts = []
    for q in range(-n_ring, n_ring + 1):
        for r in range(-n_ring, n_ring + 1):
            s = -q - r
            if abs(s) <= n_ring:
                # Convert axial to Cartesian (pointy-top, unit side)
                x = np.sqrt(3.0) * (q + r / 2.0)
                y = 1.5 * r
                pts.append((x, y))
    return pts

def _generate_candidate_sets(n_ring=2):
    # Enumerate all 11-subsets of hex-grid positions (distance-2 ring).
    # To keep it tractable, we use a heuristic: start from compact clusters.
    # We'll generate a set of candidate position lists.
    base = _hex_grid_positions(n_ring)
    # We want 11 positions. The hex grid with n_ring=2 has 1+6+12=19 points.
    # We'll pick all 11-subsets that are "connected" or compact. For simplicity,
    # we'll use a few hand-crafted compact arrangements plus random samples.
    candidates = []
    # Hand-crafted: center + 6 around + 4 more
    # Axial coords: center (0,0), ring1: (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)
    # ring2: (2,0),(1,1),(0,2),(-1,2),(-2,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-2),(2,-1)
    # Pick center + ring1 (7) + 4 from ring2 that are adjacent to ring1.
    # Let's just use a fixed compact set: center, 6 ring1, and 4 ring2 at (2,0),(0,2),(-2,1),(0,-2)
    fixed_axial = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(0,2),(-2,1),(0,-2)]
    pts = []
    for (q, r) in fixed_axial:
        x = np.sqrt(3.0) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    candidates.append(pts)

    # Another arrangement: 2 rows
    # Row 0: 6 hexes, Row 1: 5 hexes offset
    # Axial: row r=0: q=-2..3? Let's do a 3-4-3 or similar.
    # Actually, let's use a different compact shape: a "flower" of 7 + 4 in a line.
    fixed2 = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(3,0),(-2,0),(-3,0)]
    pts2 = []
    for (q, r) in fixed2:
        x = np.sqrt(3.0) * (q + r / 2.0)
        y = 1.5 * r
        pts2.append((x, y))
    candidates.append(pts2)

    # Random samples
    rng = np.random.default_rng(42)
    for _ in range(200):
        idx = rng.choice(len(base), size=11, replace=False)
        pts = [base[i] for i in idx]
        # Check compactness: bounding box
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if (max(xs) - min(xs)) < 8 and (max(ys) - min(ys)) < 8:
            candidates.append(pts)
    return candidates

def _center_hexes(hexes):
    # Shift so centroid is at origin
    cx = np.mean([h[0] for h in hexes])
    cy = np.mean([h[1] for h in hexes])
    return [(x - cx, y - cy, a) for (x, y, a) in hexes]

def hexagon_packing_11():
    best_score = -1.0
    best_result = None

    # Generate candidate position sets
    candidates = _generate_candidate_sets(n_ring=2)

    # For each candidate, try several outer orientations
    orientations = [0.0, 15.0, 30.0, 45.0, 60.0]

    for cand in candidates:
        if time.time() > _DEADLINE - 60:
            break
        # Center the hexes
        hexes = _center_hexes([(x, y, 0.0) for (x, y) in cand])
        # Ensure separation (hex grid positions are separated by sqrt(3) > 2, so unit hexes are separated)
        if not _all_pairs_separated(hexes):
            continue
        for a_deg in orientations:
            if time.time() > _DEADLINE - 30:
                break
            # Center outer at origin (since hexes are centered)
            ox, oy = 0.0, 0.0
            S = _min_outer_side(hexes, ox, oy, a_deg)
            if S < np.sqrt(11.0):
                S = np.sqrt(11.0)
            if _validity(hexes, ox, oy, S, a_deg):
                sc = 1.0 / S
                if sc > best_score:
                    best_score = sc
                    best_result = (hexes, ox, oy, a_deg, S)

    # Nelder-Mead polish on the best result
    if best_result is not None:
        hexes0, ox0, oy0, a0, S0 = best_result
        # Flatten parameters: 11*3 + 3 (outer x,y,a) = 36 params
        # We'll optimize positions and outer center/orientation, with S computed.
        def _objective(params):
            hexes = [(params[i*3], params[i*3+1], params[i*3+2]) for i in range(11)]
            ox = params[33]
            oy = params[34]
            a = params[35]
            # Check separation
            if not _all_pairs_separated(hexes):
                return 1e6
            S = _min_outer_side(hexes, ox, oy, a)
            if S < np.sqrt(11.0):
                S = np.sqrt(11.0)
            if not _inside_outer(hexes, ox, oy, S, a):
                return 1e6
            return 1.0 / S

        x0 = np.array([h[0] for h in hexes0] + [h[1] for h in hexes0] + [h[2] for h in hexes0] + [ox0, oy0, a0])
        # Actually, let's reshape properly
        x0 = np.zeros(36)
        for i in range(11):
            x0[i*3] = hexes0[i][0]
            x0[i*3+1] = hexes0[i][1]
            x0[i*3+2] = hexes0[i][2]
        x0[33] = ox0
        x0[34] = oy0
        x0[35] = a0

        try:
            res = minimize(_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if time.time() < _DEADLINE - 10:
                hexes_opt = [(res.x[i*3], res.x[i*3+1], res.x[i*3+2]) for i in range(11)]
                ox_opt = res.x[33]
                oy_opt = res.x[34]
                a_opt = res.x[35]
                if _all_pairs_separated(hexes_opt):
                    S_opt = _min_outer_side(hexes_opt, ox_opt, oy_opt, a_opt)
                    if S_opt < np.sqrt(11.0):
                        S_opt = np.sqrt(11.0)
                    if _validity(hexes_opt, ox_opt, oy_opt, S_opt, a_opt):
                        sc_opt = 1.0 / S_opt
                        if sc_opt > best_score:
                            best_score = sc_opt
                            best_result = (hexes_opt, ox_opt, oy_opt, a_opt, S_opt)
        except Exception:
            pass

    # Return best result
    if best_result is None:
        # Fallback: return a valid (though not optimal) result
        hexes = _center_hexes([(x, y, 0.0) for (x, y) in _hex_grid_positions(2)[:11]])
        ox, oy, a = 0.0, 0.0, 0.0
        S = _min_outer_side(hexes, ox, oy, a)
        if S < np.sqrt(11.0):
            S = np.sqrt(11.0)
        best_result = (hexes, ox, oy, a, S)

    hexes, ox, oy, a, S = best_result
    inner = np.array([[h[0], h[1], h[2]] for h in hexes])
    outer = np.array([ox, oy, a])
    return inner, outer, float(S)

# EVOLVE-BLOCK-END
