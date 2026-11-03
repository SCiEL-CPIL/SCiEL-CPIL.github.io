"""Pack 11 unit hexagons in a minimal regular hexagon via honeycomb clustering."""
import math
import time
import numpy as np

N = 11
_DEADLINE = 120.0

def _hex_vertices(cx, cy, side, rot_deg):
    r = side
    ang = math.radians(rot_deg)
    pts = []
    for k in range(6):
        a = ang + math.radians(60 * k)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.array(pts)

def _min_outer(inner, center):
    """Return (side, rot_deg) of the minimal regular hexagon (center `center`)
    enclosing all unit hexagons in `inner` (each given as (x,y,angle_deg))."""
    cx, cy = center
    verts = []
    for (ix, iy, iang) in inner:
        for k in range(6):
            a = math.radians(iang) + math.radians(60 * k)
            verts.append((ix + math.cos(a), iy + math.sin(a)))
    verts = np.array(verts)
    best = (1e18, 0.0)
    # coarse scan
    for tdeg in np.linspace(0, 60, 1200):
        t = math.radians(tdeg)
        rho = 0.0
        for k in range(6):
            phi = t + math.radians(60 * k)
            nx = math.cos(phi); ny = math.sin(phi)
            proj = verts[:, 0] * nx + verts[:, 1] * ny
            m = proj.max() - (nx * cx + ny * cy)
            if m > rho:
                rho = m
        S = rho * 2.0 / math.sqrt(3.0)
        if S < best[0]:
            best = (S, float(tdeg))
    # refine around best
    b, t0 = best
    for tdeg in np.linspace(t0 - 2, t0 + 2, 400):
        t = math.radians(tdeg % 60)
        rho = 0.0
        for k in range(6):
            phi = t + math.radians(60 * k)
            nx = math.cos(phi); ny = math.sin(phi)
            proj = verts[:, 0] * nx + verts[:, 1] * ny
            m = proj.max() - (nx * cx + ny * cy)
            if m > rho:
                rho = m
        S = rho * 2.0 / math.sqrt(3.0)
        if S < b:
            b, t0 = S, float(tdeg % 60)
    return b, t0

def _construct_honeycomb():
    inner = [(0.0, 0.0, 0.0)]
    for k in range(6):
        a = math.radians(60 * k)
        inner.append((2.0 * math.cos(a), 2.0 * math.sin(a), 0.0))
    # 4 edge-mid ring2 positions (distance 2*sqrt(3)), chosen symmetric-ish
    for k in [0, 2, 4, 5]:
        a = math.radians(60 * k + 30)
        inner.append((2.0 * math.sqrt(3.0) * math.cos(a), 2.0 * math.sqrt(3.0) * math.sin(a), 0.0))
    return inner

def construct():
    t0 = time.time()
    candidates = []
    # honeycomb
    hb = _construct_honeycomb()
    candidates.append(hb)
    # honeycomb rotated 30 deg
    candidates.append([(x, y, (a + 30.0) % 360.0) for (x, y, a) in hb])
    # tight linear row
    candidates.append([[2.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)])

    best = None
    for cand in candidates:
        if time.time() - t0 > _DEADLINE - 10.0:
            break
        side, ang = _min_outer(cand, (0.0, 0.0))
        if best is None or side < best[0]:
            best = (side, ang, cand)
    if best is None:
        side, ang = _min_outer(hb, (0.0, 0.0))
        best = (side, ang, hb)
    side, ang, cand = best
    return cand, (0.0, 0.0), float(side), float(ang)
