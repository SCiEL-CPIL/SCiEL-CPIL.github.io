"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def hex_corners(cx, cy, side, rot_deg):
        """Corners of a regular hexagon (flat-top base orientation) rotated by rot_deg."""
        rot = math.radians(rot_deg)
        c, s = math.cos(rot), math.sin(rot)
        pts = []
        for k in range(6):
            a = math.radians(60 * k)
            x0 = side * math.cos(a)
            y0 = side * math.sin(a)
            pts.append((cx + c * x0 - s * y0, cy + s * x0 + c * y0))
        return np.array(pts)

    def enclosing_side(inner, center, side, angle):
        """True minimal regular hexagon side (center, angle fixed) via support function."""
        # support function: for each of 6 outward normals, max dot of inner corners
        rot = math.radians(angle)
        # outward normals of the regular hexagon (pointing to each vertex direction)
        best = 0.0
        for k in range(6):
            a = math.radians(60 * k)
            nx, ny = math.cos(a), math.sin(a)
            # support in this direction = max over all inner corners of (x*nx + y*ny) - center dot n
            mx = -1e18
            for (ix, iy, ia) in inner:
                corners = hex_corners(ix, iy, 1.0, ia)
                d = corners @ np.array([nx, ny])
                mx = max(mx, d.max())
            # subtract center projection
            mx -= center[0] * nx + center[1] * ny
            best = max(best, mx)
        # side length of regular hexagon = support / cos(30deg) = support / (sqrt(3)/2)
        return best / (math.sqrt(3) / 2.0)

    def smooth_side(inner, center, side, angle, beta=50.0):
        """Smooth log-sum-exp surrogate of the support function (differentiable)."""
        rot = math.radians(angle)
        vals = []
        for k in range(6):
            a = math.radians(60 * k)
            nx, ny = math.cos(a), math.sin(a)
            dvals = []
            for (ix, iy, ia) in inner:
                corners = hex_corners(ix, iy, 1.0, ia)
                d = corners @ np.array([nx, ny])
                dvals.append(d)
            darr = np.concatenate(dvals)
            darr = darr - (center[0] * nx + center[1] * ny)
            vals.append(darr)
        vals = np.array(vals)  # shape (6, M)
        # log-sum-exp per direction, then max over 6 directions (use soft max)
        m = vals.max(axis=1, keepdims=True)
        lse = m + np.log(np.sum(np.exp(vals - m), axis=1, keepdims=True)) / 1.0
        lse = lse[:, 0]  # per direction
        # soft max over the 6 directions
        lmax = lse.max()
        softmax = lmax + np.log(np.sum(np.exp(lse - lmax)))
        return softmax / (math.sqrt(3) / 2.0)

    # --- Build the triangular 12-hexagon cluster ---
    # Center at origin. Ring of 6 at distance sqrt(3) (hex centers of adjacent unit hexes).
    # Outer ring: 5 hexes attached to 5 of the 6 ring hexes (leaving one gap to keep it compact).
    inner = [[0.0, 0.0, 0.0]]
    d = math.sqrt(3.0)
    for k in range(6):
        a = math.radians(60 * k)
        inner.append([d * math.cos(a), d * math.sin(a), 0.0])
    # 5 outer hexes: attach to ring hexes at angles 0,60,120,180,240 (skip 300)
    d2 = 2.0 * math.sqrt(3.0)
    for k in range(5):
        a = math.radians(60 * k)
        inner.append([d2 * math.cos(a), d2 * math.sin(a), 0.0])

    inner = np.array(inner)
    center = np.array([0.0, 0.0])
    side = 4.0
    angle = 0.0

    def x_to_vars(x):
        # x layout: [cx, cy, side, angle, x1,y1,a1, ...]
        cx, cy = x[0], x[1]
        s = x[2]
        ang = x[3]
        inner_pts = x[4:].reshape(N, 3)
        return inner_pts, center + np.array([cx, cy]), s, ang

    def obj(x):
        inner_pts, c, s, ang = x_to_vars(x)
        return smooth_side(inner_pts, c, s, ang, beta=50.0)

    # Initial variables
    x0 = np.zeros(4 + 3 * N)
    x0[4:] = inner.flatten()
    x0[2] = side

    # Local refinement with L-BFGS-B
    from scipy.optimize import minimize
    try:
        res = minimize(obj, x0, method='L-BFGS-B',
                       options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8})
        if time.time() < DEADLINE and res.x is not None:
            x = res.x
            inner_pts, c, s, ang = x_to_vars(x)
            # snap angle to nearest multiple of 60 for cleanliness
            ang = round(ang / 60.0) * 60.0
            inner_pts[:, 2] = [round(a / 60.0) * 60.0 for a in inner_pts[:, 2]]
            # recompute true side
            s_true = enclosing_side(inner_pts.tolist(), c.tolist(), s, ang)
            inner_out = [[float(ix), float(iy), float(ia)] for (ix, iy, ia) in inner_pts]
            return inner_out, (float(c[0]), float(c[1])), float(s_true), float(ang)
    except Exception:
        pass

    # Fallback: return the analytic cluster with a safe enclosing side
    s_true = enclosing_side(inner.tolist(), (0.0, 0.0), side, angle)
    inner_out = [[float(ix), float(iy), float(ia)] for (ix, iy, ia) in inner]
    return inner_out, (0.0, 0.0), float(s_true), 0.0
# EVOLVE-BLOCK-END
