"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side.
Uses simulated annealing with exact SAT collision checking and multi-start strategy.
"""
import time
import math
import random
import numpy as np
from scipy.optimize import minimize as _scipy_minimize

N = 12
DEADLINE = 145.0  # safety margin before the 150s kill

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _hex_vertices_local(theta):
    """Vertices of unit regular hexagon (flat-top) at origin, rotated by theta."""
    return np.array([[math.cos(theta + k * math.pi / 3.0),
                      math.sin(theta + k * math.pi / 3.0)] for k in range(6)])

def _norms(theta):
    """6 outward edge-normal directions of a regular hexagon oriented by theta."""
    return np.array([[math.cos(theta + math.pi / 6.0 + k * math.pi / 3.0),
                      math.sin(theta + math.pi / 6.0 + k * math.pi / 3.0)] for k in range(6)])

def _bounding_side(pts, theta, center):
    """Side length of smallest regular hexagon (orientation theta, center) containing pts."""
    nrm = _norms(theta)
    P = pts @ nrm.T
    c = center @ nrm.T
    req = (P - c).max(axis=0)
    return 2.0 * float(np.max(req)) / math.sqrt(3.0)

def _hexagon_overlap(A, B):
    """SAT test: True if two convex hexagons (arrays of (x,y)) overlap."""
    for poly in (A, B):
        for i in range(6):
            p1 = poly[i]
            p2 = poly[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            projA = A @ normal
            projB = B @ normal
            if projA.max() < projB.min() - 1e-9 or projB.max() < projA.min() - 1e-9:
                return False
    return True

def _all_points(inner):
    """Get all vertex points for the inner hexagons."""
    verts = [_hex_vertices_local(math.radians(r[2])) for r in inner]
    centers = np.array([[r[0], r[1]] for r in inner])
    pts = []
    for k in range(N):
        pts.append(verts[k] + centers[k])
    return np.vstack(pts), centers

def _is_valid(inner, outer, side, outer_theta, eps=1e-6):
    """Check if all inner hexagons are disjoint and contained in the outer hexagon."""
    # Check containment
    pts, _ = _all_points(inner)
    nrm = _norms(outer_theta)
    P = pts @ nrm.T
    c = np.array(outer) @ nrm.T
    for k in range(6):
        if (P[:, k] - c[k]).max() > side * math.sqrt(3.0) / 2.0 + eps:
            return False
    
    # Check pairwise disjointness
    verts = [_hex_vertices_local(math.radians(r[2])) + np.array([r[0], r[1]]) for r in inner]
    for i in range(N):
        for j in range(i + 1, N):
            if _hexagon_overlap(verts[i], verts[j]):
                return False
    return True

# ---------------------------------------------------------------------------
# Seed configurations
# ---------------------------------------------------------------------------
def _seed(pattern, rng=None):
    if rng is None:
        rng = random.Random()
    d = 2.0 * math.sqrt(3.0) + 1e-6
    a1 = np.array([d, 0.0])
    a2 = np.array([d / 2.0, d * math.sqrt(3.0) / 2.0])
    pts = []
    if pattern == "rect":
        for i in range(-1, 3):
            for j in range(-1, 2):
                pts.append(i * a1 + j * a2)
    elif pattern == "hex":
        coords = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
                  (2, 0), (-2, 0), (0, 2), (0, -2), (2, -2)]
        for (i, j) in coords:
            pts.append(i * a1 + j * a2)
    elif pattern == "rect_tall":
        for i in range(-1, 2):
            for j in range(-2, 2):
                pts.append(i * a1 + j * a2)
    elif pattern == "ring":
        for k in range(12):
            ang = 2.0 * math.pi * k / 12.0
            r = 2.5 * math.sqrt(3.0)
            pts.append(np.array([r * math.cos(ang), r * math.sin(ang)]))
    elif pattern == "strip":
        for i in range(6):
            pts.append(np.array([i * d, 0.0]))
            pts.append(np.array([i * d, d * math.sqrt(3.0)]))
    else:
        # "rotated_grid" - triangular lattice rotated 30 degrees
        a1r = np.array([d * math.cos(math.pi/6), d * math.sin(math.pi/6)])
        a2r = np.array([-d * math.sin(math.pi/6), d * math.cos(math.pi/6)])
        for i in range(-1, 3):
            for j in range(-1, 2):
                pts.append(i * a1r + j * a2r)
    pts = np.array(pts)
    c = pts.mean(axis=0)
    pts = pts - c
    return [[float(pts[k][0]), float(pts[k][1]), 0.0] for k in range(N)]

# ---------------------------------------------------------------------------
# Simulated Annealing
# ---------------------------------------------------------------------------
def _anneal(inner, outer, side, outer_theta, steps=5000, t0=1.0, t1=0.01, rng=None):
    """Simulated annealing to improve packing."""
    if rng is None:
        rng = random.Random()
    
    current = [list(r) for r in inner]
    current_side = side
    best = [list(r) for r in inner]
    best_side = side
    
    for step in range(steps):
        if time.time() > DEADLINE:
            break
        
        # Temperature schedule
        t = t0 * (t1 / t0) ** (step / steps)
        
        # Pick a random hexagon to perturb
        i = rng.randrange(N)
        dx = rng.gauss(0, 0.1)
        dy = rng.gauss(0, 0.1)
        da = rng.gauss(0, 5.0) * math.pi / 180.0
        
        # Try the perturbation
        new_inner = [list(r) for r in current]
        new_inner[i][0] += dx
        new_inner[i][1] += dy
        new_inner[i][2] += math.degrees(da)
        
        # Compute new side length
        pts, _ = _all_points(new_inner)
        new_side = _bounding_side(pts, outer_theta, np.array(outer))
        
        # Check validity (disjointness)
        valid = True
        verts = [_hex_vertices_local(math.radians(r[2])) + np.array([r[0], r[1]]) for r in new_inner]
        for a in range(N):
            for b in range(a + 1, N):
                if _hexagon_overlap(verts[a], verts[b]):
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        delta = new_side - current_side
        if delta < 0 or rng.random() < math.exp(-delta / max(t, 1e-10)):
            current = new_inner
            current_side = new_side
            if new_side < best_side:
                best = [list(r) for r in new_inner]
                best_side = new_side
    
    return best, best_side

# ---------------------------------------------------------------------------
# Local refinement using Nelder-Mead on centers
# ---------------------------------------------------------------------------
def _refine_inner(inner, outer, side, outer_theta, maxiter=200):
    """Nelder-Mead on the N centers, minimizing the bounding side (outer fixed)."""
    verts = [_hex_vertices_local(math.radians(r[2])) for r in inner]
    centers = np.array([[r[0], r[1]] for r in inner])

    def obj(x):
        c = x.reshape(N, 2)
        pts = []
        for k in range(N):
            pts.append(verts[k] + c[k])
        pts = np.vstack(pts)
        # Check if still valid
        verts_new = [verts[k] + c[k] for k in range(N)]
        for a in range(N):
            for b in range(a + 1, N):
                if _hexagon_overlap(verts_new[a], verts_new[b]):
                    return 1e6
        return _bounding_side(pts, outer_theta, np.array(outer))

    x0 = centers.reshape(-1)
    res = _scipy_minimize(obj, x0, method="Nelder-Mead",
                          options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-9})
    c = res.x.reshape(N, 2)
    new_inner = []
    for k in range(N):
        new_inner.append([float(c[k][0]), float(c[k][1]), inner[k][2]])
    return new_inner

def _refine_outer(inner, outer, side, outer_theta):
    """Sweep outer angle, return best (side, center, angle)."""
    pts, centers = _all_points(inner)
    best = (side, outer, outer_theta)
    for ang in np.linspace(0.0, math.pi / 3.0, 121):
        b = _bounding_side(pts, ang, np.array(outer))
        if b < best[0] - 1e-9:
            best = (b, tuple(float(x) for x in outer), float(ang))
    return best

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def construct():
    start = time.time()
    rng = random.Random(42)

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    def _consider(inner, outer, side, outer_theta):
        nonlocal best_side, best_inner, best_center, best_angle
        if side < best_side:
            best_side = side
            best_inner = [list(r) for r in inner]
            best_center = tuple(float(x) for x in outer)
            best_angle = float(outer_theta)

    patterns = ["rect", "hex", "rect_tall", "ring", "strip", "rotated_grid"]
    
    for pat in patterns:
        if time.time() > DEADLINE - 5:
            break

        inner = _seed(pat, rng)

        # Initial outer estimate via coarse angle sweep at origin
        pts, _ = _all_points(inner)
        s0, t0 = 10.0, 0.0
        for ang in np.linspace(0.0, math.pi / 3.0, 61):
            b = _bounding_side(pts, ang, np.zeros(2))
            if b < s0:
                s0, t0 = b, ang
        outer = (0.0, 0.0)

        # Phase 1: Simulated annealing
        inner, s1 = _anneal(inner, outer, s0, t0, steps=3000, t0=0.5, t1=0.05, rng=rng)
        
        # Phase 2: Refine outer angle
        s1, outer1, t1 = _refine_outer(inner, outer, s1, t0)
        
        # Phase 3: Nelder-Mead refinement on centers
        inner = _refine_inner(inner, outer1, s1, t1, maxiter=100)
        s2, outer2, t2 = _refine_outer(inner, outer1, s1, t1)
        
        # Phase 4: More annealing with lower temperature
        inner, s3 = _anneal(inner, outer2, s2, t2, steps=2000, t0=0.2, t1=0.01, rng=rng)
        
        # Phase 5: Final refinement
        inner = _refine_inner(inner, outer2, s3, t2, maxiter=50)
        s4, outer3, t3 = _refine_outer(inner, outer2, s3, t2)

        _consider(inner, outer3, s4, t3)

    # ---- final fine sweep of the outer angle on the best inner set ----
    if best_inner is not None:
        pts, _ = _all_points(best_inner)
        fine = np.linspace(0.0, math.pi / 3.0, 301)
        sF, cF, tF = best_side, best_center, best_angle
        for ang in fine:
            b = _bounding_side(pts, ang, np.array(best_center))
            if b < sF - 1e-12:
                sF, tF = b, ang
        _consider(best_inner, best_center, sF, tF)

    # ---- fallback ----
    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, math.degrees(best_angle)
