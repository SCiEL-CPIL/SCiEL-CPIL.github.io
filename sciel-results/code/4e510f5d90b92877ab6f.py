"""Unit regular hexagons inside a regular hexagon: force-directed + annealing."""
import math
import time
import random
import numpy as np

N = 11
EPS = 1e-6
DEADLINE = time.time() + 140.0  # safety margin vs 150s kill

# Precomputed geometry constants
SQ3 = math.sqrt(3.0)

def _hex_vertices(cx, cy, side, angle_deg):
    a = math.radians(angle_deg)
    return np.array([[cx + side * math.cos(a + k * math.pi / 3.0),
                      cy + side * math.sin(a + k * math.pi / 3.0)] for k in range(6)])

def _segment_distance(p, a, b):
    ab = b - a
    t = np.dot(p - a, ab) / max(np.dot(ab, ab), EPS)
    t = min(max(t, 0.0), 1.0)
    return np.linalg.norm(p - (a + t * ab))

def _point_in_convex(poly, p):
    # poly CCW; check all cross products same sign
    n = len(poly)
    sign = None
    for i in range(n):
        a = poly[i]; b = poly[(i + 1) % n]
        c = np.cross(b - a, p - a)
        if c > EPS:
            if sign is None: sign = 1
            elif sign != 1: return False
        elif c < -EPS:
            if sign is None: sign = -1
            elif sign != -1: return False
    return True

def _dist_point_to_poly(p, poly):
    # positive if outside, negative if inside (approx via edge distance + inside test)
    if _point_in_convex(poly, p):
        # distance to nearest edge (negative)
        dmin = min(_segment_distance(p, poly[i], poly[(i+1)%len(poly)]) for i in range(len(poly)))
        return -dmin
    dmin = min(_segment_distance(p, poly[i], poly[(i+1)%len(poly)]) for i in range(len(poly)))
    return dmin

def _hexagon_dist(A, B):
    # A,B are vertex arrays; positive if separated, negative if overlapping
    # min over vertices of A of (dist to poly B) and vice versa
    dA = min(_dist_point_to_poly(v, B) for v in A)
    dB = min(_dist_point_to_poly(v, A) for v in B)
    return max(dA, dB)

def _penalty_and_grad(pos, angles, S, outer_ang):
    # pos: (N,2) array, angles: (N,) array of radians
    outer = _hex_vertices(0.0, 0.0, S, outer_ang)
    verts = [_hex_vertices(pos[i,0], pos[i,1], 1.0, math.degrees(angles[i])) for i in range(N)]
    # gradient arrays
    gpos = np.zeros((N,2))
    gangle = np.zeros(N)
    penalty = 0.0
    # boundary: each inner vertex must be inside outer
    for i in range(N):
        for v in verts[i]:
            d = _dist_point_to_poly(v, outer)
            if d > 0:
                penalty += d*d
                # push vertex toward center
                gpos[i,0] += -2.0 * d * (v[0] - 0.0) / np.linalg.norm(v) if np.linalg.norm(v)>EPS else 0
                gpos[i,1] += -2.0 * d * (v[1] - 0.0) / np.linalg.norm(v) if np.linalg.norm(v)>EPS else 0
    # pairwise overlap
    for i in range(N):
        for j in range(i+1, N):
            d = _hexagon_dist(verts[i], verts[j])
            if d < 0:
                penalty += d*d
                # repulsion: push apart
                dx = pos[i,0]-pos[j,0]; dy = pos[i,1]-pos[j,1]
                r = math.hypot(dx,dy)
                if r < EPS: dx, dy, r = 1.0, 0.0, 1.0
                f = -2.0 * d
                gpos[i,0] += f * dx / r
                gpos[i,1] += f * dy / r
                gpos[j,0] -= f * dx / r
                gpos[j,1] -= f * dy / r
    return penalty, gpos, gangle

def _build_honeycomb():
    # 3-4-3 cluster in honeycomb, flat-top, centers spaced 2*SQ3
    pts = []
    # rows y = 0, +/- SQ3 ... use axial
    # 4 in middle row, 3 above, 3 below
    for k in range(4):
        pts.append((2*SQ3*k - 3*SQ3, 0.0))
    for k in range(3):
        pts.append((2*SQ3*k - 2*SQ3, SQ3))
    for k in range(3):
        pts.append((2*SQ3*k - 2*SQ3, -SQ3))
    return np.array(pts, dtype=float)

def _build_line():
    pts = []
    for k in range(N):
        pts.append((2*SQ3*k, 0.0))
    return np.array(pts, dtype=float)

def _build_random(seed):
    rng = random.Random(seed)
    pts = []
    for _ in range(N):
        r = rng.uniform(0, 6)
        a = rng.uniform(0, 2*math.pi)
        pts.append((r*math.cos(a), r*math.sin(a)))
    return np.array(pts, dtype=float)

def _refine(pos0, angles0, S0, outer_ang, steps=2000, T0=2.0, T1=0.01):
    pos = pos0.copy().astype(float)
    angles = angles0.copy().astype(float)
    S = S0
    best_pen = float('inf')
    best_state = (pos.copy(), angles.copy(), S)
    rng = random.Random(12345)
    n = len(pos)
    for it in range(steps):
        if time.time() > DEADLINE - 5: break
        pen, gpos, gangle = _penalty_and_grad(pos, angles, S, outer_ang)
        if pen < best_pen:
            best_pen = pen
            best_state = (pos.copy(), angles.copy(), S)
        T = T0 * (T1/T0) ** (it/steps)
        # gradient step on positions and angles
        lr = 0.02 * (1.0 + 0.5*math.cos(it*0.01))
        pos -= lr * gpos
        angles -= lr * gangle
        # random perturbations (annealing)
        if rng.random() < 0.3:
            i = rng.randrange(n)
            pos[i,0] += rng.gauss(0, 0.1)
            pos[i,1] += rng.gauss(0, 0.1)
        if rng.random() < 0.2:
            i = rng.randrange(n)
            angles[i] += rng.gauss(0, 0.05)
        # occasionally shrink S to drive tighter packing
        if it % 50 == 0 and S > 3.5:
            S = max(3.5, S - 0.05)
    return best_state, best_pen

def construct():
    t0 = time.time()
    best = None
    best_score = float('inf')

    # candidate initial configurations
    inits = []
    inits.append((_build_honeycomb(), np.zeros(N), 4.2, 0.0))
    inits.append((_build_line(), np.zeros(N), 6.5, 0.0))
    inits.append((_build_random(7), np.array([rng_a for rng_a in np.zeros(N)]), 6.0, 0.0))

    # outer orientation variants
    for outer_ang in [0.0, 30.0, 15.0, 60.0]:
        for (p0, a0, S0, oa) in inits:
            if time.time() > DEADLINE - 10: break
            # adjust initial S estimate from bounding
            # try a few starting S
            for Sstart in [S0, S0*1.1, S0*0.95]:
                if time.time() > DEADLINE - 10: break
                try:
                    (bp, ba, bS), pen = _refine(p0, a0, Sstart, outer_ang,
                                                 steps=1500, T0=1.5, T1=0.005)
                except Exception:
                    continue
                # verify validity and compute true minimal S by scaling
                # check if valid
                outer = _hex_vertices(0.0, 0.0, bS, outer_ang)
                verts = [_hex_vertices(bp[i,0], bp[i,1], 1.0, math.degrees(ba[i])) for i in range(N)]
                valid = True
                for i in range(N):
                    for v in verts[i]:
                        if not _point_in_convex(outer, v):
                            valid = False; break
                    if not valid: break
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            if _hexagon_dist(verts[i], verts[j]) < -EPS:
                                valid = False; break
                        if not valid: break
                if valid and bS < best_score:
                    best_score = bS
                    best = (bp, ba, bS, outer_ang)
    if best is None:
        # fallback: return honeycomb with generous S
        p0 = _build_honeycomb()
        return [[float(p0[i,0]), float(p0[i,1]), 0.0] for i in range(N)], (0.0,0.0), 4.2, 0.0
    bp, ba, bS, oa = best
    inner = [[float(bp[i,0]), float(bp[i,1]), float(math.degrees(ba[i]))] for i in range(N)]
    return inner, (0.0, 0.0), float(bS), float(oa)

# EVOLVE-BLOCK-END
