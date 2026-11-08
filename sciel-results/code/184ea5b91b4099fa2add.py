import math
import numpy as np

N = 11
SQ3 = math.sqrt(3)

def _hexdist(i, j):
    return (abs(i) + abs(j) + abs(i + j)) // 2

def _lattice(i, j, d):
    x = d * i + (d / 2.0) * j
    y = (d * SQ3 / 2.0) * j
    return (x, y)

def _unit_hex_verts(cx, cy, rot_deg):
    rot = math.radians(rot_deg)
    cs, sn = math.cos(rot), math.sin(rot)
    out = []
    for k in range(6):
        a = math.radians(60.0 * k)
        vx, vy = math.cos(a), math.sin(a)
        rx = vx * cs - vy * sn
        ry = vx * sn + vy * cs
        out.append((cx + rx, cy + ry))
    return out

def _hexnorm_flat(x, y):
    return max(abs(x), abs(x + y / SQ3), abs(x - y / SQ3))

def _required_S(centers, inner_rot, big_rot):
    # returns minimal S over big hexagon center (free), for fixed big_rot
    # S(c) = max over all inner vertices v of hexnorm_flat(R(-big_rot)(v - c))
    # We minimize over c. Use grid + refinement.
    # gather all vertices
    verts = []
    for (cx, cy) in centers:
        for (vx, vy) in _unit_hex_verts(cx, cy, inner_rot):
            verts.append((vx, vy))
    # centroid
    mx = sum(p[0] for p in verts) / len(verts)
    my = sum(p[1] for p in verts) / len(verts)
    br = math.radians(big_rot)
    cbs, sns = math.cos(br), math.sin(br)
    def S_of(bx, by):
        m = 0.0
        for (vx, vy) in verts:
            dx, dy = vx - bx, vy - by
            lx = dx * cbs + dy * sns
            ly = -dx * sns + dy * cbs
            m = max(m, _hexnorm_flat(lx, ly))
        return m
    # coarse grid
    best = (mx, my, S_of(mx, my))
    R = 2.0
    step = 0.1
    bx = mx - R
    while bx <= mx + R + 1e-9:
        by = my - R
        while by <= my + R + 1e-9:
            s = S_of(bx, by)
            if s < best[2]:
                best = (bx, by, s)
            by += step
        bx += step
    # refine around best
    bx0, by0, s0 = best
    for _ in range(3):
        step /= 3.0
        for dx in (-step, 0, step):
            for dy in (-step, 0, step):
                s = S_of(bx0 + dx, by0 + dy)
                if s < s0:
                    bx0, by0, s0 = bx0 + dx, by0 + dy, s
    return s0, (bx0, by0)

def _blob_candidates():
    # return list of sets of (i,j)
    cands = []
    core = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
    ring2 = [(i,j) for i in range(-3,4) for j in range(-3,4) if _hexdist(i,j)==2]
    # A: core + 4 contiguous ring2
    for start in range(12):
        sel = [ring2[(start + k) % 12] for k in range(4)]
        cands.append(set(core) | set(sel))
    # B: core + 4 axis ring2
    cands.append(set(core) | {(2,0),(0,2),(-2,0),(0,-2)})
    # C: a 3x3-ish block
    # rows of a triangular lattice
    # Let's make a blob: (i,j) with i in 0..2, j in -1..2 etc.
    # D: hex radius 2 minus 8 (keep 11) - try a few trims
    ring2_all = ring2
    import itertools
    # too many; skip
    # E: a "stadium" / row arrangement
    # 4-3-4 rows
    rows = []
    # define by explicit (i,j)
    cands.append({(-1,0),(0,0),(1,0),(2,0),  (-1,1),(0,1),(1,1),  (-1,-1),(0,-1),(1,-1),  (2,1)})
    # dedupe
    seen = set()
    uniq = []
    for c in cands:
        key = frozenset(c)
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq

def construct():
    import time
    t0 = time.time()
    d = SQ3 + 0.01  # small gap for strict disjointness
    best = None
    blobs = _blob_candidates()
    inner_rot = 0.0
    for blob in blobs:
        if time.time() - t0 > 110:
            break
        centers = [_lattice(i, j, d) for (i, j) in blob]
        for big_rot in (0.0, 30.0, 15.0, 45.0, 20.0, 25.0):
            if time.time() - t0 > 110:
                break
            S, (bx, by) = _required_S(centers, inner_rot, big_rot)
            # validity: check all vertices inside with margin
            ok = True
            br = math.radians(big_rot)
            cbs, sns = math.cos(br), math.sin(br)
            for (cx, cy) in centers:
                for (vx, vy) in _unit_hex_verts(cx, cy, inner_rot):
                    dx, dy = vx - bx, vy - by
                    lx = dx * cbs + dy * sns
                    ly = -dx * sns + dy * cbs
                    if not (abs(lx) <= S + 1e-6 and abs(lx + ly/SQ3) <= S + 1e-6 and abs(lx - ly/SQ3) <= S + 1e-6):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                if best is None or S < best[0]:
                    best = (S, centers, (bx, by), big_rot)
    if best is None:
        # fallback: seed-like
        inner = [[d * k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    S, centers, (bx, by), big_rot = best
    S += 1e-4  # tiny safety margin
    inner = [[cx, cy, 0.0] for (cx, cy) in centers]
    return inner, (bx, by), S, big_rot
