import numpy as np
import math

N = 11

def hex_verts(c, angle_deg, s=1.0):
    a = math.radians(angle_deg)
    return np.array([[c[0] + s*math.cos(a + k*math.pi/3.0), c[1] + s*math.sin(a + k*math.pi/3.0)] for k in range(6)])

def min_dist_two_hex(c1, a1, c2, a2):
    v1 = hex_verts(c1, a1)
    v2 = hex_verts(c2, a2)
    # Containment check
    def pt_in_hex(pt, hv):
        for i in range(6):
            e = hv[(i+1)%6] - hv[i]
            n = np.array([-e[1], e[0]])
            cen = hv.mean(axis=0)
            if np.dot(n, cen - hv[i]) < 0: n = -n
            if np.dot(n, pt - hv[i]) < -1e-9: return False
        return True
    for pt in v1:
        if pt_in_hex(pt, v2): return 0.0
    for pt in v2:
        if pt_in_hex(pt, v1): return 0.0
    # Separating axis
    md = float('inf')
    axes = []
    for p in (v1, v2):
        for i in range(6):
            e = p[(i+1)%6] - p[i]
            n = np.array([-e[1], e[0]])
            nr = np.linalg.norm(n)
            if nr > 1e-12: axes.append(n/nr)
    for ax in axes:
        p1 = v1 @ ax; p2 = v2 @ ax
        if p1.max() < p2.min(): md = min(md, p2.min() - p1.max())
        elif p2.max() < p1.min(): md = min(md, p1.min() - p2.max())
    return 0.0 if md == float('inf') else md

def enclosing_hex_side(pts):
    pts = np.array(pts)
    c = pts.mean(axis=0)
    best_s, best_a, best_c = float('inf'), 0.0, (0.0, 0.0)
    for pd in range(0, 60, 2):
        phi = math.radians(pd)
        r = 0.0
        for j in range(3):
            t = phi + j*math.pi/3.0
            n = np.array([math.cos(t), math.sin(t)])
            r = max(r, np.abs((pts - c) @ n).max())
        s = 2.0*r/math.sqrt(3.0)
        if s < best_s:
            best_s, best_a, best_c = s, pd, (c[0], c[1])
    return best_s, best_a, best_c

def construct():
    d = math.sqrt(3)
    pos = [[x, 0.0] for x in [-d, 0, d]]
    pos += [[x, 1.5*d] for x in [-1.5*d, -0.5*d, 0.5*d, 1.5*d]]
    pos += [[x, 3.0*d] for x in [-d, 0, d]]
    pos.append([0.0, 4.5*d])
    ang = [0.0]*11
    assert len(pos) == 11
    
    bs, ba, bc = enclosing_hex_side(pos)
    bp, ba_ = [p[:] for p in pos], ang[:]
    
    for _ in range(100):
        imp = False
        for i in range(11):
            for dx, dy in [(-0.1,0),(0.1,0),(0,-0.1),(0,0.1)]:
                np_ = [p[:] for p in bp]; np_[i] = [np_[i][0]+dx, np_[i][1]+dy]
                if all(min_dist_two_hex(np_[i], ba_[i], np_[j], ba_[j]) >= -1e-9 for j in range(11) if j!=i):
                    s,a,c = enclosing_hex_side(np_)
                    if s < bs-1e-7: bp,bs,ba,bc = np_,s,a,c; imp=True
            for da in [-5,5,-10,10]:
                na = ba_[:]; na[i] = ba_[i]+da
                if all(min_dist_two_hex(bp[i], na[i], bp[j], na[j]) >= -1e-9 for j in range(11) if j!=i):
                    s,a,c = enclosing_hex_side(bp)
                    if s < bs-1e-7: ba_,bs,ba,bc = na,s,a,c; imp=True
        if not imp: break
    
    return [[bp[i][0], bp[i][1], ba_[i]] for i in range(11)], bc, bs, ba
