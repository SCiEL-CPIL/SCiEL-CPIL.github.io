"""Differentiable rigid-body contact-network packer for 12 unit hexagons in a regular hexagon."""
import time
import math
import numpy as np

N = 12
T_START = time.time()
DEADLINE = 150.0
SAFETY = 12.0

# ---------- geometry helpers ----------
def hex_verts(cx, cy, ang):
    """6 vertices of a unit (circumradius=1) regular hexagon, CCW."""
    a = ang + np.arange(6) * (np.pi / 3.0)
    return np.stack([cx + np.cos(a), cy + np.sin(a)], axis=1)

def _support_min(d, va, vb):
    # min over i,j of (va[i]-vb[j])·d = min_i(va·d) - max_j(vb·d)
    return np.min(va @ d) - np.max(vb @ d)

def hex_separation(va, vb):
    """Minimum (signed) distance between two convex hexagons (positive=separated)."""
    # candidate directions: edge normals of A and B
    def normals(v):
        e = np.roll(v, -1, axis=0) - v
        n = np.stack([-e[:, 1], e[:, 0]], axis=1)
        nn = np.linalg.norm(n, axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        return n / nn
    na = normals(va); nb = normals(vb)
    best = 0.0
    for d in list(na) + list(nb):
        s = _support_min(d, va, vb)
        if s > best:
            best = s
    return best

def hex_sep_grad(va, vb):
    """(separation, grad_va, grad_vb) using the separating direction that gives max support."""
    def normals(v):
        e = np.roll(v, -1, axis=0) - v
        n = np.stack([-e[:, 1], e[:, 0]], axis=1)
        nn = np.linalg.norm(n, axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        return n / nn
    na = normals(va); nb = normals(vb)
    cands = list(na) + list(nb)
    best = -1e9; bd = None
    for d in cands:
        s = _support_min(d, va, vb)
        if s > best:
            best = s; bd = d
    # separation s = min_i(va·d) - max_j(vb·d)
    iA = int(np.argmin(va @ bd)); iB = int(np.argmax(vb @ bd))
    gva = np.zeros_like(va); gva[iA] = bd
    gvb = np.zeros_like(vb); gvb[iB] = -bd
    return best, gva, gvb

def max_vertex_radius(va):
    return float(np.max(np.linalg.norm(va, axis=1)))

# ---------- contact network ----------
def build_network(verts, P, S, outer_ang):
    """Return list of (i, j, d) critical contacts + outer contact info.
    d = signed separation (neg=overlap). Outer separation handled separately."""
    contacts = []
    n = len(verts)
    for i in range(n):
        for j in range(i + 1, n):
            s = hex_separation(verts[i], verts[j])
            # keep only near-critical contacts
            if s < 0.05:
                contacts.append((i, j, s))
    return contacts

def outer_violation_and_grad(verts, P, S, outer_ang):
    """For each hex, max over vertices of (support along 6 outer normals) - S.
    Returns (max_violation, list of (hex_idx, vertex_idx, normal))."""
    # outer hexagon side S, center 0, rotation outer_ang
    on = []
    for k in range(6):
        a = outer_ang + k * (np.pi / 3.0)
        # outward normal of regular hexagon edge k
        on.append(np.array([math.cos(a + np.pi / 3.0), math.sin(a + np.pi / 3.0)]))
    # For a regular hexagon with circumradius S, the support function along an edge
    # normal equals S*cos(pi/6) = S*sqrt(3)/2 (the apothem). Constraint: vertex·n <= apothem.
    apothem = S * math.sqrt(3.0) / 2.0
    maxv = -1e9; info = None
    for i in range(len(verts)):
        for vi in range(6):
            val = float(verts[i][vi] @ on[vi % 6]) - apothem
            # use the normal that actually faces this vertex (max over normals)
            m = max(float(verts[i][vi] @ o) for o in on) - apothem
            if m > maxv:
                maxv = m
                # pick argmax normal
                nn = on[int(np.argmax([float(verts[i][vi] @ o) for o in on]))]
                info = (i, vi, nn)
    return maxv, info

# ---------- objective ----------
def objective(P, S, outer_ang, mu_inter, mu_outer):
    verts = [hex_verts(P[i, 0], P[i, 1], P[i, 2]) for i in range(N)]
    gP = np.zeros_like(P)
    loss = 0.0
    # inter
    for i in range(N):
        for j in range(i + 1, N):
            s = hex_separation(verts[i], verts[j])
            if s < 0:
                loss += mu_inter * s * s
                _, gva, gvb = hex_sep_grad(verts[i], verts[j])
                gP[i, 0] += gva[:, 0].sum(); gP[i, 1] += gva[:, 1].sum()
                gP[j, 0] += gvb[:, 0].sum(); gP[j, 1] += gvb[:, 1].sum()
    # outer
    maxv, info = outer_violation_and_grad(verts, P, S, outer_ang)
    if maxv > 0:
        loss += mu_outer * maxv * maxv
        i, vi, nn = info
        gP[i, 0] += nn[0]; gP[i, 1] += nn[1]
    # S gradient: outer constraint pushes S up when violated
    gS = mu_outer * max(0.0, maxv)
    return loss, gP, gS

def evaluate(P, S, outer_ang):
    verts = [hex_verts(P[i, 0], P[i, 1], P[i, 2]) for i in range(N)]
    worst = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            s = hex_separation(verts[i], verts[j])
            if s < worst:
                worst = s
    maxv, _ = outer_violation_and_grad(verts, P, S, outer_ang)
    # valid iff no overlap and no outer penetration
    valid = (worst >= -1e-6) and (maxv <= 1e-6)
    return S, worst, maxv, valid

# ---------- optimizer: projected penalty gradient on S, free P ----------
def run():
    rng = np.random.default_rng(0)
    # initial guess: ring of 12 hexagons around center at radius ~1.2, varied angles
    R0 = 1.4
    P = np.zeros((N, 3))
    for i in range(N):
        a = i * (2 * math.pi / N)
        P[i, 0] = R0 * math.cos(a)
        P[i, 1] = R0 * math.sin(a)
        P[i, 2] = a  # align
    outer_ang = 0.0
    S = max_vertex_radius(hex_verts(P[0, 0], P[0, 1], P[0, 2])) * 2.2 + 0.5
    # ensure outer big enough initially
    S = max(S, 4.0)

    best_S = S
    best_P = P.copy()
    best_ang = outer_ang

    mu_inter = 20.0
    mu_outer = 20.0
    lr = 0.02
    iters = 4000
    for it in range(iters):
        if time.time() - T_START > DEADLINE - SAFETY:
            break
        loss, gP, gS = objective(P, S, outer_ang, mu_inter, mu_outer)
        # step
        P = P - lr * gP
        S = max(1.5, S - lr * gS)
        # small random jitter to escape plateaus
        if it % 200 == 199:
            P += rng.normal(0, 0.01, P.shape)
            mu_inter *= 1.1
            mu_outer *= 1.1
            lr *= 0.95
        # track best valid
        Sv, worst, maxv, valid = evaluate(P, S, outer_ang)
        if valid and Sv < best_S:
            best_S = Sv
            best_P = P.copy()
            best_ang = outer_ang
    # final polish: try to shrink S greedily until invalid, then back off
    P = best_P.copy()
    S = best_S
    outer_ang = best_ang
    for _ in range(300):
        if time.time() - T_START > DEADLINE - SAFETY:
            break
        S_try = S - 0.002
        _, _, maxv, valid = evaluate(P, S_try, outer_ang)
        if valid:
            S = S_try
        else:
            break
    return P, S, outer_ang

# ---------- entry point ----------
def construct():
    P, S, outer_ang = run()
    inner = [[float(P[i, 0]), float(P[i, 1]), float(math.degrees(P[i, 2]))] for i in range(N)]
    return inner, (0.0, 0.0), float(S), float(math.degrees(outer_ang))
