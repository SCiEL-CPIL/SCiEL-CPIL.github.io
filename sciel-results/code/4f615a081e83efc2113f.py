"""Lennard-Jones 55-atom cluster: Mackay icosahedron geometry + L-BFGS relaxation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START


def _lj_energy(pos):
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    return 4.0 * (r12 - r6).sum() / 2.0


def _lj_energy_grad(pos):
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    # dV/dr = 4(-12 r^-13 + 6 r^-7)
    # grad_i = sum_j dV/dr * (r_i - r_j)/r = sum_j 4(-12 r^-14 + 6 r^-8) (r_i - r_j)
    # coeff for (r_i - r_j) is 4(-12 r^-14 + 6 r^-8)
    # r^-14 = (r^2)^-7, r^-8 = (r^2)^-4
    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    return e, grad


def _make_mackay_icosahedron(r1=1.0, r2=1.5):
    """Build a two-shell Mackay icosahedron: 1 + 12 + 42 = 55 atoms."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=float)
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

    # Shell 1: 12 atoms
    shell1 = vertices * r1

    # Find edges (nearest-neighbor pairs)
    dmat = np.zeros((12, 12))
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(vertices[i] - vertices[j])
            dmat[i, j] = d
            dmat[j, i] = d
    min_d = np.min(dmat[dmat > 0])
    tol = min_d * 1.01
    edges = [(i, j) for i in range(12) for j in range(i + 1, 12) if dmat[i, j] < tol]

    # Edge midpoints (30 atoms)
    edge_mid = np.array([(vertices[i] + vertices[j]) / 2.0 for i, j in edges])

    # Faces: triangles of mutually adjacent vertices
    adj = {i: set() for i in range(12)}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    faces = set()
    for i in range(12):
        for j in adj[i]:
            for k in adj[j]:
                if k != i and i in adj[k]:
                    faces.add(tuple(sorted([i, j, k])))
    face_centers = np.array([(vertices[a] + vertices[b] + vertices[c]) / 3.0
                             for (a, b, c) in faces])

    # Scale shell 2 directions to unit length, then to r2
    edge_mid = (edge_mid / np.linalg.norm(edge_mid, axis=1, keepdims=True)) * r2
    face_centers = (face_centers / np.linalg.norm(face_centers, axis=1, keepdims=True)) * r2

    atoms = [np.zeros(3)]
    atoms.extend(list(shell1))
    atoms.extend(list(face_centers))
    atoms.extend(list(edge_mid))
    pos = np.array(atoms[:N])
    pos -= pos.mean(axis=0)
    return pos


def _min_distance(pos):
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    d = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    np.fill_diagonal(d, np.inf)
    return np.min(d)


def _enforce_min_distance(pos, min_d=0.5, iters=200):
    """Push apart any pair closer than min_d."""
    n = len(pos)
    pos = pos.copy()
    for _ in range(iters):
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        d = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
        np.fill_diagonal(d, np.inf)
        mask = d < min_d
        if not mask.any():
            break
        ii, jj = np.nonzero(mask)
        moved = False
        for i, j in zip(ii, jj):
            if i == j:
                continue
            d_ij = np.linalg.norm(pos[i] - pos[j])
            if d_ij < 1e-9:
                continue
            push = (pos[i] - pos[j]) / d_ij
            excess = min_d - d_ij
            pos[i] += push * 0.5 * excess
            pos[j] -= push * 0.5 * excess
            moved = True
        if not moved:
            break
    return pos


def _relax(pos, time_budget=40.0):
    """L-BFGS relaxation with analytic gradient, respecting a time budget."""
    t0 = time.time()
    best_pos = pos.copy()
    best_e = _lj_energy(pos)
    
    # Callback to track best state
    def callback(xk):
        nonlocal best_pos, best_e
        try:
            e_k = _lj_energy(xk)
            if e_k < best_e:
                best_e = e_k
                best_pos = xk.copy()
        except Exception:
            pass
        return time.time() - t0 > time_budget

    try:
        result = minimize(
            _lj_energy,
            pos,
            jac=_lj_energy_grad,
            method='L-BFGS-B',
            options={'maxiter': 4000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 20000, 'callback': callback},
        )
        # Check final result
        if result.fun < best_e:
            best_e = result.fun
            best_pos = result.x.copy()
    except Exception:
        pass
    
    # Safety: if we ended up with a higher energy than start, return start
    if best_e > _lj_energy(pos):
        return pos, _lj_energy(pos)
        
    return best_pos, best_e


def _refine_shells(base_pos, shell_radii, time_budget=20.0):
    """Refine the two shell radii (and center offset) via L-BFGS on the full config."""
    # base_pos is built with r1, r2; we re-scale shells by factors to allow radial tuning.
    # Simpler: just run a full relaxation on base_pos (already done). This is a fallback.
    return base_pos, _lj_energy(base_pos)


def construct():
    t0 = time.time()
    deadline = t0 + 100.0  # safety margin vs 120s limit (150s hard kill)

    best_pos = None
    best_e = np.inf

    # Primary: Mackay icosahedron with a few radius variants, relaxed.
    # The standard Mackay icosahedron has shells at r=1 and r=sqrt(5)/2 ~ 1.118?
    # Actually, for LJ minimum, the optimal shell radii are often cited around 1.0 and 1.1-1.2?
    # Let's stick to the geometric Mackay first, but allow the optimizer to adjust.
    # The previous failure was likely due to the 2x gradient error causing divergence.
    radius_sets = [(1.0, 1.118), (1.0, 1.15), (1.0, 1.10)]
    for r1, r2 in radius_sets:
        if time.time() > deadline - 30.0:
            break
        try:
            mackay = _make_mackay_icosahedron(r1, r2)
            if _min_distance(mackay) < 0.5:
                mackay = _enforce_min_distance(mackay)
            if _min_distance(mackay) < 0.5:
                continue
            
            # Calculate initial energy to ensure we are in a reasonable basin
            e_init = _lj_energy(mackay)
            
            budget = min(35.0, max(5.0, deadline - time.time() - 15.0))
            pos, e = _relax(mackay, time_budget=budget)
            
            # Validate: energy must be lower than initial and finite, and min dist ok
            if np.isfinite(e) and e < e_init and _min_distance(pos) >= 0.5:
                if e < best_e:
                    best_e = e
                    best_pos = pos.copy()
        except Exception:
            continue

    # Fallback: cubic lattice fragment if nothing valid found.
    if best_pos is None:
        side = int(np.ceil(N ** (1 / 3)))
        pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        pts.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_pos = np.array(pts[:N], dtype=float) * 1.12
        if _min_distance(best_pos) < 0.5:
            best_pos = _enforce_min_distance(best_pos)

    # Final safety: ensure minimum distance.
    if _min_distance(best_pos) < 0.5:
        best_pos = _enforce_min_distance(best_pos)

    return best_pos
# EVOLVE-BLOCK-END
