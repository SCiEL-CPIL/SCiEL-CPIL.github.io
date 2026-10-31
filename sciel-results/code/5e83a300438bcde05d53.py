import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.optimize import minimize as _minimize

N = 38
DEADLINE = 142.0
START_TIME = time.time()

def lj_energy(positions):
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r2 = float(np.dot(rij, rij))
            if r2 < 1e-12:
                return 1e20
            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2 * inv_r2 * inv_r2
            inv_r12 = inv_r6 * inv_r6
            energy += 4.0 * (inv_r12 - inv_r6)
    return energy

def lj_energy_vectorized(positions):
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    return 4.0 * np.sum(inv_r12 - inv_r6) / 2.0

def check_validity(positions, min_dist=0.5):
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            if np.dot(rij, rij) < min_dist * min_dist:
                return False
    return True

def make_fcc_truncated_octahedron(n_atoms=38):
    scale = 1.0 / math.sqrt(2.0)
    max_coord = 8
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = i * scale, j * scale, k * scale
                    r2 = x*x + y*y + z*z
                    points.append((x, y, z, r2))
    points.sort(key=lambda p: (p[3], abs(p[0]), abs(p[1]), abs(p[2])))
    selected = [p[:3] for p in points[:n_atoms]]
    coords = np.array(selected, dtype=float)
    coords *= 2.0 ** (1.0 / 6.0)
    return coords

def make_fcc_with_noise(n_atoms=38, noise_scale=0.05, seed=42):
    base = make_fcc_truncated_octahedron(n_atoms)
    rng = np.random.default_rng(seed)
    return base + rng.normal(0, noise_scale, base.shape)

def _lj_energy_and_grad(pos_flat):
    pos = pos_flat.reshape(N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    inv_r14 = inv_r12 * inv_r2
    inv_r8 = inv_r6 * inv_r2
    coeff = -24.0 * inv_r14 + 12.0 * inv_r8
    grad_matrix = coeff[:, :, np.newaxis] * diff
    grad_flat = np.sum(grad_matrix, axis=1).flatten()
    return energy, grad_flat

def _make_initial_clusters():
    clusters = []
    # Primary: FCC truncated octahedron
    try:
        fcc = make_fcc_truncated_octahedron(N)
        if check_validity(fcc):
            clusters.append(fcc)
    except Exception:
        pass

    # Small-noise FCC variants
    for s in (1, 2, 3, 4, 5, 6):
        try:
            c = make_fcc_with_noise(N, noise_scale=0.04, seed=s)
            if check_validity(c):
                clusters.append(c)
        except Exception:
            pass

    # Larger-noise FCC variants
    for s in (7, 8, 9, 10):
        try:
            c = make_fcc_with_noise(N, noise_scale=0.10, seed=s)
            if check_validity(c):
                clusters.append(c)
        except Exception:
            pass

    # BCC-like fragment: cubic lattice scaled
    try:
        side = int(math.ceil(N ** (1.0/3.0)))
        pts = []
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    pts.append((i - side//2, j - side//2, k - side//2))
        pts.sort(key=lambda p: p[0]*p[0] + p[1]*p[1] + p[2]*p[2])
        pts = pts[:N]
        arr = np.array(pts, dtype=float)
        arr *= 1.28 * 2.0 ** (1.0/6.0)
        if check_validity(arr):
            clusters.append(arr)
    except Exception:
        pass

    # Icosahedral-inspired: two concentric shells with icosahedral symmetry
    try:
        ic = _make_icosahedral(N)
        if check_validity(ic):
            clusters.append(ic)
    except Exception:
        pass

    return clusters

def _make_icosahedral(n_atoms=38):
    # Build an icosahedron + inner core
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    verts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                verts.append((sx, sy*phi, sz*phi))
    # icosahedron has 12 vertices
    verts = np.array(verts, dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    # scale outer shell
    outer_r = 2.8
    inner_r = 1.3
    # place 12 on outer, 12 on inner rotated, 1 at center, plus extras
    outer = verts * outer_r
    inner = verts * inner_r
    # rotate inner by 30 deg around z
    ang = math.pi / 6.0
    R = np.array([[math.cos(ang), -math.sin(ang), 0],
                  [math.sin(ang), math.cos(ang), 0],
                  [0, 0, 1]])
    inner = inner @ R.T
    center = np.array([[0.0, 0.0, 0.0]])
    pts = np.vstack([center, inner, outer])
    if len(pts) >= n_atoms:
        pts = pts[:n_atoms]
    else:
        # add more random
        rng = np.random.default_rng(0)
        extra = rng.normal(0, 0.5, (n_atoms - len(pts), 3))
        pts = np.vstack([pts, extra])
    # scale overall
    pts *= 2.0 ** (1.0 / 6.0) / 1.4
    return pts

def relax(positions, maxiter=400, gtol=1e-9):
    x0 = positions.flatten()
    try:
        result = _minimize(_lj_energy_and_grad, x0, method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 60, 'ftol': 1e-12})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def polish(positions, maxiter=1000, gtol=1e-12):
    x0 = positions.flatten()
    try:
        result = _minimize(_lj_energy_and_grad, x0, method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-15})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def basin_hopping(positions, n_iter=15, step_size=0.4, maxiter=120, temperature=0.05):
    x0 = positions.flatten()
    current_energy, _ = _lj_energy_and_grad(x0)
    best_energy = current_energy
    best_x = x0.copy()
    rng = np.random.default_rng(99)

    for i in range(n_iter):
        if time.time() - START_TIME > DEADLINE - 12:
            break
        noise = rng.normal(0, step_size, x0.shape)
        perturbed = x0 + noise
        try:
            result = _minimize(_lj_energy_and_grad, perturbed, method='L-BFGS-B',
                               jac=True,
                               options={'maxiter': maxiter, 'gtol': 1e-7, 'maxls': 60})
            new_x = result.x
            new_energy, _ = _lj_energy_and_grad(new_x)
        except Exception:
            continue

        if new_energy < current_energy:
            x0 = new_x
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_x = new_x.copy()
        else:
            delta = new_energy - current_energy
            if delta < 0:
                accept = 1.0
            else:
                accept = math.exp(-delta / temperature)
            if rng.uniform() < accept:
                x0 = new_x
                current_energy = new_energy

    return best_x.reshape(N, 3), best_energy

def construct():
    global START_TIME
    START_TIME = time.time()

    best_energy = float('inf')
    best_positions = None

    candidates = _make_initial_clusters()

    for cand in candidates:
        if time.time() - START_TIME > DEADLINE - 20:
            break
        try:
            relaxed = relax(cand, maxiter=300, gtol=1e-8)
            if check_validity(relaxed):
                energy = lj_energy_vectorized(relaxed)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed
        except Exception:
            pass

    if best_positions is not None and time.time() - START_TIME < DEADLINE - 25:
        try:
            bh_res, bh_e = basin_hopping(best_positions, n_iter=12, step_size=0.35, maxiter=100, temperature=0.04)
            if check_validity(bh_res) and bh_e < best_energy:
                best_energy = bh_e
                best_positions = bh_res
        except Exception:
            pass

    if best_positions is not None and time.time() - START_TIME < DEADLINE - 12:
        try:
            polished = polish(best_positions, maxiter=800, gtol=1e-12)
            if check_validity(polished):
                energy = lj_energy_vectorized(polished)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = polished
        except Exception:
            pass

    if best_positions is None:
        try:
            best_positions = make_fcc_truncated_octahedron(N)
        except Exception:
            side = int(math.ceil(N ** (1/3)))
            pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
            pts.sort(key=lambda p: (p[0]-(side-1)/2)**2 + (p[1]-(side-1)/2)**2 + (p[2]-(side-1)/2)**2)
            best_positions = np.array(pts[:N], dtype=float) * 1.3

    return best_positions
