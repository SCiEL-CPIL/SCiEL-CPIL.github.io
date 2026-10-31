import numpy as np
import time
import math
from scipy.optimize import minimize

N = 38
DEADLINE = 142.0
START_TIME = time.time()


def lj_energy(positions):
    """Compute total Lennard-Jones energy: 4 * sum(r^-12 - r^-6) over all pairs."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r2 = np.dot(rij, rij)
            if r2 < 1e-12:
                return 1e20
            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2 * inv_r2 * inv_r2
            inv_r12 = inv_r6 * inv_r6
            energy += 4.0 * (inv_r12 - inv_r6)
    return energy


def lj_energy_vectorized(positions):
    """Vectorized LJ energy computation."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy


def lj_gradient(positions):
    """Compute gradient of LJ energy with respect to positions."""
    n = len(positions)
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r2 = np.dot(rij, rij)
            if r2 < 1e-12:
                continue
            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2 * inv_r2 * inv_r2
            inv_r12 = inv_r6 * inv_r6
            coeff = 4.0 * (12.0 * inv_r12 - 6.0 * inv_r6)
            grad[i] -= coeff * rij
            grad[j] += coeff * rij
    return grad


def check_validity(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    return min_r2 >= min_dist * min_dist


def make_fcc_cluster(n_atoms=38):
    """Generate an FCC truncated octahedron cluster for LJ38."""
    a = 1.0 / math.sqrt(2.0)
    max_coord = 4
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = i * a, j * a, k * a
                    r2 = x * x + y * y + z * z
                    points.append((x, y, z, r2))
    points.sort(key=lambda p: p[3])
    selected = [p[:3] for p in points[:n_atoms]]
    return np.array(selected, dtype=float)


def make_bcc_cluster(n_atoms=38):
    """Generate a BCC cluster."""
    a = 2.0 / math.sqrt(3.0)
    max_coord = 4
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                points.append((i * a, j * a, k * a))
                points.append(((i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * a))
    points.sort(key=lambda p: p[0] ** 2 + p[1] ** 2 + p[2] ** 2)
    return np.array(points[:n_atoms], dtype=float)


def make_icosahedral_cluster(n_atoms=38):
    """Generate a nested icosahedral cluster."""
    phi = (1 + math.sqrt(5)) / 2.0
    r_ico = 1.0 / math.sqrt(1 + phi * phi)
    ico_verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]) * r_ico
    points = [np.array([0.0, 0.0, 0.0])]
    points.extend([v * 1.12 for v in ico_verts])
    rng = np.random.default_rng(42)
    for _ in range(n_atoms - len(points)):
        u = rng.uniform(0, 1)
        v = rng.uniform(0, 1)
        theta = 2 * math.pi * u
        z = 2 * v - 1
        r = math.sqrt(1 - z * z)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        points.append(np.array([x, y, z]) * 1.6)
    return np.array(points, dtype=float)


def relax(positions, maxiter=500, gtol=1e-10):
    """Relax using L-BFGS-B with gradient."""
    def energy_func(pos_flat):
        pos = pos_flat.reshape(N, 3)
        return lj_energy(pos)

    def grad_func(pos_flat):
        pos = pos_flat.reshape(N, 3)
        return lj_gradient(pos).flatten()

    x0 = positions.flatten()
    try:
        result = minimize(energy_func, x0, jac=grad_func, method='L-BFGS-B',
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 200})
        return result.x.reshape(N, 3)
    except Exception:
        return positions


def polish(positions):
    """High-precision polish pass."""
    if time.time() - START_TIME > DEADLINE - 12:
        return positions
    try:
        polished = relax(positions, maxiter=2000, gtol=1e-13)
        if check_validity(polished):
            e1 = lj_energy_vectorized(positions)
            e2 = lj_energy_vectorized(polished)
            if e2 < e1:
                return polished
    except Exception:
        pass
    return positions


def optimize_scale(positions):
    """Optimize the overall scale factor."""
    def energy_with_scale(scale):
        scaled = positions * scale[0]
        return lj_energy_vectorized(scaled)
    try:
        result = minimize(energy_with_scale, [1.0], method='Nelder-Mead',
                         options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 300})
        if result.success:
            optimal_scale = result.x[0]
            if 0.3 < optimal_scale < 3.0:
                scaled = positions * optimal_scale
                if check_validity(scaled):
                    return scaled
    except Exception:
        pass
    return positions


def basin_hopping(positions, n_steps=8, sigma=0.3):
    """Simple basin hopping: random perturbation + relaxation."""
    if time.time() - START_TIME > DEADLINE - 10:
        return positions
    best = positions
    best_e = lj_energy_vectorized(positions)
    rng = np.random.default_rng(int(time.time() * 1000) % 100000)
    for _ in range(n_steps):
        if time.time() - START_TIME > DEADLINE - 8:
            break
        perturbed = positions + rng.normal(0, sigma, positions.shape)
        relaxed = relax(perturbed, maxiter=300, gtol=1e-10)
        if check_validity(relaxed):
            e = lj_energy_vectorized(relaxed)
            if e < best_e:
                best_e = e
                best = relaxed
    return best


def construct():
    global START_TIME
    START_TIME = time.time()

    best_energy = float('inf')
    best_positions = None

    def try_candidate(cand):
        nonlocal best_energy, best_positions
        if time.time() - START_TIME > DEADLINE - 25:
            return
        try:
            relaxed = relax(cand, maxiter=500, gtol=1e-10)
            if not check_validity(relaxed):
                return
            # Scale optimize
            scaled = optimize_scale(relaxed)
            e_s = lj_energy_vectorized(scaled) if check_validity(scaled) else float('inf')
            e_r = lj_energy_vectorized(relaxed)
            if e_s < best_energy:
                best_energy = e_s
                best_positions = scaled
            elif e_r < best_energy:
                best_energy = e_r
                best_positions = relaxed
        except Exception:
            pass

    # 1. FCC base
    try:
        fcc = make_fcc_cluster(N)
        if check_validity(fcc):
            try_candidate(fcc)
    except Exception:
        pass

    # 2. Perturbed FCC (multiple noise levels)
    for noise in [0.03, 0.08, 0.15]:
        if time.time() - START_TIME > DEADLINE - 25:
            break
        try:
            rng = np.random.default_rng(100 + int(noise * 100))
            fcc_p = make_fcc_cluster(N) + rng.normal(0, noise, (N, 3))
            if check_validity(fcc_p):
                try_candidate(fcc_p)
        except Exception:
            pass

    # 3. BCC
    try:
        bcc = make_bcc_cluster(N)
        if check_validity(bcc):
            try_candidate(bcc)
    except Exception:
        pass

    # 4. Icosahedral
    try:
        ico = make_icosahedral_cluster(N)
        if check_validity(ico):
            try_candidate(ico)
    except Exception:
        pass

    # 5. Basin hopping from best so far
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 30:
        try:
            bh_result = basin_hopping(best_positions, n_steps=6, sigma=0.25)
            e_bh = lj_energy_vectorized(bh_result)
            if e_bh < best_energy:
                best_energy = e_bh
                best_positions = bh_result
        except Exception:
            pass

    # 6. Second round basin hopping with larger kicks if time permits
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 35:
        try:
            bh2 = basin_hopping(best_positions, n_steps=4, sigma=0.4)
            e2 = lj_energy_vectorized(bh2)
            if e2 < best_energy:
                best_energy = e2
                best_positions = bh2
        except Exception:
            pass

    # 7. Final polish
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 10:
        try:
            polished = polish(best_positions)
            if check_validity(polished):
                e_pol = lj_energy_vectorized(polished)
                if e_pol < best_energy:
                    best_energy = e_pol
                    best_positions = polished
        except Exception:
            pass

    if best_positions is None:
        try:
            best_positions = make_fcc_cluster(N)
        except Exception:
            best_positions = np.zeros((N, 3))
            for i in range(N):
                best_positions[i] = [i * 0.6, (i % 3) * 0.6, (i // 3) % 3 * 0.6]

    return best_positions
