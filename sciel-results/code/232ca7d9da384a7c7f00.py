# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_GLOBAL_START_TIME = None
_DEADLINE = 325.0  # 325s wall clock limit (evaluator kills at 330s)

def _get_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return _DEADLINE - (time.time() - _GLOBAL_START_TIME)

def _morse_energy(x, rho):
    """Compute Morse energy for a set of positions (vectorized)."""
    n = x.shape[0]
    if n < 2:
        return 0.0
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    er = rho * (1.0 - dist)
    er = np.clip(er, -500, 500)
    exp_er = np.exp(er)
    energy = np.sum(exp_er * (exp_er - 2.0)) / 2.0
    return energy

def _min_distance(x):
    """Compute minimum distance between any pair of atoms (vectorized)."""
    n = x.shape[0]
    if n < 2:
        return np.inf
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    return np.min(dist)

def _is_valid(x):
    """Check if the configuration is valid (atoms at least 0.5 apart)."""
    return _min_distance(x) >= 0.5

def _rotation_matrix(angle, axis):
    """Create a rotation matrix for angle (radians) around unit axis."""
    axis = np.asarray(axis, dtype=float)
    nrm = np.linalg.norm(axis)
    if nrm < 1e-12:
        return np.eye(3)
    axis = axis / nrm
    c = math.cos(angle)
    s = math.sin(angle)
    x, y, z = axis
    return np.array([
        [c + x*x*(1-c),   x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s, c + y*y*(1-c),   y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)]
    ])

def _generate_icospiral(n, seed=42):
    """
    Icosahedral spiral: 1 center + 12 icosahedron vertices + 20 face-centers
    + Fibonacci-sphere shell. Compact and highly symmetric.
    """
    rng = np.random.default_rng(seed)
    pts = [np.array([0.0, 0.0, 0.0])]

    # Icosahedron 12 vertices
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    verts = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                # 12 distinct icosahedron vertices (cyclic permutations of (0, +-1, +-phi))
                pass
    ico = [
        ( 0.0,  1.0,  phi), ( 0.0,  1.0, -phi), ( 0.0, -1.0,  phi), ( 0.0, -1.0, -phi),
        ( 1.0,  phi,  0.0), ( 1.0, -phi,  0.0), (-1.0,  phi,  0.0), (-1.0, -phi,  0.0),
        ( phi,  0.0,  1.0), ( phi,  0.0, -1.0), (-phi,  0.0,  1.0), (-phi,  0.0, -1.0),
    ]
    norm = math.sqrt(1.0 + phi*phi)
    for v in ico:
        pts.append(np.array(v) / norm)

    # 20 face-centers (unit normals)
    faces = [
        (0, 1, 4), (0, 4, 2), (0, 2, 6), (0, 6, 8), (0, 8, 1),
        (1, 8, 3), (1, 3, 5), (1, 5, 4), (2, 4, 5), (2, 5, 7),
        (2, 7, 6), (3, 8, 9), (3, 9, 10), (3, 10, 5), (4, 5, 10),
        (4, 10, 9), (6, 7, 11), (6, 11, 9), (6, 9, 8), (7, 5, 10),
    ]
    for f in faces:
        v = (np.array(ico[f[0]]) + np.array(ico[f[1]]) + np.array(ico[f[2]])) / 3.0
        nv = np.linalg.norm(v)
        if nv > 1e-9:
            pts.append(v / nv)

    # Fibonacci sphere for remaining
    remaining = n - len(pts)
    if remaining > 0:
        golden = math.pi * (3.0 - math.sqrt(5.0))
        for k in range(remaining):
            y = 1.0 - 2.0 * (k + 0.5) / remaining
            r = math.sqrt(max(0.0, 1.0 - y*y))
            th = golden * k
            pts.append(np.array([r * math.cos(th), y, r * math.sin(th)]))

    pts = pts[:n]
    # Small deterministic jitter to break exact symmetry (helps escape saddle points)
    jit = rng.uniform(-0.02, 0.02, (len(pts), 3))
    return np.array(pts) + jit

def _generate_fibonacci_shell(n, seed=42):
    """Pure Fibonacci-sphere shell (no center) - good for larger n."""
    rng = np.random.default_rng(seed)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    pts = []
    for k in range(n):
        y = 1.0 - 2.0 * (k + 0.5) / n
        r = math.sqrt(max(0.0, 1.0 - y*y))
        th = golden * k
        pts.append(np.array([r * math.cos(th), y, r * math.sin(th)]))
    # Scale up a bit so center is empty and shell is at radius ~1.2
    arr = np.array(pts) * 1.2
    jit = rng.uniform(-0.03, 0.03, (n, 3))
    return arr + jit

def _generate_fcc_chunk(n, seed=42):
    """FCC lattice chunk, sorted by radius, with small jitter."""
    rng = np.random.default_rng(seed)
    basis = [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]
    radius = 1.6 * (n ** (1.0/3.0)) + 1.0
    mc = int(radius) + 2
    pts = []
    for i in range(-mc, mc+1):
        for j in range(-mc, mc+1):
            for k in range(-mc, mc+1):
                for dx, dy, dz in basis:
                    p = np.array([i+dx, j+dy, k+dz])
                    if np.linalg.norm(p) <= radius:
                        pts.append(p)
    pts.sort(key=lambda p: np.linalg.norm(p))
    if len(pts) >= n:
        arr = np.array(pts[:n])
    else:
        while len(pts) < n:
            pts.append(rng.uniform(-radius, radius, 3))
        arr = np.array(pts)
    jit = rng.uniform(-0.04, 0.04, (n, 3))
    return arr + jit

def _generate_bcc_chunk(n, seed=42):
    """BCC lattice chunk, sorted by radius, with small jitter."""
    rng = np.random.default_rng(seed)
    basis = [(0,0,0), (0.5,0.5,0.5)]
    radius = 1.6 * (n ** (1.0/3.0)) + 1.0
    mc = int(radius) + 2
    pts = []
    for i in range(-mc, mc+1):
        for j in range(-mc, mc+1):
            for k in range(-mc, mc+1):
                for dx, dy, dz in basis:
                    p = np.array([i+dx, j+dy, k+dz])
                    if np.linalg.norm(p) <= radius:
                        pts.append(p)
    pts.sort(key=lambda p: np.linalg.norm(p))
    if len(pts) >= n:
        arr = np.array(pts[:n])
    else:
        while len(pts) < n:
            pts.append(rng.uniform(-radius, radius, 3))
        arr = np.array(pts)
    jit = rng.uniform(-0.04, 0.04, (n, 3))
    return arr + jit

def _generate_hcp_chunk(n, seed=42):
    """HCP-like chunk: two interpenetrating hexagonal layers."""
    rng = np.random.default_rng(seed)
    pts = []
    a = 1.0  # in-plane spacing
    c = 1.633  # layer spacing (ideal HCP)
    n_layers = max(2, int(2 * (n / 12.0) ** 0.5))
    for layer in range(n_layers):
        z = layer * c
        # hex grid in layer
        for ix in range(-6, 7):
            for iy in range(-6, 7):
                x = a * (ix + 0.5 * (iy & 1))
                y = a * math.sqrt(3.0) / 2.0 * iy
                if layer % 2 == 1:
                    x += 0.5 * a  # shift B layer
                r = math.sqrt(x*x + y*y)
                if r <= 2.5 * (n ** (1.0/3.0)):
                    pts.append(np.array([x, y, z]))
    pts.sort(key=lambda p: np.linalg.norm(p))
    if len(pts) >= n:
        arr = np.array(pts[:n])
    else:
        while len(pts) < n:
            pts.append(rng.uniform(-3, 3, 3))
        arr = np.array(pts)
    jit = rng.uniform(-0.03, 0.03, (n, 3))
    return arr + jit

def _relax_and_validate(positions, potential, rho):
    """Relax positions and check validity."""
    try:
        energy, relaxed = relax(positions, potential, rho)
        if _min_distance(relaxed) >= 0.5:
            return energy, relaxed
        else:
            return float('inf'), None
    except Exception:
        return float('inf'), None

def _polish(positions, rho, max_iter=300):
    """
    High-precision L-BFGS-B polish with exact gradient.
    """
    x_flat = positions.flatten()

    def objective(x):
        pos = x.reshape(-1, 3)
        if _min_distance(pos) < 0.5:
            return 1e10
        return _morse_energy(pos, rho)

    def gradient(x):
        pos = x.reshape(-1, 3)
        if _min_distance(pos) < 0.5:
            return np.zeros_like(x)
        return morse_energy_gradient(pos, rho).flatten()

    try:
        res = minimize(objective, x_flat, jac=gradient, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10,
                                'maxls': 40, 'maxcor': 20})
        if res.fun < 1e10:
            pos_res = res.x.reshape(-1, 3)
            if _min_distance(pos_res) >= 0.5:
                return float(res.fun), pos_res
    except Exception:
        pass

    return _morse_energy(positions, rho), positions

def _perturb_rotate(positions, angle, rng):
    """
    Structured perturbation: rotate a random subset of atoms around a random axis
    through the centroid. Preserves pairwise distances within the subset, so the
    relaxed configuration is less likely to collapse into an invalid state.
    """
    n = positions.shape[0]
    new_pos = positions.copy()
    # Pick a random axis through centroid
    centroid = positions.mean(axis=0)
    axis = rng.standard_normal(3)
    R = _rotation_matrix(angle, axis)
    # Rotate a random subset (50-80% of atoms) to create a "twist"
    frac = 0.5 + 0.3 * rng.random()
    k = max(1, int(frac * n))
    idx = rng.choice(n, size=k, replace=False)
    new_pos[idx] = (positions[idx] - centroid) @ R.T + centroid
    return new_pos

def _basin_hop_rotation(start_positions, potential, rho, max_iterations=80,
                        angle_range=(0.1, 0.6), seed=0):
    """
    Basin hopping with rotation-based perturbations.
    Returns (best_energy, best_positions).
    """
    rng = np.random.default_rng(seed)
    current_energy, current_positions = _relax_and_validate(start_positions, potential, rho)

    if current_positions is None:
        # Fallback: small random displacement
        perturbed = start_positions + rng.standard_normal(start_positions.shape) * 0.1
        current_energy, current_positions = _relax_and_validate(perturbed, potential, rho)
        if current_positions is None:
            return float('inf'), None

    best_energy = current_energy
    best_positions = current_positions.copy()
    temp = 3.0
    a_lo, a_hi = angle_range

    for it in range(max_iterations):
        if _get_remaining_time() < 5.0:
            break

        # Random rotation angle, shrinking over time
        lo = a_lo * (0.5 ** (it / max(1, max_iterations)))
        hi = a_hi * (0.5 ** (it / max(1, max_iterations)))
        angle = rng.uniform(lo, hi)
        new_positions = _perturb_rotate(current_positions, angle, rng)

        new_energy, new_relaxed = _relax_and_validate(new_positions, potential, rho)

        if new_relaxed is None:
            # Try a smaller rotation
            angle2 = angle * 0.3
            new_positions = _perturb_rotate(current_positions, angle2, rng)
            new_energy, new_relaxed = _relax_and_validate(new_positions, potential, rho)

        if new_relaxed is None:
            continue

        delta_e = new_energy - current_energy
        if delta_e < 0 or (temp > 1e-6 and rng.random() < math.exp(-delta_e / temp)):
            current_energy = new_energy
            current_positions = new_relaxed
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_positions.copy()

        temp *= 0.95

    return best_energy, best_positions

def construct(instance):
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()

    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    potential = instance["potential"]

    if rho is None:
        rho = 6

    best_energy = float('inf')
    best_positions = None

    # Structured initial guesses (deterministic, compact, symmetric)
    initial_guesses = []
    generators = [
        lambda: _generate_icospiral(n, seed=42),
        lambda: _generate_icospiral(n, seed=7),
        lambda: _generate_fibonacci_shell(n, seed=42),
        lambda: _generate_fcc_chunk(n, seed=42),
        lambda: _generate_bcc_chunk(n, seed=42),
        lambda: _generate_hcp_chunk(n, seed=42),
    ]
    for gen in generators:
        try:
            g = gen()
            if g.shape[0] == n:
                initial_guesses.append(g)
        except Exception:
            pass

    # Relax all initial guesses
    relaxed_results = []
    for guess in initial_guesses:
        if _get_remaining_time() < 10.0:
            break
        energy, positions = _relax_and_validate(guess, potential, rho)
        if positions is not None and energy < float('inf'):
            relaxed_results.append((energy, positions))

    relaxed_results.sort(key=lambda x: x[0])

    if relaxed_results:
        best_energy = relaxed_results[0][0]
        best_positions = relaxed_results[0][1].copy()

    # Rotation-based basin hopping from top starts
    num_bh = min(len(relaxed_results), 4)
    # Angle range tuned by rho: low rho -> long-range -> larger rotations OK
    if rho <= 3:
        angle_range = (0.15, 0.8)
    elif rho <= 6:
        angle_range = (0.10, 0.5)
    else:
        angle_range = (0.05, 0.3)

    if n <= 40:
        max_iter = 100
    elif n <= 60:
        max_iter = 70
    else:
        max_iter = 45

    for idx in range(num_bh):
        if _get_remaining_time() < 15.0:
            break
        start_pos = relaxed_results[idx][1]
        bh_seed = 100 + idx * 17
        bh_energy, bh_positions = _basin_hop_rotation(
            start_pos, potential, rho,
            max_iterations=max_iter,
            angle_range=angle_range,
            seed=bh_seed,
        )
        if bh_positions is not None and bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_positions.copy()

        # Second pass with different seed and wider angles
        if _get_remaining_time() > 20.0 and idx < 2:
            bh_energy2, bh_positions2 = _basin_hop_rotation(
                start_pos, potential, rho,
                max_iterations=max_iter // 2,
                angle_range=(angle_range[0] * 1.5, angle_range[1] * 1.5),
                seed=bh_seed + 7,
            )
            if bh_positions2 is not None and bh_energy2 < best_energy:
                best_energy = bh_energy2
                best_positions = bh_positions2.copy()

    # Final high-precision polish
    if best_positions is not None and _get_remaining_time() > 5.0:
        polished_energy, polished_pos = _polish(best_positions, rho, max_iter=300)
        if polished_pos is not None and polished_energy < best_energy:
            best_energy = polished_energy
            best_positions = polished_pos

    # Fallback
    if best_positions is None:
        best_positions = _generate_icospiral(n, seed=0)
        if not _is_valid(best_positions):
            best_positions = best_positions * 1.5
            if not _is_valid(best_positions):
                best_positions = _generate_fibonacci_shell(n, seed=99)

    return best_positions
# EVOLVE-BLOCK-END
