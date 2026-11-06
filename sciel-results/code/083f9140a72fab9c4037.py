"""Dzugutov cluster global minima: multi-polyhedral seeds + basin hopping with adaptive kicks."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import dzugutov_energy_gradient

_START = time.time()
TIME_LIMIT = 285.0  # safety margin under the 330 s kill

# Dzugutov parameters
_A, _B = 5.82, 1.28
_a, _b = 1.87, 1.94
_c, _d = 1.10, 0.27
_m = 16
_MIN_SEP = 0.5


def _remaining():
    return TIME_LIMIT - (time.time() - _START)


def _energy(pos):
    """Vectorised Dzugutov energy (flat array, n*3)."""
    p = np.asarray(pos, dtype=float)
    n = p.size // 3
    p = p.reshape(n, 3)
    diff = p[:, None, :] - p[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    r_ij = r[iu]
    
    energy = 0.0
    mask1 = r_ij < _a
    if np.any(mask1):
        r1 = r_ij[mask1]
        energy += np.sum(_A * (r1 ** (-_m) - _B) * np.exp(_c / (r1 - _a)))
    mask2 = (r_ij >= _a) & (r_ij < _b)
    if np.any(mask2):
        r2 = r_ij[mask2]
        energy += np.sum(_B * np.exp(_d / (r2 - _b)))
    return energy


def _grad_fn(pos):
    """Gradient via the provided helper, reshaped to flat."""
    p = np.asarray(pos, dtype=float)
    n = p.size // 3
    p = p.reshape(n, 3)
    _, grad = dzugutov_energy_gradient(p)
    return grad.flatten()


def _min_sep_ok(pos, min_sep=_MIN_SEP):
    """Check minimum separation constraint."""
    n = pos.shape[0]
    if n < 2:
        return True
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    return np.min(r[iu]) >= min_sep


def _enforce_min_sep(pos, min_sep=_MIN_SEP, max_iter=100):
    """Enforce minimum separation by pushing overlapping atoms apart."""
    pos = pos.copy()
    n = pos.shape[0]
    for _ in range(max_iter):
        if _min_sep_ok(pos, min_sep):
            break
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_sep:
                    if d < 1e-10:
                        direction = np.array([1.0, 0.0, 0.0])
                    else:
                        direction = (pos[i] - pos[j]) / d
                    push = (min_sep - d) / 2 + 0.005
                    pos[i] += direction * push
                    pos[j] -= direction * push
    return pos


def _get_icosahedral_points():
    """Get icosahedron vertices, edge midpoints, and face centers."""
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico /= np.linalg.norm(ico[0])
    
    # 30 edge midpoints
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(ico[i] - ico[j])
            if 0.9 < d < 1.1:
                edges.append((i, j))
    
    edge_midpoints = []
    for i, j in edges:
        mid = (ico[i] + ico[j]) / 2
        mid /= np.linalg.norm(mid)
        edge_midpoints.append(mid)
    edge_midpoints = np.array(edge_midpoints[:30])
    
    # 20 face centers
    face_centers = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                dij = np.linalg.norm(ico[i] - ico[j])
                djk = np.linalg.norm(ico[j] - ico[k])
                dik = np.linalg.norm(ico[i] - ico[k])
                if (0.9 < dij < 1.1 and 0.9 < djk < 1.1 and 0.9 < dik < 1.1):
                    fc = (ico[i] + ico[j] + ico[k]) / 3
                    fc /= np.linalg.norm(fc)
                    face_centers.append(fc)
    
    unique_fc = []
    for fc in face_centers:
        if not any(np.linalg.norm(fc - ufc) < 0.01 for ufc in unique_fc):
            unique_fc.append(fc)
    face_centers = np.array(unique_fc[:20])
    
    return ico, edge_midpoints, face_centers


def _get_dodecahedral_points():
    """Get dodecahedron vertices (20 vertices)."""
    phi = (1 + np.sqrt(5)) / 2
    psi = 1 / phi
    
    verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, psi, phi], [0, psi, -phi], [0, -psi, phi], [0, -psi, -phi],
        [psi, phi, 0], [psi, -phi, 0], [-psi, phi, 0], [-psi, -phi, 0],
        [phi, 0, psi], [phi, 0, -psi], [-phi, 0, psi], [-phi, 0, -psi]
    ], dtype=float)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    return verts


def _get_cuboctahedral_points():
    """Get cuboctahedron vertices (12 vertices)."""
    # Cuboctahedron has 12 vertices at permutations of (0, 1, 1)
    verts = []
    for perm in [(0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
                 (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
                 (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)]:
        v = np.array(perm, dtype=float)
        v /= np.linalg.norm(v)
        verts.append(v)
    
    return np.array(verts)


def _generate_macky_icosahedron(n, rng, scale=1.0):
    """Generate Mackay icosahedral cluster with n atoms."""
    ico, edge_pts, face_pts = _get_icosahedral_points()
    
    points = []
    remaining = n
    
    # Central atom
    if remaining > 0:
        points.append(np.array([0.0, 0.0, 0.0]))
        remaining -= 1
    
    # First shell: 12 icosahedron vertices
    count1 = min(12, remaining)
    for i in range(count1):
        p = ico[i] * scale * 1.2
        p += rng.normal(0, 0.02, 3)
        points.append(p)
    remaining -= count1
    
    # Second shell: 30 edge midpoints
    if remaining > 0:
        count2 = min(30, remaining)
        for i in range(count2):
            p = edge_pts[i % 30] * scale * 1.9
            p += rng.normal(0, 0.02, 3)
            points.append(p)
        remaining -= count2
    
    # Third shell: 20 face centers
    if remaining > 0:
        count3 = min(20, remaining)
        for i in range(count3):
            p = face_pts[i % 20] * scale * 2.6
            p += rng.normal(0, 0.02, 3)
            points.append(p)
        remaining -= count3
    
    # Fourth shell: more icosahedron vertices at larger radius
    if remaining > 0:
        for i in range(remaining):
            idx = i % 12
            p = ico[idx] * scale * (3.2 + 0.15 * (i // 12))
            p += rng.normal(0, 0.03, 3)
            points.append(p)
    
    points = np.array(points[:n])
    return _enforce_min_sep(points)


def _generate_dodecahedral(n, rng, scale=1.0):
    """Generate dodecahedral cluster with n atoms."""
    dodeca = _get_dodecahedral_points()
    
    points = []
    remaining = n
    
    # Central atom
    if remaining > 0:
        points.append(np.array([0.0, 0.0, 0.0]))
        remaining -= 1
    
    # First shell: 20 dodecahedron vertices
    count1 = min(20, remaining)
    for i in range(count1):
        p = dodeca[i % 20] * scale * 1.3
        p += rng.normal(0, 0.02, 3)
        points.append(p)
    remaining -= count1
    
    # Second shell: icosahedron vertices (dual polyhedron)
    ico, _, _ = _get_icosahedral_points()
    if remaining > 0:
        count2 = min(12, remaining)
        for i in range(count2):
            p = ico[i] * scale * 2.0
            p += rng.normal(0, 0.02, 3)
            points.append(p)
        remaining -= count2
    
    # Third shell: more dodecahedron vertices
    if remaining > 0:
        for i in range(remaining):
            p = dodeca[i % 20] * scale * (2.7 + 0.15 * (i // 20))
            p += rng.normal(0, 0.03, 3)
            points.append(p)
    
    points = np.array(points[:n])
    return _enforce_min_sep(points)


def _generate_cuboctahedral(n, rng, scale=1.0):
    """Generate cuboctahedral cluster with n atoms."""
    cubocta = _get_cuboctahedral_points()
    
    points = []
    remaining = n
    
    # Central atom
    if remaining > 0:
        points.append(np.array([0.0, 0.0, 0.0]))
        remaining -= 1
    
    # First shell: 12 cuboctahedron vertices
    count1 = min(12, remaining)
    for i in range(count1):
        p = cubocta[i % 12] * scale * 1.2
        p += rng.normal(0, 0.02, 3)
        points.append(p)
    remaining -= count1
    
    # Second shell: FCC-like positions
    fcc_dirs = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                if i + j + k <= 2:
                    v = np.array([i-1.5, j-1.5, k-1.5])
                    if np.linalg.norm(v) > 0.1:
                        v /= np.linalg.norm(v)
                        fcc_dirs.append(v)
    
    if remaining > 0:
        count2 = min(len(fcc_dirs), remaining)
        for i in range(count2):
            p = fcc_dirs[i] * scale * 1.8
            p += rng.normal(0, 0.02, 3)
            points.append(p)
        remaining -= count2
    
    # Third shell: more cuboctahedron vertices
    if remaining > 0:
        for i in range(remaining):
            p = cubocta[i % 12] * scale * (2.5 + 0.15 * (i // 12))
            p += rng.normal(0, 0.03, 3)
            points.append(p)
    
    points = np.array(points[:n])
    return _enforce_min_sep(points)


def _generate_seeds(n, rng, n_variants=6):
    """Generate diverse polyhedral seeds."""
    seeds = []
    generators = [
        _generate_macky_icosahedron,
        _generate_dodecahedral,
        _generate_cuboctahedral
    ]
    
    for i in range(n_variants):
        gen = generators[i % len(generators)]
        scale = rng.uniform(0.85, 1.25)
        try:
            seed = gen(n, rng, scale=scale)
            if seed.shape[0] == n:
                seeds.append(seed)
        except Exception:
            continue
    
    # If we don't have enough seeds, generate random compact ones
    while len(seeds) < n_variants and _remaining() > 20:
        # Random compact seed
        points = []
        for i in range(n):
            if i == 0:
                points.append(np.array([0.0, 0.0, 0.0]))
            else:
                r = 0.5 + 0.3 * np.sqrt(i)
                theta = rng.uniform(0, 2*np.pi)
                phi = rng.uniform(-np.pi/2, np.pi/2)
                p = np.array([
                    r * np.cos(phi) * np.cos(theta),
                    r * np.cos(phi) * np.sin(theta),
                    r * np.sin(phi)
                ])
                points.append(p)
        points = np.array(points[:n])
        points = _enforce_min_sep(points)
        seeds.append(points)
    
    return seeds[:n_variants]


def _relax_lbfgsb(pos_flat, maxiter=200):
    """Relax using L-BFGS-B with tight tolerances."""
    try:
        result = minimize(
            _energy,
            pos_flat,
            jac=_grad_fn,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 30, 'disp': False}
        )
        return result
    except Exception:
        return None


def _basin_hopping(pos, rng, n_iterations=10, kick_scale=0.3):
    """Perform basin hopping: kick, relax, accept/reject."""
    best = pos.copy()
    best_e = _energy(pos.flatten())
    
    current_scale = kick_scale
    
    for it in range(n_iterations):
        if _remaining() < 10:
            break
        
        # Adaptive kick scale
        kick = rng.normal(0, current_scale, pos.shape)
        kicked = pos + kick
        kicked = _enforce_min_sep(kicked)
        
        result = _relax_lbfgsb(kicked.flatten(), maxiter=150)
        if result is None:
            continue
        
        new_e = result.fun
        new_pos = result.x.reshape(pos.shape)
        
        if new_e < best_e:
            best_e = new_e
            best = new_pos
            pos = new_pos
            # Reduce kick scale on improvement
            current_scale *= 0.85
        else:
            # Increase kick scale on rejection
            current_scale = min(0.8, current_scale * 1.15)
    
    return best, best_e


def construct(instance):
    """Find low-energy Dzugutov cluster using multi-polyhedral seeds + basin hopping."""
    size = instance["n"]
    rng = np.random.default_rng(size)
    
    best_energy = np.inf
    best_positions = None
    
    # Phase 1: Generate diverse multi-polyhedral seeds
    seeds = _generate_seeds(size, rng, n_variants=6)
    
    # Phase 2: Relax each seed
    for i, seed in enumerate(seeds):
        if _remaining() < 15.0:
            break
        
        result = _relax_lbfgsb(seed.flatten(), maxiter=200)
        if result is not None:
            e = result.fun
            pos = result.x.reshape(size, 3)
            if e < best_energy and _min_sep_ok(pos):
                best_energy = e
                best_positions = pos
    
    # Phase 3: Basin hopping from best position
    if best_positions is not None and _remaining() > 30.0:
        bh_best, bh_energy = _basin_hopping(best_positions, rng, n_iterations=15, kick_scale=0.3)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_best
    
    # Phase 4: Basin hopping from other seeds if time permits
    if _remaining() > 20.0:
        for seed in seeds[:3]:
            if _remaining() < 15.0:
                break
            
            # First relax the seed
            result = _relax_lbfgsb(seed.flatten(), maxiter=100)
            if result is None:
                continue
            
            start_pos = result.x.reshape(size, 3)
            start_e = result.fun
            
            # Basin hopping from this seed
            bh_best, bh_energy = _basin_hopping(start_pos, rng, n_iterations=8, kick_scale=0.35)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_best
    
    # Phase 5: Final polish
    if best_positions is not None and _remaining() > 10.0:
        result = _relax_lbfgsb(best_positions.flatten(), maxiter=150)
        if result is not None:
            e = result.fun
            pos = result.x.reshape(size, 3)
            if e < best_energy and _min_sep_ok(pos):
                best_energy = e
                best_positions = pos
    
    # Fallback
    if best_positions is None:
        best_positions = seeds[0] if seeds else np.zeros((size, 3))
    
    return best_positions
# EVOLVE-BLOCK-END
