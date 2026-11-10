"""Morse cluster global minima: multi-restart basin hopping with vectorized gradients."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(pos, rho):
    """Compute Morse energy E = sum_{i<j} exp(rho(1-r_ij))(exp(rho(1-r_ij)) - 2)."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    iu = np.triu_indices(n, k=1)
    r_ij = r[iu]
    x = rho * (1.0 - r_ij)
    e = np.exp(x) * (np.exp(x) - 2.0)
    return float(np.sum(e))


def _morse_energy_and_grad(pos, rho):
    """Compute Morse energy and gradient (fully vectorized)."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    iu = np.triu_indices(n, k=1)
    r_ij = r[iu]
    x = rho * (1.0 - r_ij)
    ex = np.exp(x)
    e = ex * (ex - 2.0)
    energy = float(np.sum(e))
    
    # dE/dr_ij = 2*rho*ex*(ex-1)
    de_dr = 2.0 * rho * ex * (ex - 1.0)
    # factor[k] = de_dr[k] / r_ij[k], shape (n_pairs,)
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = de_dr / r_ij
    factor = np.where(r_ij > 1e-10, factor, 0.0)
    
    # Build weight matrix W[i,j] = factor for i<j, -factor for j<i, 0 on diagonal
    W = np.zeros((n, n))
    W[iu] = factor
    W.T[iu] = -factor
    
    # grad[i] = sum_j W[i,j] * (pos[i] - pos[j])
    # = sum_j W[i,j]*pos[i] - sum_j W[i,j]*pos[j]
    # = pos[i] * row_sum[i] - W @ pos
    row_sum = W.sum(axis=1)
    grad = pos * row_sum[:, None] - W @ pos
    return energy, grad


def _is_valid(pos, min_dist=0.5):
    """Check if all pairwise distances are at least min_dist."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    return np.min(r) >= min_dist - 1e-9


def _repair_distances(pos, min_dist=0.5):
    """Nudge overlapping atoms apart to satisfy minimum distance constraint."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(20):
        diff = pos[:, None, :] - pos[None, :, :]
        r = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(r, np.inf)
        # Find pairs too close
        too_close = r < min_dist
        if not np.any(too_close):
            break
        # For each violating pair, push apart
        for i in range(n):
            for j in range(i + 1, n):
                if too_close[i, j]:
                    d = diff[i, j]
                    r_ij = r[i, j]
                    if r_ij < 1e-10:
                        d = np.array([1.0, 0.0, 0.0])
                        r_ij = 1.0
                    push = (min_dist - r_ij + 1e-6) / 2.0
                    unit = d / r_ij
                    pos[i] += unit * push
                    pos[j] -= unit * push
    return pos


def _generate_fcc(n, spacing=1.0):
    """Generate an FCC-like cluster of n atoms."""
    a = spacing
    points = []
    max_coord = int(np.ceil(n ** (1/3))) + 2
    for i in range(-max_coord, max_coord+1):
        for j in range(-max_coord, max_coord+1):
            for k in range(-max_coord, max_coord+1):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    if np.linalg.norm(p) <= max_coord * a * 0.7:
                        points.append(p)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points -= np.mean(points, axis=0)
    return points


def _generate_icosahedral(n, spacing=1.0):
    """Generate an icosahedral cluster: 12-vertex icosahedron shells."""
    # Icosahedron vertices (unit)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = [np.zeros(3)]  # center atom
    shell_radii = [1.0, np.sqrt(2.0), 2.0, np.sqrt(5.0)]
    
    for radius in shell_radii:
        if len(points) >= n:
            break
        # Scale icosahedron vertices to this radius
        shell_pts = verts * radius * spacing
        # Also add edge midpoints for denser shells
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                mid = (verts[i] + verts[j]) / 2.0
                mid_norm = np.linalg.norm(mid)
                if mid_norm > 1e-6:
                    mid = mid / mid_norm * radius * spacing
                    shell_pts = np.vstack([shell_pts, mid.reshape(1, -1)])
        # Deduplicate
        shell_pts = np.unique(np.round(shell_pts, 8), axis=0)
        for p in shell_pts:
            if len(points) >= n:
                break
            points.append(p)
    
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points


def _generate_hcp(n, spacing=1.0):
    """Generate an HCP-like cluster of n atoms."""
    a = spacing
    points = []
    max_coord = int(np.ceil(n ** (1/3))) + 2
    for i in range(-max_coord, max_coord+1):
        for j in range(-max_coord, max_coord+1):
            for k in range(-max_coord, max_coord+1):
                if k % 2 == 0:
                    p1 = np.array([i, j, k]) * a
                    p2 = np.array([i + 0.5, j + 0.5, k]) * a
                else:
                    p1 = np.array([i + 1/3.0, j + 2/3.0, k]) * a
                    p2 = np.array([i + 2/3.0, j + 1/3.0, k]) * a
                for p in [p1, p2]:
                    if np.linalg.norm(p) <= max_coord * a * 0.7:
                        points.append(p)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points -= np.mean(points, axis=0)
    return points


def _generate_bcc(n, spacing=1.0):
    """Generate a BCC-like cluster of n atoms."""
    a = spacing
    points = []
    max_coord = int(np.ceil(n ** (1/3))) + 2
    for i in range(-max_coord, max_coord+1):
        for j in range(-max_coord, max_coord+1):
            for k in range(-max_coord, max_coord+1):
                p1 = np.array([i, j, k]) * a
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5]) * a
                for p in [p1, p2]:
                    if np.linalg.norm(p) <= max_coord * a * 0.7:
                        points.append(p)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points -= np.mean(points, axis=0)
    return points


def _generate_random_compact(n, spacing=1.0, rng=None):
    """Generate a random compact cluster with minimum separation."""
    if rng is None:
        rng = np.random.default_rng()
    radius = spacing * n ** (1.0 / 3.0) * 1.0
    points = []
    min_sep = 0.85 * spacing
    attempts = 0
    max_attempts = n * 500
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < min_sep:
                    ok = False
                    break
            if ok:
                points.append(p)
        attempts += 1
    
    # Fill remaining with random points (will be repaired later)
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    return points


def _basin_hopping(pos, rho, max_steps=30, step_size=0.3, rng=None, deadline=None):
    """Perform basin hopping with adaptive step size and Metropolis acceptance."""
    if rng is None:
        rng = np.random.default_rng()
    
    energy, pos = _morse_energy_and_grad(pos, rho)
    current_step = step_size
    accept_count = 0
    total_count = 0
    
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Adaptive step size: shrink if many acceptances, grow if few
        if total_count > 5:
            frac = accept_count / total_count
            if frac > 0.6:
                current_step *= 0.85
            elif frac < 0.2:
                current_step *= 1.15
            current_step = np.clip(current_step, 0.1, 1.0)
        
        # Random perturbation
        delta = rng.normal(0, current_step, size=pos.shape)
        new_pos = pos + delta
        
        # Repair distances if needed
        if not _is_valid(new_pos):
            new_pos = _repair_distances(new_pos)
        
        new_energy, _ = _morse_energy_and_grad(new_pos, rho)
        total_count += 1
        
        # Accept if lower energy or with Metropolis probability
        if new_energy < energy:
            energy, pos = new_energy, new_pos
            accept_count += 1
        else:
            delta_e = new_energy - energy
            if delta_e < 20.0:
                prob = np.exp(-delta_e / 1.5)
                if rng.random() < prob:
                    energy, pos = new_energy, new_pos
                    accept_count += 1
    
    # Final local relaxation
    try:
        energy, pos = relax(pos, 'morse', rho)
    except:
        pass
    
    return energy, pos


def construct(instance):
    """Multi-restart basin hopping with diverse initializations and deadline guard."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 285  # 285s deadline with 45s safety margin
    
    rng = np.random.default_rng(size)
    
    # Generate diverse starting configurations
    starts = []
    
    # Crystalline starts
    for gen_fn in [_generate_fcc, _generate_hcp, _generate_bcc, _generate_icosahedral]:
        try:
            pts = gen_fn(size, spacing=1.0)
            if len(pts) == size:
                starts.append(pts)
        except:
            pass
    
    # Random compact starts with varying density
    for i in range(4):
        try:
            rand = _generate_random_compact(size, spacing=1.0, rng=rng)
            starts.append(rand)
        except:
            pass
    
    # Additional perturbed starts from best crystalline
    if starts:
        for i in range(2):
            try:
                base = starts[0].copy()
                base += rng.normal(0, 0.3, size=base.shape)
                base = _repair_distances(base)
                starts.append(base)
            except:
                pass
    
    best_energy = np.inf
    best_pos = None
    
    # Phase 1: Relax all starts to find good local minima
    relaxed_starts = []
    for start_pos in starts:
        if time.time() > deadline:
            break
        try:
            energy, relaxed_pos = relax(start_pos, 'morse', parameter)
            relaxed_starts.append((energy, relaxed_pos))
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos.copy()
        except:
            continue
    
    # Phase 2: Basin hopping from the best relaxed starts
    # Sort by energy and take top candidates
    relaxed_starts.sort(key=lambda x: x[0])
    n_bh = min(len(relaxed_starts), 5)
    
    for idx in range(n_bh):
        if time.time() > deadline:
            break
        e0, pos0 = relaxed_starts[idx]
        try:
            bh_energy, bh_pos = _basin_hopping(pos0, parameter, max_steps=25,
                                                step_size=0.3, rng=rng, deadline=deadline)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos.copy()
        except:
            continue
    
    # Phase 3: If time remains, do additional basin hopping from best
    if best_pos is not None and time.time() < deadline - 10:
        try:
            bh_energy, bh_pos = _basin_hopping(best_pos, parameter, max_steps=20,
                                                step_size=0.2, rng=rng, deadline=deadline)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos.copy()
        except:
            pass
    
    # Fallback
    if best_pos is None:
        try:
            rand = _generate_random_compact(size, spacing=1.0, rng=rng)
            best_pos = _repair_distances(rand)
            best_energy, best_pos = relax(best_pos, potential, parameter)
        except:
            best_pos = np.zeros((size, 3))
            best_pos[:, 0] = np.arange(size) * 1.0
    
    return best_pos
# EVOLVE-BLOCK-END
