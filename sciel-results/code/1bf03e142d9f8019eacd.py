"""Sutton-Chen transition-metal cluster global minima: LAMMPS-style sequential growth + multi-start relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _ensure_min_dist(pts, dmin=0.355):
    """Push apart any pair closer than dmin so the result is always valid."""
    pts = np.asarray(pts, dtype=float).copy()
    n = pts.shape[0]
    if n < 2:
        return pts
    for _ in range(200):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[i] - pts[j]
                r = np.linalg.norm(d)
                if r < dmin:
                    if r < 1e-9:
                        pts[j] += np.array([dmin, 0.0, 0.0])
                    else:
                        push = (dmin - r) * 0.5
                        pts[i] += push * d / r
                        pts[j] -= push * d / r
                    moved = True
        if not moved:
            break
    return pts


def _validate_positions(pos):
    """Check if positions are finite and no atoms are closer than 0.35."""
    if not np.all(np.isfinite(pos)):
        return False
    if pos.shape[0] < 2:
        return True
    for i in range(pos.shape[0]):
        for j in range(i + 1, pos.shape[0]):
            if np.linalg.norm(pos[i] - pos[j]) < 0.35:
                return False
    return True


def _fcc_cluster(n, spacing=0.72):
    """Generate an n-atom fcc-like cluster by filling fcc lattice sites nearest to origin."""
    if n <= 0:
        return np.empty((0, 3))
    basis = np.array([[0, 0, 0],
                      [0, 0.5, 0.5],
                      [0.5, 0, 0.5],
                      [0.5, 0.5, 0]], dtype=float)
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                cubic = np.array([i, j, k], dtype=float)
                for b in basis:
                    p = (cubic + b) * spacing
                    sites.append(p)
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    selected = sites[idx[:n]]
    selected = selected - selected.mean(axis=0)
    return selected


def _hcp_cluster(n, spacing=0.72):
    """Generate an n-atom hcp-like cluster by filling hcp lattice sites nearest to origin."""
    if n <= 0:
        return np.empty((0, 3))
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for layer in range(-max_shell, max_shell + 1):
        z = layer * c
        for i in range(-max_shell, max_shell + 1):
            for j in range(-max_shell, max_shell + 1):
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3.0) / 2.0 * j
                sites.append([x, y, z])
                sites.append([x + a * 0.5, y + a * np.sqrt(3.0) / 6.0, z])
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    selected = sites[idx[:n]]
    selected = selected - selected.mean(axis=0)
    return selected


def _icosahedral_cluster(n, spacing=0.72):
    """Generate an n-atom icosahedral shell cluster."""
    if n <= 0:
        return np.empty((0, 3))
    phi = (1 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi, 1/phi], [0, -phi, 1/phi], [0, phi, -1/phi], [0, -phi, -1/phi],
        [phi, 1/phi, 0], [-phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, -1/phi, 0],
        [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    points = [np.zeros(3)]
    for v in ico_verts[:12]:
        points.append(v * spacing * 1.0)
    
    if n <= 13:
        selected = np.array(points[:n])
        return selected - selected.mean(axis=0)
    
    num_second = min(42, n - 13)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for k in range(num_second):
        z = 1.0 - 2.0 * (k + 0.5) / num_second
        r = np.sqrt(1.0 - z * z)
        theta = golden_angle * k
        p = np.array([r * np.cos(theta), r * np.sin(theta), z])
        points.append(p * spacing * 1.8)
    
    remaining = n - len(points)
    if remaining > 0:
        num_third = remaining
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for k in range(num_third):
            z = 1.0 - 2.0 * (k + 0.5) / num_third
            r = np.sqrt(1.0 - z * z)
            theta = golden_angle * (k + 0.3)
            p = np.array([r * np.cos(theta), r * np.sin(theta), z])
            points.append(p * spacing * 2.6)
    
    selected = np.array(points[:n])
    return selected - selected.mean(axis=0)


def _random_compact(n, spacing=0.72, seed=42):
    """Generate a random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    min_dist = 0.55 * spacing
    points = []
    max_attempts = 20000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    ok = False
                    break
            if ok:
                points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    selected = np.array(points)
    return selected - selected.mean(axis=0)


def _lammps_growth(n, potential, kind, t0, deadline, seed=42):
    """
    LAMMPS-style sequential growth: start with a small core, add atoms one by one
    by minimizing the energy contribution of the new atom given the existing core.
    This produces a well-relaxed structure that is a true local minimum.
    """
    rng = np.random.default_rng(seed)
    spacing = 0.72
    
    # Start with a small icosahedral core (13 atoms) or smaller if n < 13
    core_size = min(13, n)
    if core_size <= 13:
        core = _icosahedral_cluster(core_size, spacing)
    else:
        core = _icosahedral_cluster(13, spacing)
    
    # Relax the core
    try:
        e_core, core = relax(core, potential, kind)
        if not np.all(np.isfinite(core)):
            return None
    except Exception:
        return None
    
    # Add atoms one by one
    for new_idx in range(core_size, n):
        if time.time() > deadline - 30.0:
            break
        
        # Generate candidate positions for the new atom on a spherical shell
        # around the current cluster
        center = core.mean(axis=0)
        radius = spacing * (new_idx + 1) ** (1.0 / 3.0) * 1.1
        
        # Try multiple candidate positions
        best_e = np.inf
        best_pos = None
        
        # Candidate 1: on the surface, golden angle distribution
        for trial in range(8):
            z = 1.0 - 2.0 * (trial + 0.5) / 8.0
            r = np.sqrt(1.0 - z * z)
            theta = rng.uniform(0, 2 * np.pi)
            candidate = center + radius * np.array([r * np.cos(theta), r * np.sin(theta), z])
            
            # Test this candidate
            test_cluster = np.vstack([core, candidate])
            try:
                e_test, _ = sutton_chen_energy_gradient(test_cluster, kind)
                if np.isfinite(e_test) and e_test < best_e:
                    best_e = e_test
                    best_pos = candidate
            except Exception:
                continue
        
        # Candidate 2: near an existing atom (to form a bond)
        for trial in range(4):
            existing_idx = rng.integers(0, len(core))
            direction = rng.normal(0, 1, 3)
            direction = direction / np.linalg.norm(direction)
            candidate = core[existing_idx] + spacing * direction
            test_cluster = np.vstack([core, candidate])
            try:
                e_test, _ = sutton_chen_energy_gradient(test_cluster, kind)
                if np.isfinite(e_test) and e_test < best_e:
                    best_e = e_test
                    best_pos = candidate
            except Exception:
                continue
        
        # If no good candidate found, use a random position
        if best_pos is None:
            best_pos = center + radius * rng.normal(0, 1, 3)
            best_pos = best_pos / np.linalg.norm(best_pos - center) * radius + center
        
        # Add the new atom
        core = np.vstack([core, best_pos])
        
        # Relax the entire cluster
        try:
            e_relaxed, core = relax(core, potential, kind)
            if not np.all(np.isfinite(core)):
                return None
        except Exception:
            return None
    
    return core


def _jitter_cluster(base, n, spacing=0.72, seed=0, jitter_frac=0.15):
    """Add random jitter to a base cluster to create diverse starting points."""
    rng = np.random.default_rng(seed)
    jitter = rng.normal(0, jitter_frac * spacing, size=base.shape)
    return base + jitter


def construct(instance):
    """Multi-start structured seed relaxation for Sutton-Chen clusters with LAMMPS growth."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    # Time management: total budget ~330s, leave 30s safety margin
    t_start = time.time()
    deadline = t_start + 300.0  # 300s out of 330s
    
    # Generate diverse initial geometries
    spacing = 0.72
    geometries = []
    
    # 1. LAMMPS-style growth (primary method)
    try:
        lammps_result = _lammps_growth(size, potential, kind, t_start, deadline, seed=42)
        if lammps_result is not None and lammps_result.shape[0] == size and _validate_positions(lammps_result):
            geometries.append(lammps_result)
    except Exception:
        pass
    
    # 2. FCC-like
    try:
        fcc = _fcc_cluster(size, spacing)
        if fcc.shape[0] == size and _validate_positions(fcc):
            geometries.append(fcc)
    except Exception:
        pass
    
    # 3. HCP-like
    try:
        hcp = _hcp_cluster(size, spacing)
        if hcp.shape[0] == size and _validate_positions(hcp):
            geometries.append(hcp)
    except Exception:
        pass
    
    # 4. Icosahedral
    try:
        ico = _icosahedral_cluster(size, spacing)
        if ico.shape[0] == size and _validate_positions(ico):
            geometries.append(ico)
    except Exception:
        pass
    
    # 5. Random compact (multiple seeds)
    for seed in [42, 123, 777, 2024]:
        try:
            rand = _random_compact(size, spacing, seed=seed)
            if rand.shape[0] == size and _validate_positions(rand):
                geometries.append(rand)
        except Exception:
            pass
    
    # 6. Jittered versions of structured geometries
    for i, base in enumerate(geometries[:4]):
        for jseed in [10, 20, 30]:
            try:
                jit = _jitter_cluster(base, size, spacing, seed=jseed + i * 100, jitter_frac=0.12)
                if jit.shape[0] == size and _validate_positions(jit):
                    geometries.append(jit)
            except Exception:
                pass

    # Ensure we have at least one valid geometry
    if not geometries:
        positions = np.zeros((size, 3))
        positions[:, 0] = np.arange(size) * 0.5
        geometries.append(positions)
    
    # Relax each geometry and keep the best
    best_energy = np.inf
    best_positions = None
    
    # Phase 1: Relax initial seeds
    for geom in geometries:
        if time.time() > deadline - 20.0:
            break
        try:
            energy, positions = relax(geom, potential, kind)
            if np.isfinite(energy) and energy < best_energy and _validate_positions(positions):
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
    
    if best_positions is None:
        best_positions = geometries[0]
    
    # Phase 2: Basin Hopping - shake and relax from best solution
    if time.time() < deadline - 10.0:
        rng = np.random.default_rng(42)
        shake_count = 0
        base_noise = 0.08
        while time.time() < deadline - 5.0:
            noise_scale = base_noise * (1.0 + 0.05 * shake_count)
            noise = rng.normal(0, noise_scale, size=best_positions.shape)
            new_pos = best_positions + noise
            
            if not _validate_positions(new_pos):
                shake_count += 1
                continue
            
            try:
                energy, positions = relax(new_pos, potential, kind)
                if np.isfinite(energy) and energy < best_energy and _validate_positions(positions):
                    best_energy = energy
                    best_positions = positions
                    shake_count = 0
                else:
                    shake_count += 1
            except Exception:
                shake_count += 1
            
            if shake_count > 80:
                break

    # Final validation
    if not np.all(np.isfinite(best_positions)):
        best_positions = np.zeros((size, 3))
        best_positions[:, 0] = np.arange(size) * 0.5
    
    return _ensure_min_dist(best_positions)
# EVOLVE-BLOCK-END
