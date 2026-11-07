"""Program for Sutton-Chen TM cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 280.0  # Safety margin before 300s kill


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    """Check if positions are valid (finite, no atoms closer than min_dist)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n - 1):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True


def _generate_fcc(n):
    """Generate a cluster of n atoms in an FCC-like structure."""
    max_r = int(np.ceil(n ** (1 / 3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * (1.0 / np.sqrt(2.0))
                    candidates.append((np.linalg.norm(p), p))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_bcc(n):
    """Generate a cluster of n atoms in a BCC-like structure."""
    max_r = int(np.ceil(n ** (1 / 3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k])
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5])
                candidates.append((np.linalg.norm(p1), p1))
                candidates.append((np.linalg.norm(p2), p2))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_hcp(n):
    """Generate a cluster of n atoms in an HCP-like structure."""
    a = 1.0
    c = np.sqrt(8.0 / 3.0) * a
    max_r = int(np.ceil(n ** (1 / 3))) + 3
    candidates = []
    
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                x = a * (i + 0.5 * j)
                y = a * np.sqrt(3) / 2 * j
                z = c * k
                p1 = np.array([x, y, z])
                candidates.append((np.linalg.norm(p1), p1))
                
                x2 = a * (i + 0.5 * j + 1.0 / np.sqrt(3.0))
                y2 = a * np.sqrt(3) / 2 * (j + 1.0 / 3.0)
                z2 = c * (k + 1.0 / 3.0)
                p2 = np.array([x2, y2, z2])
                candidates.append((np.linalg.norm(p2), p2))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_icosahedron(n):
    """Generate a cluster with icosahedral symmetry."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    
    pts = [np.zeros(3)]
    pts.extend(verts.tolist())
    
    # Add edge midpoints
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 0.5:
                mid = (verts[i] + verts[j]) / 2
                mid = mid / np.linalg.norm(mid) * 1.05
                pts.append(mid)
    
    # Add face centroids
    while len(pts) < n:
        for i in range(12):
            for j in range(i + 1, 12):
                for k in range(j + 1, 12):
                    if len(pts) >= n:
                        break
                    d1 = np.linalg.norm(verts[i] - verts[j])
                    d2 = np.linalg.norm(verts[j] - verts[k])
                    d3 = np.linalg.norm(verts[k] - verts[i])
                    if d1 < 0.5 and d2 < 0.5 and d3 < 0.5:
                        centroid = (verts[i] + verts[j] + verts[k]) / 3
                        centroid = centroid / np.linalg.norm(centroid) * 1.15
                        pts.append(centroid)
                if len(pts) >= n:
                    break
            if len(pts) >= n:
                break
        if len(pts) < n:
            for v in verts:
                if len(pts) >= n:
                    break
                pts.append(v * 1.3)
    
    return np.array(pts[:n])


def _generate_random_compact(n, rng, radius=2.0):
    """Generate n random points within a sphere."""
    pts = []
    while len(pts) < n:
        p = rng.uniform(-1, 1, size=(10, 3))
        norms = np.linalg.norm(p, axis=1)
        valid = p[norms <= 1.0]
        for v in valid:
            if len(pts) < n:
                pts.append(v * radius)
    return np.array(pts[:n])


def _growth_construct(n, kind, rng, max_time=60.0):
    """
    Deterministic growth: build cluster atom by atom by finding the
    lowest-energy interstitial site and relaxing.
    """
    start_t = time.time()
    
    # Start with 2 atoms
    if n <= 2:
        pos = np.array([[0, 0, 0], [1.0, 0, 0]])[:n]
        try:
            energy, pos = relax(pos, 'sutton_chen', kind)
        except Exception:
            pass
        return pos, energy
    
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    try:
        energy, pos = relax(pos, 'sutton_chen', kind)
    except Exception:
        pass
    
    current_n = 2
    
    while current_n < n:
        if time.time() - start_t > max_time:
            break
        
        # Generate candidate positions for new atom
        # Use a grid of candidate positions around existing cluster
        candidates = []
        
        # Radial shells around existing atoms
        for shell_r in [0.8, 1.0, 1.2, 1.5, 2.0]:
            # Fibonacci sphere for uniform coverage
            n_pts = 20
            for idx in range(n_pts):
                y = 1 - 2 * (idx + 0.5) / n_pts
                r = np.sqrt(1 - y * y)
                theta = 2 * np.pi * idx / n_pts
                direction = np.array([r * np.cos(theta), y, r * np.sin(theta)])
                
                # Place around each existing atom (but limit to avoid too many)
                for base_idx in range(min(current_n, 5)):
                    new_pos = pos[base_idx] + shell_r * direction
                    candidates.append(new_pos)
        
        # Also add some random positions
        for _ in range(10):
            new_pos = rng.normal(0, 1.5, 3)
            candidates.append(new_pos)
        
        candidates = np.array(candidates)
        
        # Evaluate energy for each candidate (add to current cluster)
        best_energy = np.inf
        best_candidate = None
        
        for cand in candidates:
            trial = np.vstack([pos, cand.reshape(1, 3)])
            if not _is_valid(trial):
                continue
            try:
                e, _ = sutton_chen_energy_gradient(trial, kind)
                if np.isfinite(e) and e < best_energy:
                    best_energy = e
                    best_candidate = cand
            except Exception:
                continue
        
        if best_candidate is None:
            # Fallback: add atom at a reasonable distance
            best_candidate = pos[0] + np.array([1.0, 0, 0])
        
        # Add the best candidate
        pos = np.vstack([pos, best_candidate.reshape(1, 3)])
        current_n += 1
        
        # Relax the new cluster
        try:
            energy, pos = relax(pos, 'sutton_chen', kind)
            if not np.isfinite(energy) or not _is_valid(pos):
                # Revert to previous and try a different position
                pos = np.delete(pos, -1, axis=0)
                current_n -= 1
                # Try a slightly different position
                alt = best_candidate + rng.normal(0, 0.2, 3)
                pos = np.vstack([pos, alt.reshape(1, 3)])
                current_n += 1
                try:
                    energy, pos = relax(pos, 'sutton_chen', kind)
                except Exception:
                    pos = np.delete(pos, -1, axis=0)
                    current_n -= 1
        except Exception:
            pass
    
    return pos, energy


def _polish(positions, kind):
    """High-precision polish pass."""
    try:
        energy, pos = relax(positions, 'sutton_chen', kind)
        if not np.isfinite(energy):
            return None, np.inf
        
        # Additional refinement
        for _ in range(10):
            e, g = sutton_chen_energy_gradient(pos, kind)
            if not np.isfinite(e):
                break
            # Check if gradient is small enough
            if np.max(np.abs(g)) < 1e-10:
                break
            lr = 0.0005
            new_pos = pos - lr * g
            if _is_valid(new_pos):
                pos = new_pos
            else:
                break
        
        energy, pos = relax(pos, 'sutton_chen', kind)
        if np.isfinite(energy) and _is_valid(pos):
            return pos, energy
    except Exception:
        pass
    return None, np.inf


def _basin_hopping(positions, kind, rng, n_steps=30, max_time=60.0):
    """
    Simple basin hopping: perturb, relax, accept if lower energy.
    """
    start_t = time.time()
    current_energy, current_pos = relax(positions, 'sutton_chen', kind)
    best_energy, best_pos = current_energy, current_pos
    
    for step in range(n_steps):
        if time.time() - start_t > max_time:
            break
        
        # Perturb
        noise_scale = 0.1 + 0.1 * (step // 10)
        perturbed = current_pos + rng.normal(0, noise_scale, current_pos.shape)
        
        if not _is_valid(perturbed):
            continue
        
        try:
            new_energy, new_pos = relax(perturbed, 'sutton_chen', kind)
        except Exception:
            continue
        
        if not np.isfinite(new_energy) or not _is_valid(new_pos):
            continue
        
        # Accept if lower energy (Metropolis at T=0)
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos
    
    return best_pos, best_energy


def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_pos = None
    
    rng = np.random.default_rng(42 + n)
    
    # Strategy 1: Growth construction
    if _time_left() > 40.0:
        try:
            growth_pos, growth_energy = _growth_construct(n, kind, rng, max_time=40.0)
            if _is_valid(growth_pos) and np.isfinite(growth_energy):
                best_energy = growth_energy
                best_pos = growth_pos
        except Exception:
            pass
    
    # Strategy 2: Multi-start from base structures
    base_structures = []
    base_structures.append(_generate_fcc(n))
    base_structures.append(_generate_bcc(n))
    base_structures.append(_generate_hcp(n))
    base_structures.append(_generate_icosahedron(n))
    base_structures.append(_generate_random_compact(n, rng, radius=2.0))
    
    relaxed_bases = []
    for base in base_structures:
        if _time_left() < 30.0:
            break
        try:
            energy, relaxed_pos = relax(base, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed_pos):
                relaxed_bases.append(relaxed_pos)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed_pos
        except Exception:
            continue
    
    # Strategy 3: Basin hopping from best base
    if best_pos is not None and _time_left() > 40.0:
        try:
            bh_pos, bh_energy = _basin_hopping(best_pos, kind, rng, n_steps=30, max_time=40.0)
            if _is_valid(bh_pos) and bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos
        except Exception:
            pass
    
    # Strategy 4: More perturbed restarts
    max_restarts = 50
    for i in range(max_restarts):
        if _time_left() < 20.0:
            break
        
        if relaxed_bases:
            base = relaxed_bases[i % len(relaxed_bases)].copy()
        elif best_pos is not None:
            base = best_pos.copy()
        else:
            base = base_structures[i % len(base_structures)].copy()
        
        noise_scale = 0.05 + 0.1 * ((i // 5) % 4)
        pos = base + rng.normal(0, noise_scale, base.shape)
        
        try:
            energy, relaxed_pos = relax(pos, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed_pos):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed_pos
        except Exception:
            continue
    
    # Final polish
    if best_pos is not None and _time_left() > 15.0:
        try:
            polished_pos, polished_energy = _polish(best_pos, kind)
            if polished_pos is not None and polished_energy < best_energy:
                best_energy = polished_energy
                best_pos = polished_pos
        except Exception:
            pass
    
    # Fallback
    if best_pos is None:
        best_pos = _generate_fcc(n)
    
    return best_pos

# EVOLVE-BLOCK-END
