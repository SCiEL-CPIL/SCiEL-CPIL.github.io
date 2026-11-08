"""
Strategy: Multi-start global optimization for Sutton-Chen transition-metal clusters.
Uses diverse structural templates (FCC, HCP, BCC, icosahedral, random, liquid).
Adaptive basin-hopping with Metropolis acceptance, plus high-precision polish.
"""

import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE_SEC = 320.0
_START_TIME = None

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        return _DEADLINE_SEC
    return _DEADLINE_SEC - (time.time() - _START_TIME)

def _liquid_start(n, rng, spacing=1.0):
    """Generate a liquid-like random start with minimum distance."""
    radius = spacing * (n / 4.0) ** (1/3.0) * 3.0
    points = []
    min_dist = 0.55 * spacing
    max_attempts = n * 120
    
    for _ in range(max_attempts):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if not points:
            points.append(p)
            continue
        existing = np.array(points)
        dists = np.linalg.norm(existing - p, axis=1)
        if np.min(dists) >= min_dist:
            points.append(p)
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    points = np.array(points)
    points = points - np.mean(points, axis=0)
    return points

def _fcc_cluster(n, spacing=1.0, rng=None):
    """Generate n atoms from an FCC lattice, compactly centered."""
    if rng is None:
        rng = np.random.default_rng()
    
    a = spacing * np.sqrt(2)
    R = (3 * n * a**3 / (16 * np.pi)) ** (1/3) * 1.5
    n_side = int(np.ceil(R / a)) + 2
    
    points = []
    for i in range(-n_side, n_side + 1):
        for j in range(-n_side, n_side + 1):
            for k in range(-n_side, n_side + 1):
                base = np.array([i, j, k]) * a
                for offset in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = base + np.array(offset) * a
                    if np.linalg.norm(p) <= R:
                        points.append(p)
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    
    if len(idx) < n:
        R2 = R * 1.6
        n_side2 = int(np.ceil(R2 / a)) + 2
        points2 = []
        for i in range(-n_side2, n_side2 + 1):
            for j in range(-n_side2, n_side2 + 1):
                for k in range(-n_side2, n_side2 + 1):
                    base = np.array([i, j, k]) * a
                    for offset in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                        p = base + np.array(offset) * a
                        if np.linalg.norm(p) <= R2:
                            points2.append(p)
        points = np.array(points2)
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)
    
    points = points[idx[:n]]
    points = points - np.mean(points, axis=0)
    return points

def _hcp_cluster(n, spacing=1.0, rng=None):
    """Generate n atoms from an HCP lattice."""
    if rng is None:
        rng = np.random.default_rng()
    
    a = spacing
    c = a * np.sqrt(8.0/3.0)
    
    points = []
    n_layers = int(np.ceil(np.sqrt(n))) + 5
    n_side = int(np.ceil(np.sqrt(n))) + 5
    R2 = (n / 4.0) ** (2/3) * a**2 * 4.0
    
    for layer in range(-n_layers, n_layers + 1):
        z = layer * c
        if layer % 2 == 0:
            shift_x, shift_y = 0.0, 0.0
        else:
            shift_x = a / 2
            shift_y = a * np.sqrt(3) / 6
        
        for i in range(-n_side, n_side + 1):
            for j in range(-n_side, n_side + 1):
                x = i * a + j * a * 0.5 + shift_x
                y = j * a * np.sqrt(3) / 2.0 + shift_y
                if x**2 + y**2 + z**2 <= R2:
                    points.append([x, y, z])
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    
    if len(idx) < n:
        while len(points) < n:
            p = rng.uniform(-4*a, 4*a, 3)
            points = np.vstack([points, p])
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)
    
    points = points[idx[:n]]
    points = points - np.mean(points, axis=0)
    return points

def _random_compact(n, spacing=1.0, rng=None, compactness=2.8, min_dist_factor=0.65):
    """Random compact cluster with minimum distance constraint."""
    if rng is None:
        rng = np.random.default_rng()
        
    radius = spacing * (n / 4.0) ** (1/3.0) * compactness
    points = []
    min_dist = min_dist_factor * spacing
    max_attempts = n * 100
    
    for _ in range(max_attempts):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if not points:
            points.append(p)
            continue
        existing = np.array(points)
        dists = np.linalg.norm(existing - p, axis=1)
        if np.min(dists) >= min_dist:
            points.append(p)
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    points = np.array(points)
    points = points - np.mean(points, axis=0)
    return points

def _icosahedral_shell(n, spacing=1.0, rng=None):
    """Icosahedral arrangement: center + 12 + 42 + ... shells."""
    if rng is None:
        rng = np.random.default_rng()
    
    points = [np.array([0.0, 0.0, 0.0])]
    
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    if n > 1:
        shell1 = ico_verts * spacing
        for v in shell1:
            if len(points) < n:
                points.append(v)
    
    if n > 13:
        a = spacing * np.sqrt(2)
        R2 = 2.4 * spacing
        n_side = 6
        second_shell = []
        for i in range(-n_side, n_side + 1):
            for j in range(-n_side, n_side + 1):
                for k in range(-n_side, n_side + 1):
                    base = np.array([i, j, k]) * a
                    for offset in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                        p = base + np.array(offset) * a
                        r = np.linalg.norm(p)
                        if 1.3 * spacing < r < 2.8 * spacing:
                            second_shell.append(p / r * 2.0 * spacing)
        
        seen = set()
        unique_second = []
        for p in second_shell:
            key = tuple(np.round(p, 3))
            if key not in seen:
                seen.add(key)
                unique_second.append(p)
        
        for p in unique_second:
            if len(points) < n:
                points.append(p)
    
    while len(points) < n:
        p = rng.uniform(-4.0*spacing, 4.0*spacing, 3)
        if np.linalg.norm(p) < 4.5 * spacing:
            points.append(p)
    
    points = np.array(points[:n])
    points = points - np.mean(points, axis=0)
    return points

def _bcc_cluster(n, spacing=1.0, rng=None):
    """Generate n atoms from a BCC lattice."""
    if rng is None:
        rng = np.random.default_rng()
    
    a = spacing * 2.0 / np.sqrt(3)
    R = (3 * n * a**3 / (8 * np.pi)) ** (1/3) * 1.5
    n_side = int(np.ceil(R / a)) + 2
    
    points = []
    for i in range(-n_side, n_side + 1):
        for j in range(-n_side, n_side + 1):
            for k in range(-n_side, n_side + 1):
                base = np.array([i, j, k]) * a
                for offset in [(0,0,0), (0.5,0.5,0.5)]:
                    p = base + np.array(offset) * a
                    if np.linalg.norm(p) <= R:
                        points.append(p)
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    
    if len(idx) < n:
        R2 = R * 1.6
        n_side2 = int(np.ceil(R2 / a)) + 2
        points2 = []
        for i in range(-n_side2, n_side2 + 1):
            for j in range(-n_side2, n_side2 + 1):
                for k in range(-n_side2, n_side2 + 1):
                    base = np.array([i, j, k]) * a
                    for offset in [(0,0,0), (0.5,0.5,0.5)]:
                        p = base + np.array(offset) * a
                        if np.linalg.norm(p) <= R2:
                            points2.append(p)
        points = np.array(points2)
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)
    
    points = points[idx[:n]]
    points = points - np.mean(points, axis=0)
    return points

def _perturb(positions, rng, scale=0.3):
    noise = rng.normal(0, scale, positions.shape)
    return positions + noise

def _basin_hopping(positions, potential, kind, rng, n_iters=100, base_scale=0.4, min_time=5.0, use_metropolis=True, temperature_scale=0.02):
    """Run basin hopping from initial positions with optional Metropolis acceptance."""
    best_energy, best_pos = relax(positions, potential, kind)
    current_energy, current_pos = best_energy, best_pos
    
    for i in range(n_iters):
        if _time_left() < min_time:
            break
        
        # Adaptive scale: start larger, decay, but maintain minimum exploration
        scale = base_scale * (0.92 ** (i // 10)) + 0.12
        new_pos = _perturb(current_pos, rng, scale)
        
        try:
            e, pos = relax(new_pos, potential, kind)
            if not np.isfinite(e):
                continue
            # Metropolis acceptance with cooling temperature
            if e < current_energy:
                current_energy, current_pos = e, pos
            elif use_metropolis:
                T = temperature_scale * max(0.1, 1.0 - i / (2.0 * n_iters))
                if rng.random() < np.exp(-(e - current_energy) / T):
                    current_energy, current_pos = e, pos
            if e < best_energy:
                best_energy = e
                best_pos = pos
        except:
            pass
    
    return best_energy, best_pos

def _high_precision_polish(positions, potential, kind):
    """Final high-precision polish pass with tighter tolerance."""
    try:
        from scipy.optimize import minimize
        
        def obj(x):
            pos = x.reshape(-1, 3)
            e, _ = sutton_chen_energy_gradient(pos, kind)
            if not np.isfinite(e):
                return 1e10
            return e
        
        def grad(x):
            pos = x.reshape(-1, 3)
            _, g = sutton_chen_energy_gradient(pos, kind)
            if not np.all(np.isfinite(g)):
                return np.zeros_like(g).flatten()
            return g.flatten()
        
        x0 = positions.flatten()
        result = minimize(
            obj, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 500}
        )
        
        if result.fun < np.inf and np.all(np.isfinite(result.x)):
            final_pos = result.x.reshape(-1, 3)
            if np.min(np.linalg.norm(final_pos[:, np.newaxis] - final_pos[np.newaxis, :], axis=2)) > 0.35:
                return result.fun, final_pos
        return relax(positions, potential, kind)
    except:
        return relax(positions, potential, kind)

def construct(instance):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    rng = np.random.default_rng(42 + size)
    
    best_energy = np.inf
    best_positions = None
    
    templates = []
    
    # Structural templates
    try:
        fcc = _fcc_cluster(size, 1.0, rng)
        templates.append(fcc)
    except:
        pass
    
    try:
        hcp = _hcp_cluster(size, 1.0, rng)
        templates.append(hcp)
    except:
        pass
    
    try:
        ico = _icosahedral_shell(size, 1.0, rng)
        templates.append(ico)
    except:
        pass
    
    try:
        bcc = _bcc_cluster(size, 1.0, rng)
        templates.append(bcc)
    except:
        pass
    
    # Diverse random starts with varying compactness
    compactnesses = [2.2, 2.5, 2.8, 3.0, 3.2, 3.5, 4.0, 4.5]
    for comp in compactnesses:
        try:
            rand = _random_compact(size, 1.0, rng, compactness=comp, min_dist_factor=0.6)
            templates.append(rand)
        except:
            pass
    
    # Liquid-like starts
    for _ in range(4):
        try:
            liq = _liquid_start(size, rng, 1.0)
            templates.append(liq)
        except:
            pass
    
    # Phase 1: Relax all templates, save results
    relaxed_results = []
    for t in templates:
        if _time_left() < 20.0:
            break
        try:
            e, pos = relax(t, potential, kind)
            if np.isfinite(e):
                relaxed_results.append((e, pos))
                if e < best_energy:
                    best_energy = e
                    best_positions = pos
        except:
            pass
    
    if best_positions is None and templates:
        best_positions = templates[0]
        best_energy = np.inf
    
    # Sort relaxed results by energy
    relaxed_results.sort(key=lambda x: x[0])
    
    # Phase 2: Basin hopping from top structures
    # Start with the best, then try others if time permits
    if best_positions is not None:
        n_bh_best = max(60, int((_time_left() - 30.0) * 3.5))
        best_energy, best_positions = _basin_hopping(
            best_positions, potential, kind, rng, 
            n_iters=n_bh_best, base_scale=0.55, min_time=15.0, temperature_scale=0.025
        )
    
    # Phase 3: Basin hopping from other good starting points
    # Use already-relaxed positions (no need to re-relax)
    bh_count = 0
    for e_t, pos_t in relaxed_results:
        if _time_left() < 30.0:
            break
        if e_t > best_energy + 3.0:
            continue
        if bh_count >= 8:
            break
        try:
            n_bh = max(30, int((_time_left() - 25.0) * 1.5))
            e, pos = _basin_hopping(pos_t, potential, kind, rng, n_iters=n_bh, base_scale=0.5, min_time=12.0, temperature_scale=0.02)
            if e < best_energy:
                best_energy = e
                best_positions = pos
            bh_count += 1
        except:
            pass
    
    # Phase 4: Additional random basin hopping if time remains
    if _time_left() > 40.0:
        extra_starts = 3
        for _ in range(extra_starts):
            if _time_left() < 25.0:
                break
            try:
                rand = _random_compact(size, 1.0, rng, compactness=rng.uniform(2.5, 4.5), min_dist_factor=0.55)
                e_rand, pos_rand = relax(rand, potential, kind)
                if np.isfinite(e_rand) and e_rand < best_energy + 8.0:
                    n_bh = max(25, int((_time_left() - 20.0) * 1.2))
                    e, pos = _basin_hopping(pos_rand, potential, kind, rng, n_iters=n_bh, base_scale=0.55, min_time=10.0, temperature_scale=0.02)
                    if e < best_energy:
                        best_energy = e
                        best_positions = pos
            except:
                pass
    
    # Final high-precision polish
    if _time_left() > 12.0 and best_positions is not None:
        try:
            e_polished, pos_polished = _high_precision_polish(best_positions, potential, kind)
            if np.isfinite(e_polished) and e_polished < best_energy:
                best_energy = e_polished
                best_positions = pos_polished
        except:
            pass
    
    # Validate final result
    if best_positions is None:
        # Fallback: return a simple compact arrangement
        best_positions = _random_compact(size, 1.0, rng)
        try:
            best_energy, best_positions = relax(best_positions, potential, kind)
        except:
            pass
    
    # Ensure minimum distance constraint
    if best_positions is not None:
        for i in range(len(best_positions)):
            for j in range(i+1, len(best_positions)):
                d = np.linalg.norm(best_positions[i] - best_positions[j])
                if d < 0.35:
                    # Push atoms apart
                    if d > 1e-10:
                        direction = (best_positions[i] - best_positions[j]) / d
                        shift = (0.35 - d) / 2 + 0.01
                        best_positions[i] += direction * shift
                        best_positions[j] -= direction * shift
                    else:
                        best_positions[i] += np.array([0.2, 0, 0])
                        best_positions[j] -= np.array([0.2, 0, 0])
    
    return best_positions
