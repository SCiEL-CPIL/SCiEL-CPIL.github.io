"""Coarse-to-fine grid seeding with gradient-based insertion for Sutton-Chen clusters."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _sc_energy(positions, n_exp, m_exp, c):
    """Compute Sutton-Chen energy for a cluster."""
    n = len(positions)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        r2_i = np.sum((positions[i] - positions[np.arange(n) != i]) ** 2, axis=1)
        r2_i = np.maximum(r2_i, 1e-12)
        r_i = np.sqrt(r2_i)
        repulsive = 0.5 * np.sum(r_i ** (-n_exp))
        attractive = c * np.sqrt(np.sum(r_i ** (-m_exp)))
        total += repulsive - attractive
    return total


def _sc_energy_fast(positions, n_exp, m_exp, c):
    """Vectorized Sutton-Chen energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    r = np.sqrt(r2)
    
    repulsive_sum = np.sum(r ** (-n_exp), axis=1)
    attractive_sum = np.sum(r ** (-m_exp), axis=1)
    
    total = np.sum(0.5 * repulsive_sum - c * np.sqrt(np.maximum(attractive_sum, 0)))
    return total


def _grid_search_core(n_core, n_exp, m_exp, c, spacing=0.72, n_grid=4):
    """
    Coarse-to-fine grid search to find low-energy core configurations.
    Starts with diverse candidates and refines around the best.
    """
    rng = np.random.default_rng(42)
    best_core = None
    best_e = np.inf
    
    d = spacing
    candidates = []
    
    # Common low-energy motifs
    if n_core >= 2:
        candidates.append(np.array([[d/2, 0, 0], [-d/2, 0, 0]]))
    if n_core >= 3:
        candidates.append(np.array([
            [0, 0, 0],
            [d, 0, 0],
            [d/2, d*np.sqrt(3)/2, 0]
        ]))
    if n_core >= 4:
        a = d * 0.95
        candidates.append(np.array([
            [a, a, a],
            [a, -a, -a],
            [-a, a, -a],
            [-a, -a, a]
        ]))
    if n_core >= 4:
        s = d * 0.9
        candidates.append(np.array([
            [s, s, 0], [s, -s, 0], [-s, -s, 0], [-s, s, 0]
        ]))
    
    # Random compact configurations
    for _ in range(30):
        pts = rng.uniform(-d, d, (n_core, 3))
        valid = True
        for i in range(n_core):
            for j in range(i+1, n_core):
                if np.linalg.norm(pts[i] - pts[j]) < 0.4:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            candidates.append(pts)
    
    # Evaluate all candidates
    for cand in candidates:
        if len(cand) < n_core:
            continue
        min_d = np.inf
        for i in range(n_core):
            for j in range(i+1, n_core):
                min_d = min(min_d, np.linalg.norm(cand[i] - cand[j]))
        if min_d < 0.35:
            continue
        e = _sc_energy_fast(cand, n_exp, m_exp, c)
        if e < best_e:
            best_e = e
            best_core = cand.copy()
    
    # Coarse-to-fine refinement
    if best_core is not None:
        for refine_round in range(3):
            refine_factor = 0.3 ** refine_round
            for _ in range(15):
                perturbed = best_core + rng.uniform(-d*refine_factor, d*refine_factor, (n_core, 3))
                valid = True
                for i in range(n_core):
                    for j in range(i+1, n_core):
                        if np.linalg.norm(perturbed[i] - perturbed[j]) < 0.35:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                e = _sc_energy_fast(perturbed, n_exp, m_exp, c)
                if e < best_e:
                    best_e = e
                    best_core = perturbed.copy()
    
    if best_core is None:
        best_core = np.array([[i * d for i in range(n_core)], [0]*n_core, [0]*n_core]).T
        best_e = _sc_energy_fast(best_core, n_exp, m_exp, c)
    
    return best_core, best_e


def _gradient_guided_insertion(positions, n_exp, m_exp, c, spacing=0.72):
    """
    Find optimal insertion position for a new atom using energy evaluation
    at candidate sites derived from existing cluster geometry.
    """
    n_current = len(positions)
    d_opt = spacing * 1.0
    
    candidates = []
    
    # Directions from each atom
    directions = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx*dx + dy*dy + dz*dz > 0:
                    v = np.array([dx, dy, dz], dtype=float)
                    directions.append(v / np.linalg.norm(v))
    
    for i in range(n_current):
        for v in directions:
            new_p = positions[i] + v * d_opt
            candidates.append(new_p)
    
    # Spherical shells
    R_base = d_opt * (n_current ** (1/3)) * 0.8
    for shell_factor in [0.7, 1.0, 1.3]:
        R = R_base * shell_factor
        for theta in np.linspace(0, 2*np.pi, 16, endpoint=False):
            for phi in np.linspace(0, np.pi, 10, endpoint=False):
                x = R * np.sin(phi) * np.cos(theta)
                y = R * np.sin(phi) * np.sin(theta)
                z = R * np.cos(phi)
                candidates.append(np.array([x, y, z]))
    
    best_new_pos = None
    best_new_e = np.inf
    
    for new_p in candidates:
        dists = np.linalg.norm(positions - new_p, axis=1)
        if np.min(dists) < 0.35:
            continue
        trial = np.vstack([positions, new_p])
        e = _sc_energy_fast(trial, n_exp, m_exp, c)
        if e < best_new_e:
            best_new_e = e
            best_new_pos = new_p
    
    if best_new_pos is None:
        R = d_opt * (n_current ** (1/3)) * 1.2
        rng = np.random.default_rng(hash(tuple(positions[0].round(4))))
        while True:
            new_p = rng.uniform(-R, R, 3)
            if np.linalg.norm(new_p) <= R:
                dists = np.linalg.norm(positions - new_p, axis=1)
                if np.min(dists) >= 0.35:
                    best_new_pos = new_p
                    best_new_e = _sc_energy_fast(np.vstack([positions, new_p]), n_exp, m_exp, c)
                    break
    
    return best_new_pos, best_new_e


def _grow_cluster(n_atoms, n_exp, m_exp, c, spacing=0.72):
    """
    Build cluster by first finding optimal core via grid search,
    then growing via gradient-guided insertion.
    """
    core_size = min(6, n_atoms)
    core, core_e = _grid_search_core(core_size, n_exp, m_exp, c, spacing)
    
    current = core.copy()
    current_e = core_e
    
    for _ in range(n_atoms - core_size):
        new_pos, new_e = _gradient_guided_insertion(current, n_exp, m_exp, c, spacing)
        if new_pos is not None:
            current = np.vstack([current, new_pos])
            current_e = new_e
        else:
            R = spacing * (len(current) ** (1/3)) * 1.2
            rng = np.random.default_rng(hash(tuple(current[0].round(4))))
            while True:
                p = rng.uniform(-R, R, 3)
                if np.linalg.norm(p) <= R:
                    dists = np.linalg.norm(current - p, axis=1)
                    if np.min(dists) >= 0.35:
                        current = np.vstack([current, p])
                        current_e = _sc_energy_fast(current, n_exp, m_exp, c)
                        break
    
    return current


def _refine_with_relax(positions, potential, kind, n_exp, m_exp, c):
    """Use the provided relax function to polish the cluster."""
    try:
        energy, positions = relax(positions, potential, kind)
        return positions, energy
    except Exception:
        return positions, _sc_energy_fast(positions, n_exp, m_exp, c)


def construct(instance):
    """
    Coarse-to-fine grid search over core atoms, then gradient-guided insertion,
    followed by local relaxation.
    """
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho", "12-6"))
    n_exp = instance["n_exp"]
    m_exp = instance["m_exp"]
    c = instance["c"]
    
    if "n_exp" not in instance or instance.get("n_exp") is None:
        param_map = {
            "12-6": (12, 6, 144.41),
            "9-6": (9, 6, 39.432),
            "10-8": (10, 8, 34.408),
        }
        n_exp, m_exp, c = param_map.get(kind, (12, 6, 144.41))
    
    spacing = 0.72
    
    best_positions = None
    best_energy = np.inf
    
    # Strategy 1: Grid-seeded core + gradient-guided growth
    try:
        pos1 = _grow_cluster(size, n_exp, m_exp, c, spacing)
        pos1_refined, e1 = _refine_with_relax(pos1, potential, kind, n_exp, m_exp, c)
        if e1 < best_energy:
            best_energy = e1
            best_positions = pos1_refined
    except Exception:
        pass
    
    # Strategy 2: Random compact + relax (multiple tries)
    rng = np.random.default_rng(size)
    for trial in range(5):
        radius = spacing * size ** (1.0 / 3.0) * 0.8
        points = []
        attempts = 0
        while len(points) < size and attempts < 10000:
            attempts += 1
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                    points.append(p)
        if len(points) >= size:
            points = points[:size]
            try:
                pos2, e2 = _refine_with_relax(np.array(points), potential, kind, n_exp, m_exp, c)
                if e2 < best_energy:
                    best_energy = e2
                    best_positions = pos2
            except Exception:
                pass
    
    # Strategy 3: Layered growth
    try:
        pos3 = np.zeros((size, 3))
        pos3[0] = [0, 0, 0]
        d = spacing * 1.0
        idx = 1
        for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            if idx < size:
                pos3[idx] = [dx*d, dy*d, dz*d]
                idx += 1
        for dx, dy, dz in [(1,1,0),(1,-1,0),(-1,1,0),(-1,-1,0),
                           (1,0,1),(1,0,-1),(-1,0,1),(-1,0,-1),
                           (0,1,1),(0,1,-1),(0,-1,1),(0,-1,-1)]:
            if idx < size:
                v = np.array([dx, dy, dz], dtype=float)
                pos3[idx] = v / np.linalg.norm(v) * d * 1.4
                idx += 1
        if idx < size:
            R = d * (size ** (1/3)) * 0.9
            while idx < size:
                p = rng.uniform(-R, R, 3)
                if np.linalg.norm(p) <= R:
                    valid = True
                    for i in range(idx):
                        if np.linalg.norm(p - pos3[i]) < 0.4:
                            valid = False
                            break
                    if valid:
                        pos3[idx] = p
                        idx += 1
        pos3 = pos3[:size]
        pos3_refined, e3 = _refine_with_relax(pos3, potential, kind, n_exp, m_exp, c)
        if e3 < best_energy:
            best_energy = e3
            best_positions = pos3_refined
    except Exception:
        pass
    
    if best_positions is None:
        d = spacing
        best_positions = np.array([[i * d, 0, 0] for i in range(size)])
    
    # Final check: ensure no atoms too close
    for i in range(len(best_positions)):
        for j in range(i+1, len(best_positions)):
            diff = best_positions[i] - best_positions[j]
            dist = np.linalg.norm(diff)
            if dist < 0.35:
                direction = diff / max(dist, 1e-10)
                shift = (0.35 - dist) / 2 + 0.01
                best_positions[i] += direction * shift
                best_positions[j] -= direction * shift
    
    return best_positions
# EVOLVE-BLOCK-END
