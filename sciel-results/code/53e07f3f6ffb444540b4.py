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
        # Avoid division by zero (shouldn't happen if min distance > 0.35)
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
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)  # Avoid self-interaction
    r2 = np.maximum(r2, 1e-12)
    r = np.sqrt(r2)
    
    # For each atom i, sum over j != i
    repulsive_sum = np.sum(r ** (-n_exp), axis=1)  # (n,)
    attractive_sum = np.sum(r ** (-m_exp), axis=1)  # (n,)
    
    total = np.sum(0.5 * repulsive_sum - c * np.sqrt(np.maximum(attractive_sum, 0)))
    return total


def _grid_seed_core(n_atoms, n_exp, m_exp, c, spacing=0.72):
    """
    Build a dense low-energy core by grid-searching positions of the first few atoms,
    then greedily inserting remaining atoms at positions that minimize incremental energy.
    """
    rng = np.random.default_rng(42)
    
    # Start with a small core (2-4 atoms) found via grid/coarse search
    core_size = min(4, n_atoms)
    
    if core_size == 1:
        positions = np.zeros((1, 3))
    elif core_size == 2:
        # Optimal 2-atom distance for Sutton-Chen
        # For 12-6: r_min where 12/r^13 = 6*c^2/r^7 * ... actually let's just use spacing
        d = spacing * 1.0
        positions = np.array([[d/2, 0, 0], [-d/2, 0, 0]])
    else:
        # Try several compact configurations for the core
        best_core = None
        best_core_e = np.inf
        
        # FCC-like triangular arrangements
        d = spacing * 1.0
        candidates = []
        
        # Linear
        candidates.append(np.array([[i * d for i in range(core_size)], [0]*core_size, [0]*core_size]).T)
        # Triangle (for 3+)
        if core_size >= 3:
            candidates.append(np.array([
                [0, 0, 0],
                [d, 0, 0],
                [d/2, d*np.sqrt(3)/2, 0]
            ])[:core_size])
        # Tetrahedral-ish (for 4)
        if core_size >= 4:
            a = d * 0.9
            candidates.append(np.array([
                [a, a, a],
                [a, -a, -a],
                [-a, a, -a],
                [-a, -a, a]
            ])[:core_size])
        # Square
        if core_size >= 4:
            s = d * 0.95
            candidates.append(np.array([
                [s, s, 0], [s, -s, 0], [-s, -s, 0], [-s, s, 0]
            ])[:core_size])
        # Random compact
        for _ in range(20):
            pts = rng.uniform(-d, d, (core_size, 3))
            # Ensure min distance
            valid = True
            for i in range(core_size):
                for j in range(i+1, core_size):
                    if np.linalg.norm(pts[i] - pts[j]) < 0.4:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                candidates.append(pts)
        
        for cand in candidates:
            if len(cand) < core_size:
                continue
            # Check min distance
            min_d = np.inf
            for i in range(core_size):
                for j in range(i+1, core_size):
                    min_d = min(min_d, np.linalg.norm(cand[i] - cand[j]))
            if min_d < 0.35:
                continue
            e = _sc_energy_fast(cand, n_exp, m_exp, c)
            if e < best_core_e:
                best_core_e = e
                best_core = cand.copy()
        
        if best_core is None:
            # Fallback: simple linear
            positions = np.array([[i * d for i in range(core_size)], [0]*core_size, [0]*core_size]).T
        else:
            positions = best_core
    
    # Now greedily insert remaining atoms
    current = positions.copy()
    current_e = _sc_energy_fast(current, n_exp, m_exp, c)
    
    for _ in range(n_atoms - core_size):
        # Try inserting a new atom at various candidate positions
        best_new_pos = None
        best_new_e = np.inf
        
        # Candidate positions: near existing atoms at optimal distance, and on a shell
        d_opt = spacing * 1.0
        candidates = []
        
        # Near each existing atom
        for i in range(len(current)):
            # 8 directions around atom i
            for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1),
                               (1,1,0),(1,-1,0),(-1,1,0),(-1,-1,0),(1,0,1),(1,0,-1),
                               (-1,0,1),(-1,0,-1),(0,1,1),(0,1,-1),(0,-1,1),(0,-1,-1)]:
                v = np.array([dx, dy, dz], dtype=float)
                v = v / np.linalg.norm(v) * d_opt
                new_p = current[i] + v
                candidates.append(new_p)
        
        # On a sphere of radius proportional to cluster size
        R = d_opt * (len(current) ** (1/3)) * 0.8
        for theta in np.linspace(0, 2*np.pi, 12, endpoint=False):
            for phi in np.linspace(0, np.pi, 8, endpoint=False):
                x = R * np.sin(phi) * np.cos(theta)
                y = R * np.sin(phi) * np.sin(theta)
                z = R * np.cos(phi)
                candidates.append(np.array([x, y, z]))
        
        # Random positions in bounding box
        bbox = np.max(np.abs(current), axis=0) + d_opt
        for _ in range(30):
            candidates.append(rng.uniform(-bbox, bbox))
        
        # Evaluate each candidate
        for new_p in candidates:
            # Check min distance
            dists = np.linalg.norm(current - new_p, axis=1)
            if np.min(dists) < 0.35:
                continue
            trial = np.vstack([current, new_p])
            e = _sc_energy_fast(trial, n_exp, m_exp, c)
            if e < best_new_e:
                best_new_e = e
                best_new_pos = new_p
        
        if best_new_pos is not None:
            current = np.vstack([current, best_new_pos])
            current_e = best_new_e
        else:
            # Fallback: add at a random far position
            new_p = rng.uniform(-2, 2, 3)
            current = np.vstack([current, new_p])
    
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
    
    # Map kind to parameters if not directly given
    if "n_exp" not in instance or instance.get("n_exp") is None:
        param_map = {
            "12-6": (12, 6, 144.41),
            "9-6": (9, 6, 39.432),
            "10-8": (10, 8, 34.408),
        }
        n_exp, m_exp, c = param_map.get(kind, (12, 6, 144.41))
    
    spacing = 0.72
    
    # Generate multiple diverse initial clusters using different strategies
    best_positions = None
    best_energy = np.inf
    
    # Strategy 1: Grid-seeded core + greedy insertion
    try:
        pos1 = _grid_seed_core(size, n_exp, m_exp, c, spacing)
        pos1_refined, e1 = _refine_with_relax(pos1, potential, kind, n_exp, m_exp, c)
        if e1 < best_energy:
            best_energy = e1
            best_positions = pos1_refined
    except Exception:
        pass
    
    # Strategy 2: Random compact + relax (multiple tries)
    rng = np.random.default_rng(size)
    for trial in range(6):
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
    
    # Strategy 3: Layered growth (add atoms in shells)
    try:
        pos3 = np.zeros((size, 3))
        pos3[0] = [0, 0, 0]
        d = spacing * 1.0
        idx = 1
        # Shell 1: 6 neighbors
        for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            if idx < size:
                pos3[idx] = [dx*d, dy*d, dz*d]
                idx += 1
        # Shell 2: 12 face neighbors
        for dx, dy, dz in [(1,1,0),(1,-1,0),(-1,1,0),(-1,-1,0),
                           (1,0,1),(1,0,-1),(-1,0,1),(-1,0,-1),
                           (0,1,1),(0,1,-1),(0,-1,1),(0,-1,-1)]:
            if idx < size:
                v = np.array([dx, dy, dz], dtype=float)
                pos3[idx] = v / np.linalg.norm(v) * d * 1.4
                idx += 1
        # Fill remaining with random in bounding box
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
        # Last resort fallback
        d = spacing
        best_positions = np.array([[i * d, 0, 0] for i in range(size)])
    
    # Final check: ensure no atoms too close
    # If any pair is too close, nudge them apart
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
