"""Sutton-Chen cluster global minima: gradient-insertion seeding + basin hopping."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_PARAMS = {
    "12-6": (12, 6, 144.41),
    "9-6":  (9, 6, 39.432),
    "10-8": (10, 8, 34.408),
}

_DEADLINE = 320.0  # wall-clock safety margin (evaluator kills at 330)
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _sc_energy(positions, n_exp, m_exp, c):
    """Compute full Sutton-Chen energy (vectorized)."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r = np.sqrt(r2)
    inv_r = 1.0 / r
    inv_r_n = inv_r ** n_exp
    inv_r_m = inv_r ** m_exp
    
    # Pairwise terms
    pair_rep = inv_r_n  # (n, n)
    pair_att = inv_r_m  # (n, n)
    np.fill_diagonal(pair_rep, 0.0)
    np.fill_diagonal(pair_att, 0.0)
    
    # Cohesive term
    rho_i = np.sum(pair_att, axis=1)
    energy = np.sum(pair_rep) - c * np.sum(np.sqrt(rho_i))
    return energy


def _incremental_energy(new_pos, existing, n_exp, m_exp, c):
    """Energy contribution of adding new_pos to existing cluster."""
    n = len(existing)
    if n == 0:
        return 0.0
    
    # Distances from new atom to all existing
    diff = new_pos - existing
    r2 = np.sum(diff ** 2, axis=1)
    r = np.sqrt(r2)
    inv_r = 1.0 / r
    inv_r_n = inv_r ** n_exp
    inv_r_m = inv_r ** m_exp
    
    # Repulsive contribution from new atom
    rep_new = np.sum(inv_r_n)
    
    # Cohesive: need to update rho for existing atoms
    # rho_i_new = rho_i_old + inv_r_m[i]
    # We don't have old rho, so compute full energy difference
    # E_new - E_old = sum_{j} [inv_r_n - c*(sqrt(rho_j + inv_r_m_j) - sqrt(rho_j))]
    # This is complex; use full energy for simplicity in insertion
    
    # Full energy with new atom
    positions = np.vstack([existing, new_pos.reshape(1, -1)])
    e_full = _sc_energy(positions, n_exp, m_exp, c)
    e_old = _sc_energy(existing, n_exp, m_exp, c)
    return e_full - e_old


def _relax(positions, kind):
    """Relax positions using the provided relax function."""
    try:
        energy, new_pos = relax(positions, "sutton_chen", kind)
        if np.all(np.isfinite(new_pos)):
            return energy, new_pos
    except Exception:
        pass
    return np.inf, positions


def _build_insertion_seed(n, kind, rng, time_budget=10.0):
    """Build a seed by sequentially inserting atoms to minimize incremental energy."""
    n_exp, m_exp, c = _PARAMS[kind]
    
    # Start with a small cluster
    if n <= 2:
        if n == 1:
            return np.array([[0.0, 0.0, 0.0]])
        else:
            return np.array([[0.0, 0.0, 0.0], [0.72, 0.0, 0.0]])
    
    # Start with 2 atoms
    positions = np.array([[0.0, 0.0, 0.0], [0.72, 0.0, 0.0]])
    
    for i in range(2, n):
        if _remaining() < 5.0:
            break
        
        # Try multiple candidate positions
        best_energy = np.inf
        best_pos = None
        n_candidates = 8
        
        for _ in range(n_candidates):
            # Random position near the cluster
            center = np.mean(positions, axis=0)
            radius = 0.72 * len(positions) ** (1.0/3.0)
            candidate = center + rng.uniform(-radius, radius, 3)
            
            # Check minimum distance
            if len(positions) > 0:
                dists = np.linalg.norm(positions - candidate, axis=1)
                if np.min(dists) < 0.35:
                    continue
            
            # Compute incremental energy
            inc_e = _incremental_energy(candidate, positions, n_exp, m_exp, c)
            if inc_e < best_energy:
                best_energy = inc_e
                best_pos = candidate
        
        if best_pos is not None:
            positions = np.vstack([positions, best_pos.reshape(1, -1)])
        else:
            # Fallback: place at a reasonable distance
            center = np.mean(positions, axis=0)
            new_pos = center + rng.normal(0, 0.5, 3)
            positions = np.vstack([positions, new_pos.reshape(1, -1)])
    
    return positions


def _basin_hopping(positions, kind, n_kicks=5, kick_scale=0.3, time_budget=20.0):
    """Simple basin hopping with random kicks."""
    n_exp, m_exp, c = _PARAMS[kind]
    current_energy, current_pos = _relax(positions, kind)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    start_time = time.time()
    for kick in range(n_kicks):
        if time.time() - start_time > time_budget:
            break
        if _remaining() < 10.0:
            break
        
        # Random kick
        kick_vec = np.random.default_rng(kick).normal(0, kick_scale, current_pos.shape)
        kicked = current_pos + kick_vec
        
        # Check minimum distance
        if len(kicked) > 1:
            diff = kicked[:, np.newaxis, :] - kicked[np.newaxis, :, :]
            r2 = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(r2, 1.0)
            min_dist = np.sqrt(np.min(r2[r2 > 0]))
            if min_dist < 0.35:
                continue
        
        # Relax
        new_energy, new_pos = _relax(kicked, kind)
        
        # Metropolis acceptance
        if new_energy < current_energy or np.random.default_rng(kick + 1000).random() < np.exp(-(new_energy - current_energy) / 0.05):
            current_energy = new_energy
            current_pos = new_pos
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos.copy()
    
    return best_energy, best_pos


def construct(instance):
    """Construct optimal Sutton-Chen cluster configuration."""
    size = instance["n"]
    kind = instance["kind"]
    n_exp, m_exp, c = _PARAMS[kind]
    
    # Check deadline
    if _remaining() < 10.0:
        # Emergency: return a simple valid configuration
        positions = np.zeros((size, 3))
        for i in range(size):
            positions[i] = [i * 0.72, (i % 3) * 0.72, (i // 3) * 0.72]
        # Ensure minimum distance
        for i in range(size):
            for j in range(i + 1, size):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < 0.35:
                    positions[j] = positions[i] + 0.4 * np.array([1, 0, 0])
        return positions
    
    rng = np.random.default_rng(size)
    
    # Strategy: Multiple diverse seeds, then basin hopping on the best
    n_seeds = 4
    best_energy = np.inf
    best_pos = None
    
    # Seed 1: Gradient insertion
    if _remaining() > 30.0:
        try:
            seed1 = _build_insertion_seed(size, kind, rng, time_budget=20.0)
            e1, pos1 = _basin_hopping(seed1, kind, n_kicks=3, kick_scale=0.25, time_budget=25.0)
            if e1 < best_energy:
                best_energy = e1
                best_pos = pos1
        except Exception:
            pass
    
    # Seed 2: FCC-like truncation
    if _remaining() > 25.0 and best_pos is None:
        try:
            # Generate FCC points
            fcc_points = []
            a = 0.72
            for i in range(-2, 3):
                for j in range(-2, 3):
                    for k in range(-2, 3):
                        # FCC: (0,0,0) + (a/2)(1,1,0) type
                        for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                            p = a * np.array([i + dx, j + dy, k + dz])
                            if np.linalg.norm(p) < a * 3:
                                fcc_points.append(p)
            
            # Shuffle and take first 'size'
            rng.shuffle(fcc_points)
            if len(fcc_points) >= size:
                seed2 = np.array(fcc_points[:size])
                # Center
                seed2 -= np.mean(seed2, axis=0)
                e2, pos2 = _basin_hopping(seed2, kind, n_kicks=3, kick_scale=0.25, time_budget=25.0)
                if e2 < best_energy:
                    best_energy = e2
                    best_pos = pos2
        except Exception:
            pass
    
    # Seed 3: Random compact
    if _remaining() > 20.0:
        try:
            radius = 0.72 * size ** (1.0/3.0) * 0.8
            points = []
            for _ in range(size):
                for attempt in range(50):
                    p = rng.uniform(-radius, radius, 3)
                    if np.linalg.norm(p) <= radius:
                        if all(np.linalg.norm(p - q) >= 0.36 for q in points):
                            points.append(p)
                            break
            if len(points) == size:
                seed3 = np.array(points)
                e3, pos3 = _basin_hopping(seed3, kind, n_kicks=3, kick_scale=0.3, time_budget=25.0)
                if e3 < best_energy:
                    best_energy = e3
                    best_pos = pos3
        except Exception:
            pass
    
    # Seed 4: Another random
    if _remaining() > 15.0:
        try:
            radius = 0.72 * size ** (1.0/3.0) * 0.7
            points = []
            for _ in range(size):
                for attempt in range(50):
                    p = rng.uniform(-radius, radius, 3)
                    if np.linalg.norm(p) <= radius:
                        if all(np.linalg.norm(p - q) >= 0.36 for q in points):
                            points.append(p)
                            break
            if len(points) == size:
                seed4 = np.array(points)
                e4, pos4 = _basin_hopping(seed4, kind, n_kicks=3, kick_scale=0.3, time_budget=25.0)
                if e4 < best_energy:
                    best_energy = e4
                    best_pos = pos4
        except Exception:
            pass
    
    # Final relaxation on best
    if best_pos is not None and _remaining() > 5.0:
        try:
            final_energy, final_pos = _relax(best_pos, kind)
            if np.all(np.isfinite(final_pos)):
                return final_pos
            else:
                return best_pos
        except Exception:
            return best_pos
    
    # Fallback: simple valid configuration
    positions = np.zeros((size, 3))
    for i in range(size):
        positions[i] = [i * 0.72, (i % 3) * 0.72, (i // 3) * 0.72]
    for i in range(size):
        for j in range(i + 1, size):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < 0.35:
                positions[j] = positions[i] + 0.4 * np.array([1, 0, 0])
    return positions

# EVOLVE-BLOCK-END
