"""New program: incremental cluster growth + basin hopping + relax refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(pos, rho):
    """Compute Morse energy E = sum_{i<j} exp(rho(1-r))(exp(rho(1-r)) - 2)."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    x = rho * (1.0 - r)
    # Only upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    xij = x[iu]
    e = np.exp(xij) * (np.exp(xij) - 2.0)
    return float(np.sum(e))


def _add_atom_greedy(pos, new_atom, rho):
    """Add atom to cluster, return new energy."""
    new_pos = np.vstack([pos, new_atom.reshape(1, -1)])
    return _morse_energy(new_pos, rho)


def _grow_cluster(seed_pos, n_target, rho, rng, max_iter_per_atom=200):
    """Grow cluster from seed_pos to n_target atoms via greedy insertion."""
    pos = seed_pos.copy()
    n = len(pos)
    # Estimate radius
    radius = 1.0 * n ** (1.0 / 3.0)
    
    while n < n_target:
        best_energy = np.inf
        best_pos = None
        best_new_atom = None
        
        # Generate candidate positions for new atom
        n_candidates = max(50, n * 3)
        for _ in range(n_candidates):
            # Try positions on a shell around existing cluster
            if n <= 3:
                # For very small clusters, try random positions
                cand = rng.normal(0, radius * 0.5, 3)
            else:
                # Place near existing atoms at equilibrium distance
                idx = rng.integers(0, n)
                anchor = pos[idx]
                direction = rng.normal(0, 1, 3)
                direction /= (np.linalg.norm(direction) + 1e-10)
                # Try multiple distances
                for dist in [0.9, 1.0, 1.1, 1.2, 1.3, 1.5]:
                    cand = anchor + direction * dist
                    # Check minimum distance
                    if np.any(np.linalg.norm(cand - pos, axis=-1) < 0.5):
                        break
                    else:
                        break
                
                # Also try random positions
                if _ is n_candidates // 2:
                    cand = rng.normal(0, radius * 0.8, 3)
            
            # Check minimum distance constraint
            if np.any(np.linalg.norm(cand - pos, axis=-1) < 0.5):
                continue
            
            energy = _add_atom_greedy(pos, cand, rho)
            if energy < best_energy:
                best_energy = energy
                best_new_atom = cand
        
        if best_new_atom is None:
            # Fallback: place atom at random position far away
            best_new_atom = rng.normal(0, radius * 1.5, 3)
        
        pos = np.vstack([pos, best_new_atom.reshape(1, -1)])
        n += 1
    
    return pos


def _basin_hopping_step(pos, rho, rng, step_size=0.5, n_steps=50):
    """Perturb and re-relax to escape local minima."""
    perturbed = pos + rng.normal(0, step_size, pos.shape)
    # Ensure minimum distance
    for i in range(len(perturbed)):
        for j in range(i + 1, len(perturbed)):
            d = np.linalg.norm(perturbed[i] - perturbed[j])
            if d < 0.5:
                # Push apart
                mid = (perturbed[i] + perturbed[j]) / 2
                diff = perturbed[i] - perturbed[j]
                if np.linalg.norm(diff) < 1e-10:
                    diff = rng.normal(0, 1, 3)
                diff /= np.linalg.norm(diff)
                perturbed[i] = mid + diff * 0.25
                perturbed[j] = mid - diff * 0.25
    
    energy, relaxed = relax(perturbed, 'morse', rho)
    return energy, relaxed


def construct(instance):
    """Incremental growth + basin hopping + relax."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rho = parameter
    
    start_time = time.time()
    time_budget = 45.0  # seconds per instance (6 instances * 45 = 270 < 300)
    
    rng = np.random.default_rng(42 + size)
    
    # Generate diverse seed clusters
    def make_seed(n_seed):
        """Create a small seed cluster."""
        if n_seed <= 1:
            return np.array([[0.0, 0.0, 0.0]])
        if n_seed == 2:
            return np.array([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])
        # Use a simple icosahedral-like arrangement for 3-12 atoms
        if n_seed <= 12:
            # Icosahedron vertices
            phi = (1 + np.sqrt(5)) / 2
            ico = np.array([
                [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
                [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
                [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1]
            ], dtype=float)
            ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
            ico = ico[:n_seed] * 1.0
            # Ensure minimum distance
            for i in range(len(ico)):
                for j in range(i + 1, len(ico)):
                    d = np.linalg.norm(ico[i] - ico[j])
                    if d < 0.5:
                        ico[j] = ico[i] + (ico[j] - ico[i]) / d * 0.5
            return ico
        # For larger seeds, use random compact cluster
        points = []
        radius = 1.0 * n_seed ** (1.0 / 3.0) * 0.8
        while len(points) < n_seed:
            p = rng.uniform(-radius, radius, 3)
            if all(np.linalg.norm(p - q) >= 0.6 for q in points):
                points.append(p)
        return np.array(points)
    
    best_energy = np.inf
    best_positions = None
    
    n_seeds = min(6, max(3, int(time_budget / 8)))  # 3-6 seeds
    
    for seed_idx in range(n_seeds):
        if time.time() - start_time > time_budget:
            break
        
        # Choose seed size based on target size
        if size <= 20:
            seed_size = max(3, size // 4)
        elif size <= 50:
            seed_size = max(5, size // 5)
        else:
            seed_size = max(8, size // 6)
        
        seed = make_seed(seed_size)
        
        # Grow cluster
        pos = _grow_cluster(seed, size, rho, rng)
        
        # Refine with relax
        energy, pos = relax(pos, 'morse', rho)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = pos.copy()
        
        # Basin hopping from best position
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time > 5.0:
            n_bh_steps = min(10, int(remaining_time / 3))
            for bh_idx in range(n_bh_steps):
                if time.time() - start_time > time_budget - 2:
                    break
                step_size = 0.3 + 0.2 * rng.random()
                bh_energy, bh_pos = _basin_hopping_step(best_positions, rho, rng, step_size)
                if bh_energy < best_energy:
                    best_energy = bh_energy
                    best_positions = bh_pos.copy()
    
    # Final refinement
    if best_positions is not None:
        final_energy, final_pos = relax(best_positions, 'morse', rho)
        if final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_pos
    
    return best_positions
# EVOLVE-BLOCK-END
