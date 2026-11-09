"""Sutton-Chen transition-metal cluster global minima using Simulated Annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _validate_positions(pos):
    """Check if positions are valid (finite, min distance > 0.35)."""
    if not np.all(np.isfinite(pos)):
        return False
    n = len(pos)
    # Vectorized distance check
    # For n ~ 80, n^2 is 6400, which is fast.
    # We can do this more efficiently by chunking or just using scipy.spatial if available, 
    # but standard numpy is safer for imports.
    # However, for the SA loop, we need this to be very fast.
    # Let's use a simpler check for the SA loop: 
    # We only check the moved atoms against all others, and moved against moved.
    
    # For the final validation, we do a full check.
    dists = np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.35

def _fast_min_dist_check(pos, moved_indices):
    """Check min distance only for moved atoms against all, and moved against moved."""
    if len(moved_indices) == 0:
        return True
    moved = pos[moved_indices]
    others = np.delete(pos, moved_indices, axis=0)
    
    # Moved vs Others
    if len(others) > 0:
        dists_m_o = np.linalg.norm(moved[:, np.newaxis, :] - others[np.newaxis, :, :], axis=2)
        if np.min(dists_m_o) < 0.35:
            return False
            
    # Moved vs Moved
    if len(moved_indices) > 1:
        dists_m_m = np.linalg.norm(moved[:, np.newaxis, :] - moved[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists_m_m, np.inf)
        if np.min(dists_m_m) < 0.35:
            return False
            
    return True

def _generate_fcc_cluster(n, lattice=0.72):
    fcc_offsets = [
        [0, 0, 0],
        [lattice / 2, lattice / 2, 0],
        [lattice / 2, 0, lattice / 2],
        [0, lattice / 2, lattice / 2],
    ]
    size = int(np.ceil(n ** (1.0 / 3.0))) + 2
    all_points = []
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            for k in range(-size, size + 1):
                base = np.array([i * lattice, j * lattice, k * lattice])
                for off in fcc_offsets:
                    all_points.append(base + np.array(off))
    all_points = np.array(all_points)
    dists = np.linalg.norm(all_points, axis=1)
    max_r = 0.9 * n ** (1.0 / 3.0) + 1.0
    mask = dists <= max_r
    all_points = all_points[mask]
    d2 = np.linalg.norm(all_points, axis=1)
    order = np.argsort(d2)
    if len(all_points) >= n:
        return all_points[order[:n]]
    else:
        return _generate_random_cluster(n, seed=42)

def _generate_hcp_cluster(n, lattice=0.72):
    a = lattice
    c = a * np.sqrt(8.0 / 3.0)
    points = []
    size = int(np.ceil(n ** (1.0 / 3.0))) + 2
    for k in range(-size, size + 1):
        z = k * c / 2
        layer = k % 2
        for i in range(-size, size + 1):
            for j in range(-size, size + 1):
                if layer == 0:
                    x = i * a + j * a / 2
                    y = j * a * np.sqrt(3) / 2
                else:
                    x = i * a + j * a / 2 + a / 2
                    y = j * a * np.sqrt(3) / 2 + a * np.sqrt(3) / 6
                points.append([x, y, z])
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    max_r = 0.9 * n ** (1.0 / 3.0) + 1.0
    mask = dists <= max_r
    points = points[mask]
    d2 = np.linalg.norm(points, axis=1)
    order = np.argsort(d2)
    if len(points) >= n:
        return points[order[:n]]
    else:
        return _generate_random_cluster(n, seed=43)

def _generate_icosahedral_cluster(n):
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0])
    points = [np.zeros(3)]
    for v in ico_verts:
        points.append(v.copy())
    n_verts = len(ico_verts)
    shell2 = []
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            d = np.linalg.norm(ico_verts[i] - ico_verts[j])
            if 0.9 < d < 1.1:
                mid = (ico_verts[i] + ico_verts[j]) / 2
                mid = mid / np.linalg.norm(mid) * 1.35
                shell2.append(mid)
    for p in shell2:
        points.append(p)
    shell3 = []
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            for k in range(j + 1, n_verts):
                d1 = np.linalg.norm(ico_verts[i] - ico_verts[j])
                d2 = np.linalg.norm(ico_verts[j] - ico_verts[k])
                d3 = np.linalg.norm(ico_verts[i] - ico_verts[k])
                if all(0.9 < d < 1.1 for d in [d1, d2, d3]):
                    fc = (ico_verts[i] + ico_verts[j] + ico_verts[k]) / 3
                    fc = fc / np.linalg.norm(fc) * 1.7
                    shell3.append(fc)
    for p in shell3:
        points.append(p)
    if len(points) >= n:
        return np.array(points[:n])
    else:
        rng = np.random.default_rng(44)
        while len(points) < n:
            p = rng.normal(size=3)
            p = p / np.linalg.norm(p) * 2.0
            points.append(p)
    return np.array(points[:n])

def _generate_random_cluster(n, seed, min_dist=0.8, max_radius=None):
    rng = np.random.default_rng(seed)
    if max_radius is None:
        max_radius = 0.85 * n ** (1.0 / 3.0)
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 200:
        p = rng.uniform(-max_radius, max_radius, 3)
        if np.linalg.norm(p) <= max_radius:
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    ok = False
                    break
            if ok:
                points.append(p)
        attempts += 1
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-max_radius, max_radius, 3)
            points.append(p)
    return np.array(points)

def _simulated_annealing(pos, kind, deadline, rng):
    """Run SA on the given positions until deadline."""
    n = len(pos)
    current_energy, _ = sutton_chen_energy_gradient(pos, kind)
    if not np.isfinite(current_energy):
        return current_energy, pos
        
    best_energy = current_energy
    best_pos = pos.copy()
    
    # Initial temperature estimate: roughly the width of the energy landscape
    # We use a heuristic based on n
    T = 5.0 
    T_min = 1e-4
    cooling_rate = 0.995
    
    # We will perturb a fraction of atoms
    # For n=80, perturbing 4 atoms is a good move
    num_perturb = max(1, int(0.05 * n))
    
    while time.time() < deadline - 5.0: # Leave 5s for final polish
        # Generate move
        indices = rng.choice(n, size=num_perturb, replace=False)
        # Displacement magnitude proportional to T
        # Heuristic: step size ~ 0.1 * T
        step_size = 0.1 * T 
        if step_size < 0.01:
            step_size = 0.01
            
        old_pos = pos.copy()
        pos[indices] += rng.normal(0, step_size, (num_perturb, 3))
        
        # Check validity (fast check)
        if not _fast_min_dist_check(pos, indices):
            # Revert
            pos[:] = old_pos
            # Don't cool too fast on rejection, but do cool slightly
            T *= cooling_rate
            continue
            
        new_energy, _ = sutton_chen_energy_gradient(pos, kind)
        
        if not np.isfinite(new_energy):
            pos[:] = old_pos
            T *= cooling_rate
            continue
            
        delta_E = new_energy - current_energy
        
        # Metropolis acceptance
        if delta_E < 0:
            accept = True
        else:
            # Accept with probability exp(-delta_E / T)
            # If T is very small, this is 0
            if T < 1e-6:
                accept = False
            else:
                # Avoid overflow
                log_prob = -delta_E / T
                if log_prob > 0:
                    accept = True
                else:
                    accept = (rng.random() < np.exp(log_prob))
                    
        if accept:
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = pos.copy()
        else:
            pos[:] = old_pos
            
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best_energy, best_pos

def construct(instance):
    """
    Global search using Simulated Annealing from diverse structured seeds.
    """
    start_time = time.time()
    # 330s limit. Use 320s as deadline for safety.
    deadline = start_time + 320.0
    
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial configurations
    # We don't relax these first. We use them as starting points for SA.
    # However, starting from a relaxed local minimum is often better for SA 
    # if we are doing "restart SA" or "quenching". 
    # But standard SA from a random/structured config explores more.
    # Let's try a hybrid: Generate structured configs, do a VERY quick relax (10 steps?) 
    # or just use them as is. 
    # Given the time budget, running full SA from scratch is risky.
    # Strategy: 
    # 1. Generate 10 diverse seeds.
    # 2. For each seed, run SA for a short time (e.g., 2s).
    # 3. Keep the best.
    # 4. Run SA on the best for the remaining time.
    
    seeds = []
    # FCC variations
    for lat in [0.65, 0.70, 0.75]:
        seeds.append(_generate_fcc_cluster(n, lattice=lat))
    # HCP variations
    for lat in [0.65, 0.70, 0.75]:
        seeds.append(_generate_hcp_cluster(n, lattice=lat))
    # Icosahedral
    seeds.append(_generate_icosahedral_cluster(n))
    # Random
    seeds.append(_generate_random_cluster(n, seed=100))
    seeds.append(_generate_random_cluster(n, seed=200))
    
    # Filter valid seeds
    valid_seeds = []
    for s in seeds:
        if _validate_positions(s):
            valid_seeds.append(s)
            
    if not valid_seeds:
        # Fallback
        pos = _generate_random_cluster(n, seed=999)
        try:
            _, pos = relax(pos, potential, kind)
        except:
            pass
        return pos
        
    best_energy = np.inf
    best_pos = None
    
    # Phase 1: Quick SA on each seed
    # Allocate ~2s per seed
    time_per_seed = 2.0
    for i, seed in enumerate(valid_seeds):
        if time.time() > deadline - 10.0:
            break
            
        seed_deadline = time.time() + time_per_seed
        try:
            e, p = _simulated_annealing(seed, kind, seed_deadline, rng)
            if np.isfinite(e) and _validate_positions(p):
                if e < best_energy:
                    best_energy = e
                    best_pos = p.copy()
        except Exception:
            continue
            
    if best_pos is None:
        # Fallback to first seed relaxed
        try:
            best_energy, best_pos = relax(valid_seeds[0], potential, kind)
        except:
            best_pos = valid_seeds[0]
            
    # Phase 2: Long SA on the best found
    if time.time() < deadline - 10.0:
        try:
            e, p = _simulated_annealing(best_pos, kind, deadline - 5.0, rng)
            if np.isfinite(e) and _validate_positions(p):
                if e < best_energy:
                    best_energy = e
                    best_pos = p.copy()
        except Exception:
            pass
            
    # Phase 3: Final Polish with L-BFGS-B
    # Use the last few seconds to relax the best SA result
    if time.time() < deadline - 1.0:
        try:
            e, p = relax(best_pos, potential, kind)
            if np.isfinite(e) and _validate_positions(p):
                if e < best_energy:
                    best_energy = e
                    best_pos = p.copy()
        except Exception:
            pass
            
    return best_pos
# EVOLVE-BLOCK-END
