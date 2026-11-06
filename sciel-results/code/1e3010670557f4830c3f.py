"""Dzugutov cluster global-minima search with fragment-merger strategy and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
_TOTAL_BUDGET = 290.0  # Safety margin below 300s

def _remaining_time():
    return _TOTAL_BUDGET - (time.time() - _START_TIME)

def _icosahedron_vertices(scale=1.0):
    """Generate vertices of a regular icosahedron scaled to 'scale'."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
        [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
        [phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, 1.0], [-phi, 0.0, -1.0],
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / norms * scale
    return v

def _make_13_atom_icosahedron(spacing=1.0):
    """Create a 13-atom icosahedron: 1 center + 12 shell atoms."""
    center = np.zeros(3)
    shell = _icosahedron_vertices(scale=spacing)
    return np.vstack([center[None, :], shell])

def _make_mackay_55(spacing=1.0):
    """
    Generate a 55-atom Mackay Icosahedron.
    Structure: 1 center + 12 first shell + 42 second shell.
    """
    pts = [np.zeros(3)]
    
    # Shell 1: 12 atoms at radius 'spacing'
    shell1 = _icosahedron_vertices(scale=spacing)
    pts.extend([p.copy() for p in shell1])
    
    # Shell 2: 42 atoms
    R2 = spacing * ((1.0 + np.sqrt(5.0)) / 2.0)
    shell2_verts = _icosahedron_vertices(scale=R2)
    pts.extend([p.copy() for p in shell2_verts])
    
    # Find edge midpoints of the icosahedron defined by shell2_verts
    d_edge = np.linalg.norm(shell2_verts[0] - shell2_verts[1])
    midpoints = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(shell2_verts[i] - shell2_verts[j])
            if np.isclose(d, d_edge, rtol=1e-3):
                mid = (shell2_verts[i] + shell2_verts[j]) / 2.0
                midpoints.append(mid)
                
    if len(midpoints) == 30:
        pts.extend(midpoints)
    else:
        # Fallback: add random points near the surface
        for _ in range(42 - len(pts) + 1):
            dir = np.random.normal(0, 1, 3)
            dir = dir / np.linalg.norm(dir)
            p = dir * R2
            pts.append(p)
            
    pts = np.array(pts, dtype=float)
    return pts[:55]

def _fragment_merger_seed(n, rng, base_spacing=1.0):
    """
    Fragment-merger strategy: build cluster from 13-atom icosahedral fragments,
    then add/remove atoms to reach target size n.
    """
    fragments = []
    current_count = 0
    
    # Determine how many 13-atom fragments we need
    # For n around 50-60, use 4-5 fragments
    # For n around 38-45, use 3-4 fragments
    
    if n <= 13:
        return _make_13_atom_icosahedron(base_spacing)[:n]
    
    num_fragments = max(1, (n + 12) // 13)
    
    # Place fragments with some overlap to form a connected cluster
    # Use icosahedral directions for fragment placement
    directions = _icosahedron_vertices(scale=1.0)
    
    for i in range(num_fragments):
        if current_count >= n:
            break
            
        # Place fragment center at a position based on direction
        if i == 0:
            center_pos = np.zeros(3)
        else:
            # Place subsequent fragments near existing ones
            # Use a direction from the icosahedron set
            dir_idx = i % 12
            dir_vec = directions[dir_idx]
            
            # Distance between fragment centers: overlap slightly
            # 13-atom icosahedron has radius ~spacing, so place centers ~1.5*spacing apart
            dist = base_spacing * 1.3 * rng.uniform(0.9, 1.1)
            center_pos = dir_vec * dist * (i / max(1, num_fragments - 1) if num_fragments > 1 else 0)
            
            # Add some randomness
            center_pos += rng.normal(0, 0.1, 3)
        
        # Create fragment
        frag = _make_13_atom_icosahedron(base_spacing * rng.uniform(0.95, 1.05))
        frag = frag + center_pos
        
        # Only add atoms if we haven't exceeded n
        remaining = n - current_count
        if remaining >= 13:
            fragments.append(frag)
            current_count += 13
        else:
            # Add partial fragment
            fragments.append(frag[:remaining])
            current_count += remaining
            
    if not fragments:
        # Fallback
        return _make_mackay_55(base_spacing)[:n] if n <= 55 else _make_mackay_55(base_spacing)
    
    # Combine fragments
    all_pts = np.vstack(fragments)
    
    # If we have more atoms than needed, trim
    if len(all_pts) > n:
        # Remove outermost atoms
        all_pts = all_pts[:n]
    elif len(all_pts) < n:
        # Add random atoms near the surface
        R = base_spacing * 1.5
        while len(all_pts) < n:
            dir = rng.normal(0, 1, 3)
            dir = dir / np.linalg.norm(dir)
            p = dir * R * rng.uniform(0.8, 1.2)
            all_pts = np.vstack([all_pts, p[None, :]])
            
    return all_pts[:n]

def _make_seed(n, rng, seed_type=None):
    """
    Generate a diverse seed for n atoms.
    """
    if seed_type is None:
        seed_type = rng.integers(0, 4)
    
    if seed_type == 0:
        # Mackay icosahedron based
        base = _make_mackay_55(1.0)
        if n <= 55:
            # Remove atoms from outer shell
            outer_indices = np.arange(13, 55)
            core_indices = np.arange(0, 13)
            num_remove = 55 - n
            if num_remove <= 42:
                to_remove = rng.choice(outer_indices, size=num_remove, replace=False)
            else:
                to_remove_outer = rng.choice(outer_indices, size=42, replace=False)
                remaining = num_remove - 42
                to_remove_core = rng.choice(core_indices, size=remaining, replace=False)
                to_remove = np.concatenate([to_remove_outer, to_remove_core])
            keep_mask = np.ones(55, dtype=bool)
            keep_mask[to_remove] = False
            return base[keep_mask]
        else:
            # Add atoms
            num_add = n - 55
            new_pts = []
            R2 = 1.0 * ((1.0 + np.sqrt(5.0)) / 2.0)
            for _ in range(num_add):
                dir = rng.normal(0, 1, 3)
                dir = dir / np.linalg.norm(dir)
                r = R2 * (1.0 + rng.uniform(0.0, 0.3))
                p = dir * r
                new_pts.append(p)
            return np.vstack([base, np.array(new_pts)])
            
    elif seed_type == 1:
        # Fragment merger
        return _fragment_merger_seed(n, rng)
        
    elif seed_type == 2:
        # Random ball with icosahedral bias
        pts = [np.zeros(3)]
        for _ in range(n - 1):
            # Bias towards icosahedral directions
            if rng.random() < 0.5:
                dir = _icosahedron_vertices(1.0)[rng.integers(0, 12)]
            else:
                dir = rng.normal(0, 1, 3)
                dir = dir / np.linalg.norm(dir)
            r = rng.uniform(0.5, 2.5)
            pts.append(dir * r)
        return np.array(pts)
        
    else:
        # Pure random
        return rng.uniform(-2, 2, (n, 3))

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    t0 = time.time()
    best_energy = np.inf
    best_positions = None
    
    rng = np.random.default_rng(42 + size)
    
    # Phase 1: Generate and relax diverse seeds
    num_seeds = 20
    seeds = []
    
    for i in range(num_seeds):
        if _remaining_time() < 10.0:
            break
        seed = _make_seed(size, rng)
        seeds.append(seed)
        
    relaxed_seeds = []
    for seed in seeds:
        if _remaining_time() < 8.0:
            break
        try:
            energy, positions = relax(seed, potential, parameter)
            if np.all(np.isfinite(energy)) and np.all(np.isfinite(positions)):
                # Check minimum distance
                min_dist = np.inf
                for i in range(len(positions)):
                    for j in range(i+1, len(positions)):
                        d = np.linalg.norm(positions[i] - positions[j])
                        min_dist = min(min_dist, d)
                if min_dist >= 0.45:  # Slightly relaxed constraint for flexibility
                    relaxed_seeds.append((energy, positions))
        except Exception:
            continue
            
    if not relaxed_seeds:
        # Fallback
        seed = _make_seed(size, rng, seed_type=0)
        try:
            _, best_positions = relax(seed, potential, parameter)
        except Exception:
            best_positions = _make_seed(size, rng, seed_type=0)
        return best_positions

    # Sort by energy
    relaxed_seeds.sort(key=lambda x: x[0])
    best_energy, best_positions = relaxed_seeds[0]
    
    # Phase 2: Basin Hopping from top seeds
    bh_rng = np.random.default_rng(123 + size)
    
    top_seeds = relaxed_seeds[:min(5, len(relaxed_seeds))]
    
    for seed_energy, seed_pos in top_seeds:
        if _remaining_time() < 5.0:
            break
            
        current_energy = seed_energy
        current_pos = seed_pos.copy()
        local_best_energy = seed_energy
        local_best_pos = seed_pos.copy()
        
        scale = 0.15
        T = 0.05
        stuck_count = 0
        
        while _remaining_time() > 5.0:
            # Perturb
            perturbed = current_pos + bh_rng.normal(0, scale, current_pos.shape)
            
            try:
                energy, positions = relax(perturbed, potential, parameter)
            except Exception:
                continue
                
            if not np.all(np.isfinite(energy)):
                continue
                
            # Metropolis acceptance
            accept = False
            if energy < current_energy:
                accept = True
            else:
                delta_e = energy - current_energy
                if delta_e < 10.0:  # Cap to avoid overflow
                    if bh_rng.random() < np.exp(-delta_e / T):
                        accept = True
                        
            if accept:
                current_energy = energy
                current_pos = positions.copy()
                stuck_count = 0
                
                if energy < local_best_energy:
                    local_best_energy = energy
                    local_best_pos = positions.copy()
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = positions.copy()
            else:
                stuck_count += 1
                # Adjust scale based on stagnation
                if stuck_count % 20 == 0:
                    scale *= 0.9
                    if scale < 0.05:
                        scale = 0.05
                    T *= 0.95
                    if T < 0.01:
                        T = 0.01
                
                # Occasionally jump to a different seed
                if stuck_count > 50 and len(top_seeds) > 1:
                    other_seed = top_seeds[bh_rng.integers(0, len(top_seeds))]
                    current_energy = other_seed[0]
                    current_pos = other_seed[1].copy()
                    stuck_count = 0
                    scale = 0.15
                    T = 0.05
                    
    return best_positions
# EVOLVE-BLOCK-END
