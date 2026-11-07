"""Sutton-Chen cluster global minima search using Basin Hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _fib_sphere(n_pts, radius):
    """Generate points on a sphere using Fibonacci lattice."""
    pts = []
    golden = (1 + np.sqrt(5)) / 2
    for i in range(n_pts):
        z = 1 - 2 * (i + 0.5) / n_pts
        r = np.sqrt(1 - z*z)
        theta = 2 * np.pi * i / golden
        pts.append([r * np.cos(theta), r * np.sin(theta), z])
    return np.array(pts) * radius


def _generate_initial_configs(n, kind):
    """Generate diverse initial configurations for basin hopping."""
    configs = []
    
    # 1. Fibonacci sphere (spherical shell)
    # Estimate radius based on n and typical packing density
    # For n atoms, volume ~ n * (d_min)^3. d_min ~ 1.
    # R ~ (3n/4pi)^(1/3) * d_min
    R = 0.8 * n ** (1.0/3.0)
    
    # Add some atoms inside to make it compact
    if n <= 12:
        # Small clusters: icosahedron-like or just sphere
        if n == 1:
            configs.append(np.zeros((1, 3)))
        else:
            # Place on sphere
            configs.append(_fib_sphere(n, R))
    else:
        # For larger n, mix shell and core
        # Let's try a few different radii/distributions
        configs.append(_fib_sphere(n, R))
        
        # 2. Compact random (Lloyd-like iteration is expensive, so just random within sphere)
        rng = np.random.default_rng(42)
        pts = rng.uniform(-R, R, (n, 3))
        # Ensure no overlaps initially? Relax will fix it, but too close can cause issues.
        # Simple check: if any distance < 0.35, perturb.
        # Actually, relax handles it, but let's add a second random config
        configs.append(pts)
        
        # 3. FCC fragment
        a = 0.72
        side = int(np.ceil((n / 4) ** (1/3))) + 2
        candidates = []
        basis = [
            [0, 0, 0],
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.5, 0.5, 0],
        ]
        for i in range(-side, side + 1):
            for j in range(-side, side + 1):
                for k in range(-side, side + 1):
                    for b in basis:
                        p = np.array([i * a + b[0] * a, j * a + b[1] * a, k * a + b[2] * a])
                        candidates.append(p)
        candidates = np.array(candidates)
        center = candidates.mean(axis=0)
        candidates -= center
        dists = np.linalg.norm(candidates, axis=1)
        idx = np.argsort(dists)[:n]
        configs.append(candidates[idx])
        
        # 4. Icosahedral-like (Mackay shells approximation)
        # Shell 1: 12 atoms (icosahedron vertices)
        # Shell 2: 42 atoms
        # Shell 3: 92 atoms
        # Just use fib spheres for shells
        shell1_r = 1.0
        shell2_r = 1.8
        shell3_r = 2.6
        
        pts = []
        if n >= 1:
            pts.append(np.zeros(3))
        remaining = n - 1
        
        if remaining > 0:
            n1 = min(12, remaining)
            pts.extend(_fib_sphere(n1, shell1_r))
            remaining -= n1
            
        if remaining > 0:
            n2 = min(42, remaining)
            pts.extend(_fib_sphere(n2, shell2_r))
            remaining -= n2
            
        if remaining > 0:
            n3 = min(92, remaining)
            pts.extend(_fib_sphere(n3, shell3_r))
            remaining -= n3
            
        if remaining > 0:
            # Fill rest on a larger shell
            pts.extend(_fib_sphere(remaining, 3.4))
            
        configs.append(np.array(pts[:n]))

    return configs


def _basin_hopping(positions, potential, kind, max_iters, rng, time_limit, start_time):
    """Run Basin Hopping optimization."""
    n = len(positions)
    
    # Initial relaxation
    try:
        current_energy, current_pos = relax(positions, potential, kind)
    except:
        # Fallback if relax fails
        current_energy, _ = sutton_chen_energy_gradient(positions, kind)
        current_pos = positions.copy()
        
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    # Kick parameters
    # Start with larger kicks, reduce over time
    kick_scale_start = 0.5
    kick_scale_end = 0.1
    
    for i in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Determine kick scale (linear annealing)
        progress = i / max_iters
        kick_scale = kick_scale_start - (kick_scale_start - kick_scale_end) * progress
        
        # Generate kick
        # Randomly select a subset of atoms to kick? Or all?
        # Kicking all atoms with small noise is safer to maintain compactness.
        # Kicking a subset can break the cluster into pieces, which is good for exploration
        # but might lead to non-compact minima. Let's kick all atoms with small noise.
        
        # Alternative: "Repulsive" move - pick a random atom and move it away from neighbors.
        # Simpler: Gaussian noise on all atoms.
        delta = rng.normal(0, kick_scale, (n, 3))
        new_pos = current_pos + delta
        
        # Relax the new position
        try:
            new_energy, new_pos_relaxed = relax(new_pos, potential, kind)
        except:
            # If relaxation fails, skip
            continue
            
        # Accept if better (pure downhill) or with Metropolis
        # For global minima, pure downhill on the basin landscape is standard BH.
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos_relaxed
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
                
        # Optional: Metropolis acceptance to avoid getting stuck in a local minimum of the basin landscape
        # if new_energy < best_energy + 0.1: # Small tolerance
        #     if rng.random() < np.exp(-(new_energy - current_energy) / 0.05):
        #         current_energy = new_energy
        #         current_pos = new_pos_relaxed
        #     # else: keep current, but best is still tracked
        
    return best_energy, best_pos


def construct(instance):
    """Construct Sutton-Chen cluster using Basin Hopping."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance["kind"]
    
    start_time = time.time()
    time_limit = 250.0  # Leave buffer for 300s total
    
    # Generate initial configs
    inits = _generate_initial_configs(size, kind)
    
    best_energy_global = np.inf
    best_pos_global = None
    
    # Run BH on each initial config
    # Allocate time based on number of inits
    n_inits = len(inits)
    time_per_init = time_limit / n_inits
    
    for i, init in enumerate(inits):
        time_remaining = time_limit - (time.time() - start_time)
        if time_remaining < 5: # Need at least 5s for a few relaxations
            break
            
        # Estimate max iters based on time
        # Assume ~0.1s per relax for n=80.
        # So ~10 iters per second.
        # Let's be conservative: 5 iters per second.
        max_iters = int(time_remaining * 5)
        if max_iters < 10:
            max_iters = 10
            
        rng = np.random.default_rng(12345 + i)
        
        try:
            energy, pos = _basin_hopping(init, potential, kind, max_iters, rng, time_limit, start_time)
            if energy < best_energy_global:
                best_energy_global = energy
                best_pos_global = pos
        except:
            continue
            
    if best_pos_global is None:
        # Fallback
        rng = np.random.default_rng(size)
        radius = 0.8 * size ** (1/3)
        best_pos_global = rng.uniform(-radius, radius, (size, 3))
        try:
            best_pos_global = relax(best_pos_global, potential, kind)[1]
        except:
            pass

    return best_pos_global
# EVOLVE-BLOCK-END
