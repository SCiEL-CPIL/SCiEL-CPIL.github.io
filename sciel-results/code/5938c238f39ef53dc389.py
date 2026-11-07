"""Sutton-Chen transition-metal cluster global minima - multi-start with geometric templates."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _get_parameters(kind):
    """Get exponent and c parameters for each kind."""
    params = {
        '12-6': (12, 6, 144.41),
        '9-6': (9, 6, 39.432),
        '10-8': (10, 8, 34.408),
    }
    return params.get(kind, (12, 6, 144.41))


def _fcc_positions(n, spacing=1.0):
    """Generate positions for a truncated FCC cluster with n atoms."""
    # For FCC, the number of atoms in a sphere of radius R with spacing a is approximately:
    # n ≈ (4/3) * π * R^3 / (a^3 * 2) * 4 = (8π/3) * (R/a)^3
    # So R/a ≈ (3n/(8π))^(1/3)
    # But we need exactly n atoms, so we generate all FCC points in a large sphere and take the closest n to center.
    
    # Generate FCC lattice points
    points = []
    max_r = int(np.ceil(spacing * (3 * n / (8 * np.pi)) ** (1/3) * 1.5)) + 2
    
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                # FCC lattice: points at (i,j,k) where i+j+k is even, scaled by spacing/sqrt(2)
                # Actually, standard FCC: conventional cell has atoms at corners and face centers
                # Let's use a simpler approach: generate all integer lattice points and keep FCC subset
                pass
    
    # Simpler: generate points on FCC lattice
    # FCC basis vectors: a1 = (0, 0.5, 0.5)*a, a2 = (0.5, 0, 0.5)*a, a3 = (0.5, 0.5, 0)*a
    # Nearest neighbor distance = a/sqrt(2)
    # We want nearest neighbor distance ≈ 1.0, so a = sqrt(2)
    
    a = spacing * np.sqrt(2)
    a1 = np.array([0, 0.5, 0.5]) * a
    a2 = np.array([0.5, 0, 0.5]) * a
    a3 = np.array([0.5, 0.5, 0]) * a
    
    max_idx = int(np.ceil(spacing * (3 * n / (8 * np.pi)) ** (1/3) * 1.5 / a)) + 2
    
    candidates = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                p = i * a1 + j * a2 + k * a3
                r = np.linalg.norm(p)
                if r <= spacing * (3 * n / (8 * np.pi)) ** (1/3) * 1.3:
                    candidates.append((r, p))
    
    candidates.sort(key=lambda x: x[0])
    return np.array([c[1] for c in candidates[:n]])


def _hcp_positions(n, spacing=1.0):
    """Generate positions for a truncated HCP cluster with n atoms."""
    # HCP lattice: two interpenetrating FCC-like layers
    # Simpler: use hexagonal close packing
    a = spacing  # nearest neighbor distance
    
    # Generate HCP points
    candidates = []
    max_r = int(np.ceil(spacing * (3 * n / (8 * np.pi)) ** (1/3) * 1.5)) + 2
    
    # HCP basis: 
    # Layer 1: hexagonal lattice with spacing a
    # Layer 2: offset by (a/sqrt(3), a/3) in plane, and h/2 in z where h = sqrt(8/3)*a
    h = np.sqrt(8.0/3.0) * a
    
    for layer in range(0, 2 * max_r + 1):
        z = layer * h / 2
        offset = (layer % 2) * np.array([a / np.sqrt(3), a / 3.0])
        
        for i in range(-max_r, max_r + 1):
            for j in range(-max_r, max_r + 1):
                # Hexagonal lattice coordinates
                x = a * (i + 0.5 * j) + offset[0]
                y = a * (np.sqrt(3) / 2 * j) + offset[1]
                p = np.array([x, y, z])
                r = np.linalg.norm(p)
                if r <= spacing * (3 * n / (8 * np.pi)) ** (1/3) * 1.3:
                    candidates.append((r, p))
    
    candidates.sort(key=lambda x: x[0])
    return np.array([c[1] for c in candidates[:n]])


def _random_compact(n, spacing=1.0, seed=42):
    """Generate random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    min_dist = 0.85 * spacing
    
    points = []
    max_attempts = 10000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    
    # If we couldn't place all points, fill remaining randomly
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def _shrink_wrap_anneal(n, kind, seed=0, time_budget=30.0):
    """Shrink-wrap annealing: start with overlapping random sphere, relax with soft-core."""
    rng = np.random.default_rng(seed)
    spacing = 1.0
    radius = spacing * n ** (1.0 / 3.0) * 0.5  # Tightly packed, overlapping
    
    # Generate random points in a small sphere
    points = rng.uniform(-radius, radius, (n, 3))
    
    # Apply soft-core L-BFGS-B relaxation with time-dependent repulsion
    # We'll do multiple rounds of relaxation with decreasing overlap
    
    best_energy = np.inf
    best_positions = points.copy()
    
    start_time = time.time()
    num_rounds = 5
    for round_idx in range(num_rounds):
        if time.time() - start_time > time_budget:
            break
        
        # Add a soft-core repulsive term to prevent atoms from being too close
        # This is done by modifying the energy function during relaxation
        # For simplicity, we'll just do standard relaxation but with perturbed initial conditions
        
        # Perturb the current positions slightly
        noise_scale = 0.3 * (1 - round_idx / num_rounds)
        perturbed = best_positions + rng.normal(0, noise_scale, (n, 3))
        
        try:
            energy, positions = relax(perturbed, 'sutton_chen', kind)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions
        except:
            pass
    
    return best_energy, best_positions


def construct(instance):
    """Multi-start optimization with geometric templates and random compact starts."""
    size = instance["n"]
    kind = instance["kind"]
    
    start_time = time.time()
    deadline = start_time + 280  # 280 seconds, leaving 50s margin
    
    n_exp, m_exp, c = _get_parameters(kind)
    
    # Generate various starting configurations
    starts = []
    
    # FCC template
    try:
        fcc = _fcc_positions(size, spacing=1.0)
        starts.append(('fcc', fcc))
    except:
        pass
    
    # HCP template
    try:
        hcp = _hcp_positions(size, spacing=1.0)
        starts.append(('hcp', hcp))
    except:
        pass
    
    # Random compact starts with different seeds
    for seed in range(6):
        try:
            rc = _random_compact(size, spacing=1.0, seed=seed + 100)
            starts.append((f'random_{seed}', rc))
        except:
            pass
    
    # Shrink-wrap annealing starts
    for seed in range(3):
        try:
            sw_energy, sw_pos = _shrink_wrap_anneal(size, kind, seed=seed, time_budget=20.0)
            if np.isfinite(sw_energy):
                starts.append((f'shrinkwrap_{seed}', sw_pos))
        except:
            pass
    
    # Relax all starts and keep the best
    best_energy = np.inf
    best_positions = None
    
    for name, positions in starts:
        if time.time() > deadline - 30:
            break
        
        try:
            energy, relaxed = relax(positions, 'sutton_chen', kind)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = relaxed
        except:
            pass
    
    # If we still have time, do additional random restarts
    while time.time() < deadline - 10:
        seed = int(time.time() * 1000) % 100000
        try:
            rc = _random_compact(size, spacing=1.0, seed=seed)
            energy, relaxed = relax(rc, 'sutton_chen', kind)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = relaxed
        except:
            break
    
    if best_positions is None:
        # Fallback: return random compact
        best_positions = _random_compact(size, spacing=1.0, seed=0)
    
    return best_positions
# EVOLVE-BLOCK-END
