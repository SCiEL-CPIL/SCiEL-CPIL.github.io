"""Rewritten Dzugutov cluster global-minima search with Poly-Tetrahedral Seeds and Multi-Scale Basin Hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
_TOTAL_BUDGET = 295.0  # Safety margin below 330s kill limit

def _remaining_time():
    return _TOTAL_BUDGET - (time.time() - _START_TIME)

def _icosahedron_vertices(scale=1.0):
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
        [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
        [phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, 1.0], [-phi, 0.0, -1.0],
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / norms * scale
    return v

def _tetrahedron_vertices(scale=1.0):
    v = np.array([
        [1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / norms * scale
    return v

def _generate_seed(n, seed_type, rng):
    """
    Generate a seed configuration for n atoms.
    seed_type: 'icosahedron', 'tetrahedron', 'bcc', 'fcc', 'random'
    """
    if seed_type == 'icosahedron':
        return _make_icosahedral_seed(n, rng)
    elif seed_type == 'tetrahedron':
        return _make_polytetrahedral_seed(n, rng)
    elif seed_type == 'bcc':
        return _make_bcc_seed(n, rng)
    elif seed_type == 'fcc':
        return _make_fcc_seed(n, rng)
    else:
        return _make_random_seed(n, rng)

def _make_icosahedral_seed(n, rng):
    """
    Create an icosahedral seed.
    For n=55, this is the Mackay icosahedron.
    For other n, we add or remove atoms from the surface.
    """
    if n == 55:
        return _mackay_icosahedron_55(spacing=1.0)
    
    # Start with 55-atom Mackay icosahedron
    base = _mackay_icosahedron_55(spacing=1.0)
    
    if n < 55:
        # Remove atoms from the outer shell (indices 13 to 54)
        outer_idx = np.arange(13, 55)
        to_remove = rng.choice(outer_idx, size=55-n, replace=False)
        mask = np.ones(55, dtype=bool)
        mask[to_remove] = False
        return base[mask]
    else:
        # Add atoms to the surface
        pts = [base]
        num_add = n - 55
        for _ in range(num_add):
            # Pick a random direction from the center
            dir = rng.normal(0, 1, 3)
            dir /= np.linalg.norm(dir)
            # Place slightly outside the second shell (radius ~ 1.618)
            r = 1.618 * (1.0 + rng.uniform(0.05, 0.2))
            pts.append(dir * r)
        return np.vstack(pts)

def _mackay_icosahedron_55(spacing=1.0):
    pts = [np.zeros(3)]
    shell1 = _icosahedron_vertices(scale=spacing)
    pts.extend(shell1)
    
    shell2_verts = _icosahedron_vertices(scale=spacing * 1.618)
    pts.extend(shell2_verts)
    
    d_edge = np.linalg.norm(shell2_verts[0] - shell2_verts[1])
    midpoints = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(shell2_verts[i] - shell2_verts[j])
            if np.isclose(d, d_edge, rtol=1e-3):
                mid = (shell2_verts[i] + shell2_verts[j]) / 2.0
                midpoints.append(mid)
    pts.extend(midpoints)
    return np.array(pts)

def _make_polytetrahedral_seed(n, rng):
    """
    Create a poly-tetrahedral seed.
    The Dzugutov potential favors tetrahedral coordination.
    We build a cluster by adding tetrahedra.
    """
    pts = [np.zeros(3)]
    # Add first tetrahedron (3 atoms, since center is already there)
    tet1 = _tetrahedron_vertices(scale=1.0)
    pts.extend(tet1[:3])
    
    while len(pts) < n:
        # Pick a random existing atom
        i = rng.choice(len(pts))
        # Pick a random direction
        dir = rng.normal(0, 1, 3)
        dir /= np.linalg.norm(dir)
        # Place new atom at distance ~1.0
        new_pt = np.array(pts[i]) + dir * 1.0
        pts.append(new_pt)
        
        # Check if we can add more atoms to form a tetrahedron with new_pt and some neighbors
        # This is a simplified growth algorithm
        # For a more robust polytetrahedral structure, we might need a more complex logic
        # But for seeding, random growth is okay.
        
    return np.array(pts[:n])

def _make_bcc_seed(n, rng):
    """
    Create a BCC seed.
    """
    pts = []
    # Generate BCC lattice points within a sphere
    # BCC lattice: (0,0,0) and (0.5, 0.5, 0.5) in cubic unit cells
    # We'll generate points in a grid and filter by distance
    max_r = 2.0
    grid = np.arange(-max_r, max_r+0.5, 0.5)
    for x in grid:
        for y in grid:
            for z in grid:
                p = np.array([x, y, z])
                if np.linalg.norm(p) <= max_r:
                    pts.append(p)
                # Add body center if inside
                p_center = p + np.array([0.25, 0.25, 0.25]) # Approximation for BCC
                # Actually, BCC is corner + body center of cube.
                # Let's do it properly.
                
    # Proper BCC generation
    pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                p1 = np.array([i, j, k])
                if np.linalg.norm(p1) <= 2.0:
                    pts.append(p1)
                p2 = np.array([i+0.5, j+0.5, k+0.5])
                if np.linalg.norm(p2) <= 2.0:
                    pts.append(p2)
    
    pts = np.array(pts)
    # Shuffle and take first n
    idx = rng.permutation(len(pts))
    return pts[idx[:n]]

def _make_fcc_seed(n, rng):
    """
    Create an FCC seed.
    """
    pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                # FCC lattice points: corners + face centers
                p1 = np.array([i, j, k])
                if np.linalg.norm(p1) <= 2.0:
                    pts.append(p1)
                # Face centers
                p2 = np.array([i+0.5, j+0.5, k])
                if np.linalg.norm(p2) <= 2.0:
                    pts.append(p2)
                p3 = np.array([i+0.5, j, k+0.5])
                if np.linalg.norm(p3) <= 2.0:
                    pts.append(p3)
                p4 = np.array([i, j+0.5, k+0.5])
                if np.linalg.norm(p4) <= 2.0:
                    pts.append(p4)
    
    pts = np.array(pts)
    idx = rng.permutation(len(pts))
    return pts[idx[:n]]

def _make_random_seed(n, rng):
    """
    Create a random compact seed.
    """
    radius = 1.5 * (n / 55.0) ** (1.0/3.0)
    pts = rng.uniform(-radius, radius, (n, 3))
    return pts

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    t_start = time.time()
    # Allocate time for this instance.
    # We have 6 instances. Total budget 295s.
    # Let's assume equal distribution but with a safety margin.
    # If this is the first instance, we have ~295s.
    # If it's the last, we have less.
    # To be safe, we cap the time per instance.
    # 295 / 6 = ~49s per instance.
    # Let's use 45s per instance to be safe.
    instance_budget = 45.0
    # But if we are the last instance, we might have more time left?
    # No, the function is called sequentially.
    # The evaluator calls construct() for each n.
    # So we should use a fixed budget per call to ensure we finish all of them.
    # However, if we finish early, we can use the remaining time.
    # But we don't know if there are more instances.
    # The task says "called once for each n in (38, 45, 50, 55, 57, 60)".
    # So there are exactly 6 calls.
    # We can track the number of calls? No, the function is stateless.
    # But we can use a global variable.
    
    global _CALL_COUNT
    if '_CALL_COUNT' not in globals():
        _CALL_COUNT = 0
    _CALL_COUNT += 1
    
    # If we are the first call, we have 295s.
    # If we are the last, we have 295 - 5*45 = 70s?
    # Let's just use a fixed 45s per instance.
    # 6 * 45 = 270s. 295 - 270 = 25s buffer.
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse seeds
    seed_types = ['icosahedron', 'tetrahedron', 'bcc', 'fcc', 'random']
    seeds = []
    for st in seed_types:
        # Generate 2 seeds of each type
        for _ in range(2):
            rng = np.random.default_rng(np.random.randint(0, 10000))
            seed = _generate_seed(size, st, rng)
            seeds.append(seed)
            
    # Relax all seeds
    for seed in seeds:
        if (time.time() - t_start) > instance_budget:
            break
        try:
            energy, positions = relax(seed, potential, parameter)
        except Exception:
            continue
        if energy < best_energy:
            best_energy = energy
            best_positions = positions
            
    if best_positions is None:
        # Fallback
        seed = _generate_seed(size, 'random', np.random.default_rng(1))
        try:
            _, best_positions = relax(seed, potential, parameter)
        except Exception:
            best_positions = seed
        return best_positions
    
    # Basin Hopping from the best solution
    current_energy = best_energy
    current_pos = best_positions
    bh_rng = np.random.default_rng(size + 100)
    
    # Multi-scale perturbation
    scales = [0.05, 0.1, 0.2, 0.3, 0.5]
    
    while (time.time() - t_start) < instance_budget:
        # Pick a random scale
        scale = scales[bh_rng.choice(len(scales))]
        perturbed = current_pos + bh_rng.normal(0, scale, current_pos.shape)
        
        try:
            energy, positions = relax(perturbed, potential, parameter)
        except Exception:
            continue
            
        # Metropolis acceptance
        if energy < current_energy:
            current_energy = energy
            current_pos = positions
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        else:
            # Accept with probability exp(-(E_new - E_old)/T)
            # T should be small to avoid getting stuck in high energy states
            T = 0.05
            if bh_rng.random() < np.exp(-(energy - current_energy) / T):
                current_energy = energy
                current_pos = positions
                
    return best_positions
# EVOLVE-BLOCK-END
