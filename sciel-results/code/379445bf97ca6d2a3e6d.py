"""Dzugutov cluster global-minima search with icosahedral seeds and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
_TOTAL_BUDGET = 295.0  # Safety margin below 330s kill limit


def _remaining_time():
    return _TOTAL_BUDGET - (time.time() - _START_TIME)


def _icosahedron_vertices(scale=1.0):
    """Generate 12 vertices of a regular icosahedron."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
        [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
        [phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, 1.0], [-phi, 0.0, -1.0],
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / norms * scale
    return v


def _octahedron_vertices(scale=1.0):
    """Generate 6 vertices of a regular octahedron."""
    v = np.array([
        [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / norms * scale
    return v


def _build_icosahedral_cluster(n, spacing=1.0, rng=None):
    """
    Build an icosahedral cluster with n atoms.
    Uses concentric icosahedral shells: center (1) + shell1 (12) + shell2 (42) = 55
    For n < 55, trim from outer shells. For n > 55, add to outer shell.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Shell radii for Mackay icosahedron
    # R0 = 0 (center), R1 = spacing, R2 = spacing * 1.618 (golden ratio)
    R1 = spacing
    R2 = spacing * 1.618
    
    pts = [np.zeros(3)]  # center atom
    
    # Shell 1: 12 atoms
    shell1 = _icosahedron_vertices(scale=R1)
    pts.extend([p.copy() for p in shell1])
    
    # Shell 2: 42 atoms (12 vertices + 30 edge midpoints of larger icosahedron)
    shell2_verts = _icosahedron_vertices(scale=R2)
    pts.extend([p.copy() for p in shell2_verts])
    
    # Find edge midpoints of the icosahedron defined by shell2_verts
    # An icosahedron has 30 edges
    d_edge = np.linalg.norm(shell2_verts[0] - shell2_verts[1])
    midpoints = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(shell2_verts[i] - shell2_verts[j])
            if np.isclose(d, d_edge, rtol=1e-3):
                mid = (shell2_verts[i] + shell2_verts[j]) / 2.0
                midpoints.append(mid)
    
    pts.extend(midpoints)
    
    pts = np.array(pts, dtype=float)
    
    if n == len(pts):
        return pts
    
    if n < len(pts):
        # Remove atoms from outer shell (indices 13 to end)
        outer_start = 13
        num_to_remove = len(pts) - n
        if num_to_remove <= len(pts) - outer_start:
            outer_indices = np.arange(outer_start, len(pts))
            to_remove = rng.choice(outer_indices, size=num_to_remove, replace=False)
            keep_mask = np.ones(len(pts), dtype=bool)
            keep_mask[to_remove] = False
            return pts[keep_mask]
        else:
            # Need to remove from shell 1 as well
            to_remove = rng.choice(len(pts), size=num_to_remove, replace=False)
            keep_mask = np.ones(len(pts), dtype=bool)
            keep_mask[to_remove] = False
            return pts[keep_mask]
    
    # n > 55: add atoms at random positions near the surface
    num_add = n - len(pts)
    new_pts = []
    surface_radius = R2 * 1.05
    for _ in range(num_add):
        dir_vec = rng.normal(0, 1, 3)
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        r = surface_radius * (1.0 + rng.uniform(0.0, 0.2))
        p = dir_vec * r
        new_pts.append(p)
    pts = np.vstack([pts, np.array(new_pts)])
    return pts


def _build_polytetrahedral_seed(n, spacing=1.0, rng=None):
    """
    Build a polytetrahedral seed by combining icosahedral units.
    For the Dzugutov potential, polytetrahedral packing is optimal.
    """
    if rng is None:
        rng = np.random.default_rng(123)
    
    # Start with a central icosahedron
    pts = [np.zeros(3)]
    shell1 = _icosahedron_vertices(scale=spacing)
    pts.extend([p.copy() for p in shell1])
    pts = np.array(pts, dtype=float)
    
    # Add additional icosahedral units around it
    # Each additional unit shares a face with the central icosahedron
    # This creates polytetrahedral packing
    
    # For simplicity, add atoms at positions that maintain local icosahedral symmetry
    # by placing them at the vertices of adjacent icosahedra
    
    if n <= 13:
        if n == 13:
            return pts
        # Trim from shell 1
        num_keep = n - 1
        shell1_indices = np.arange(1, 13)
        keep_shell1 = rng.choice(shell1_indices, size=num_keep, replace=False)
        keep_mask = np.zeros(13, dtype=bool)
        keep_mask[0] = True
        keep_mask[keep_shell1] = True
        return pts[keep_mask]
    
    # Add second shell atoms for polytetrahedral packing
    # The 42 atoms of the second shell in a Mackay icosahedron
    R2 = spacing * 1.618
    shell2_verts = _icosahedron_vertices(scale=R2)
    pts = np.vstack([pts, shell2_verts])
    
    if n <= len(pts):
        return pts[:n]
    
    # Add edge midpoints
    d_edge = np.linalg.norm(shell2_verts[0] - shell2_verts[1])
    midpoints = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(shell2_verts[i] - shell2_verts[j])
            if np.isclose(d, d_edge, rtol=1e-3):
                mid = (shell2_verts[i] + shell2_verts[j]) / 2.0
                midpoints.append(mid)
    
    pts = np.vstack([pts, np.array(midpoints)])
    
    if n <= len(pts):
        return pts[:n]
    
    # Add more atoms at random positions near the surface
    num_add = n - len(pts)
    new_pts = []
    surface_radius = R2 * 1.05
    for _ in range(num_add):
        dir_vec = rng.normal(0, 1, 3)
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        r = surface_radius * (1.0 + rng.uniform(0.0, 0.2))
        p = dir_vec * r
        new_pts.append(p)
    pts = np.vstack([pts, np.array(new_pts)])
    return pts


def _build_bcc_cluster(n, spacing=1.0, rng=None):
    """Build a BCC-like cluster as an alternative seed."""
    if rng is None:
        rng = np.random.default_rng(999)
    
    pts = [np.zeros(3)]
    # BCC: 8 corners + 1 center of a cube
    half = spacing / 2.0
    corners = np.array([
        [s1 * half, s2 * half, s3 * half]
        for s1 in [-1, 1] for s2 in [-1, 1] for s3 in [-1, 1]
    ])
    pts.extend([p.copy() for p in corners])
    pts = np.array(pts, dtype=float)
    
    if n <= len(pts):
        return pts[:n]
    
    # Add more atoms in a BCC lattice pattern
    num_add = n - len(pts)
    new_pts = []
    for _ in range(num_add):
        # Random position in a cube of side ~ spacing * n^(1/3)
        size = spacing * (n ** (1.0 / 3.0))
        p = rng.uniform(-size / 2, size / 2, 3)
        new_pts.append(p)
    pts = np.vstack([pts, np.array(new_pts)])
    return pts


def _build_fcc_cluster(n, spacing=1.0, rng=None):
    """Build an FCC-like cluster as an alternative seed."""
    if rng is None:
        rng = np.random.default_rng(777)
    
    pts = [np.zeros(3)]
    # FCC: 8 corners + 6 face centers of a cube
    half = spacing / 2.0
    corners = np.array([
        [s1 * half, s2 * half, s3 * half]
        for s1 in [-1, 1] for s2 in [-1, 1] for s3 in [-1, 1]
    ])
    face_centers = np.array([
        [0, half, 0], [0, -half, 0],
        [half, 0, 0], [-half, 0, 0],
        [0, 0, half], [0, 0, -half],
    ])
    pts.extend([p.copy() for p in corners])
    pts.extend([p.copy() for p in face_centers])
    pts = np.array(pts, dtype=float)
    
    if n <= len(pts):
        return pts[:n]
    
    num_add = n - len(pts)
    new_pts = []
    size = spacing * (n ** (1.0 / 3.0))
    for _ in range(num_add):
        p = rng.uniform(-size / 2, size / 2, 3)
        new_pts.append(p)
    pts = np.vstack([pts, np.array(new_pts)])
    return pts


def _generate_seeds(n, rng=None):
    """Generate diverse seeds for basin hopping."""
    if rng is None:
        rng = np.random.default_rng(0)
    
    seeds = []
    
    # Icosahedral seeds with different spacings
    for spacing in [0.9, 1.0, 1.1]:
        seed = _build_icosahedral_cluster(n, spacing=spacing, rng=rng)
        seeds.append(seed)
    
    # Polytetrahedral seeds
    for spacing in [0.95, 1.05]:
        seed = _build_polytetrahedral_seed(n, spacing=spacing, rng=rng)
        seeds.append(seed)
    
    # BCC and FCC seeds for diversity
    seed = _build_bcc_cluster(n, spacing=1.0, rng=rng)
    seeds.append(seed)
    
    seed = _build_fcc_cluster(n, spacing=1.0, rng=rng)
    seeds.append(seed)
    
    # Random compact seed
    size = 1.0 * (n ** (1.0 / 3.0))
    random_seed = rng.uniform(-size / 2, size / 2, (n, 3))
    seeds.append(random_seed)
    
    return seeds


def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    remaining = _remaining_time()
    # Allocate time for this instance
    # 6 instances total, so ~49s each. Use 45s per instance.
    instance_budget = min(45.0, remaining - 5.0)
    if instance_budget < 5.0:
        instance_budget = max(3.0, remaining * 0.15)
    
    t0 = time.time()
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse seeds
    rng = np.random.default_rng(size * 1000 + 42)
    seeds = _generate_seeds(size, rng=rng)
    
    # Phase 1: Relax all seeds and keep the best few
    relaxed_results = []
    for i, seed in enumerate(seeds):
        if (time.time() - t0) > instance_budget * 0.4:
            break
        try:
            energy, positions = relax(seed, potential, parameter)
            if np.isfinite(energy):
                relaxed_results.append((energy, positions))
        except Exception:
            continue
    
    if not relaxed_results:
        # Fallback: return a relaxed icosahedral seed
        seed = _build_icosahedral_cluster(size, spacing=1.0, rng=np.random.default_rng(1))
        try:
            _, best_positions = relax(seed, potential, parameter)
        except Exception:
            best_positions = seed
        return best_positions
    
    # Sort by energy and keep top results
    relaxed_results.sort(key=lambda x: x[0])
    top_results = relaxed_results[:min(5, len(relaxed_results))]
    
    best_energy = top_results[0][0]
    best_positions = top_results[0][1]
    
    # Phase 2: Basin hopping from the best results
    bh_rng = np.random.default_rng(size * 777 + 13)
    
    # Adaptive perturbation scale
    base_scale = 0.2
    min_scale = 0.05
    max_scale = 0.5
    
    for start_idx, (start_energy, start_pos) in enumerate(top_results):
        if (time.time() - t0) > instance_budget:
            break
        
        current_energy = start_energy
        current_pos = start_pos
        local_best_energy = start_energy
        local_best_pos = start_pos
        
        scale = base_scale
        
        # Basin hopping iterations
        max_bh_iters = 50
        for bh_iter in range(max_bh_iters):
            if (time.time() - t0) > instance_budget:
                break
            
            # Perturb
            perturbed = current_pos + bh_rng.normal(0, scale, current_pos.shape)
            
            try:
                energy, positions = relax(perturbed, potential, parameter)
            except Exception:
                continue
            
            if not np.isfinite(energy):
                continue
            
            # Metropolis acceptance
            if energy < current_energy:
                current_energy = energy
                current_pos = positions
                scale = max(min_scale, scale * 0.95)  # Decrease scale on improvement
            else:
                # Accept with probability exp(-(E_new - E_old) / T)
                T = 0.15
                if bh_rng.random() < np.exp(-(energy - current_energy) / T):
                    current_energy = energy
                    current_pos = positions
                    scale = min(max_scale, scale * 1.05)  # Increase scale on acceptance
                else:
                    scale = min(max_scale, scale * 1.1)  # Increase scale on rejection
            
            if energy < local_best_energy:
                local_best_energy = energy
                local_best_pos = positions
            
            if local_best_energy < best_energy:
                best_energy = local_best_energy
                best_positions = local_best_pos
        
        # If we found a new global best, restart BH from it
        if local_best_energy < best_energy:
            best_energy = local_best_energy
            best_positions = local_best_pos
    
    return best_positions
# EVOLVE-BLOCK-END
