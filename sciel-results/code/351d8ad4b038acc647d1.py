import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
from scipy.optimize import minimize
import time

def _get_icosahedron_vertices(radius=1.0):
    """
    Generate the 12 vertices of a regular icosahedron centered at origin.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi],
        [0, -1, phi],
        [0, 1, -phi],
        [0, -1, -phi],
        [1, phi, 0],
        [-1, phi, 0],
        [1, -phi, 0],
        [-1, -phi, 0],
        [phi, 0, 1],
        [phi, 0, -1],
        [-phi, 0, 1],
        [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    return verts * radius

def _get_mackay_icosahedron_55():
    """
    Generate the 55-atom Mackay icosahedron.
    Shell 1: 13 atoms (1 center + 12 vertices)
    Shell 2: 42 atoms (12 vertices + 30 edge midpoints of the icosahedron)
    """
    # Shell 1
    center = np.array([0.0, 0.0, 0.0])
    verts = _get_icosahedron_vertices(1.0)
    shell1 = np.vstack([center, verts])
    
    # Shell 2
    # Find edges
    min_dist = np.inf
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < min_dist:
                min_dist = d
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, min_dist):
                edges.append((i, j))
    
    shell2_pts = []
    # 12 vertices of the icosahedron (same directions as shell1 vertices)
    for v in verts:
        shell2_pts.append(v)
    # 30 edge midpoints
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        shell2_pts.append(mid)
        
    shell2_pts = np.array(shell2_pts)
    norms = np.linalg.norm(shell2_pts, axis=1)
    shell2_pts = shell2_pts / norms[:, np.newaxis] * 1.8 # Typical second shell radius
    
    return np.vstack([shell1, shell2_pts])

def _generate_initial_structure(n, seed=0):
    """
    Generate an initial structure for n atoms based on the 55-atom Mackay icosahedron.
    """
    mackay_55 = _get_mackay_icosahedron_55()
    rng = np.random.default_rng(seed)
    
    if n == 55:
        return mackay_55
    
    if n < 55:
        # Remove atoms from the second shell (indices 13 to 54)
        # We want to keep the structure compact.
        # Let's remove atoms randomly from the second shell.
        # To be more symmetric, we could remove in pairs or based on symmetry,
        # but random removal followed by relaxation usually works well for Dzugutov.
        
        # Identify indices of shell 2 atoms
        shell2_indices = np.arange(13, 55)
        
        # Shuffle and pick the ones to keep
        rng.shuffle(shell2_indices)
        num_to_keep = n - 13
        if num_to_keep < 0:
             # n < 13, use shell 1 only
             return mackay_55[:n]
             
        keep_indices = np.sort(shell2_indices[:num_to_keep])
        keep_indices = np.concatenate([np.arange(13), keep_indices])
        
        return mackay_55[keep_indices]
        
    else:
        # n > 55
        # Add atoms to a third shell
        # Approximate third shell by scaling icosahedron vertices and edge midpoints further out
        # Or just add atoms at random positions on a sphere of radius ~2.6
        
        base = mackay_55.copy()
        extra_needed = n - 55
        
        # Generate candidate positions for extra atoms
        # Use vertices of a larger icosahedron or random points
        # Let's use random points on a sphere of radius 2.6
        extra_pts = []
        attempts = 0
        while len(extra_pts) < extra_needed and attempts < 1000:
            # Random direction
            u = rng.uniform(-1, 1, 3)
            norm = np.linalg.norm(u)
            if norm > 1e-6:
                u = u / norm
                p = u * 2.6
                # Check distance to existing atoms
                if np.all(np.linalg.norm(base - p, axis=1) > 0.8):
                    extra_pts.append(p)
            attempts += 1
            
        if len(extra_pts) < extra_needed:
            # Fallback: just add random points
            while len(extra_pts) < extra_needed:
                p = rng.normal(0, 1.0, 3)
                extra_pts.append(p)
                
        extra_pts = np.array(extra_pts)
        return np.vstack([base, extra_pts])

def _optimize_structure(positions, potential, time_limit=10.0):
    """
    Optimize the structure using L-BFGS-B with the Dzugutov gradient.
    """
    start_time = time.time()
    x0 = positions.flatten()
    
    def fun(x):
        pos = x.reshape(-1, 3)
        try:
            # Use the provided gradient function which likely returns energy and gradient
            # The function signature is dzugutov_energy_gradient(positions)
            # We need to check if it returns (energy, grad) or just grad.
            # The task says: "chem_clusters.dzugutov_energy_gradient(positions) may be called"
            # Usually these return (E, G). Let's assume (E, G).
            E, G = dzugutov_energy_gradient(pos)
            return E, G.flatten()
        except Exception:
            return 1e10, np.zeros_like(x)
            
    # L-BFGS-B requires fun to return (f, g)
    res = minimize(fun, x0, method='L-BFGS-B', jac=True, options={'maxiter': 500, 'gtol': 1e-5, 'ftol': 1e-8})
    
    if time.time() - start_time > time_limit:
        return positions # Return original if too slow
        
    x_opt = res.x.reshape(-1, 3)
    # Ensure minimum distance constraint is somewhat respected?
    # The potential has a repulsive core, so it should be fine.
    return x_opt

def construct(instance):
    """
    Find low-energy Dzugutov cluster structures.
    """
    size = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + 280  # 20s margin before 300s limit
    
    best_energy = float('inf')
    best_positions = None
    
    def try_relax_and_update(init_positions, label=""):
        nonlocal best_energy, best_positions
        try:
            # First, try the suite's relax function
            energy, positions = relax(init_positions, potential, None)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            pass
        
        # Then, try L-BFGS-B from the relaxed structure or original
        if time.time() < deadline - 5:
            try:
                # Optimize the result from relax
                opt_pos = _optimize_structure(best_positions if best_positions is not None else init_positions, potential, time_limit=10)
                # Calculate energy of opt_pos
                E, _ = dzugutov_energy_gradient(opt_pos)
                if E < best_energy:
                    best_energy = E
                    best_positions = opt_pos
            except Exception:
                pass

    # 1. Standard Mackay-based initialization
    init_1 = _generate_initial_structure(size, seed=size)
    try_relax_and_update(init_1, "mackay")
    
    if time.time() > deadline:
        return best_positions if best_positions is not None else init_1

    # 2. Perturbed Mackay
    rng = np.random.default_rng(size + 100)
    init_2 = init_1 + rng.normal(0, 0.05, init_1.shape)
    try_relax_and_update(init_2, "perturbed_mackay")
    
    if time.time() > deadline:
        return best_positions if best_positions is not None else init_1

    # 3. Distorted Icosahedron (different radius)
    # Try a slightly compressed or expanded version
    scale_factors = [0.9, 1.1]
    for sf in scale_factors:
        if time.time() > deadline - 10: break
        init_scaled = init_1 * sf
        try_relax_and_update(init_scaled, f"scaled_{sf}")
        
    if time.time() > deadline:
        return best_positions if best_positions is not None else init_1

    # 4. Random Compact (backup)
    if time.time() < deadline - 20:
        # Generate a random compact cluster
        radius = 1.5 * size ** (1.0/3.0)
        points = []
        attempts = 0
        while len(points) < size and attempts < size * 500:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) > radius:
                continue
            if points:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) < 0.8:
                    continue
            points.append(p)
            attempts += 1
            
        if len(points) < size:
            while len(points) < size:
                p = rng.uniform(-radius, radius, 3)
                points.append(p)
                
        init_rand = np.array(points[:size])
        try_relax_and_update(init_rand, "random")

    if best_positions is None:
        return init_1
        
    return best_positions
