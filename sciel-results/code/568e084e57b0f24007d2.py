import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient
import scipy.optimize

def _min_distance_check(pos):
    """Check if all pairwise distances are at least 0.5."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.5

def _generate_fcc_cluster(n, rng):
    """
    Generate an FCC cluster of approximately n atoms, centered.
    Uses a cubic block of FCC lattice points, taking the n closest to the center.
    """
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ])
    
    side = int(np.ceil((0.25 * n) ** (1.0/3.0))) + 2
    
    pts = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for b in basis:
                    p = np.array([i, j, k]) + b
                    pts.append(p)
    
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    cluster = pts[idx]
    
    cluster += rng.normal(0, 0.05, size=cluster.shape)
    
    return cluster

def _generate_hcp_cluster(n, rng):
    """
    Generate an HCP cluster of approximately n atoms, centered.
    """
    a = 1.0
    c = np.sqrt(8.0/3.0) * a
    
    pts = []
    radius = np.sqrt(n / (np.pi * 0.866))
    num_layers = int(np.ceil(2 * radius / (c/2))) + 2
    
    for layer_idx in range(-num_layers//2, num_layers//2 + 1):
        z = layer_idx * (c / 2.0)
        if layer_idx % 2 == 0:
            offset = np.array([0.0, 0.0])
        else:
            offset = np.array([a/2.0, a*np.sqrt(3)/6.0])
            
        for i in range(-int(radius), int(radius) + 1):
            for j in range(-int(radius), int(radius) + 1):
                x = a * (i + j/2.0) + offset[0]
                y = a * (np.sqrt(3)/2.0 * j) + offset[1]
                p = np.array([x, y, z])
                if np.linalg.norm(p[:2]) <= radius:
                    pts.append(p)
                    
    if len(pts) < n:
        return _generate_fcc_cluster(n, rng)
        
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    cluster = pts[idx]
    cluster += rng.normal(0, 0.05, size=cluster.shape)
    
    return cluster

def _generate_icosahedron(n, rng):
    """
    Generate a Mackay icosahedron cluster.
    """
    phi = (1 + np.sqrt(5)) / 2.0
    
    v1 = np.array([0, 1, phi])
    v2 = np.array([0, -1, phi])
    v3 = np.array([0, 1, -phi])
    v4 = np.array([0, -1, -phi])
    v5 = np.array([1, phi, 0])
    v6 = np.array([-1, phi, 0])
    v7 = np.array([1, -phi, 0])
    v8 = np.array([-1, -phi, 0])
    v9 = np.array([phi, 0, 1])
    v10 = np.array([-phi, 0, 1])
    v11 = np.array([phi, 0, -1])
    v12 = np.array([-phi, 0, -1])
    
    verts = np.array([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12])
    norm = np.linalg.norm(verts, axis=1)
    verts = verts / norm[:, np.newaxis]
    
    points = [np.zeros(3)]
    points.extend(verts)
    
    if n <= 13:
        pts = np.array(points[:n])
        return pts + rng.normal(0, 0.01, size=pts.shape)
        
    pts = np.array(points)
    if n > len(pts):
        num_to_add = n - len(pts)
        u = rng.uniform(0, 1, num_to_add)
        v = rng.uniform(0, 1, num_to_add)
        theta = 2 * np.pi * u
        phi_s = np.arccos(2 * v - 1)
        
        x = 2.5 * np.sin(phi_s) * np.cos(theta)
        y = 2.5 * np.sin(phi_s) * np.sin(theta)
        z = 2.5 * np.cos(phi_s)
        
        new_pts = np.column_stack((x, y, z))
        pts = np.vstack((pts, new_pts))
        
    pts += rng.normal(0, 0.05, size=pts.shape)
    return pts

def _polish(positions, potential, rho, max_iter=1000):
    """
    Perform a high-precision polish pass using L-BFGS-B with tight tolerances.
    This is a refinement of the local minimum found by relax().
    """
    try:
        # Use scipy.optimize.minimize with L-BFGS-B for tighter control
        # We need to define the objective and gradient functions
        
        def objective(x):
            pos = x.reshape(-1, 3)
            # Check min distance
            if not _min_distance_check(pos):
                return 1e10
            e, g = morse_energy_gradient(pos, rho)
            return e
            
        def gradient(x):
            pos = x.reshape(-1, 3)
            if not _min_distance_check(pos):
                return np.zeros_like(x)
            e, g = morse_energy_gradient(pos, rho)
            return g.flatten()
            
        x0 = positions.flatten()
        
        # L-BFGS-B with tight tolerance
        res = scipy.optimize.minimize(
            objective, 
            x0, 
            jac=gradient, 
            method='L-BFGS-B', 
            options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-10}
        )
        
        if res.success or res.fun < 1e9:
            return res.fun, res.x.reshape(-1, 3)
        else:
            return positions  # Return original if polish fails
            
    except Exception:
        return positions

def construct(instance):
    """
    Find the lowest-energy arrangement of n atoms bound by Morse potentials.
    Uses structured seeds (FCC, HCP, Icosahedron) and L-BFGS-B relaxation with polish.
    """
    n = instance["n"]
    rho = instance.get("rho", 3)
    potential = instance["potential"]
    
    start_time = time.time()
    # Safety margin: 330s limit, use 45s per instance
    deadline = start_time + 45.0
    
    rng = np.random.default_rng(42 + n + int(rho*10))
    
    best_energy = np.inf
    best_positions = None
    
    seeds = []
    
    # Always try FCC
    try:
        seeds.append(_generate_fcc_cluster(n, rng))
    except Exception:
        pass
        
    # Always try HCP
    try:
        seeds.append(_generate_hcp_cluster(n, rng))
    except Exception:
        pass
        
    # Try Icosahedron if rho is high or n is small
    if rho >= 6 or n <= 50:
        try:
            seeds.append(_generate_icosahedron(n, rng))
        except Exception:
            pass
            
    # Add a random seed as fallback
    try:
        radius = n ** (1.0/3.0) * 0.8
        pts = rng.uniform(-radius, radius, size=(n, 3))
        valid_pts = []
        for p in pts:
            if all(np.linalg.norm(p - q) >= 0.5 for q in valid_pts):
                valid_pts.append(p)
        if len(valid_pts) == n:
            seeds.append(np.array(valid_pts))
        else:
            seeds.append(pts)
    except Exception:
        pass
        
    if not seeds:
        seeds.append(rng.uniform(-n**0.33, n**0.33, size=(n, 3)))
        
    # Relax each seed
    for seed in seeds:
        if time.time() > deadline - 15.0:
            break
            
        if not _min_distance_check(seed):
            continue
            
        try:
            energy, positions = relax(seed, potential, rho)
            
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue
            
    # If we have time, do basin hopping from the best seed
    if best_positions is not None and time.time() < deadline - 20.0:
        current_energy = best_energy
        current_positions = best_positions
        kick_sigma = 0.1
        
        while time.time() < deadline - 10.0:
            new_positions = current_positions + rng.normal(0, kick_sigma, size=(n, 3))
            if not _min_distance_check(new_positions):
                continue
                
            try:
                energy, positions = relax(new_positions, potential, rho)
            except Exception:
                continue
                
            if energy < current_energy:
                current_energy = energy
                current_positions = positions
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
            else:
                delta_e = energy - current_energy
                T = 0.05
                if np.random.rand() < np.exp(-delta_e / T):
                    current_energy = energy
                    current_positions = positions
                    
    # Final polish pass
    if best_positions is not None and time.time() < deadline - 5.0:
        polished = _polish(best_positions, potential, rho, max_iter=500)
        if isinstance(polished, tuple):
            e, pos = polished
            if e < best_energy:
                best_energy = e
                best_positions = pos
        else:
            best_positions = polished
            
    if best_positions is None:
        return seeds[0]
        
    return best_positions
