import numpy as np
from scipy.optimize import minimize
import time

N = 38

def _lj_energy_and_grad(pos):
    """
    Calculate Lennard-Jones energy and gradient for a set of positions.
    pos: (N, 3) array of coordinates.
    Returns: energy, gradient (N, 3)
    """
    # Compute pairwise differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero for i==j
    np.fill_diagonal(dist_sq, 1.0)
    
    # Inverse distances
    inv_r2 = 1.0 / dist_sq
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    pair_energy = inv_r12 - inv_r6
    np.fill_diagonal(pair_energy, 0.0)
    energy = 2.0 * np.sum(pair_energy)
    
    inv_r8 = inv_r2 ** 4
    inv_r14 = inv_r2 ** 7
    
    grad_factor = 4.0 * (-12.0 * inv_r14 + 6.0 * inv_r8)
    np.fill_diagonal(grad_factor, 0.0)
    
    grad = np.sum(grad_factor[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def _make_fcc_cluster():
    """
    Generate an FCC lattice fragment of 38 atoms.
    """
    # FCC basis vectors
    v1 = np.array([1.0, 1.0, 0.0])
    v2 = np.array([1.0, 0.0, 1.0])
    v3 = np.array([0.0, 1.0, 1.0])
    
    points = []
    for n1 in range(-4, 5):
        for n2 in range(-4, 5):
            for n3 in range(-4, 5):
                p = n1 * v1 + n2 * v2 + n3 * v3
                points.append(p)
                
    points = np.array(points)
    
    dists = np.linalg.norm(points, axis=1)
    sorted_indices = np.argsort(dists)
    
    cluster = points[sorted_indices[:N]]
    
    centroid = np.mean(cluster, axis=0)
    cluster = cluster - centroid
    
    scale = (2.0 ** (1.0/6.0)) / np.sqrt(2.0)
    cluster = cluster * scale
    
    return cluster

def _make_icosahedral_cluster():
    """
    Generate an icosahedral cluster of 38 atoms (1 + 12 + 25).
    """
    # 12 vertices of a regular icosahedron
    phi = (1 + np.sqrt(5)) / 2
    r = 1.0 / np.sqrt(1 + phi**2)
    
    verts = []
    for i in range(3):
        for j in range(2):
            for k in range(2):
                pass # This is getting complicated, let's use a simpler heuristic or just perturb FCC
    
    # Actually, let's just use a perturbed FCC with different scales as multi-start
    # Or generate a "spherical" shell distribution
    # For now, return FCC to keep it simple, but we can add more starts in construct
    
    return _make_fcc_cluster()

def _make_perturbed_cluster(base_pos, scale, noise_level, rng):
    """
    Create a perturbed version of the base cluster.
    """
    pos = base_pos * scale
    pos += rng.normal(0, noise_level, pos.shape)
    return pos

def _make_icosahedral_start():
    """
    Create an icosahedral-like starting structure with 1 + 12 + 25 atoms.
    Layer 1: 1 atom at center
    Layer 2: 12 atoms at icosahedron vertices  
    Layer 3: 25 atoms at face centers of the icosahedron (approximately)
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of regular icosahedron (normalized to unit sphere)
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Center atom
    center = np.array([[0, 0, 0]])
    
    # 12 shell atoms at icosahedron vertices
    shell1 = verts * 1.0
    
    # For the 25 outer atoms, place them at midpoints of edges and face centers
    # Edge midpoints (30 edges, pick 20) and some face centers (20 faces, pick 5)
    # For simplicity, place 25 atoms on a slightly larger sphere at quasi-random positions
    # that avoid the icosahedral symmetry
    
    # Better: use the fact that 38 = 1 + 12 + 25
    # Place 25 atoms at positions that form a truncated icosahedron-like shell
    # Use golden ratio spiral for quasi-uniform distribution on sphere
    
    n_outer = 25
    outer = []
    for i in range(n_outer):
        # Fibonacci sphere
        y = 1 - 2 * (i + 0.5) / n_outer
        radius_at_y = np.sqrt(1 - y**2)
        golden_angle = 2.39996322972865332  # 2*pi*(3-sqrt(5))
        theta = golden_angle * i
        x = radius_at_y * np.cos(theta)
        z = radius_at_y * np.sin(theta)
        outer.append([x, y, z])
    
    outer = np.array(outer)
    outer = outer / np.linalg.norm(outer, axis=1, keepdims=True) * 1.8
    
    pos = np.vstack([center, shell1, outer])
    
    # Scale to appropriate LJ distance
    # The optimal nearest-neighbor distance for LJ is 2^(1/6) ≈ 1.122
    # In icosahedron, center-to-vertex is 1.0, so scale by 2^(1/6)
    scale = 2.0 ** (1.0/6.0)
    pos = pos * scale
    
    return pos

def _lj_energy_and_grad(pos):
    """
    Calculate Lennard-Jones energy and gradient for a set of positions.
    pos: (N, 3) array of coordinates.
    Returns: energy, gradient (N, 3)
    """
    # Compute pairwise differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero for i==j
    np.fill_diagonal(dist_sq, 1.0)
    
    # Inverse distances
    inv_r2 = 1.0 / dist_sq
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    pair_energy = inv_r12 - inv_r6
    np.fill_diagonal(pair_energy, 0.0)
    energy = 2.0 * np.sum(pair_energy)
    
    inv_r8 = inv_r2 ** 4
    inv_r14 = inv_r2 ** 7
    
    grad_factor = 4.0 * (-12.0 * inv_r14 + 6.0 * inv_r8)
    np.fill_diagonal(grad_factor, 0.0)
    
    grad = np.sum(grad_factor[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def _make_fcc_cluster():
    """
    Generate an FCC lattice fragment of 38 atoms.
    """
    # FCC basis vectors
    v1 = np.array([1.0, 1.0, 0.0])
    v2 = np.array([1.0, 0.0, 1.0])
    v3 = np.array([0.0, 1.0, 1.0])
    
    points = []
    for n1 in range(-4, 5):
        for n2 in range(-4, 5):
            for n3 in range(-4, 5):
                p = n1 * v1 + n2 * v2 + n3 * v3
                points.append(p)
                
    points = np.array(points)
    
    dists = np.linalg.norm(points, axis=1)
    sorted_indices = np.argsort(dists)
    
    cluster = points[sorted_indices[:N]]
    
    centroid = np.mean(cluster, axis=0)
    cluster = cluster - centroid
    
    scale = (2.0 ** (1.0/6.0)) / np.sqrt(2.0)
    cluster = cluster * scale
    
    return cluster

def _make_icosahedral_cluster():
    """
    Generate an icosahedral cluster of 38 atoms (1 + 12 + 25).
    Layer 1: 1 atom at center
    Layer 2: 12 atoms at icosahedron vertices
    Layer 3: 25 atoms on a quasi-uniform outer shell
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of regular icosahedron (normalized to unit sphere)
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Center atom
    center = np.array([[0, 0, 0]])
    
    # 12 shell atoms at icosahedron vertices
    shell1 = verts * 1.0
    
    # 25 outer atoms via Fibonacci sphere
    n_outer = 25
    outer = []
    for i in range(n_outer):
        y = 1 - 2 * (i + 0.5) / n_outer
        radius_at_y = np.sqrt(1 - y**2)
        golden_angle = 2.39996322972865332
        theta = golden_angle * i
        x = radius_at_y * np.cos(theta)
        z = radius_at_y * np.sin(theta)
        outer.append([x, y, z])
    
    outer = np.array(outer)
    outer = outer / np.linalg.norm(outer, axis=1, keepdims=True) * 1.8
    
    pos = np.vstack([center, shell1, outer])
    
    # Scale to appropriate LJ distance
    scale = 2.0 ** (1.0/6.0)
    pos = pos * scale
    
    return pos

def _make_perturbed_cluster(base_pos, scale, noise_level, rng):
    """
    Create a perturbed version of the base cluster.
    """
    pos = base_pos * scale
    pos += rng.normal(0, noise_level, pos.shape)
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Generate initial guesses from multiple structural motifs
    fcc_pos = _make_fcc_cluster()
    ico_pos = _make_icosahedral_cluster()
    
    # Initial energy
    e0, g0 = _lj_energy_and_grad(fcc_pos)
    best_pos = fcc_pos.copy()
    best_e = e0
    
    # Also try icosahedral
    e1, g1 = _lj_energy_and_grad(ico_pos)
    if e1 < best_e:
        best_pos = ico_pos.copy()
        best_e = e1
    
    # Multi-start optimization
    rng = np.random.default_rng(42)
    
    # Define a set of scales and noise levels to explore
    scales = [0.9, 0.95, 1.0, 1.05, 1.1]
    noise_levels = [0.0, 0.01, 0.05, 0.1]
    
    # We have ~110s. Let's allocate time per start.
    # 5 scales * 4 noise levels = 20 starts.
    # 110 / 20 = 5.5s per start.
    # L-BFGS is fast for 38 atoms, so this should be feasible.
    
    starts = []
    for s in scales:
        for n in noise_levels:
            if time.time() > deadline - 5.0: # Stop generating if close to deadline
                break
            starts.append((s, n))
    
    for s, n in starts:
        if time.time() > deadline - 2.0:
            break
            
        pos = _make_perturbed_cluster(fcc_pos, s, n, rng)
        
        # Check minimum distance constraint (0.5)
        # If violated, skip or shrink
        # For small noise, it should be fine.
        
        try:
            res = minimize(
                _lj_energy_and_grad, 
                pos, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-10}
            )
            
            if res.fun < best_e:
                best_pos = res.x
                best_e = res.fun
                
        except Exception:
            pass
        
        # If we found a very good energy, we might tighten the search
        # But for now, just continue
    
    # Also try icosahedral-based starts
    ico_starts = [(0.95, 0.0), (1.0, 0.01), (1.0, 0.05), (1.05, 0.0)]
    for s, n in ico_starts:
        if time.time() > deadline - 2.0:
            break
        pos = _make_perturbed_cluster(ico_pos, s, n, rng)
        try:
            res = minimize(
                _lj_energy_and_grad, 
                pos, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-10}
            )
            if res.fun < best_e:
                best_pos = res.x
                best_e = res.fun
        except Exception:
            pass
    
    # Final high-precision polish on the best result found
    if time.time() < deadline - 5.0:
        try:
            res = minimize(
                _lj_energy_and_grad, 
                best_pos, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 2000, 'gtol': 1e-14, 'ftol': 1e-16}
            )
            if res.fun < best_e:
                best_pos = res.x
                best_e = res.fun
        except Exception:
            pass
    
    # Second polish pass if time remains
    if time.time() < deadline - 2.0 and best_e > -173.9:
        try:
            res = minimize(
                _lj_energy_and_grad, 
                best_pos, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 5000, 'gtol': 1e-15, 'ftol': 1e-18}
            )
            if res.fun < best_e:
                best_pos = res.x
                best_e = res.fun
        except Exception:
            pass

    return best_pos
