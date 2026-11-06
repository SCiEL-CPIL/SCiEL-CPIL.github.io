# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import sys

def _dzugutov_energy(positions, A=5.82, B=1.28, a=1.87, b=1.94, c=1.10, d=0.27, m=16):
    """Compute Dzugutov potential energy (vectorized)."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    energy = 0.0
    i_idx, j_idx = np.triu_indices(n, k=1)
    r = dists[i_idx, j_idx]
    
    # Mask for r < a
    mask_a = r < a
    if np.any(mask_a):
        r_a = r[mask_a]
        energy += np.sum(A * (r_a**(-m) - B) * np.exp(c / (r_a - a)))
        
    # Mask for a <= r < b
    mask_b = (r >= a) & (r < b)
    if np.any(mask_b):
        r_b = r[mask_b]
        energy += np.sum(B * np.exp(d / (r_b - b)))
        
    return energy

def _generate_mackay_icosahedron(n, rng=None):
    """
    Generate a Mackay icosahedral cluster.
    This is the preferred structure for Dzugutov potential.
    We construct it by adding shells of icosahedral symmetry.
    """
    if rng is None:
        rng = np.random.default_rng(42)
        
    # Icosahedral vertices (12)
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts[0])
    
    points = [np.zeros(3)] # Center
    
    # Shell 1: 12 vertices
    r1 = 1.15 # Typical bond length for Dzugutov
    points.extend([v * r1 for v in verts])
    
    # Shell 2: 30 edges (midpoints of edges of icosahedron, projected to sphere)
    # Actually, Mackay icosahedron shell 2 has 30 atoms at distance sqrt(2)*r1?
    # Let's use the standard Mackay construction:
    # Shell 1: 12 atoms at distance d1
    # Shell 2: 30 atoms at distance d2 = d1 * sqrt(2) ? No, usually d2 is slightly different in relaxed clusters.
    # For a geometric seed, we use ideal icosahedral geometry.
    
    # Generate edge midpoints of the icosahedron
    edges = []
    # Find all pairs of vertices that are close (edges)
    for i in range(12):
        for j in range(i+1, 12):
            dist = np.linalg.norm(verts[i] - verts[j])
            if dist < 1.5: # Heuristic for edge length in normalized icosahedron
                edges.append((i, j))
                
    shell2_points = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        # Project to sphere of radius sqrt(2) * r1? 
        # In ideal icosahedron, edge midpoint distance from center is r1 * sqrt(2).
        norm_mid = mid / np.linalg.norm(mid)
        shell2_points.append(norm_mid * (r1 * np.sqrt(2)))
        
    points.extend(shell2_points)
    
    # Shell 3: 20 faces (centers of faces)
    # This gets complicated. For n > 42, we need more shells.
    # A simpler approach for larger n:
    # Generate points on concentric icosahedral shells until we reach n.
    
    if n <= 13:
        return np.array(points[:n])
        
    # For larger n, we can add more layers.
    # Layer 3 has 20 atoms (face centers).
    # We can approximate by generating more points on the icosahedral symmetry axes/faces.
    # Or simply expand the radius and add points.
    
    # A robust method: Generate a large icosahedral cluster and take the first n.
    # But we need to ensure connectivity and validity.
    
    # Let's try adding a 3rd shell: 20 face centers.
    # Triangles of icosahedron:
    triangles = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                # Check if i,j,k form a triangle (all edges exist)
                if (i,j) in [(e[0],e[1]) for e in edges] or (j,i) in [(e[0],e[1]) for e in edges]:
                    pass # This logic is fragile.
                    
    # Alternative: Just generate points on a sphere with icosahedral symmetry distribution
    # and take the closest n to the center? No, that's just a shell.
    
    # Let's stick to the 42-atom Mackay icosahedron (1+12+30) as the primary seed.
    # For n > 42, we add a 4th shell (120 atoms in full Mackay, but we only need up to 60).
    # The 4th shell atoms are at the vertices of the dual icosahedron (dodecahedron) scaled up?
    # Actually, for n=50-60, adding a partial 4th shell or expanding the 3rd shell is good.
    
    # Let's generate a "dense" icosahedral-like set by adding points on the edges of the 30-face shell?
    # Simpler: If n > 42, we take the 42 atoms and add n-42 atoms at random positions
    # on the surface of the 42-atom cluster, then relax.
    
    if n > 42:
        base = np.array(points[:42])
        extra_needed = n - 42
        extra_points = []
        for _ in range(extra_needed):
            # Pick a random face of the icosahedron (approx by picking 3 random nearby vertices)
            # Or just pick a random direction and place at radius ~ r1 * 2.2
            direction = rng.normal(0, 1, 3)
            direction /= np.linalg.norm(direction)
            # Place slightly outside the 3rd shell
            r_new = r1 * 2.2 
            p = direction * r_new
            extra_points.append(p)
        points = list(base) + extra_points
        
    return np.array(points[:n])

def _generate_fcc_cluster(n, rng):
    """Generate a cluster from FCC lattice (alternative seed)."""
    points = []
    L = 1.74
    for x in range(-3, 4):
        for y in range(-3, 4):
            for z in range(-3, 4):
                points.append([x*L, y*L, z*L])
                points.append([(x+0.5)*L, (y+0.5)*L, z*L])
                points.append([(x+0.5)*L, y*L, (z+0.5)*L])
                points.append([x*L, (y+0.5)*L, (z+0.5)*L])
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    selected = points[order[:n]]
    return selected

def construct(instance):
    """Find low-energy Dzugutov cluster using Mackay seeds and Gradient-based local relaxation."""
    n = instance["n"]
    potential = instance["potential"]
    param = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 290  # 300s limit, 10s safety margin
    
    best_energy = np.inf
    best_positions = None
    
    from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
    
    # Define the objective function for L-BFGS-B
    def objective(x):
        """
        Objective function for L-BFGS-B.
        It takes a 1D array of positions, reshapes to (n, 3), and returns energy.
        """
        x_reshaped = x.reshape(n, 3)
        # Check minimum distance constraint
        i_idx, j_idx = np.triu_indices(n, k=1)
        min_dist = np.min(np.linalg.norm(x_reshaped[i_idx] - x_reshaped[j_idx], axis=1))
        if min_dist < 0.5:
            return 1e10 # Penalty for invalid structure
            
        return _dzugutov_energy(x_reshaped)
        
    def gradient(x):
        """
        Gradient function for L-BFGS-B.
        """
        x_reshaped = x.reshape(n, 3)
        i_idx, j_idx = np.triu_indices(n, k=1)
        min_dist = np.min(np.linalg.norm(x_reshaped[i_idx] - x_reshaped[j_idx], axis=1))
        if min_dist < 0.5:
            return np.zeros_like(x)
            
        try:
            # dzugutov_energy_gradient returns (energy, gradient)
            e, grad = dzugutov_energy_gradient(x_reshaped)
            return grad.flatten()
        except Exception:
            return np.zeros_like(x)

    # Strategy: Run Gradient-based local optimization starting from different seeds.
    
    seeds = []
    
    # Seed 1: Mackay Icosahedron
    try:
        seed_ico = _generate_mackay_icosahedron(n, np.random.default_rng(n))
        # Center the seed
        seed_ico = seed_ico - np.mean(seed_ico, axis=0)
        seeds.append(seed_ico.flatten())
    except Exception:
        pass
        
    # Seed 2: FCC Cluster
    try:
        seed_fcc = _generate_fcc_cluster(n, np.random.default_rng(n+1))
        seed_fcc = seed_fcc - np.mean(seed_fcc, axis=0)
        # Check validity
        i_idx, j_idx = np.triu_indices(n, k=1)
        min_dist = np.min(np.linalg.norm(seed_fcc[i_idx] - seed_fcc[j_idx], axis=1))
        if min_dist >= 0.5:
            seeds.append(seed_fcc.flatten())
    except Exception:
        pass
        
    # Seed 3: Random Compact (fallback)
    try:
        rng = np.random.default_rng(n+2)
        points = [np.zeros(3)]
        radius = 0.5
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                min_dist = min(np.linalg.norm(p - q) for q in points) if points else np.inf
                if min_dist >= 0.5:
                    points.append(p)
            else:
                radius += 0.1
        if len(points) == n:
            seed_rand = np.array(points) - np.mean(np.array(points), axis=0)
            seeds.append(seed_rand.flatten())
    except Exception:
        pass

    if not seeds:
        # Absolute fallback
        seed = np.random.default_rng(0).normal(0, 1, n*3).reshape(n, 3)
        seeds.append(seed.flatten())

    # Run L-BFGS-B for each seed
    # We limit the number of iterations and time per seed.
    # Total time budget is ~290s.
    # Let's allocate ~60s per seed if we have 4 seeds, or adjust dynamically.
    
    import scipy.optimize as so
    
    for i_seed, seed in enumerate(seeds):
        if time.time() > deadline - 10: # Stop if close to deadline
            break
            
        # Determine max funevals based on remaining time
        remaining = deadline - time.time()
        
        # Use a reasonable number of iterations. 
        # L-BFGS-B is very efficient for smooth potentials.
        # 100 iterations should be enough to converge to a local minimum.
        maxiter = 100
        
        try:
            result = so.minimize(
                objective, 
                seed, 
                method='L-BFGS-B', 
                jac=gradient,
                options={'maxiter': maxiter, 'gtol': 1e-5, 'disp': False}
            )
            
            if result.success and result.fun < best_energy:
                best_energy = result.fun
                best_positions = result.x.reshape(n, 3)
                
        except Exception as e:
            # If L-BFGS-B fails, try a simple relaxation of the seed
            try:
                x_reshaped = seed.reshape(n, 3)
                energy, relaxed = relax(x_reshaped, potential, param)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed.copy()
            except Exception:
                pass

    # If no valid result was found, use the first seed relaxed
    if best_positions is None:
        try:
            x_reshaped = seeds[0].reshape(n, 3)
            energy, relaxed = relax(x_reshaped, potential, param)
            best_positions = relaxed
        except Exception:
            # Last resort: generate a valid random structure
            rng = np.random.default_rng(n)
            points = [np.zeros(3)]
            radius = 0.5
            while len(points) < n:
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius:
                    min_dist = min(np.linalg.norm(p - q) for q in points) if points else np.inf
                    if min_dist >= 0.5:
                        points.append(p)
                else:
                    radius += 0.1
            best_positions = np.array(points[:n])

    # Final check: ensure minimum distance constraint
    if len(best_positions) >= 2:
        i_idx, j_idx = np.triu_indices(len(best_positions), k=1)
        dists = np.linalg.norm(best_positions[i_idx] - best_positions[j_idx], axis=1)
        if np.min(dists) < 0.5:
            # Attempt to fix by expanding
            center = np.mean(best_positions, axis=0)
            best_positions = (best_positions - center) * 1.2 + center
            # If still invalid, we might have to accept it or try again
            # For safety, if it's still invalid, we return the best we have, 
            # but the evaluator might reject it. 
            # Let's try to relax again after expansion.
            try:
                energy, relaxed = relax(best_positions, potential, param)
                best_positions = relaxed
            except Exception:
                pass

    return best_positions
# EVOLVE-BLOCK-END
