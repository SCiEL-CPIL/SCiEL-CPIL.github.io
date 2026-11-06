# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Dzugutov potential parameters (for internal checks if needed, though we use the provided gradient)
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16

def _get_mackay_icosahedron(n):
    """
    Generate a Mackay icosahedron-like structure for n atoms.
    For n=55, this is the perfect Mackay icosahedron.
    For other n, we add/remove atoms from the outer shell or inner shell.
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of a regular icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    # Normalize to unit radius
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Mackay icosahedron shells:
    # Shell 0: 1 atom at center (0,0,0)
    # Shell 1: 12 atoms at radius ~1.0 (vertices of icosahedron)
    # Shell 2: 42 atoms at radius ~2.0 (vertices of larger icosahedron + face centers + edge midpoints?)
    # Actually, the 55-atom Mackay icosahedron is constructed by:
    # 1 center
    # 12 vertices of an icosahedron
    # 42 atoms on the surface of a larger icosahedron (vertices, edge centers, face centers)
    
    # Let's construct the 55-atom structure specifically.
    # The 55-atom cluster is often described as having icosahedral symmetry.
    # Coordinates for a Mackay icosahedron (Mackay, 1962):
    # It's easier to generate the 55 points directly or via a known generator.
    
    # A simpler robust approach for "Mackay-like":
    # Use the 12 icosahedron vertices scaled to ~1.1
    # Use the 30 edge midpoints scaled to ~1.8?
    # Use the 20 face centers scaled to ~1.9?
    # 1 + 12 + 30 + 20 = 63. That's too many.
    
    # The 55-atom Mackay icosahedron has:
    # 1 center
    # 12 inner shell (radius ~1.0)
    # 42 outer shell (radius ~1.8-2.0)
    
    # Let's use a known set of 55 coordinates or generate them via a script.
    # Since I can't run scripts, I will generate a high-quality icosahedral seed.
    
    # Vertices of icosahedron
    v1 = verts * 1.05
    
    # For the outer shell of the 55-atom Mackay icosahedron, the atoms are located at:
    # The 12 vertices of a larger icosahedron
    # The 30 midpoints of the edges of the larger icosahedron
    # Total 42 atoms in the outer shell.
    
    # Let's generate the 30 edge midpoints of the icosahedron.
    # Edges of icosahedron:
    # Each vertex connects to 5 others.
    # We can find edges by distance.
    dists = np.linalg.norm(v1[:, np.newaxis] - v1[np.newaxis, :], axis=2)
    # The shortest distance between vertices in a unit icosahedron is ~1.051
    # Let's find pairs with distance < 1.2 * min_dist
    min_dist = np.min(dists[dists > 1e-6])
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if dists[i, j] < 1.1 * min_dist:
                edges.append((i, j))
    
    midpoints = []
    for i, j in edges:
        mid = (v1[i] + v1[j]) / 2.0
        # Normalize to a specific radius for the outer shell
        # In Mackay, outer shell radius is approx 1.8-2.0 relative to inner 1.0
        # Let's scale midpoints to radius ~1.8
        r_mid = np.linalg.norm(mid)
        if r_mid > 1e-6:
            mid_scaled = mid / r_mid * 1.8
            midpoints.append(mid_scaled)
            
    # We have 30 midpoints.
    # We also need the 12 vertices of the outer icosahedron.
    # These are just v1 scaled to 1.8?
    # In a Mackay icosahedron, the outer shell is a larger icosahedron.
    # The 12 vertices are at radius ~1.8.
    # The 30 edge midpoints are also on the surface.
    # Wait, the 55-atom Mackay icosahedron consists of:
    # 1 center
    # 12 atoms at radius ~1.0
    # 42 atoms at radius ~1.8 (12 vertices + 30 edge midpoints of a larger icosahedron)
    
    outer_verts = verts * 1.8
    outer_mids = []
    for i, j in edges:
        mid = (outer_verts[i] + outer_verts[j]) / 2.0
        outer_mids.append(mid)
        
    points = [np.zeros(3)] # Center
    points.extend(list(v1)) # Inner shell
    points.extend(list(outer_verts)) # Outer vertices
    points.extend(outer_mids) # Outer midpoints
    
    # Total points: 1 + 12 + 12 + 30 = 55. Perfect.
    
    if n == 55:
        return np.array(points)
    
    # For other n, we modify the 55-atom structure.
    # If n < 55, remove atoms from the outer shell (midpoints first, then vertices) to keep symmetry as much as possible.
    # If n > 55, add atoms. But the task is 38-60.
    # 38, 45, 50 are < 55.
    # 57, 60 are > 55.
    
    # Strategy for n < 55:
    # Remove midpoints (30) first, then outer vertices (12), then inner (12).
    # To maintain low energy, we should remove atoms that are "least bound" or just take the first n of the list?
    # Better: Keep the core.
    # Core: 1 + 12 = 13.
    # Next: 12 outer verts. Total 25.
    # Next: 30 mids. Total 55.
    
    # For n=50: 1+12+12+25 mids?
    # For n=45: 1+12+12+20 mids?
    # For n=38: 1+12+12+13 mids?
    # For n=38, maybe we should drop some outer verts too?
    # Let's just take the first n points from the 55-point list, but sorted by "importance".
    # Importance: Center > Inner > Outer Verts > Outer Mids.
    
    sorted_points = points[:1] + points[1:13] + points[13:25] + points[25:55]
    
    if n <= 55:
        return np.array(sorted_points[:n])
    else:
        # For n > 55, add atoms randomly on a shell of radius ~2.5?
        # Or just return the 55 and let the optimizer add? No, shape must be (n,3).
        # We can add atoms at random positions on a sphere of radius 2.5.
        rng = np.random.default_rng(42)
        extra = []
        for _ in range(n - 55):
            # Random point on sphere
            u = rng.normal(size=3)
            u = u / np.linalg.norm(u)
            extra.append(u * 2.5)
        return np.array(points + extra)

def _energy_and_grad(x, n):
    """Flattened energy and gradient for L-BFGS-B."""
    positions = x.reshape(-1, 3)
    # Use the provided gradient function
    # Note: dzugutov_energy_gradient returns (energy, gradient)
    # The gradient is likely flattened or (n,3)? 
    # Standard scipy minimize expects a flat gradient vector.
    # Let's assume dzugutov_energy_gradient returns (energy, grad_array) where grad_array is (n,3) or (3n,).
    # If it returns (n,3), we need to flatten it.
    
    # Checking typical behavior of such wrappers:
    # Usually returns (E, dE/dx) where dE/dx is same shape as x if x is (n,3), or flat.
    # The prompt says: chem_clusters.dzugutov_energy_gradient(positions)
    # Let's assume it returns (energy, gradient) where gradient is (n,3).
    
    energy, grad = dzugutov_energy_gradient(positions)
    
    # Check if grad is flat or 2D
    if grad.shape == (n, 3):
        grad_flat = grad.flatten()
    else:
        grad_flat = grad
        
    return energy, grad_flat

def construct(instance):
    n = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    # 1. Generate initial template
    try:
        init_pos = _get_mackay_icosahedron(n)
    except:
        # Fallback to random compact if template fails
        rng = np.random.default_rng(42)
        radius = 1.5 * n**(1.0/3.0)
        init_pos = rng.uniform(-radius, radius, (n, 3))
        
    # Center it
    init_pos = init_pos - np.mean(init_pos, axis=0)
    
    # 2. Optimization
    # We will try L-BFGS-B. It is very efficient for smooth functions.
    # We need to be careful with the number of iterations.
    
    x0 = init_pos.flatten()
    
    best_energy = np.inf
    best_pos = init_pos.copy()
    
    # Option 1: L-BFGS-B
    try:
        # First pass: Coarse optimization
        res = minimize(
            _energy_and_grad,
            x0,
            args=(n,),
            method='L-BFGS-B',
            jac=True, # We provide the gradient in the same function
            options={'maxiter': 100, 'ftol': 1e-5, 'gtol': 1e-4}
        )
        
        if time.time() < deadline:
            if res.success or res.fun < best_energy:
                best_energy = res.fun
                best_pos = res.x.reshape(-1, 3)
                
                # Second pass: Fine optimization from the result of the first
                x1 = best_pos.flatten()
                res2 = minimize(
                    _energy_and_grad,
                    x1,
                    args=(n,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                
                if res2.fun < best_energy:
                    best_energy = res2.fun
                    best_pos = res2.x.reshape(-1, 3)
                    
    except Exception as e:
        # If L-BFGS fails, fall back to the provided relax
        pass
    
    # Option 2: If L-BFGS didn't improve much or failed, try the provided relax
    # Also, we can try a small perturbation + L-BFGS to escape local minima if needed,
    # but Mackay is usually the global minimum for these sizes.
    
    # Check if we should try the built-in relax as a backup or refinement
    # The built-in relax might be robust.
    try:
        # Use the best position found so far as input to relax
        # Note: relax might modify the array in place or return new? 
        # "chem_clusters.relax(positions, 'dzugutov')"
        # Let's assume it returns relaxed positions or updates in place.
        # To be safe, we pass a copy.
        test_pos = best_pos.copy()
        relaxed_pos = relax(test_pos, potential)
        
        # Calculate energy of relaxed pos
        # We need to compute energy. The gradient function returns energy.
        e_relax, _ = dzugutov_energy_gradient(relaxed_pos)
        
        if e_relax < best_energy:
            best_energy = e_relax
            best_pos = relaxed_pos
    except:
        pass
    
    # Final check: ensure no atoms are too close (min distance 0.5)
    # L-BFGS might violate this if the potential is not hard-core.
    # The Dzugutov potential has a repulsive core (r^-16), so it should be fine.
    # But let's verify.
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    if min_dist < 0.5:
        # Fallback to a safer random structure if constraints are violated
        # This is unlikely with Dzugutov, but good for safety.
        rng = np.random.default_rng(123)
        radius = 2.0 * n**(1.0/3.0)
        safe_pos = rng.uniform(-radius, radius, (n, 3))
        safe_pos = safe_pos - np.mean(safe_pos, axis=0)
        
        # Relax the safe pos
        try:
            relaxed_safe = relax(safe_pos, potential)
            e_safe, _ = dzugutov_energy_gradient(relaxed_safe)
            if e_safe < best_energy:
                best_pos = relaxed_safe
        except:
            best_pos = safe_pos
            
    return best_pos
# EVOLVE-BLOCK-END
