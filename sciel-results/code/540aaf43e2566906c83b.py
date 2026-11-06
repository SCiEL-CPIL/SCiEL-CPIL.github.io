# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import basinhopping
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16

def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy for a set of positions."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = dists[i, j]
            if r < 1e-8:
                continue
            if r < a:
                energy += A * (r**(-m) - B) * np.exp(c / (r - a))
            elif r < b:
                energy += B * np.exp(d / (r - b))
    return energy

def _generate_mackay_icosahedron(n_target):
    """
    Generate a Mackay icosahedron structure and truncate to n_target atoms.
    Mackay icosahedra have 1, 13, 55, 147 atoms.
    We generate the 55-atom and 147-atom structures and truncate as needed.
    """
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Generate points for a Mackay icosahedron of order M (M=1 is 13 atoms, M=2 is 55, M=3 is 147)
    # We need to cover up to 60 atoms, so we generate M=3 (147 atoms) and truncate.
    M = 3
    points = []
    
    # Center atom
    points.append(np.zeros(3))
    
    # Generate lattice points within the icosahedron
    # Use the icosahedral symmetry to generate points
    # The icosahedron can be defined by vertices at distance R.
    # For Mackay, the lattice is a subset of a simple cubic lattice rotated appropriately,
    # or more simply, we can use the known coordinates for the 147-atom cluster.
    
    # A simpler approach: Use the 12 vertices, 30 edge midpoints, 20 face centers, etc.
    # But generating the full 147-atom Mackay icosahedron exactly is complex.
    # Instead, we use a robust method: generate points on a 3D grid, rotate to align with icosahedral axes,
    # and keep those inside the icosahedron defined by the 12 vertices at a certain radius.
    
    # Actually, for Dzugutov, the "global minima" are often close to the 13, 55, 147 Mackay icosahedra.
    # Let's generate a dense set of points in a spherical volume and filter by icosahedral symmetry constraints?
    # No, better to use the explicit construction.
    
    # Explicit construction of 55-atom Mackay icosahedron:
    # 1 center, 12 vertices, 30 edge midpoints, 12 face centers? No.
    # 1 + 12 + 30 + 12 = 55.
    # Let's use the 147-atom one for n>55.
    
    # For simplicity and robustness, let's generate a BCC-like cluster but constrained to icosahedral symmetry?
    # Or just use the 13-atom and 55-atom structures as seeds and add atoms for larger n.
    
    # Let's generate the 55-atom Mackay icosahedron.
    # Vertices (12): distance R
    # Edge midpoints (30): distance R * cos(pi/10) ? No.
    # Face centers (12): distance R * cos(2*pi/5) ? No.
    
    # Let's use a different strategy: Generate a BCC cluster and an Icosahedral cluster.
    # The Icosahedral cluster will be built by placing atoms at the vertices, edge midpoints, and face centers of a regular icosahedron.
    
    # Vertices
    v = verts
    # Edge midpoints
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            # Check if i and j are connected in icosahedron
            # Distance between connected vertices is 2 * sin(pi/5) * R
            # R = 1.0
            d = np.linalg.norm(v[i] - v[j])
            # In a regular icosahedron with edge length a, distance between non-adjacent vertices is a*phi
            # Here our verts are normalized to unit sphere.
            # Edge length in unit sphere: 2 * sin(pi/5)
            edge_len = 2 * np.sin(np.pi/5)
            if abs(d - edge_len) < 1e-5:
                edges.append((v[i] + v[j]) / 2)
                
    # Face centers
    # 20 faces. We need to identify the triangles.
    # This is getting complicated. Let's use a simpler heuristic for the seed:
    # 1. Start with 13-atom icosahedron (center + 12 vertices).
    # 2. Add atoms at edge midpoints (30 atoms) -> 43 atoms.
    # 3. Add atoms at face centers (12 atoms? No, 20 faces) -> 63 atoms.
    # This gives 1 + 12 + 30 + 20 = 63 atoms.
    # We can truncate this to any n <= 63.
    
    # Let's build the 63-atom "super-icosahedron" seed.
    points_63 = [np.zeros(3)]
    points_63.extend(v)
    
    # Edge midpoints
    for mid in edges:
        points_63.append(mid)
        
    # Face centers
    # Identify faces by checking triples of vertices that form a triangle.
    # In a regular icosahedron, each face is an equilateral triangle.
    # Side length is edge_len.
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(v[i] - v[j])
                d2 = np.linalg.norm(v[j] - v[k])
                d3 = np.linalg.norm(v[k] - v[i])
                if abs(d1 - edge_len) < 1e-5 and abs(d2 - edge_len) < 1e-5 and abs(d3 - edge_len) < 1e-5:
                    # Check if this face is already found (avoid duplicates)
                    # The center of the face is the average of the 3 vertices
                    fc = (v[i] + v[j] + v[k]) / 3
                    # Normalize to unit sphere? No, the face center is inside the sphere.
                    # We want to place an atom there.
                    # But we need to ensure we don't add duplicates.
                    # There are 20 faces.
                    faces.append(fc)
                    
    # There might be duplicates in faces if we're not careful, but with i<j<k it should be unique.
    # However, the icosahedron has 20 faces.
    points_63.extend(faces)
    
    # Scale the structure. The Dzugutov potential minimum is around r ~ 1.8-2.0.
    # Our unit sphere has radius 1.
    # Let's scale by a factor to match the expected bond length.
    # Bond length in icosahedron is edge_len = 2*sin(pi/5) ~ 1.175.
    # We want bond length ~ 1.9.
    # Scale factor = 1.9 / 1.175 ~ 1.6.
    scale = 1.6
    points_63 = np.array(points_63) * scale
    
    # Center the cluster
    points_63 = points_63 - np.mean(points_63, axis=0)
    
    # Sort by distance from center to ensure consistent truncation
    dists = np.linalg.norm(points_63, axis=1)
    sorted_idx = np.argsort(dists)
    points_63 = points_63[sorted_idx]
    
    # Truncate to n_target
    if n_target <= len(points_63):
        return points_63[:n_target]
    else:
        # If n_target > 63, we need more atoms.
        # Generate a larger structure or just add random atoms around the surface.
        # For n > 63, we can use a BCC cluster as a fallback or extend the icosahedron.
        # Given n is at most 60, this branch is not taken.
        return points_63

def _generate_bcc_cluster(n):
    """Generate a BCC-like cluster structure."""
    lattice_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                lattice_points.append(np.array([i, j, k]))
                lattice_points.append(np.array([i + 0.5, j + 0.5, k + 0.5]))
    
    lattice_points = np.array(lattice_points)
    centers = np.linalg.norm(lattice_points, axis=1)
    sorted_idx = np.argsort(centers)
    points = lattice_points[sorted_idx[:n]]
    points = points - np.mean(points, axis=0)
    # Scale to match Dzugutov bond length
    points *= 1.6
    return points

def _generate_random_compact(n, rng):
    """Generate a random compact cluster."""
    radius = 1.1 * n**(1.0/3.0) * 0.75
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 for q in points):
            points.append(p)
    return np.array(points)

def construct(instance):
    """Find low-energy Dzugutov cluster structures."""
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 295.0  # 5 second safety margin before 300s limit
    
    # Generate diverse initial structures
    initial_structures = []
    
    # 1. Mackay Icosahedron (Best seed for Dzugutov)
    try:
        ico = _generate_mackay_icosahedron(n)
        initial_structures.append(ico)
    except Exception as e:
        print(f"Ico gen error: {e}")
        
    # 2. BCC
    try:
        bcc = _generate_bcc_cluster(n)
        initial_structures.append(bcc)
    except Exception as e:
        print(f"BCC gen error: {e}")
        
    # 3. Random compact
    rng = np.random.default_rng(n)
    try:
        rand = _generate_random_compact(n, rng)
        initial_structures.append(rand)
    except Exception as e:
        print(f"Rand gen error: {e}")
        
    best_energy = np.inf
    best_positions = None
    
    # Define the objective function for basin hopping
    def objective(x_flat):
        x = x_flat.reshape(n, 3)
        # Check minimum distance constraint
        if np.any(np.linalg.norm(x[:, np.newaxis, :] - x[np.newaxis, :, :], axis=2) < 0.5):
            return 1e10
        return _dzugutov_energy(x)
    
    def gradient(x_flat):
        x = x_flat.reshape(n, 3)
        e, g = dzugutov_energy_gradient(x, potential)
        return g.flatten()
        
    # Local minimizer using L-BFGS-B
    def local_minimizer(x_flat):
        from scipy.optimize import minimize
        res = minimize(objective, x_flat, jac=gradient, method='L-BFGS-B', 
                       options={'maxiter': 100, 'gtol': 1e-6})
        return res.fun, res.x, {'nit': res.nit, 'nfev': res.nfev, 'njev': res.njev}
        
    # Basin Hopping parameters
    bh_kwargs = {
        'niter': 50,  # Number of basin hopping iterations
        'T': 0.5,     # Temperature
        'stepsize': 0.5, # Step size for Monte Carlo moves
        'minimizer_kwargs': {'maxiter': 100},
        'seed': 42
    }
    
    # Run Basin Hopping from each initial structure
    for i, init_pos in enumerate(initial_structures):
        if time.time() > deadline - 10:
            break
            
        init_flat = init_pos.flatten()
        
        # Check initial validity
        if objective(init_flat) > 1e9:
            continue
            
        try:
            res = basinhopping(objective, init_flat, minimizer=local_minimizer, 
                               **bh_kwargs)
            if res.fun < best_energy:
                best_energy = res.fun
                best_positions = res.x.reshape(n, 3)
        except Exception as e:
            print(f"Basin hopping error: {e}")
            
    # Final polish with high-precision local relaxation
    if best_positions is not None and time.time() < deadline - 5:
        try:
            # Use the provided relax function for final precision
            energy, relaxed_pos = relax(best_positions, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos
        except Exception as e:
            print(f"Final relax error: {e}")
            
    # Fallback
    if best_positions is None:
        try:
            energy, positions = relax(initial_structures[0], potential, parameter)
            return positions
        except:
            return initial_structures[0]
            
    return best_positions
# EVOLVE-BLOCK-END
