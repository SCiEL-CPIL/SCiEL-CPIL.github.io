"""Dzugutov cluster global minima search using gradient-based L-BFGS-B on icosahedral templates."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
import time
from scipy.optimize import minimize

WORKING_LIMIT = 325.0
START_TIME = time.time()

def _remaining_time():
    return WORKING_LIMIT - (time.time() - START_TIME)

def _icosahedron_vertices():
    """12 vertices of a regular icosahedron centered at origin, normalized to unit circumradius."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / norms
    return v

def _build_mackay_55(spacing=1.0):
    """Build a 55-atom Mackay icosahedron: 1+12+42."""
    verts_unit = _icosahedron_vertices()
    
    # Layer 0: center
    layer0 = np.array([[0.0, 0.0, 0.0]])
    
    # Layer 1: 12 vertices at radius = spacing
    layer1 = verts_unit * spacing
    
    # Layer 2: 42 atoms - these are at the midpoints of edges of the icosahedron (12*5/2=30) 
    # plus additional atoms. Actually Mackay 55 = 1 + 12 + 42.
    # The 42 outer atoms form a truncated icosahedron-like shell.
    # Simpler approach: place them on a spherical shell at radius ~1.8*spacing
    # with icosahedral symmetry.
    
    # For a proper Mackay icosahedron, the outer 42 atoms are at positions:
    # - 30 edge midpoints of the inner icosahedron (at radius ~0.85*spacing from center... no)
    # Actually, let's use a simpler but effective approach:
    # Place 42 atoms on a sphere of radius 1.8*spacing with icosahedral distribution.
    
    # Better: use the known Mackay structure. The 55-atom Mackay icosahedron has:
    # 1 center, 12 first-shell, 42 second-shell.
    # Second shell positions can be generated from icosahedral symmetry operations.
    
    # Let me use a different approach: generate 42 points with icosahedral symmetry on a sphere.
    # The 42 points in the second shell of a Mackay icosahedron are at:
    # - 12 face centers (one per triangular face) at radius r2
    # - 30 edge midpoints at radius r2
    # That's 42! 
    
    # Face centers: centroids of the 20 triangular faces
    # Edge midpoints: midpoints of the 30 edges
    
    # First, get the 20 faces and 30 edges of the icosahedron
    # Icosahedron faces (20 triangles) - need to define them
    # Icosahedron edges (30) - need to define them
    
    # Let me compute these from the vertex positions
    v = verts_unit
    # Find edges: pairs with minimum distance
    min_dist = np.inf
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(v[i] - v[j])
            if d < min_dist:
                min_dist = d
    
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(v[i] - v[j])
            if abs(d - min_dist) < 1e-6:
                edges.append((i, j))
    
    # Find faces: triples of vertices that form triangles with all three edges present
    edge_set = set()
    for i, j in edges:
        edge_set.add((min(i,j), max(i,j)))
    
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                if (min(i,j), max(i,j)) in edge_set and \
                   (min(j,k), max(j,k)) in edge_set and \
                   (min(i,k), max(i,k)) in edge_set:
                    faces.append((i, j, k))
    
    # Now build the 42 outer atoms
    outer_atoms = []
    
    # 30 edge midpoints, scaled to appropriate radius
    # The edge midpoint radius should be such that the structure is compact
    # For Mackay, the outer shell radius is about 1.8 * inner shell radius
    r_outer = 1.8 * spacing
    r_edge_mid = 0.85 * spacing  # approximate
    
    for i, j in edges:
        mid = (v[i] + v[j]) / 2.0
        # Scale to the outer shell radius
        mid_norm = np.linalg.norm(mid)
        if mid_norm > 1e-10:
            mid_scaled = mid / mid_norm * r_outer
            outer_atoms.append(mid_scaled)
    
    # 12 face centers
    for i, j, k in faces:
        fc = (v[i] + v[j] + v[k]) / 3.0
        fc_norm = np.linalg.norm(fc)
        if fc_norm > 1e-10:
            fc_scaled = fc / fc_norm * r_outer
            outer_atoms.append(fc_scaled)
    
    layer2 = np.array(outer_atoms)
    
    return np.vstack([layer0, layer1, layer2])

def _build_icosahedral_template(n, spacing=1.0):
    """Build an icosahedral template for n atoms."""
    if n <= 13:
        verts = _icosahedron_vertices() * spacing
        center = np.array([[0.0, 0.0, 0.0]])
        if n == 1:
            return center
        elif n <= 12:
            return np.vstack([center, verts[:n-1]])
        else:
            return np.vstack([center, verts])
    
    # For larger n, start with Mackay 55 and trim/expand
    if n == 55:
        return _build_mackay_55(spacing)
    
    # For other sizes, build from icosahedral shells
    # Shells: 1, 12, 42, ...
    # n=38: 1+12+25 (partial second shell)
    # n=45: 1+12+32
    # n=50: 1+12+37
    # n=57: 1+12+42+2
    # n=60: 1+12+42+5
    
    verts_unit = _icosahedron_vertices()
    layer0 = np.array([[0.0, 0.0, 0.0]])
    layer1 = verts_unit * spacing
    
    # Generate all 42 outer shell positions
    v = verts_unit
    min_dist = np.inf
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(v[i] - v[j])
            if d < min_dist:
                min_dist = d
    
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(v[i] - v[j])
            if abs(d - min_dist) < 1e-6:
                edges.append((i, j))
    
    edge_set = set()
    for i, j in edges:
        edge_set.add((min(i,j), max(i,j)))
    
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                if (min(i,j), max(i,j)) in edge_set and \
                   (min(j,k), max(j,k)) in edge_set and \
                   (min(i,k), max(i,k)) in edge_set:
                    faces.append((i, j, k))
    
    r_outer = 1.8 * spacing
    outer_atoms = []
    for i, j in edges:
        mid = (v[i] + v[j]) / 2.0
        mid_norm = np.linalg.norm(mid)
        if mid_norm > 1e-10:
            mid_scaled = mid / mid_norm * r_outer
            outer_atoms.append(mid_scaled)
    for i, j, k in faces:
        fc = (v[i] + v[j] + v[k]) / 3.0
        fc_norm = np.linalg.norm(fc)
        if fc_norm > 1e-10:
            fc_scaled = fc / fc_norm * r_outer
            outer_atoms.append(fc_scaled)
    
    # We have 42 outer positions. Need n-13 of them.
    n_outer_needed = n - 13
    if n_outer_needed <= 0:
        return np.vstack([layer0, layer1[:max(0, n-1)]])
    elif n_outer_needed <= 42:
        # Take the first n_outer_needed outer atoms
        # Sort by some criterion - take those closest to ideal spacing
        outer_atoms = np.array(outer_atoms[:n_outer_needed])
        return np.vstack([layer0, layer1, outer_atoms])
    else:
        # More than 55 atoms - add more shell
        # For now, just return 55 and let optimization handle it
        return _build_mackay_55(spacing)

def _dzugutov_energy_flat(x, n, A=5.82, B=1.28, a=1.87, b=1.94, c=1.10, d=0.27, m=16):
    """Compute Dzugutov energy from flat coordinate array."""
    pos = x.reshape(n, 3)
    E = 0.0
    for i in range(n):
        for j in range(i+1, n):
            r = np.linalg.norm(pos[i] - pos[j])
            if r < 1e-10:
                r = 1e-10
            if r < a:
                E += A * (r**(-m) - B) * np.exp(c / (r - a))
            elif r < b:
                E += B * np.exp(d / (r - b))
    return E

def _dzugutov_gradient_flat(x, n, A=5.82, B=1.28, a=1.87, b=1.94, c=1.10, d=0.27, m=16):
    """Compute Dzugutov gradient from flat coordinate array."""
    pos = x.reshape(n, 3)
    grad = np.zeros_like(x)
    for i in range(n):
        for j in range(i+1, n):
            rij = pos[i] - pos[j]
            r = np.linalg.norm(rij)
            if r < 1e-10:
                continue
            if r < a:
                # V = A(r^-m - B)exp(c/(r-a))
                # dV/dr = A[-m*r^(-m-1)*exp(c/(r-a)) + (r^-m - B)*exp(c/(r-a))*(-c/(r-a)^2)]
                term1 = -m * r**(-m-1)
                term2 = (r**(-m) - B) * (-c / (r - a)**2)
                dVdr = A * (term1 + term2) * np.exp(c / (r - a))
            elif r < b:
                # V = B*exp(d/(r-b))
                # dV/dr = B*exp(d/(r-b))*(-d/(r-b)^2)
                dVdr = B * np.exp(d / (r - b)) * (-d / (r - b)**2)
            else:
                dVdr = 0.0
            
            # Force on i: -dV/dr * rij/r
            # Force on j: +dV/dr * rij/r
            if r > 1e-10:
                f = dVdr * rij / r
                grad[i*3:(i+1)*3] += f
                grad[j*3:(j+1)*3] -= f
    return grad

def _lbfgs_optimize(positions, n, max_time=15.0):
    """Optimize positions using L-BFGS-B with Dzugutov gradient."""
    x0 = positions.flatten()
    
    def fun(x):
        return _dzugutov_energy_flat(x, n)
    
    def jac(x):
        return _dzugutov_gradient_flat(x, n)
    
    start = time.time()
    try:
        result = minimize(fun, x0, jac=jac, method='L-BFGS-B', 
                         options={'maxiter': 200, 'maxfun': 500, 'gtol': 1e-8,
                                  'disp': False})
        if result.success or result.fun < 1e10:
            return result.fun, result.x.reshape(n, 3)
        else:
            return result.fun, result.x.reshape(n, 3)
    except Exception:
        return _dzugutov_energy_flat(x0, n), positions

def _basin_hopping(positions, n, max_iterations=20, rng=None):
    """Basin hopping with L-BFGS-B relaxation."""
    if rng is None:
        rng = np.random.default_rng()
    
    current_energy, current_pos = _lbfgs_optimize(positions, n)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    amplitude = 0.3
    beta = 1.0 / 0.3
    
    for it in range(max_iterations):
        if _remaining_time() < 10.0:
            break
        
        perturbed = current_pos + rng.normal(0, amplitude, positions.shape)
        new_energy, new_pos = _lbfgs_optimize(perturbed, n, max_time=5.0)
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-beta * delta_e):
            current_energy = new_energy
            current_pos = new_pos
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
        
        amplitude *= 1.05
    
    return best_energy, best_pos

def _ensure_min_distance(positions, min_dist=0.5, max_iter=30):
    """Ensure all pairs are at least min_dist apart."""
    positions = positions.copy()
    n = len(positions)
    
    for _ in range(max_iter):
        if _remaining_time() < 5.0:
            break
        min_d = np.inf
        min_pair = (0, 1)
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_d:
                    min_d = d
                    min_pair = (i, j)
        
        if min_d >= min_dist:
            break
        
        i, j = min_pair
        direction = (positions[i] - positions[j]) / (min_d + 1e-10)
        push = (min_dist - min_d) / 2 + 0.01
        positions[i] += direction * push
        positions[j] -= direction * push
    
    return positions

def construct(instance):
    """Find low-energy Dzugutov clusters using gradient-based optimization on icosahedral templates."""
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    global START_TIME
    START_TIME = time.time()
    
    rng = np.random.default_rng(n * 7 + 13)
    
    best_energy = np.inf
    best_positions = None
    
    # Strategy: multiple icosahedral templates with different spacings, optimized via L-BFGS-B
    # Then basin hopping from the best results
    
    n_starts = 4
    spacings = [0.95, 1.0, 1.05, 1.1]
    
    for start_idx in range(n_starts):
        if _remaining_time() < 30.0:
            break
        
        spacing = spacings[start_idx % len(spacings)]
        
        # Build icosahedral template
        template = _build_icosahedral_template(n, spacing)
        
        # Add small random perturbation to break symmetry
        perturbed = template + rng.normal(0, 0.05, template.shape)
        
        # Optimize with L-BFGS-B
        energy, pos = _lbfgs_optimize(perturbed, n, max_time=20.0)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = pos.copy()
        
        # Basin hopping from this result
        if _remaining_time() > 20.0:
            bh_energy, bh_pos = _basin_hopping(pos, n, max_iterations=15, rng=rng)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_pos.copy()
    
    # Final relaxation using the provided relax function
    if best_positions is not None:
        try:
            final_energy, final_pos = relax(best_positions, potential, parameter)
            if final_energy < best_energy:
                best_energy = final_energy
                best_positions = final_pos
        except Exception:
            pass
    
    # Ensure minimum distance
    if best_positions is not None:
        best_positions = _ensure_min_distance(best_positions)
    
    return best_positions
# EVOLVE-BLOCK-END
