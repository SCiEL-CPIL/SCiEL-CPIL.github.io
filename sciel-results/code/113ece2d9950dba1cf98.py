"""Lennard-Jones 55-atom cluster global minimum search via radial icosahedral parameterization."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 55
TIME_LIMIT = 140.0  # safety margin before 150s kill

def _lennard_jones_energy(positions):
    """Compute total Lennard-Jones energy for a set of positions."""
    n = len(positions)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = positions[i] - positions[j]
            r2 = np.dot(d, d)
            r6 = r2 ** 3
            r12 = r6 * r6
            total += 4.0 * (r12 / r6 - 1.0)
    return total

def _lennard_jones_energy_vec(positions):
    """Vectorized Lennard-Jones energy computation."""
    n = len(positions)
    # Compute all pairwise differences
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    # Mask diagonal (self-interactions)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** 3
    r12 = r6 * r6
    energy = 4.0 * np.sum(r12 / r6 - 1.0)
    # We counted each pair twice, so divide by 2
    return energy / 2.0

def _generate_mackay_icosahedron():
    """Generate the 55-atom Mackay icosahedron structure.
    
    The Mackay icosahedron has:
    - 1 atom at center
    - 12 atoms on the inner shell (icosahedron vertices) at radius r1
    - 42 atoms on the outer shell (icosahedron vertices + edge midpoints + face centers) at radius r2
    
    Actually, the standard Mackay icosahedron for 55 atoms:
    - Shell 0: 1 atom at origin
    - Shell 1: 12 atoms at vertices of regular icosahedron, radius ~1.05
    - Shell 2: 42 atoms, radius ~2.10
    
    The 42 outer atoms consist of:
    - 12 at vertices (same directions as inner shell)
    - 30 at edge midpoints
    
    Let me construct this properly using icosahedron geometry.
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2.0
    
    # 12 vertices of a regular icosahedron (centered at origin)
    # Using the standard construction with (0, ±1, ±phi), (±1, ±phi, 0), (±phi, 0, ±1)
    verts = []
    for a in [1, -1]:
        for b in [1, -1]:
            verts.append([0, a, b * phi])
            verts.append([a, b * phi, 0])
            verts.append([b * phi, 0, a])
    verts = np.array(verts, dtype=float)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    
    # The 55-atom Mackay icosahedron:
    # 1 center + 12 inner + 42 outer
    # Inner shell: 12 atoms at icosahedron vertices, radius r1
    # Outer shell: 42 atoms at radius r2
    #   - 12 at icosahedron vertex directions
    #   - 30 at edge midpoint directions
    
    # Edge midpoints of icosahedron:
    # An icosahedron has 30 edges. Each edge connects two vertices.
    # We need to find all unique edges.
    
    # Two vertices are connected by an edge if their dot product is close to a specific value.
    # For a regular icosahedron with unit vertices, edge length = 2/sqrt(phi^2+1) ... 
    # Actually, let's compute: the edge length of the icosahedron with these vertices.
    # Distance between (0,1,phi) and (1,phi,0):
    # sqrt(1 + (phi-1)^2 + phi^2) = sqrt(1 + (1/phi)^2 + phi^2)
    # = sqrt(1 + 1/phi^2 + phi^2)
    # phi^2 = phi + 1 = 2.618..., 1/phi^2 = 1/(phi+1) = phi-1 = 0.618...
    # So sqrt(1 + 0.618 + 2.618) = sqrt(4.236) ≈ 2.058
    # The dot product of adjacent vertices: (0,1,phi)·(1,phi,0) = phi
    # So adjacent vertices have dot product = phi ≈ 1.618
    
    # Find all pairs with dot product close to phi
    dot_products = verts @ verts.T
    edge_pairs = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dot_products[i, j] - phi) < 1e-6:
                edge_pairs.append((i, j))
    
    # There should be 30 edges
    # Generate edge midpoint directions
    edge_mid_dirs = []
    for i, j in edge_pairs:
        mid = verts[i] + verts[j]
        mid_norm = np.linalg.norm(mid)
        if mid_norm > 1e-10:
            edge_mid_dirs.append(mid / mid_norm)
    edge_mid_dirs = np.array(edge_mid_dirs, dtype=float)
    
    # Now construct the 55-atom structure
    # Optimal radii for LJ: r1 ≈ 1.05, r2 ≈ 2.10 (roughly)
    # Let's use initial estimates and let optimization refine
    r1 = 1.05
    r2 = 2.10
    
    positions = np.zeros((55, 3))
    # Atom 0: center
    positions[0] = [0, 0, 0]
    # Atoms 1-12: inner shell
    for i in range(12):
        positions[1 + i] = r1 * verts[i]
    # Atoms 13-24: outer shell at vertex directions
    for i in range(12):
        positions[13 + i] = r2 * verts[i]
    # Atoms 25-54: outer shell at edge midpoint directions
    for i in range(30):
        positions[25 + i] = r2 * edge_mid_dirs[i]
    
    return positions, verts, edge_mid_dirs

def _radial_to_cartesian(radii, verts, edge_mid_dirs):
    """Convert 55 radial parameters to Cartesian coordinates.
    
    radii[0] = center (should be 0 for symmetry, but we allow it)
    radii[1:13] = inner shell radii (12 atoms)
    radii[13:25] = outer shell vertex radii (12 atoms)
    radii[25:55] = outer shell edge midpoint radii (30 atoms)
    """
    positions = np.zeros((55, 3))
    # Center atom - keep at origin for symmetry
    positions[0] = [0, 0, 0]
    
    # Inner shell
    for i in range(12):
        positions[1 + i] = radii[1 + i] * verts[i]
    
    # Outer shell vertices
    for i in range(12):
        positions[13 + i] = radii[13 + i] * verts[i]
    
    # Outer shell edge midpoints
    for i in range(30):
        positions[25 + i] = radii[25 + i] * edge_mid_dirs[i]
    
    return positions

def _energy_from_radii(radii, verts, edge_mid_dirs):
    """Compute LJ energy from radial parameters."""
    positions = _radial_to_cartesian(radii, verts, edge_mid_dirs)
    return _lennard_jones_energy_vec(positions)

def _minimize_cartesian(positions, time_budget):
    """Minimize energy in Cartesian space using L-BFGS-B."""
    start_time = time.time()
    deadline = time_budget - 5.0  # safety margin
    
    best_energy = _lennard_jones_energy_vec(positions)
    best_pos = positions.copy()
    
    # Try multiple restarts with different initial perturbations
    rng = np.random.RandomState(42)
    
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Add small random perturbation
        if restart == 0:
            pos = positions.copy()
        else:
            pos = positions + rng.normal(0, 0.1, size=positions.shape)
        
        # Ensure minimum distance constraint
        # Check pairwise distances
        valid = True
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                if np.linalg.norm(pos[i] - pos[j]) < 0.5:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        try:
            result = minimize(
                _lennard_jones_energy_vec,
                pos,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.copy()
        except Exception:
            continue
        
        if time.time() > deadline:
            break
    
    return best_pos, best_energy

def _differential_evolution_radial(verts, edge_mid_dirs, time_budget):
    """Use differential evolution on radial parameters."""
    start_time = time.time()
    deadline = time_budget - 10.0  # leave time for final polish
    
    # Define bounds for radii
    # Center atom radius: keep small (0 to 0.5)
    # Inner shell: 0.8 to 1.5
    # Outer shell: 1.8 to 2.5
    
    bounds = [(0.0, 0.5)]  # center
    bounds += [(0.8, 1.5)] * 12  # inner shell
    bounds += [(1.8, 2.5)] * 12  # outer shell vertices
    bounds += [(1.8, 2.5)] * 30  # outer shell edge midpoints
    
    def objective(radii):
        pos = _radial_to_cartesian(radii, verts, edge_mid_dirs)
        # Check minimum distance constraint
        for i in range(55):
            for j in range(i + 1, 55):
                if np.linalg.norm(pos[i] - pos[j]) < 0.5:
                    return 1e6
        return _lennard_jones_energy_vec(pos)
    
    # Initial population around Mackay structure
    r1 = 1.05
    r2 = 2.10
    x0 = np.zeros(55)
    x0[0] = 0.0
    x0[1:13] = r1
    x0[13:25] = r2
    x0[25:55] = r2
    
    # Run differential evolution with time limit
    # We'll do a limited number of iterations and check time
    max_iter = 100
    tol = 1e-8
    
    # Use scipy's differential evolution
    try:
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=max_iter,
            tol=tol,
            popsize=15,
            polish=False,  # We'll do our own polish
            updating='immediate'
        )
        
        best_radii = result.x
        best_energy = result.fun
    except Exception:
        best_radii = x0
        best_energy = objective(x0)
    
    # Check time and do local polish
    if time.time() < deadline:
        positions = _radial_to_cartesian(best_radii, verts, edge_mid_dirs)
        polished_pos, polished_energy = _minimize_cartesian(positions, time.time() - start_time + time_budget)
        
        if polished_energy < best_energy:
            best_energy = polished_energy
            # Convert back to radii for consistency
            # Actually, we'll just return the Cartesian positions
            return polished_pos, polished_energy
    
    positions = _radial_to_cartesian(best_radii, verts, edge_mid_dirs)
    return positions, best_energy

def construct():
    """Construct 55-atom LJ cluster configuration."""
    start_time = time.time()
    
    # Generate Mackay icosahedron geometry
    mackay_pos, verts, edge_mid_dirs = _generate_mackay_icosahedron()
    
    # Initial energy
    initial_energy = _lennard_jones_energy_vec(mackay_pos)
    best_pos = mackay_pos.copy()
    best_energy = initial_energy
    
    # Strategy 1: Radial optimization with differential evolution
    if time.time() - start_time < TIME_LIMIT - 30:
        try:
            pos_de, energy_de = _differential_evolution_radial(verts, edge_mid_dirs, TIME_LIMIT)
            if energy_de < best_energy:
                best_energy = energy_de
                best_pos = pos_de.copy()
        except Exception:
            pass
    
    # Strategy 2: Direct Cartesian optimization from Mackay structure
    if time.time() - start_time < TIME_LIMIT - 20:
        try:
            pos_cart, energy_cart = _minimize_cartesian(mackay_pos, TIME_LIMIT)
            if energy_cart < best_energy:
                best_energy = energy_cart
                best_pos = pos_cart.copy()
        except Exception:
            pass
    
    # Strategy 3: Additional Cartesian optimization from best found
    if time.time() - start_time < TIME_LIMIT - 15:
        try:
            pos_final, energy_final = _minimize_cartesian(best_pos, TIME_LIMIT)
            if energy_final < best_energy:
                best_energy = energy_final
                best_pos = pos_final.copy()
        except Exception:
            pass
    
    # Final check: ensure minimum distance constraint
    # If violated, project to satisfy constraint
    for _ in range(10):
        min_dist = float('inf')
        min_pair = (0, 1)
        for i in range(55):
            for j in range(i + 1, 55):
                d = np.linalg.norm(best_pos[i] - best_pos[j])
                if d < min_dist:
                    min_dist = d
                    min_pair = (i, j)
        
        if min_dist >= 0.5:
            break
        
        # Push apart the closest pair
        i, j = min_pair
        diff = best_pos[i] - best_pos[j]
        norm = np.linalg.norm(diff)
        if norm < 1e-10:
            diff = np.array([0.01, 0.01, 0.01])
            norm = np.linalg.norm(diff)
        direction = diff / norm
        shift = (0.5 - min_dist) / 2.0 + 0.01
        best_pos[i] += direction * shift
        best_pos[j] -= direction * shift
    
    return best_pos
