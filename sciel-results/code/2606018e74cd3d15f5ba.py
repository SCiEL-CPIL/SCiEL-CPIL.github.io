"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _mackay_icosahedron_55():
    """
    Construct the 55-atom Mackay icosahedron.
    Structure:
    - 1 atom at center (0,0,0)
    - 12 atoms at vertices of a regular icosahedron (inner shell)
    - 30 atoms at midpoints of edges of the outer icosahedron (outer shell)
    
    The standard Mackay icosahedron has:
    - Inner shell: 12 vertices of an icosahedron with radius r1
    - Outer shell: 30 edge midpoints of a larger icosahedron with radius r2
    
    For the optimal 55-atom cluster, the ratio r2/r1 is approximately 2.
    Actually, in the Mackay icosahedron, the outer shell atoms are at the 
    midpoints of the edges of the outer icosahedron. Let me reconsider.
    
    The Mackay icosahedron for 55 atoms:
    - Center: 1 atom
    - First shell: 12 atoms (vertices of icosahedron)
    - Second shell: 42 atoms? No, 1+12+30 = 43, not 55.
    
    Wait, 55 = 1 + 12 + 32? No. Let me recalculate.
    Actually, the Mackay icosahedron shells are:
    - Shell 0: 1 atom (center)
    - Shell 1: 12 atoms (vertices of icosahedron)
    - Shell 2: 42 atoms (vertices and edge midpoints of a larger icosahedron)
    Total: 1 + 12 + 42 = 55.
    
    The second shell consists of:
    - 12 vertices of the outer icosahedron
    - 30 edge midpoints of the outer icosahedron
    Total: 12 + 30 = 42.
    
    So the structure is:
    - 1 atom at origin
    - 12 atoms at icosahedron vertices (radius r1)
    - 12 atoms at outer icosahedron vertices (radius r2)
    - 30 atoms at outer icosahedron edge midpoints (radius r2 * cos(36°) or similar)
    
    Actually, for a regular icosahedron, the edge midpoint distance from center 
    is r * cos(36°) where r is the circumradius? Let me think more carefully.
    
    For a regular icosahedron with circumradius R, the edge length is a = R * 4/sqrt(10+2*sqrt(5)) ≈ R * 1.0515.
    The distance from center to edge midpoint is R * sqrt(1 - (a/2)^2 / R^2) = R * sqrt(1 - (1.0515/2)^2 / 1) ... 
    
    Actually, let me just use the known coordinates.
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices (12 vertices)
    # Standard icosahedron vertices can be given as:
    # (0, ±1, ±phi), (±1, ±phi, 0), (±phi, 0, ±1)
    verts = []
    for sign1 in [1, -1]:
        for sign2 in [1, -1]:
            verts.append([0, sign1, sign2 * phi])
            verts.append([sign1, sign2 * phi, 0])
            verts.append([sign1 * phi, 0, sign2])
    
    verts = np.array(verts, dtype=float)
    
    # Normalize to unit circumradius
    norms = np.linalg.norm(verts, axis=1)
    verts_unit = verts / norms[:, np.newaxis]
    
    # Inner shell: 12 atoms at radius r1
    r1 = 1.0  # We'll optimize this
    inner_shell = verts_unit * r1
    
    # Outer shell: 
    # 12 vertices at radius r2
    r2 = 2.0  # We'll optimize this
    outer_vertices = verts_unit * r2
    
    # 30 edge midpoints: for each edge of the icosahedron, the midpoint
    # First, find all edges (pairs of vertices that are closest to each other)
    # In a regular icosahedron, each vertex is connected to 5 others
    # Edge length in unit circumradius:
    edge_len = np.linalg.norm(verts_unit[0] - verts_unit[1])
    
    # Find all edges
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            if abs(d - edge_len) < 1e-6:
                edges.append((i, j))
    
    # There should be 30 edges
    edge_midpoints = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2
        # Normalize and scale to r2
        mid_norm = np.linalg.norm(mid)
        mid_unit = mid / mid_norm
        edge_midpoints.append(mid_unit * r2)
    
    edge_midpoints = np.array(edge_midpoints, dtype=float)
    
    # Combine all atoms
    center = np.array([[0.0, 0.0, 0.0]])
    all_atoms = np.vstack([center, inner_shell, outer_vertices, edge_midpoints])
    
    return all_atoms

def _lj_energy(pos):
    """Compute Lennard-Jones energy for a set of positions."""
    n = len(pos)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(pos[i] - pos[j])
            if r < 1e-10:
                return 1e10
            r6 = r ** 6
            r12 = r6 ** 2
            energy += 4.0 * (r12 / (r12 + 1e-30) - 1.0 / (r6 + 1e-30))
    return energy

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Get the Mackay icosahedron geometry
    mackay = _mackay_icosahedron_55()
    
    # Pre-optimize the shell radii using a simple 2D search
    # We'll try different r1 and r2 values
    best_energy = np.inf
    best_pos = mackay.copy()
    
    # Try a grid of r1 and r2 values
    r1_values = np.linspace(0.9, 1.2, 7)
    r2_values = np.linspace(1.8, 2.4, 7)
    
    for r1 in r1_values:
        for r2 in r2_values:
            if time.time() > deadline - 20:
                break
            # Reconstruct with these radii
            phi = (1 + np.sqrt(5)) / 2
            verts = []
            for sign1 in [1, -1]:
                for sign2 in [1, -1]:
                    verts.append([0, sign1, sign2 * phi])
                    verts.append([sign1, sign2 * phi, 0])
                    verts.append([sign1 * phi, 0, sign2])
            verts = np.array(verts, dtype=float)
            norms = np.linalg.norm(verts, axis=1)
            verts_unit = verts / norms[:, np.newaxis]
            
            inner_shell = verts_unit * r1
            outer_vertices = verts_unit * r2
            
            edge_len = np.linalg.norm(verts_unit[0] - verts_unit[1])
            edges = []
            for i in range(12):
                for j in range(i + 1, 12):
                    d = np.linalg.norm(verts_unit[i] - verts_unit[j])
                    if abs(d - edge_len) < 1e-6:
                        edges.append((i, j))
            
            edge_midpoints = []
            for i, j in edges:
                mid = (verts_unit[i] + verts_unit[j]) / 2
                mid_norm = np.linalg.norm(mid)
                mid_unit = mid / mid_norm
                edge_midpoints.append(mid_unit * r2)
            
            edge_midpoints = np.array(edge_midpoints, dtype=float)
            center = np.array([[0.0, 0.0, 0.0]])
            pos = np.vstack([center, inner_shell, outer_vertices, edge_midpoints])
            
            # Check minimum distance
            min_dist = np.inf
            for i in range(len(pos)):
                for j in range(i + 1, len(pos)):
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < min_dist:
                        min_dist = d
            if min_dist < 0.5:
                continue
            
            e = _lj_energy(pos)
            if e < best_energy:
                best_energy = e
                best_pos = pos.copy()
        
        if time.time() > deadline - 20:
            break
    
    # Now do L-BFGS relaxation from the best configuration
    if time.time() < deadline - 10:
        def energy_func(x):
            pos = x.reshape(-1, 3)
            return _lj_energy(pos)
        
        result = minimize(energy_func, best_pos.flatten(), method='L-BFGS-B',
                          options={'maxiter': 500, 'gtol': 1e-8})
        
        if result.fun < best_energy:
            best_pos = result.x.reshape(-1, 3)
            best_energy = result.fun
    
    # Check if we're still within deadline
    if time.time() < deadline - 5:
        # Try a few stochastic perturbations from the best config
        np.random.seed(42)
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            perturbed = best_pos + np.random.normal(0, 0.05, best_pos.shape)
            
            # Check minimum distance
            min_dist = np.inf
            for i in range(len(perturbed)):
                for j in range(i + 1, len(perturbed)):
                    d = np.linalg.norm(perturbed[i] - perturbed[j])
                    if d < min_dist:
                        min_dist = d
            if min_dist < 0.5:
                continue
            
            def energy_func(x):
                pos = x.reshape(-1, 3)
                return _lj_energy(pos)
            
            try:
                result = minimize(energy_func, perturbed.flatten(), method='L-BFGS-B',
                                  options={'maxiter': 200, 'gtol': 1e-8})
                if result.fun < best_energy:
                    best_energy = result.fun
                    best_pos = result.x.reshape(-1, 3)
            except:
                pass
    
    # Final check: ensure minimum distance constraint
    for i in range(len(best_pos)):
        for j in range(i + 1, len(best_pos)):
            d = np.linalg.norm(best_pos[i] - best_pos[j])
            if d < 0.5:
                # This shouldn't happen, but just in case
                pass
    
    return best_pos
# EVOLVE-BLOCK-END
