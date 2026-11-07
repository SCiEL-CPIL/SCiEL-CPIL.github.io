"""
Strategy: Vectorized Multi-Start Basin Hopping with Potential-Specific Templates.
Key differences from previous approach:
1. Explicit handling of the 10-8 potential (Pd-like) which favors BCC/Icosahedral structures,
   whereas 12-6 (Ni-like) and 9-6 (Al-like) favor FCC.
2. Replaces the slow, single-step basin hopping with a tight loop that maximizes
   the number of (perturb + relax) cycles within the time budget.
3. Uses a global deadline guard to ensure we return the best valid result found
   before the evaluator kills the process.
"""

# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Deadline guard: Evaluator kills at 330s. We aim for 290s to be safe.
_DEADLINE_SEC = 290.0
_START_TIME = None

def _time_left():
    return _DEADLINE_SEC - (time.time() - _START_TIME)

def _fcc_fragment(n, spacing=1.0):
    """Generate n atoms from an FCC lattice fragment."""
    # FCC lattice points: (x, y, z) where x, y, z are integers and x+y+z is even.
    # Scaled by a/2.
    points = []
    max_coord = int(np.ceil(n ** (1/3))) + 4
    a = spacing
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            for z in range(-max_coord, max_coord + 1):
                if (x + y + z) % 2 == 0:
                    points.append([x * 0.5 * a, y * 0.5 * a, z * 0.5 * a])
    
    if len(points) < n:
        return None
        
    points = np.array(points)
    # Sort by distance to center (0,0,0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _bcc_fragment(n, spacing=1.0):
    """Generate n atoms from a BCC lattice fragment."""
    points = []
    max_coord = int(np.ceil(n ** (1/3))) + 4
    a = spacing
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # Corner
                points.append([i*a, j*a, k*a])
                # Body center
                points.append([(i+0.5)*a, (j+0.5)*a, (k+0.5)*a])
    
    if len(points) < n:
        return None
        
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _icosahedral_fragment(n, spacing=1.0):
    """
    Generate a Mackay icosahedral cluster.
    This is a strong local minimum for many SC potentials, especially when
    the bulk structure is not strictly FCC (like 10-8 Pd).
    """
    points = []
    # Mackay icosahedron layers: 1, 12, 42, 92, ...
    # We need to generate the vertices.
    # This is complex to generate exactly. 
    # Alternative: Use a dense random compact cluster and let relaxer find icosahedron?
    # Or use a specific icosahedral grid.
    # For simplicity and robustness, we will use a "random compact" with high density 
    # which often falls into icosahedral traps for small n, or BCC for larger n.
    # However, for n ~ 50-80, icosahedra are very competitive.
    # Let's implement a simple icosahedral generator for the first few shells.
    
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of a regular icosahedron
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    # Normalize to unit radius
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0])
    
    # Center
    points.append([0, 0, 0])
    
    # First shell: 12 atoms
    shell1_radius = spacing * 1.0
    for v in ico_verts:
        points.append(v * shell1_radius)
        
    if n <= 13:
        return np.array(points[:n])
        
    # Second shell: 42 atoms
    # These are located at the vertices of the icosahedron scaled up, 
    # plus points on the faces.
    # A simpler approximation for a "compact icosahedral-like" cluster 
    # is to take the FCC fragment but distort it, or just rely on 
    # random compact + relax. 
    # Given the complexity of exact Mackay generation, we will 
    # generate a dense random cluster here as a proxy for "non-FCC" 
    # high-symmetry local minima, as random compact often finds icosahedra 
    # for n < 50 and BCC-like for larger n in SC potentials.
    
    # Actually, let's just return None here and let the random compact 
    # handles the "exotic" structures, or we can use a BCC fragment 
    # which is a good competitor for 10-8.
    return None

def _random_compact(n, spacing=0.72, seed=0):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    # Radius estimation for close packing
    # Volume of sphere ~ n * volume of atom
    # r_atom ~ spacing / 2
    # R ~ (3n/4)^(1/3) * spacing
    radius = spacing * (3 * n / 4.0) ** (1.0 / 3.0) * 1.2
    points = []
    # Use rejection sampling to ensure minimum distance
    min_dist = 0.6 * spacing
    attempts = 0
    while len(points) < n and attempts < n * 200:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Vectorized distance check for speed
            if not points:
                points.append(p)
            else:
                pts_arr = np.array(points)
                dists = np.linalg.norm(pts_arr - p, axis=1)
                if np.min(dists) >= min_dist:
                    points.append(p)
        attempts += 1
    if len(points) < n:
        # Fallback: just random points, relaxer will fix overlaps
        points = rng.uniform(-radius, radius, (n, 3)).tolist()
    return np.array(points)

def construct(instance):
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance["kind"]
    potential = instance["potential"]
    
    # Sutton-Chen equilibrium distance is roughly 0.72-0.75 in these reduced units
    spacing = 0.72
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate diverse starting points based on potential kind
    starts = []
    
    # 1. Structure-specific templates
    if kind == '12-6' or kind == '9-6':
        # FCC is the bulk structure
        fcc = _fcc_fragment(n, spacing)
        if fcc is not None:
            starts.append(fcc)
        # HCP is a close competitor
        # (Implementing HCP is similar to FCC but with hexagonal layers. 
        # For speed, we skip explicit HCP generation and rely on 
        # random compact + basin hopping to find HCP if it's better, 
        # or we can add a simple HCP generator if time permits. 
        # Given time constraints, FCC + Random is usually sufficient.)
        
    elif kind == '10-8':
        # BCC or Icosahedral is often the bulk/local minimum
        bcc = _bcc_fragment(n, spacing)
        if bcc is not None:
            starts.append(bcc)
        # Also try FCC as a competitor
        fcc = _fcc_fragment(n, spacing)
        if fcc is not None:
            starts.append(fcc)

    # 2. Random compact starts (multiple seeds)
    # We generate a few random starts to cover other local minima
    for seed in range(8):
        rand = _random_compact(n, spacing, seed)
        starts.append(rand)
        
    # Relax each start
    for start in starts:
        if _time_left() < 10:
            break
        try:
            energy, positions = relax(start, potential, kind)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
            
    # If we have time, do Basin Hopping on the best found
    if best_positions is not None and _time_left() > 10:
        current_best = best_positions.copy()
        current_energy = best_energy
        
        # Adaptive step size
        step_size = 0.15
        rng = np.random.default_rng(42)
        
        # Tight loop for basin hopping
        while _time_left() > 5:
            # Perturb
            noise = rng.normal(0, step_size, (n, 3))
            perturbed = current_best + noise
            
            try:
                energy, positions = relax(perturbed, potential, kind)
                
                # Check validity (min distance > 0.35)
                # The relaxer should handle this, but let's be safe.
                # If invalid, skip.
                # Note: sutton_chen_energy_gradient returns energy. 
                # If positions are invalid, energy might be inf or nan.
                if np.isfinite(energy) and energy < current_energy:
                    current_energy = energy
                    current_best = positions
                    # Decrease step size slightly on success
                    step_size *= 0.98
                else:
                    # Increase step size slightly if no improvement
                    step_size *= 1.02
                    # Cap step size
                    if step_size > 0.5:
                        step_size = 0.5
            except Exception:
                continue
                
        if current_energy < best_energy:
            best_energy = current_energy
            best_positions = current_best

    # Final safety check: ensure no atoms are too close
    if best_positions is not None:
        # Check min distance
        # O(n^2) check is fast for n < 100
        min_d = float('inf')
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(best_positions[i] - best_positions[j])
                if d < min_d:
                    min_d = d
        if min_d < 0.35:
            # If invalid, we must return something valid.
            # Try to find a valid start from our initial list.
            for start in starts:
                try:
                    energy, positions = relax(start, potential, kind)
                    # Check validity
                    min_d_check = float('inf')
                    for i in range(n):
                        for j in range(i+1, n):
                            d = np.linalg.norm(positions[i] - positions[j])
                            if d < min_d_check:
                                min_d_check = d
                    if min_d_check > 0.35:
                        best_positions = positions
                        best_energy = energy
                        break
                except Exception:
                    continue

    return best_positions
# EVOLVE-BLOCK-END
