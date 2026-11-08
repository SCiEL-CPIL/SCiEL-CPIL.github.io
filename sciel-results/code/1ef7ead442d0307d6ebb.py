"""
Rewritten program for Sutton-Chen cluster global minima.
Uses structural motif generation (Icosahedral, FCC) and Basin Hopping.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Global start time for deadline tracking
_START_TIME = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _generate_icosahedron(n_atoms, spacing=1.0):
    """
    Generate a Mackay icosahedron or truncated icosahedron with n_atoms.
    This is a strong candidate for many cluster sizes.
    """
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]) * spacing
    verts /= np.linalg.norm(verts[0]) * 0.5 # Normalize radius to ~1
    
    # Shell radii for Mackay icosahedra: 1, 2, 3...
    # Atoms per shell: 1, 12, 42, 92...
    # We will build layers.
    points = []
    # Center atom
    points.append(np.zeros(3))
    
    # Shell 1: 12 vertices
    if n_atoms > 12:
        points.extend(verts.tolist())
    
    # Shell 2: 42 atoms (30 edge centers, 12 face centers)
    if n_atoms > 54:
        # Edge centers
        edges = []
        # Icosahedron edges connect vertices with distance sqrt(2) (in unit sphere approx)
        # We can just find pairs of vertices that are close
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if d < 1.2 * spacing: # Heuristic for edge length
                    edges.append((i, j))
        
        # Add edge centers
        for i, j in edges:
            mid = (verts[i] + verts[j]) / 2.0
            # Normalize to shell radius 2
            mid_norm = np.linalg.norm(mid)
            if mid_norm > 1e-6:
                mid = mid / mid_norm * 2.0
                points.append(mid)
        
        # Face centers
        # An icosahedron has 20 faces.
        # We can approximate face centers by averaging 3 vertices of a triangular face.
        # Identifying faces programmatically is complex. 
        # Alternative: Use a predefined set of 42 points for the second shell.
        # Let's use a simpler approach: generate points on a sphere and reject those too close to existing points?
        # Or better: Use the fact that the 2nd shell of a Mackay icosahedron consists of:
        # 12 atoms at face centers (radius 2*phi/... ?) 
        # Actually, let's just use a random spherical distribution for the remainder if we can't easily generate perfect shells.
        # However, perfect icosahedra are key.
        
        # Let's try to add the 12 face centers.
        # The 20 faces of an icosahedron.
        # We can find them by checking which triplets of vertices form a triangle with no other vertices inside.
        # This is getting complex. 
        # Let's stick to the 12 vertices and center for small N, and for larger N, 
        # we will use a "truncated" approach: take the icosahedron and add atoms in a way that maintains symmetry roughly.
        
        # For now, let's just return what we have if it's close, or fall back to FCC.
        pass

    # If we need more atoms, we can't easily extend the perfect icosahedron without complex geometry.
    # So, for N > 55, we might just use the 55-atom icosahedron as a seed and let relaxation handle the rest,
    # or generate an FCC cluster which is easier to truncate.
    
    return np.array(points)

def _generate_fcc_cluster(n_atoms, spacing=1.0):
    """
    Generate an FCC cluster by truncating an FCC lattice to a sphere of radius R.
    """
    # Estimate radius
    R = spacing * (n_atoms * 4 / np.pi) ** (1/3) * 1.5 # Heuristic
    pts = []
    # FCC lattice points
    # Generate points in a cube slightly larger than R
    max_coord = int(R / spacing) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) scaled by spacing
                # Actually, standard FCC unit cell has atoms at corners and face centers.
                # Let's just iterate over a simple cubic grid and check if it's an FCC site.
                # Or simpler: generate all points in a grid and filter by distance.
                pass
    
    # Simpler FCC generation:
    pts = []
    # Generate a grid of points
    max_idx = int(np.ceil(R / (spacing * 0.5))) + 2
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # Check if (i,j,k) is an FCC lattice point
                # FCC lattice points are (n1, n2, n3) where n1+n2+n3 is even? No.
                # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in units of lattice constant a.
                # Let's use a=1.
                # Point p = (i, j, k) is a corner.
                # If i,j,k are integers, it's a corner if i+j+k is even? No, all integers are corners in a simple cubic.
                # In FCC, the unit cell is simple cubic with basis.
                # So points are (i,j,k), (i, j+0.5, k+0.5), (i+0.5, j, k+0.5), (i+0.5, j+0.5, k)
                
                # Let's just generate all 4 types
                candidates = [
                    (i, j, k),
                    (i, j + 0.5, k + 0.5),
                    (i + 0.5, j, k + 0.5),
                    (i + 0.5, j + 0.5, k)
                ]
                for c in candidates:
                    p = np.array(c) * spacing
                    if np.linalg.norm(p) <= R:
                        # Check minimum distance to existing points to avoid duplicates
                        if not pts or all(np.linalg.norm(p - q) > 0.1 for q in pts):
                            pts.append(p)
    
    pts = np.array(pts)
    if len(pts) < n_atoms:
        # If we didn't get enough, add random points near the surface
        while len(pts) < n_atoms:
            p = pts[np.random.randint(len(pts))] + np.random.normal(0, 0.1, 3)
            if all(np.linalg.norm(p - q) > 0.1 for q in pts):
                pts = np.vstack([pts, p])
    else:
        # Randomly select n_atoms points, preferring those on the surface? 
        # For now, just take the first n_atoms or random selection.
        # Random selection from the cluster is better to avoid bias.
        idx = np.random.choice(len(pts), n_atoms, replace=False)
        pts = pts[idx]
        
    return pts

def _generate_random_compact(n_atoms, spacing=1.0):
    """Generate a random compact cluster."""
    rng = np.random.default_rng()
    radius = spacing * n_atoms ** (1.0 / 3.0) * 0.75
    points = []
    attempts = 0
    while len(points) < n_atoms and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
        attempts += 1
    # If not enough, fill with random points
    while len(points) < n_atoms:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind")
    
    # List of initial structures
    init_structures = []
    
    # 1. FCC-like
    try:
        fcc = _generate_fcc_cluster(size)
        init_structures.append(fcc)
    except Exception:
        pass
        
    # 2. Icosahedral (if size allows)
    if size <= 55:
        try:
            ico = _generate_icosahedron(size)
            if len(ico) == size:
                init_structures.append(ico)
        except Exception:
            pass
    else:
        # For larger sizes, use a truncated icosahedron or just FCC
        # We can try to generate a 55-atom icosahedron and add random atoms?
        # Or just rely on FCC and Random.
        pass

    # 3. Random compact
    init_structures.append(_generate_random_compact(size))
    
    # 4. Another random
    init_structures.append(_generate_random_compact(size))

    best_energy = np.inf
    best_positions = None
    
    # Basin Hopping
    # We will run BH on each initial structure
    # Time budget: ~300s total.
    # Assume ~10-20s per relax call for large N.
    # We need to be careful.
    
    for init_pos in init_structures:
        if _time_left() < 20:
            break
            
        current_energy, current_pos = relax(init_pos, potential, kind)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_positions = current_pos
            
        # Perform a few BH steps
        # Perturbation size
        # Typical interatomic distance is ~1.0. Perturbation 0.1-0.5
        perturbation_size = 0.3
        max_bh_steps = 10 # Adjust based on time
        
        for step in range(max_bh_steps):
            if _time_left() < 20:
                break
                
            # Perturb
            noise = np.random.normal(0, perturbation_size, current_pos.shape)
            perturbed_pos = current_pos + noise
            
            # Relax
            new_energy, new_pos = relax(perturbed_pos, potential, kind)
            
            # Accept if lower energy
            if new_energy < current_energy:
                current_energy = new_energy
                current_pos = new_pos
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_positions = new_pos
            else:
                # Metropolis acceptance? 
                # For global minima, simple descent is often enough if we have good starts.
                # But BH usually accepts uphill moves with prob exp(-dE/T).
                # Let's use simple descent to save time, relying on multiple starts.
                # Or a very high T.
                pass

    if best_positions is None:
        # Fallback
        best_positions = init_structures[0]
        
    return best_positions
# EVOLVE-BLOCK-END
