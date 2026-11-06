"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Global start time for budget enforcement
_START_TIME = time.time()
TIME_LIMIT = 280.0  # Leave 20s safety margin for 300s limit

def _check_time():
    return time.time() - _START_TIME < TIME_LIMIT

def _generate_icosahedron(n, scale=1.0):
    """
    Generates an icosahedral cluster of n atoms.
    The Dzugutov potential favors icosahedral order.
    We construct a central atom, a first shell of 12, a second shell of 42, etc.
    If n is not a magic number, we add atoms to the outermost shell or randomly near the surface.
    """
    phi = (1 + np.sqrt(5)) / 2
    # Regular icosahedron vertices (12)
    verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    verts = np.array(verts)
    # Normalize to unit radius
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    verts *= scale
    
    positions = [np.zeros(3)] # Central atom
    if n > 1:
        positions.extend(verts.tolist())
        
    # Second shell: 42 atoms (icosahedral symmetry)
    # These are the midpoints of the edges of the icosahedron, scaled up.
    # Or more simply, we can use a known set of 42 points.
    # For simplicity and robustness, let's generate points on a sphere of radius ~2*scale
    # with icosahedral symmetry.
    # A simpler approach for arbitrary n: 
    # 1. Start with icosahedron (13 atoms).
    # 2. If n > 13, add atoms on a second shell.
    
    if n > 13:
        # Second shell radius is roughly 2 * scale
        r2 = 2.0 * scale
        # Generate 42 points. We can approximate by taking the icosahedron vertices 
        # and rotating them, or just placing them on a sphere.
        # To maintain some symmetry, let's place them at the "face centers" and "edge midpoints" 
        # or just use a dense spherical distribution constrained to the icosahedral symmetry group.
        
        # Let's use a simpler heuristic: 
        # Place atoms on a sphere of radius r2. 
        # We want to approximate the 42-atom shell.
        # We can generate 42 points by taking the 12 vertices, adding their reflections/rotations?
        # Actually, the 42 points are the vertices of a snub icosahedron or similar.
        # Let's just place 42 points uniformly on a sphere of radius r2, 
        # but snap them to be close to the icosahedral directions if possible.
        
        # A robust way: generate 42 random points on sphere, but ensure they are well separated.
        # However, random points might break symmetry.
        # Let's try to construct the 42 shell properly.
        # The 42 points are the images of a single point under the icosahedral group.
        # Pick a point p not on a symmetry axis.
        # The group has 60 elements. We need 42 points, so it's not a single orbit.
        # It's a union of orbits.
        
        # Alternative: Use the fact that the global minima are often "icosahedral" with defects.
        # Let's just fill the space with a BCC-like lattice clipped to a sphere?
        # No, the prompt says "polytetrahedral".
        
        # Let's use a simpler generator: 
        # 1. Central atom.
        # 2. 12 atoms at distance 1.
        # 3. Remaining atoms placed on a sphere of radius ~2, ~3, etc.
        # To be efficient, we will place the remaining atoms on a sphere of radius r_k 
        # such that they are separated by at least 0.5.
        
        current_n = 13
        shell_radius = 2.0 * scale
        # Add atoms to second shell
        while current_n < n and _check_time():
            # Try to place a point on the sphere of radius shell_radius
            # that is at least 0.5 away from all existing points
            found = False
            for _ in range(50): # 50 attempts per atom
                p = rng_uniform_sphere() * shell_radius
                if all(np.linalg.norm(p - q) >= 0.5 for q in positions):
                    positions.append(p)
                    current_n += 1
                    found = True
                    break
            if not found:
                # If we can't fit more at this radius, increase radius
                shell_radius *= 1.2
                if shell_radius > 10 * scale: # Safety break
                    break
                    
    # If we still don't have enough atoms (e.g., very dense packing required),
    # add random points in the volume
    if current_n < n:
        while current_n < n and _check_time():
            p = rng_uniform_sphere() * (shell_radius + 0.5)
            if all(np.linalg.norm(p - q) >= 0.5 for q in positions):
                positions.append(p)
                current_n += 1
            else:
                # If stuck, shrink min distance slightly for initialization (relax will fix)
                if all(np.linalg.norm(p - q) >= 0.4 for q in positions):
                    positions.append(p)
                    current_n += 1
                else:
                    # Give up and use random
                    p = rng.uniform(-shell_radius, shell_radius, 3)
                    positions.append(p)
                    current_n += 1

    return np.array(positions[:n])

def rng_uniform_sphere():
    """Generate a random point on the unit sphere."""
    v = np.random.normal(0, 1, 3)
    return v / np.linalg.norm(v)

def _generate_bcc_cluster(n, spacing=1.0):
    """Generate a BCC-like cluster, which is a reasonable fallback."""
    points = []
    # Simple cubic grid with BCC center
    L = int(np.ceil((n / 2.0) ** (1/3))) # Rough size
    for x in range(-L, L+1):
        for y in range(-L, L+1):
            for z in range(-L, L+1):
                # Corner
                p = np.array([x, y, z]) * spacing
                if np.linalg.norm(p) < 5*spacing:
                    points.append(p)
                # Center
                p_c = (np.array([x, y, z]) + 0.5) * spacing
                if np.linalg.norm(p_c) < 5*spacing:
                    points.append(p_c)
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    
    # If we have too many or too few, adjust
    if len(points) > n:
        # Keep the ones closest to the center
        points = sorted(points, key=lambda p: np.linalg.norm(p))[:n]
    else:
        # Add random points
        while len(points) < n:
            p = np.random.uniform(-3*spacing, 3*spacing, 3)
            points.append(p)
            
    return np.array(points)

def _basin_hopping(positions, potential, parameter, max_iter=100, step_size=0.5):
    """
    Perform basin hopping: 
    1. Relax to local min.
    2. Perturb.
    3. Repeat.
    """
    current_energy, current_pos = relax(positions, potential, parameter)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    rng = np.random.default_rng(42)
    
    for i in range(max_iter):
        if not _check_time():
            break
            
        # Perturb
        perturbed = current_pos + rng.normal(0, step_size, current_pos.shape)
        
        # Relax
        try:
            new_energy, new_pos = relax(perturbed, potential, parameter)
        except:
            continue
            
        # Metropolis acceptance (T=0 for simplicity, just accept if lower)
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
            
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos.copy()
            
        # Adaptive step size? Keep simple for now.
        
    return best_energy, best_pos

def construct(instance):
    """
    Strategy:
    1. Generate icosahedral initial guess (matches Dzugutov symmetry).
    2. Generate BCC initial guess (fallback).
    3. Run basin hopping on both.
    4. Return the best result.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Time budget check
    if not _check_time():
        # Emergency: return a valid but unoptimized cluster
        return _generate_bcc_cluster(size)
    
    best_energy = float('inf')
    best_pos = None
    
    # Candidate 1: Icosahedral
    try:
        pos_ico = _generate_icosahedron(size, scale=1.0)
        # Ensure minimum distance constraint is roughly met before relax
        # Relax will handle it, but if initial overlap is too high, it might fail or be slow
        # Let's do a quick check
        if np.linalg.norm(pos_ico, axis=1).max() < 50:
            e, p = _basin_hopping(pos_ico, potential, parameter, max_iter=50, step_size=0.5)
            if e < best_energy:
                best_energy = e
                best_pos = p
    except Exception as ex:
        pass
        
    if not _check_time():
        if best_pos is not None:
            return best_pos
        return _generate_bcc_cluster(size)
        
    # Candidate 2: BCC
    try:
        pos_bcc = _generate_bcc_cluster(size, spacing=1.0)
        e, p = _basin_hopping(pos_bcc, potential, parameter, max_iter=50, step_size=0.5)
        if e < best_energy:
            best_energy = e
            best_pos = p
    except Exception as ex:
        pass
        
    if not _check_time():
        if best_pos is not None:
            return best_pos
        return _generate_bcc_cluster(size)
        
    # Candidate 3: Random compact (from original, as a fallback)
    try:
        spacing = 1.1
        rng = np.random.default_rng(size)
        radius = spacing * size ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
                points.append(p)
        pos_rand = np.array(points)
        e, p = _basin_hopping(pos_rand, potential, parameter, max_iter=50, step_size=0.5)
        if e < best_energy:
            best_energy = e
            best_pos = p
    except Exception as ex:
        pass
        
    if best_pos is not None:
        return best_pos
    else:
        return _generate_bcc_cluster(size)

# EVOLVE-BLOCK-END
