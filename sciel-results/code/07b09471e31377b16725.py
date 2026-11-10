"""
Combined program for Dzugutov cluster global minima.
Uses a portfolio of structural heuristics (FCC, Icosahedral, Random)
with parallel relaxation and global deadline management.
"""
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Global time tracking
_START_TIME = None
_DEADLINE = 320.0  # 10s safety margin before 330s kill


def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)


def _generate_fcc(n, rng):
    """Generate an FCC lattice chunk roughly spherical."""
    # Estimate radius based on density
    # For Dzugutov, equilibrium distance is around 1.8-2.0
    # Let's use a spacing of 1.1 for initial generation (as in seed)
    # but scale up for FCC to be denser
    # Actually, let's just use a reasonable box size
    R = (3.0 * n / (4.0 * np.pi) * 10.0) ** (1.0 / 3.0) * 1.2
    a = 1.1  # Lattice constant approx equilibrium distance
    
    # Generate FCC points in a box
    # FCC primitive vectors
    # Points: (0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5) * a
    points = []
    # We need to generate enough points to fill the sphere
    # Box size
    box = 2 * R + a
    nx = int(box / a) + 2
    ny = int(box / a) + 2
    nz = int(box / a) + 2
    
    center = np.array([box/2, box/2, box/2])
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                base = np.array([i, j, k]) * a + center - np.array([box/2, box/2, box/2])
                # 4 FCC points
                p1 = base + np.array([0, 0, 0]) * a
                p2 = base + np.array([0.5, 0.5, 0]) * a
                p3 = base + np.array([0.5, 0, 0.5]) * a
                p4 = base + np.array([0, 0.5, 0.5]) * a
                
                for p in [p1, p2, p3, p4]:
                    if np.linalg.norm(p) <= R:
                        points.append(p)
    
    # Shuffle and take n
    if len(points) < n:
        # If box was too small, expand R and retry (simple fallback: random)
        return None
        
    points = np.array(points)
    idx = rng.choice(len(points), n, replace=False)
    return points[idx]


def _generate_random(n, rng):
    """Generate random points in a sphere with minimum distance constraint."""
    R = (3.0 * n / (4.0 * np.pi) * 10.0) ** (1.0 / 3.0) * 1.2
    points = []
    max_attempts = 10000
    min_dist = 0.8 # Slightly larger than 0.5 to avoid immediate rejection
    
    # Use rejection sampling
    while len(points) < n and max_attempts > 0:
        # Uniform in sphere
        u = rng.random(3)
        r = R * (rng.random() ** (1.0/3.0))
        theta = rng.random() * 2 * np.pi
        phi = np.arccos(2 * rng.random() - 1)
        p = np.array([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi)
        ])
        
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
        max_attempts -= 1
        
    if len(points) < n:
        # Fallback: just take random, relax will fix overlaps if it can, or we add random
        # Actually if we can't fit, increase R
        R *= 1.2
        points = []
        while len(points) < n:
             u = rng.random(3)
             r = R * (rng.random() ** (1.0/3.0))
             theta = rng.random() * 2 * np.pi
             phi = np.arccos(2 * rng.random() - 1)
             p = np.array([
                 r * np.sin(phi) * np.cos(theta),
                 r * np.sin(phi) * np.sin(theta),
                 r * np.cos(phi)
             ])
             if all(np.linalg.norm(p - q) >= 0.6 for q in points):
                 points.append(p)
             else:
                 # Force add with jitter to avoid infinite loop, relax will separate
                 if len(points) % 100 == 99:
                     points.append(p + rng.normal(scale=0.1, size=3))
             
    return np.array(points)


def _generate_icosahedral(n, rng):
    """
    Generate an icosahedral-like structure.
    Start with 1 center + 12 vertices.
    Add face centers.
    Add next shell vertices if needed.
    Jitter everything.
    """
    if n == 1:
        return np.array([[0,0,0]])
        
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    a = 1.0
    # Icosahedron vertices
    verts = [
        (-1,  phi,  0), ( 1,  phi,  0), (-1, -phi,  0), ( 1, -phi,  0),
        ( 0, -1,  phi), ( 0,  1,  phi), ( 0, -1, -phi), ( 0,  1, -phi),
        ( phi,  0, -1), ( phi,  0,  1), (-phi,  0, -1), (-phi,  0,  1)
    ]
    verts = np.array(verts)
    # Normalize to unit radius
    verts = verts / np.linalg.norm(verts[0])
    
    points = [np.array([0,0,0])]
    # Scale to roughly equilibrium distance 1.8
    # Distance from center to vertex in icosahedron is 1. 
    # We want r ~ 1.8.
    scale = 1.8
    for v in verts:
        points.append(v * scale)
        
    current_n = 13
    
    if n > 13:
        # Add face centers
        # Triangles of icosahedron
        # This is complex to generate programmatically without geometry lib.
        # Simpler: Add atoms on the surface of the sphere defined by the icosahedron
        # Or just add random atoms near the surface.
        # Let's add atoms at the midpoints of the edges of the icosahedron?
        # Number of edges in icosahedron is 30.
        # 13 + 30 = 43. Close to 45, 50.
        
        edges = []
        # Identify edges by distance between vertices
        verts_scaled = verts * scale
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(verts_scaled[i] - verts_scaled[j])
                if d < 1.5 * scale: # Edge length is ~1.05 * scale
                     edges.append((i, j))
        
        for i, j in edges:
            if current_n >= n:
                break
            mid = (verts_scaled[i] + verts_scaled[j]) / 2.0
            points.append(mid)
            current_n += 1
            
        if current_n < n:
             # Add atoms on the sphere surface
             R_surf = scale
             while current_n < n:
                 # Random on sphere
                 theta = rng.random() * 2 * np.pi
                 phi = np.arccos(2 * rng.random() - 1)
                 p = np.array([
                     R_surf * np.sin(phi) * np.cos(theta),
                     R_surf * np.sin(phi) * np.sin(theta),
                     R_surf * np.cos(phi)
                 ])
                 # Check distance to existing points
                 if all(np.linalg.norm(p - q) >= 0.8 for q in points):
                     points.append(p)
                     current_n += 1
                 else:
                     # Jitter and try again, or just add with jitter
                     points.append(p + rng.normal(scale=0.05, size=3))
                     current_n += 1
                     
    # Jitter all points to break symmetry
    points = np.array(points)
    points = points + rng.normal(scale=0.05, size=points.shape)
    
    return points


def _relax_candidate(positions, potential, parameter):
    """Relax a single candidate and return (energy, positions)."""
    try:
        energy, pos = relax(positions, potential, parameter)
        return (energy, pos)
    except Exception:
        return (float('inf'), positions)


def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Initialize global time
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    rng_seed = size
    rng = np.random.default_rng(rng_seed)
    
    candidates = []
    
    # 1. FCC
    fcc = _generate_fcc(size, rng)
    if fcc is not None:
        candidates.append(fcc)
        
    # 2. Random
    rand = _generate_random(size, rng)
    candidates.append(rand)
    
    # 3. Icosahedral
    ico = _generate_icosahedral(size, rng)
    candidates.append(ico)
    
    # 4. Another Random with different seed/density
    rng2 = np.random.default_rng(size + 100)
    rand2 = _generate_random(size, rng2)
    candidates.append(rand2)
    
    # 5. FCC with different lattice constant
    rng3 = np.random.default_rng(size + 200)
    # Modify _generate_fcc to accept lattice constant? 
    # For now, just jitter the FCC one
    if fcc is not None:
        fcc_jittered = fcc + rng3.normal(scale=0.1, size=fcc.shape)
        candidates.append(fcc_jittered)
    
    # Relax all candidates in parallel
    best_energy = float('inf')
    best_positions = None
    
    # Use a thread pool for parallel relaxation
    # Limit to 4 threads as per constraints
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for pos in candidates:
            # Check time remaining
            remaining = _get_time_remaining()
            if remaining < 10:  # Need at least 10s for relax
                break
            futures.append(executor.submit(_relax_candidate, pos, potential, parameter))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                energy, positions = future.result(timeout=60)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass
            
            # Check time remaining
            remaining = _get_time_remaining()
            if remaining < 5:
                break
    
    # If no valid candidate was found, return the first one (unrelaxed)
    if best_positions is None:
        if candidates:
            best_positions = candidates[0]
        else:
            # Fallback to random
            best_positions = _generate_random(size, rng)
            
    return best_positions
