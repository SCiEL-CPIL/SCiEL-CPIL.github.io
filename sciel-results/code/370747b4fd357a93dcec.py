"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

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
    verts = np.array(verts, dtype=float)
    # Normalize to unit radius
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    # Scale to desired nearest neighbor distance (approx 1.8 for Dzugutov)
    # The distance between nearest neighbors in a unit icosahedron is 2/sqrt(phi^2+1) * 1? 
    # Actually, let's just scale such that min distance is approx 1.8
    # For unit icosahedron, min dist is 2/sqrt(1+phi^2) * 1? No, vertices are on unit sphere.
    # Distance between (0,1,phi) and (0,-1,phi) is 2.
    # Distance between (0,1,phi) and (1,phi,0) is sqrt(1 + (phi-1)^2 + phi^2) = sqrt(1 + phi^2 - 2phi + 1 + phi^2) = sqrt(2 + 2phi^2 - 2phi)
    # phi^2 = phi + 1. So 2 + 2(phi+1) - 2phi = 4. Dist = 2.
    # So min dist in unit sphere icosahedron is 2.
    # We want min dist ~ 1.8. So scale = 1.8 / 2 = 0.9.
    verts = verts * 0.9 * scale
    
    points = [np.zeros(3)]  # Central atom
    if n > 1:
        points.extend([v.copy() for v in verts[:min(n-1, 12)]])
    
    # Second shell: 30 face centers + 12 edge midpoints? Or just more random on sphere?
    # For simplicity and robustness, if n > 13, we add atoms on a second shell of radius ~ 1.8 * sqrt(2) ~ 2.5
    # Or we can just use a "random on sphere" approach for the rest, but that's less structured.
    # Let's try to add a second icosahedral shell.
    # Vertices of a dodecahedron? Or just scale up the icosahedron?
    # A common approach is to take the 12 vertices of the icosahedron, and for each, place neighbors.
    # Simpler: Generate random points on a sphere of radius R2, ensuring min distance.
    
    while len(points) < n:
        # Try to place on a sphere of radius 2.5 (approx second shell)
        # Or just random compact
        r = 2.5 * scale
        theta = np.random.uniform(0, 2*np.pi)
        z = np.random.uniform(-1, 1)
        x = np.sqrt(1 - z**2) * np.cos(theta)
        y = np.sqrt(1 - z**2) * np.sin(theta)
        p = np.array([x, y, z]) * r
        
        # Check distance
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < 0.5: # Hard constraint from task
                valid = False
                break
        if valid:
            points.append(p)
            
    return np.array(points)

def _generate_bcc_cluster(n, scale=1.0):
    """Generate a BCC-like cluster."""
    points = []
    # BCC lattice points
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                # Corner
                p1 = np.array([i, j, k]) * 1.4 * scale
                if len(points) < n and all(np.linalg.norm(p1 - q) >= 0.5 for q in points):
                    points.append(p1)
                # Body center
                p2 = np.array([i+0.5, j+0.5, k+0.5]) * 1.4 * scale
                if len(points) < n and all(np.linalg.norm(p2 - q) >= 0.5 for q in points):
                    points.append(p2)
    if len(points) < n:
        # Fill with random
        while len(points) < n:
            p = np.random.uniform(-3, 3, 3) * scale
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
    return np.array(points[:n])

def construct(instance):
    """
    Strategy: 
    1. Generate multiple diverse structured initializations (Icosahedral, BCC, Perturbed Icosa).
    2. Perform Simulated Annealing / Basin Hopping on the best ones to find global minima.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    # List of initializers
    initializers = [
        lambda: _generate_icosahedron(size, 1.0),
        lambda: _generate_icosahedron(size, 0.9),
        lambda: _generate_icosahedron(size, 1.1),
        lambda: _generate_bcc_cluster(size, 1.0),
        lambda: _generate_icosahedron(size, 1.0) + np.random.normal(0, 0.1, (size, 3)) # Perturbed
    ]
    
    # Pre-relax all initializers
    candidates = []
    for init_func in initializers:
        if not _check_time():
            break
        try:
            pos = init_func()
            # Basic relaxation to ensure stability and fix any minor overlaps
            e, pos_relaxed = relax(pos, potential, parameter)
            if e < best_energy:
                best_energy = e
                best_positions = pos_relaxed
            candidates.append(pos_relaxed)
        except Exception:
            continue
            
    if best_positions is None:
        # Fallback to random if all fail
        pos = np.random.uniform(-2, 2, (size, 3))
        e, pos_relaxed = relax(pos, potential, parameter)
        return pos_relaxed

    # Simulated Annealing on the best candidates
    # We will run SA for a limited time on the top few candidates
    top_candidates = sorted(candidates, key=lambda x: relax(x, potential, parameter)[0])[:3]
    
    current_time = time.time()
    sa_time_limit = min(TIME_LIMIT - (current_time - _START_TIME) - 10, 60) # Max 60s per SA run, total budget aware
    
    for start_pos in top_candidates:
        if not _check_time():
            break
            
        # Simulated Annealing
        pos = start_pos.copy()
        e, _ = relax(pos, potential, parameter)
        current_energy = e
        
        T = 1.0
        cooling_rate = 0.99
        max_steps = 5000
        
        for step in range(max_steps):
            if time.time() - _START_TIME > TIME_LIMIT - 10:
                break
            if not _check_time():
                break
                
            # Perturb a random atom
            i = np.random.randint(size)
            displacement = np.random.normal(0, 0.2, 3)
            new_pos = pos.copy()
            new_pos[i] += displacement
            
            # Check hard constraint
            if any(np.linalg.norm(new_pos[i] - q) < 0.5 for j, q in enumerate(new_pos) if j != i):
                continue
                
            new_e, new_pos_relaxed = relax(new_pos, potential, parameter)
            
            delta_e = new_e - current_energy
            if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
                pos = new_pos_relaxed
                current_energy = new_e
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_positions = pos.copy()
                    
            T *= cooling_rate
            if T < 0.01:
                break
                
    return best_positions
# EVOLVE-BLOCK-END
