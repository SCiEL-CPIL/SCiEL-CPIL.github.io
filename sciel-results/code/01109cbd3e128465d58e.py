"""
Strategy: Basin Hopping with Polytetrahedral Seeding.
The current program performs single-shot relaxations from PT and random starts.
This program implements a basin hopping algorithm:
1. Generate diverse initial structures (PT shells, random compact).
2. Relax each to find local minima.
3. Iteratively perturb the best structure (Langevin dynamics step),
   relax again, and accept if energy decreases.
This active search allows escaping local minima that trap single-shot relaxations,
exploiting the PT motif's deep basins while navigating the rugged landscape.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Global deadline tracking
_START_TIME = time.time()
_HARD_LIMIT = 320.0  # Safety margin before 330s kill

def _get_remaining_time():
    return max(0.1, _HARD_LIMIT - (time.time() - _START_TIME))

def _generate_icosahedron(n_atoms, rng):
    """
    Generate a structure based on icosahedral shells (Mackay-like).
    """
    phi = (1 + np.sqrt(5)) / 2
    # 12 unique icosahedral vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    positions = [np.zeros(3)] # Center
    
    # Shell 1: 12 atoms
    r1 = 1.9
    for v in verts:
        positions.append(v * r1)
        
    if n_atoms <= 13:
        return np.array(positions[:n_atoms])
        
    # Shell 2: Generate remaining atoms on a sphere with radius r2
    # To maintain PT character, we place them near the "midpoints" of the icosahedron surface
    # or just on a slightly larger sphere.
    # For n=55, we need 42 more.
    r2 = 1.9 * 1.75
    num_outer = n_atoms - 13
    if num_outer > 0:
        # Generate num_outer points on sphere r2
        # Use a spiral distribution for better coverage
        # Or just random, but biased towards icosahedral directions?
        # Let's use random for now, but ensure they are distinct.
        outer_pts = []
        for _ in range(num_outer):
            v = rng.normal(size=3)
            v = v / np.linalg.norm(v) * r2
            # Add small jitter to avoid exact duplicates if any
            v += rng.normal(scale=0.01, size=3)
            outer_pts.append(v)
        positions.extend(outer_pts)
        
    return np.array(positions[:n_atoms])

def _random_compact(n_atoms, rng, spacing=1.0):
    """
    Generate random compact structure.
    """
    radius = spacing * n_atoms ** (1.0 / 3.0) * 0.9
    points = []
    # Try to pack in a sphere
    max_tries = 10000
    tries = 0
    while len(points) < n_atoms and tries < max_tries:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check distance to existing points
            if not points:
                points.append(p)
            else:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) >= 0.6 * spacing:
                    points.append(p)
        tries += 1
    
    if len(points) < n_atoms:
        # Fallback: just random in a cube if packing failed
        points = [rng.uniform(-radius, radius, 3) for _ in range(n_atoms)]
        
    return np.array(points)

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    # Budget allocation: 
    # We have ~50s per instance if sequential.
    # Let's use 40s for safety.
    time_budget = 40.0
    
    rng = np.random.default_rng(size)
    
    best_energy = np.inf
    best_positions = None
    
    # Phase 1: Generate diverse initial structures
    starts = []
    
    # 1. PT-inspired
    try:
        pt_pos = _generate_icosahedron(size, rng)
        # Ensure min distance 0.5
        # Simple check
        valid = True
        for i in range(len(pt_pos)):
            for j in range(i+1, len(pt_pos)):
                if np.linalg.norm(pt_pos[i] - pt_pos[j]) < 0.5:
                    valid = False
                    break
            if not valid: break
        if valid:
            starts.append(pt_pos)
    except Exception:
        pass
        
    # 2. Random Compact (Seed 1)
    try:
        rc1 = _random_compact(size, rng)
        starts.append(rc1)
    except Exception:
        pass
        
    # 3. Random Compact (Seed 2)
    try:
        rc2 = _random_compact(size, rng)
        starts.append(rc2)
    except Exception:
        pass
        
    # 4. "Deformed" PT: Perturb PT structure
    if starts:
        pt_deformed = starts[0] + rng.normal(scale=0.2, size=starts[0].shape)
        # Check validity roughly
        valid = True
        for i in range(len(pt_deformed)):
            for j in range(i+1, len(pt_deformed)):
                if np.linalg.norm(pt_deformed[i] - pt_deformed[j]) < 0.5:
                    valid = False
                    break
            if not valid: break
        if valid:
            starts.append(pt_deformed)

    # Phase 2: Relax all starts
    for i, start_pos in enumerate(starts):
        if time.time() - start_time > time_budget * 0.3:
            break
        try:
            energy, positions = relax(start_pos, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            pass

    if best_positions is None:
        # Fallback
        return starts[-1] if starts else np.zeros((size, 3))

    # Phase 3: Basin Hopping
    # We have remaining time: time_budget - (time.time() - start_time)
    # Use this for basin hopping iterations.
    
    current_energy = best_energy
    current_positions = best_positions
    
    # Adaptive step size
    step_size = 0.2
    min_step_size = 0.05
    max_iter = 100 # Cap iterations just in case
    
    while time.time() - start_time < time_budget * 0.95 and max_iter > 0:
        max_iter -= 1
        
        # Generate perturbation
        # Langevin-like: x' = x + sigma * N(0,1)
        perturbed = current_positions + rng.normal(scale=step_size, size=current_positions.shape)
        
        # Check validity (min distance 0.5)
        # Optimization: Only check if step_size is small enough that it's likely valid?
        # Or just check. For n=60, 60*60/2 = 1800 checks. Fast enough.
        valid = True
        for i in range(size):
            for j in range(i+1, size):
                if np.linalg.norm(perturbed[i] - perturbed[j]) < 0.5:
                    valid = False
                    break
            if not valid: break
            
        if not valid:
            # Retry with smaller step or skip?
            # Let's skip and try a smaller step next time
            step_size = max(min_step_size, step_size * 0.8)
            continue
            
        # Relax
        try:
            energy, positions = relax(perturbed, potential, parameter)
            
            if energy < current_energy:
                # Accept
                current_energy = energy
                current_positions = positions
                best_energy = energy
                best_positions = positions
                # Increase step size slightly if we've been stuck?
                # Or keep it? Standard BH often keeps it or cools it.
                # Let's keep it constant for now, or increase if we accept often?
                # Let's try a simple scheme: if accepted, keep step_size. 
                # If rejected, decrease step_size.
            else:
                # Reject
                step_size = max(min_step_size, step_size * 0.9)
                
        except Exception:
            step_size = max(min_step_size, step_size * 0.9)
            
    return best_positions

# EVOLVE-BLOCK-END
