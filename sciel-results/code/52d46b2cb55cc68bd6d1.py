# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
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

def _generate_mackay_icosahedron(n):
    """
    Generate a Mackay icosahedron cluster.
    Structure: 1 (center) + 12 (inner shell) + 42 (outer shell) = 55.
    For n < 55, we strip atoms from the outer shell.
    For n > 55, we add a third shell (simplified).
    """
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of an icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = [np.zeros(3)]
    
    # Inner shell: 12 atoms at distance r1
    # Optimal r1 for Dzugutov is roughly around 1.0-1.1
    r1 = 1.05 
    points.extend([v * r1 for v in verts])
    
    if n <= 13:
        return np.array(points[:n])
    
    # Outer shell: 42 atoms
    # These correspond to the vertices of a truncated icosahedron or specific directions
    # The 42 directions are the 12 vertices of the inner shell (scaled) and 30 edge midpoints
    # Actually, the Mackay icosahedron outer shell consists of:
    # - 12 atoms along the vertices of the inner shell (scaled by 2)
    # - 30 atoms along the edge midpoints of the inner shell (scaled)
    
    outer_verts = []
    # 12 vertex directions
    for v in verts:
        outer_verts.append(v * 2.0)
        
    # 30 edge midpoints
    # We need to find edges. In an icosahedron, each vertex connects to 5 others.
    # Total edges = 30.
    # Let's compute distances between all pairs of verts to find edges.
    # Edge length in unit icosahedron is 2.
    # Let's just generate all midpoints and filter by distance to ensure they are unique edges.
    edge_midpoints = []
    for i in range(12):
        for j in range(i+1, 12):
            mid = (verts[i] + verts[j]) / 2.0
            mid = mid / np.linalg.norm(mid)
            # Scale to appropriate radius. 
            # In Mackay, the radius is such that the bond lengths are similar.
            # r_outer ~ 2 * r_inner. The midpoint direction is between two inner vertices.
            # Distance from center to midpoint direction is 1.0.
            # We want the distance to be ~ 2.0 * r1? No, the geometry is specific.
            # Let's use a scaling factor.
            # If inner is at 1.05, outer is typically around 2.1.
            # The midpoint direction has length 1.0.
            # Let's place them at radius 2.1.
            edge_midpoints.append(mid * 2.1)
            
    # Remove duplicates if any (though midpoints of distinct edges in icosahedron are distinct directions)
    # There are exactly 30 edges.
    outer_verts.extend(edge_midpoints)
    
    # We have 12 + 30 = 42 directions for the outer shell.
    # We need n - 13 atoms from this shell.
    needed_outer = n - 13
    if needed_outer > 0:
        # Shuffle or just take first N? 
        # To maintain symmetry for the first few, we should take them in a balanced way.
        # But for basin hopping, the starting point just needs to be close.
        # Let's just take the first `needed_outer` from the list.
        # To be more symmetric, we could interleave vertex and midpoint directions.
        # But simple slicing is fine for a seed.
        points.extend(outer_verts[:needed_outer])
        
    # If n > 55, add more atoms
    if n > 55:
        # Add a third shell. 120 vertices of a dodecahedron? Or just random compact around.
        # For now, let's just add random points on a sphere of radius 3.2
        rng = np.random.default_rng(42)
        current_count = len(points)
        while current_count < n:
            # Random point on sphere
            u = rng.uniform(0, 1)
            v = rng.uniform(0, 1)
            theta = 2 * np.pi * u
            z = 2 * v - 1
            r = 3.2
            x = r * np.sqrt(1 - z*z) * np.cos(theta)
            y = r * np.sqrt(1 - z*z) * np.sin(theta)
            z = r * z
            points.append(np.array([x, y, z]))
            current_count += 1
            
    points = np.array(points[:n])
    # Center
    points = points - np.mean(points, axis=0)
    return points

def construct(instance):
    """
    Basin Hopping strategy.
    1. Start with a Mackay Icosahedron.
    2. Relax.
    3. Kick (perturb) + Relax.
    4. Repeat until time limit.
    """
    n = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin
    
    # 1. Generate initial structure
    try:
        pos = _generate_mackay_icosahedron(n)
    except Exception as e:
        # Fallback to random if Mackay fails
        rng = np.random.default_rng(n)
        pos = rng.uniform(-2, 2, (n, 3))
        
    # 2. Initial Relaxation
    try:
        best_energy, best_pos = relax(pos, potential)
    except Exception as e:
        best_energy = _dzugutov_energy(pos)
        best_pos = pos.copy()
    
    # 3. Basin Hopping Loop
    rng = np.random.default_rng(12345)
    kick_size = 0.5  # Initial perturbation scale
    
    # Adaptive kick size: if we get stuck (energy doesn't improve for many steps), increase kick
    # Or simply decay kick size slowly? 
    # For global minima, larger kicks are often needed to escape deep local minima.
    # Let's start moderate and keep it moderate.
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        
        # Perturb
        # Gaussian noise
        noise = rng.normal(0, kick_size, best_pos.shape)
        new_pos = best_pos + noise
        
        # Relax
        try:
            new_energy, relaxed_new_pos = relax(new_pos, potential)
            
            # Accept if better
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = relaxed_new_pos
                # Optionally reduce kick size slightly if we found a new minimum?
                # No, keep it constant to explore.
            else:
                # If not better, we can still move to the new basin if we are doing a random walk,
                # but for "best so far" strategy, we only keep the best.
                # However, standard basin hopping often accepts worse moves with Metropolis criterion
                # to escape. But since we want the GLOBAL minimum, and relax is expensive,
                # a "greedy" basin hopping (only accept improvements) is safer and faster
                # if the initial seed is good.
                # Let's try a slight Metropolis acceptance to avoid getting stuck in one basin.
                # T = 0.1 * (best_energy / 100) # Relative temperature?
                # Simpler: just keep the best.
                pass
                
        except Exception as e:
            # Relaxation failed, try smaller kick next time?
            pass
        
        # Check time
        if time.time() > deadline - 5.0:
            break
            
    # 4. Final Polish
    # Run a very tight relaxation on the best position found
    try:
        # Use the provided relax again, maybe it converges better from the current state
        # Or use gradient descent if available?
        # relax() is likely robust.
        final_energy, final_pos = relax(best_pos, potential)
        if final_energy < best_energy:
            best_energy = final_energy
            best_pos = final_pos
    except Exception as e:
        pass

    return best_pos
# EVOLVE-BLOCK-END
