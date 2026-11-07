import numpy as np
import time
from autoresearch_suite.chem_clusters import relax

def _generate_fcc(n, rng):
    # Generate FCC points in a sphere
    # FCC lattice vectors: a/2 (0,1,1), a/2 (1,0,1), a/2 (1,1,0)
    # a = 1.0 (arbitrary scale, relax will optimize)
    a = 1.0
    # Density of FCC is 4 / a^3.
    # Volume of sphere R is 4/3 pi R^3.
    # n approx 4/3 pi R^3 / a^3 => R approx a * (3n/4pi)^(1/3)
    R = a * (3 * n / (4 * np.pi)) ** (1.0 / 3.0) * 1.5 # 1.5 factor for surface
    
    # Generate grid
    # Max index
    max_idx = int(R / a) + 3
    
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # FCC basis
                for b in [[0,0,0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]:
                    p = np.array([i, j, k]) * a + np.array(b) * a
                    if np.linalg.norm(p) <= R:
                        points.append(p)
    
    points = np.array(points)
    if len(points) < n:
        return None
        
    # Sort by distance from center and take first n
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)[:n]
    return points[indices]

def _generate_random(n, rng, min_dist=0.5):
    # Generate random points in a sphere with minimum distance
    R = 1.0 * n ** (1.0/3.0)
    pts = []
    max_attempts = 1000
    attempts = 0
    while len(pts) < n and attempts < max_attempts:
        p = rng.uniform(-R, R, 3)
        if np.linalg.norm(p) <= R:
            # Check min distance
            if all(np.linalg.norm(p - q) >= min_dist for q in pts):
                pts.append(p)
        attempts += 1
        
    if len(pts) < n:
        # If we couldn't fit them, just return random points (might be invalid, but relax might fix?)
        # Better to return None if we can't generate valid start
        return None
        
    return np.array(pts)

def construct(instance):
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "sutton_chen")
    
    rng = np.random.default_rng(n)
    
    best_energy = np.inf
    best_pos = None
    
    start_time = time.time()
    deadline = start_time + 280.0 # 280s limit to be safe
    
    # Generate initial structures
    starts = []
    
    fcc_pos = _generate_fcc(n, rng)
    if fcc_pos is not None:
        starts.append(fcc_pos)
        
    rand_pos = _generate_random(n, rng)
    if rand_pos is not None:
        starts.append(rand_pos)
        
    # Add jittered versions
    if fcc_pos is not None:
        for sigma in [0.1, 0.2, 0.3]:
            starts.append(fcc_pos + rng.normal(0, sigma, fcc_pos.shape))
            
    if rand_pos is not None:
        for sigma in [0.1, 0.2]:
            starts.append(rand_pos + rng.normal(0, sigma, rand_pos.shape))

    # Basin Hopping
    # For each start, do a few steps of perturb-relax
    for start_pos in starts:
        if time.time() > deadline:
            break
            
        # Check validity of start
        if np.any(~np.isfinite(start_pos)):
            continue
        # Simple min dist check
        min_d = np.inf
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(start_pos[i] - start_pos[j])
                if d < min_d:
                    min_d = d
                if min_d < 0.35:
                    break
            if min_d < 0.35:
                break
        if min_d < 0.35:
            continue
            
        current_pos = start_pos
        current_energy = np.inf
        
        # Relax initial
        try:
            current_energy, current_pos = relax(current_pos, potential, kind)
        except Exception:
            continue
            
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
            
        # Basin Hopping steps
        # Number of steps depends on time
        # Let's do up to 20 steps per start if time permits
        for step in range(20):
            if time.time() > deadline:
                break
                
            # Perturb
            sigma = 0.2 + 0.1 * step # Increase perturbation
            perturbed_pos = current_pos + rng.normal(0, sigma, current_pos.shape)
            
            # Check validity
            if np.any(~np.isfinite(perturbed_pos)):
                continue
            min_d = np.inf
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(perturbed_pos[i] - perturbed_pos[j])
                    if d < 0.35:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
                
            # Relax
            try:
                new_energy, new_pos = relax(perturbed_pos, potential, kind)
            except Exception:
                continue
                
            if new_energy < current_energy:
                current_energy = new_energy
                current_pos = new_pos
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pos = new_pos.copy()
            else:
                # Optional: Metropolis accept, but simple hill climbing is faster and often sufficient for global min search with multiple starts
                pass

    if best_pos is None:
        # Fallback: random
        R = 1.0 * n ** (1.0/3.0)
        best_pos = rng.uniform(-R, R, (n, 3))
        
    return best_pos
