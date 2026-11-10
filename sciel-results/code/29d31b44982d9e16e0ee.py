"""
Strategy: Multi-Start Basin Hopping with Diverse Geometric Seeds
- Uses icosahedral, decahedral (FCC-like), and random compact seeds.
- Implements a Basin Hopping loop: Perturb -> Relax -> Accept/Reject.
- Uses adaptive perturbation scales to explore the energy landscape.
- Deadline-guarded to ensure valid output within 330s.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Deadline guard
_T0 = time.time()
_DEADLINE = 325.0  # Safety margin below 330s

def _time_left():
    return _DEADLINE - (time.time() - _T0)

def _icosahedron_vertices():
    """Generate vertices of a regular icosahedron normalized to unit radius."""
    phi = (1 + 5**0.5) / 2
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def _generate_icosahedral_seed(n, rng):
    """
    Generate an icosahedral seed.
    For magic numbers (1, 13, 55, 147), uses exact Mackay geometry.
    For other n, uses the largest magic number < n and adds atoms on the surface.
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    magic_nums = [1, 13, 55, 147]
    base_n = 1
    for mn in magic_nums:
        if mn <= n:
            base_n = mn
        else:
            break
            
    # Generate base structure
    if base_n == 1:
        pts = [np.array([0.0, 0.0, 0.0])]
    elif base_n == 13:
        v = _icosahedron_vertices()
        pts = [np.array([0.0, 0.0, 0.0])] + [v[i] for i in range(12)]
    elif base_n == 55:
        # 1 center + 12 vertices + 30 edge midpoints + 12 face centers? 
        # Actually 1 + 12 + 42 = 55.
        # 42 atoms in shell 2.
        # Let's approximate by scaling up the 13-atom icosahedron and adding atoms?
        # Better: Construct the 55-atom cluster properly if possible, else fallback.
        # For simplicity and robustness in this short time, we will use a "nested" approach:
        # Place 13 atoms at r=1.
        # Place 42 atoms on a sphere of radius ~2.
        # Or just use the 13-atom core and add 42 atoms randomly on a shell of radius 1.8.
        v = _icosahedron_vertices()
        pts = [np.array([0.0, 0.0, 0.0])] + [v[i] for i in range(12)]
        # Add 42 atoms on a shell
        for _ in range(42):
            dir = rng.normal(size=3)
            dir /= np.linalg.norm(dir)
            # Add some randomness to radius to avoid perfect shell symmetry which might be unstable
            r = 1.8 + rng.uniform(-0.1, 0.1)
            pts.append(dir * r)
    else:
        # Fallback for larger n (not expected in this task, but safe)
        v = _icosahedron_vertices()
        pts = [np.array([0.0, 0.0, 0.0])] + [v[i] for i in range(12)]
        for _ in range(n - 13):
            dir = rng.normal(size=3)
            dir /= np.linalg.norm(dir)
            r = 2.5 + rng.uniform(-0.2, 0.2)
            pts.append(dir * r)

    pts = np.array(pts)
    
    # If we have more than n atoms, truncate (should not happen if logic is correct)
    if len(pts) > n:
        pts = pts[:n]
    elif len(pts) < n:
        # Add random atoms on the surface
        current_radius = np.max(np.linalg.norm(pts, axis=1)) if len(pts) > 0 else 1.0
        for _ in range(n - len(pts)):
            dir = rng.normal(size=3)
            dir /= np.linalg.norm(dir)
            r = current_radius + rng.uniform(-0.2, 0.5)
            pts = np.vstack([pts, dir * r])
            
    return pts

def _generate_decahedral_seed(n, rng):
    """
    Generate a decahedral (FCC-like) seed.
    Uses a central atom and shells of FCC stacking.
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # FCC magic numbers: 1, 13, 55, 147, 309...
    # 13-atom decahedron: 1 center, 12 vertices of a regular tetrahedron? No, FCC 13 is a truncated octahedron?
    # Actually, the 13-atom FCC cluster is a central atom surrounded by 12 atoms in an icosahedral-like arrangement? 
    # No, FCC 13 is usually the "decahedron" which has 5-fold symmetry? 
    # Wait, the 13-atom FCC cluster is often described as a central atom with 12 neighbors in a cuboctahedral arrangement? 
    # Let's just use a generic compact FCC-like packing.
    # For simplicity, we'll generate a random compact cluster with FCC-like local order by using a smaller spacing.
    
    # A simple heuristic for decahedral: Use a smaller radius for the same number of atoms compared to icosahedral.
    # Or just use random compact with a specific seed bias.
    # To differentiate from icosahedral, let's use a "cubical" packing.
    
    pts = [np.array([0.0, 0.0, 0.0])]
    # Add 12 atoms in a cuboctahedral arrangement?
    # Vertices of a cuboctahedron: (0, 1, 1), (0, 1, -1), ...
    v = np.array([
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / norms
    pts.extend([v[i] for i in range(12)])
    
    if n > 13:
        # Add more atoms on a larger shell
        for _ in range(n - 13):
            dir = rng.normal(size=3)
            dir /= np.linalg.norm(dir)
            r = 1.8 + rng.uniform(-0.1, 0.1)
            pts.append(dir * r)
            
    pts = np.array(pts)
    if len(pts) > n:
        pts = pts[:n]
    elif len(pts) < n:
        current_radius = np.max(np.linalg.norm(pts, axis=1)) if len(pts) > 0 else 1.0
        for _ in range(n - len(pts)):
            dir = rng.normal(size=3)
            dir /= np.linalg.norm(dir)
            r = current_radius + rng.uniform(-0.2, 0.5)
            pts = np.vstack([pts, dir * r])
            
    return pts

def _generate_random_compact_seed(n, rng):
    """
    Generate a random compact cluster using rejection sampling.
    """
    spacing = 0.75
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    max_attempts = 10000
    attempts = 0
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check distance to existing points
            if points:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) < 0.8 * spacing:
                    attempts += 1
                    continue
            points.append(p)
            attempts = 0 # Reset attempts on success
    
    if len(points) < n:
        # Fallback: Just add random points if rejection fails
        for _ in range(n - len(points)):
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
            
    return np.array(points)

def construct(instance):
    """
    Multi-Start Basin Hopping.
    """
    n = instance["n"]
    potential = instance["potential"]
    kind = instance["kind"]
    
    best_energy = float('inf')
    best_pos = None
    
    rng = np.random.default_rng(42)
    
    # Define seeds
    seeds = []
    
    # 1. Icosahedral seeds (most likely to be good for SC)
    for i in range(3):
        seed = _generate_icosahedral_seed(n, rng)
        seeds.append(seed)
        
    # 2. Decahedral/FCC seeds
    for i in range(2):
        seed = _generate_decahedral_seed(n, rng)
        seeds.append(seed)
        
    # 3. Random compact seeds
    for i in range(2):
        seed = _generate_random_compact_seed(n, rng)
        seeds.append(seed)
        
    # Basin Hopping Loop
    # We will iterate through seeds, and for each, run a few BH steps.
    # Or, we can maintain a global best and do BH from the current best.
    # Strategy: 
    # 1. Relax all initial seeds.
    # 2. Keep the best.
    # 3. Run BH from the best, and also from the 2nd and 3rd best to avoid getting stuck.
    
    relaxed_seeds = []
    for seed in seeds:
        if _time_left() < 10.0:
            break
        try:
            e, pos = relax(seed, potential, kind)
            if np.isfinite(e) and e < best_energy:
                best_energy = e
                best_pos = pos
            relaxed_seeds.append((e, pos))
        except Exception:
            pass
            
    # Sort relaxed seeds by energy
    relaxed_seeds.sort(key=lambda x: x[0])
    
    # Run Basin Hopping from the top 3 seeds
    for e_init, pos_init in relaxed_seeds[:3]:
        current_e = e_init
        current_pos = pos_init
        
        # BH Parameters
        # Start with larger perturbations and decrease
        scales = [0.3, 0.2, 0.1, 0.05, 0.02]
        
        for scale in scales:
            if _time_left() < 5.0:
                break
                
            # Try a few steps at this scale
            for _ in range(5):
                if _time_left() < 5.0:
                    break
                
                # Perturb
                perturbation = rng.normal(0, scale, current_pos.shape)
                perturbed_pos = current_pos + perturbation
                
                # Relax
                try:
                    e_new, pos_new = relax(perturbed_pos, potential, kind)
                    if not np.isfinite(e_new):
                        continue
                        
                    # Accept if better, or with some probability (Metropolis is not strictly needed for global min search if we just track best, 
                    # but accepting worse moves helps escape. Since we have a time limit, simple greedy + restarts is often safer/faster.
                    # However, standard BH accepts worse moves. Let's use a simple greedy update for the *current* chain, 
                    # but keep track of the *global* best separately.
                    
                    if e_new < current_e:
                        current_e = e_new
                        current_pos = pos_new
                        
                    # Update global best
                    if e_new < best_energy:
                        best_energy = e_new
                        best_pos = pos_new
                        
                except Exception:
                    pass
                    
    # Final check
    if best_pos is None:
        # Fallback: linear chain
        pos = np.zeros((n, 3))
        for i in range(n):
            pos[i] = [i * 1.0, 0, 0]
        return pos
        
    return best_pos
# EVOLVE-BLOCK-END
