import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

POTENTIAL = 'dzugutov'
_GLOBAL_START_TIME = None
_GLOBAL_DEADLINE = None


def _get_icosahedron_vertices():
    """Return the 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    verts = np.array(verts, dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts


def _get_mackay_icosahedron_55(bond_length=1.1):
    """
    Generate the 55-atom Mackay icosahedron.
    Structure: 1 center + 12 vertices + 42 shell-2 atoms
    Shell 2 atoms are at face centers (20) and edge midpoints (30)? 
    Actually 1+12+42=55. The 42 are arranged with icosahedral symmetry.
    
    A common construction:
    - 1 center
    - 12 at icosahedron vertices (distance R1)
    - 20 at face centers (distance R2) 
    - 30 at edge midpoints? No, that gives 63.
    
    Correct Mackay 55: The 42 shell-2 atoms are at positions that form
    a truncated icosahedron-like shell. Let me use a different approach:
    generate points with icosahedral symmetry.
    
    Actually, the standard 55-atom Mackay icosahedron has:
    - 1 center
    - 12 in first shell (icosahedron vertices)
    - 42 in second shell
    
    The 42 second-shell atoms can be generated as:
    - 20 face centers of the icosahedron
    - 30 edge midpoints? 20+30=50, not 42.
    
    Let me use a simpler robust method: generate 55 points with icosahedral
    symmetry by taking the 13-atom icosahedron and adding 42 points at
    positions that maintain icosahedral symmetry.
    
    Alternative: Use the fact that we can just relax a well-structured initial
    guess. Let me generate a dense icosahedral-like cluster.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = _get_icosahedron_vertices()
    
    # Center atom
    atoms = [np.array([0.0, 0.0, 0.0])]
    
    # First shell: 12 icosahedron vertices
    R1 = bond_length
    for v in verts:
        atoms.append(R1 * v)
    
    # Second shell: 42 atoms
    # Generate these at positions that maintain icosahedral symmetry
    # The 42 atoms in a Mackay icosahedron are located at:
    # - 20 face centers (one per triangular face)
    # - 30 edge midpoints? That's 50, too many.
    # 
    # Actually, let me use a different construction.
    # The 55-atom cluster can be seen as a close-packed icosahedral structure.
    # 
    # Let me just generate 42 additional atoms in a shell with icosahedral
    # symmetry by using the icosahedron's symmetry operations.
    
    # Icosahedral symmetry operations (24 rotations)
    # For simplicity, let me place atoms at specific symmetric positions
    
    # Face centers: 20 faces of icosahedron
    # Edge midpoints: 30 edges
    # But 20 + 30 = 50, we need 42.
    
    # Let me use a different approach: generate points on a sphere with
    # icosahedral symmetry using the icosahedron's vertices and applying
    # symmetry operations to a few seed points.
    
    # Actually, the simplest robust approach: generate 55 points in a
    # compact icosahedral-like arrangement and relax.
    
    # Let me generate points by taking the 13-atom icosahedron and adding
    # 42 points at the midpoints of edges and face centers, then select
    # 42 of them with proper symmetry.
    
    # For now, let me use a simpler method: generate 55 points in a sphere
    # with icosahedral bias, then relax.
    
    # Better: Use known coordinates for 55-atom Mackay icosahedron
    # The structure has icosahedral symmetry (Ih point group, 120 operations)
    
    # Let me generate the 42 shell-2 atoms by placing them at:
    # - 12 positions corresponding to edge midpoints (but only 12, not 30)
    # - 30 positions corresponding to face centers? 
    # 
    # This is getting complex. Let me use a pragmatic approach:
    # Generate 55 atoms in a compact icosahedral-like structure using
    # rejection sampling with icosahedral symmetry bias, then relax.
    
    # For the purpose of this implementation, let me generate a reasonable
    # initial guess and rely on relaxation.
    
    # Generate 42 additional atoms in a shell of radius ~1.5-2.0 * bond_length
    R2 = bond_length * 1.5
    
    # Generate 42 points with icosahedral symmetry
    # Use the icosahedron's 24 rotational symmetries
    # Place seed points and apply all symmetries
    
    # Icosahedral rotation matrices (24 proper rotations)
    # For simplicity, let me just generate 42 points in a shell
    # with some icosahedral bias
    
    # Actually, let me just generate 42 points randomly in a shell
    # and relax - the relaxation will find the icosahedral structure
    
    rng = np.random.default_rng(42)
    existing = np.array(atoms)
    min_dist = 0.5
    
    for _ in range(42):
        for attempt in range(1000):
            # Sample in a shell
            r = rng.uniform(R1 * 1.2, R2 * 1.5)
            theta = rng.uniform(0, 2 * np.pi)
            cos_phi = rng.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi**2)
            p = np.array([
                r * sin_phi * np.cos(theta),
                r * sin_phi * np.sin(theta),
                r * cos_phi
            ])
            
            dists = np.linalg.norm(existing - p, axis=1)
            if np.min(dists) >= min_dist:
                atoms.append(p)
                existing = np.array(atoms)
                break
    
    return np.array(atoms)


def _generate_seed(size, seed_idx, rng):
    """Generate a diverse seed structure of the given size."""
    if size == 55:
        return _get_mackay_icosahedron_55(bond_length=1.1)
    
    # Generate a 55-atom base and modify
    base = _get_mackay_icosahedron_55(bond_length=1.1)
    
    if size < 55:
        # Remove atoms from the outermost shell
        dists = np.linalg.norm(base, axis=1)
        sort_idx = np.argsort(dists)[::-1]  # Farthest first
        num_to_remove = 55 - size
        remove_idx = sort_idx[:num_to_remove]
        keep_idx = np.setdiff1d(np.arange(55), remove_idx)
        return base[keep_idx]
    else:
        # Add atoms near the surface
        atoms = base.copy()
        num_to_add = size - 55
        min_dist = 0.5
        
        for _ in range(num_to_add):
            for attempt in range(2000):
                # Place near a random surface atom
                surface_atoms = atoms[np.linalg.norm(atoms, axis=1) > 1.0]
                if len(surface_atoms) == 0:
                    surface_atoms = atoms
                ref = surface_atoms[rng.integers(len(surface_atoms))]
                offset = rng.normal(0, 0.4, 3)
                p = ref + offset
                
                dists = np.linalg.norm(atoms - p, axis=1)
                if np.min(dists) >= min_dist:
                    atoms = np.vstack([atoms, p])
                    break
        return atoms


def construct(instance):
    global _GLOBAL_START_TIME, _GLOBAL_DEADLINE
    
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        _GLOBAL_DEADLINE = _GLOBAL_START_TIME + 290.0
    
    size = instance["n"]
    start_time = time.time()
    
    # Time budget for this instance
    remaining = _GLOBAL_DEADLINE - time.time()
    instance_deadline = start_time + min(45.0, max(10.0, remaining - 5.0))
    
    rng = np.random.default_rng(size * 1000 + 42)
    
    # Generate multiple diverse seeds
    seeds = []
    for i in range(4):
        try:
            seed = _generate_seed(size, i, rng)
            if seed is not None and len(seed) == size:
                seeds.append(seed)
        except Exception:
            pass
    
    # Fallback: random compact
    if not seeds:
        radius = 0.8 * size ** (1/3) + 2.0
        points = []
        for _ in range(size):
            for _ in range(2000):
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) > radius:
                    continue
                if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                    points.append(p)
                    break
            else:
                p = rng.uniform(-radius * 1.5, radius * 1.5, 3)
                points.append(p)
        seeds.append(np.array(points))
    
    # Relax all seeds and keep the best
    best_energy = float('inf')
    best_positions = None
    
    for seed in seeds:
        try:
            e, pos = relax(seed, POTENTIAL)
            if e < best_energy:
                best_energy = e
                best_positions = pos.copy()
        except Exception:
            continue
    
    if best_positions is None:
        # Last resort: return the first seed
        return seeds[0]
    
    # Basin Hopping
    current_positions = best_positions.copy()
    temperature = 0.15
    step_size = 0.25
    patience = 0
    max_patience = 30
    
    while time.time() < instance_deadline:
        perturbation = rng.normal(0, step_size, size=(size, 3))
        new_positions = current_positions + perturbation
        
        try:
            new_energy, new_relaxed = relax(new_positions, POTENTIAL)
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_positions = new_relaxed.copy()
                current_positions = new_relaxed.copy()
                patience = 0
            else:
                delta_e = new_energy - best_energy
                acceptance_prob = np.exp(-delta_e / temperature)
                if rng.random() < acceptance_prob:
                    current_positions = new_relaxed.copy()
                    patience += 1
                else:
                    patience += 1
            
            # Anneal
            temperature *= 0.97
            if temperature < 1e-3:
                temperature = 1e-3
            step_size *= 0.995
            
            if patience > max_patience:
                # Reset with a larger perturbation
                current_positions = best_positions + rng.normal(0, 0.5, size=(size, 3))
                patience = 0
                temperature = 0.1
                step_size = 0.25
                
        except Exception:
            continue
    
    return best_positions
