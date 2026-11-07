"""Basin-Hopping search program for Sutton-Chen TM cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from multiprocessing import Pool, cpu_count
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _is_valid(positions, min_dist=0.35):
    """Check that all coordinates are finite and pairwise distances >= min_dist."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True

def _generate_icosahedral_shells(n, rng):
    """
    Generate points on concentric shells based on icosahedral symmetry.
    This creates a high-symmetry, dense packing suitable for TM clusters.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = []
    
    # Shell 1: 12 vertices (Icosahedron)
    if n >= 12:
        points.extend([v.copy() for v in verts])
        n -= 12
    
    # Shell 2: 30 vertices (Cuboctahedron-like / edge midpoints scaled)
    # For n=38, 38-12=26. We need 26 more.
    # We can use the 30 edge midpoints of the icosahedron, or just random on a sphere of radius ~1.5
    # Let's use a dense random spherical shell for the rest to ensure coverage
    if n > 0:
        radius2 = 1.5
        # Add points on this shell
        for _ in range(n):
            theta = rng.uniform(0, 2 * np.pi)
            cos_phi = rng.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi**2)
            p = radius2 * np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
            points.append(p)
        n -= 0 # n becomes 0 effectively if we filled it, but let's be precise
        
    # If n was very large, we'd add more shells, but for 38-80 this is sufficient.
    # For n=75, 75-12=63. 63 points on shell 2 (radius 1.5) might be too dense?
    # Ideal radius for N atoms in shell k is roughly proportional to sqrt(N).
    # Let's just fill remaining with random spherical points at radius 1.5.
    # Actually, for n=75, 63 points on a sphere of radius 1.5 is dense but relaxable.
    
    return np.array(points[:len(points)]) # Ensure exact count? No, we generated exactly n if logic holds.
    # Wait, the logic above: 
    # if n>=12, add 12, n-=12.
    # if n>0, add n points.
    # So total is 12 + (original_n - 12) = original_n. Correct.

def _relax_one(args):
    """Relax a single starting configuration."""
    positions, potential, kind = args
    try:
        energy, relaxed_positions = relax(positions, potential, kind)
        if np.isfinite(energy) and _is_valid(relaxed_positions):
            return (energy, relaxed_positions)
    except Exception:
        pass
    return (np.inf, None)

def _basin_hopping_step(positions, rng, scale, potential, kind):
    """
    Perform a single basin-hopping step:
    1. Perturb positions
    2. Relax
    3. Return energy and new positions
    """
    perturbed = positions + rng.normal(0, scale, positions.shape)
    return _relax_one((perturbed, potential, kind))

def construct(instance):
    """
    Basin-Hopping search:
    1. Generate high-symmetry IC (Icosahedral shells)
    2. Relax
    3. Iteratively perturb and relax with Metropolis acceptance
    4. Use multiprocessing for parallel perturb/relax attempts to explore landscape faster
    """
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho", "12-6"))
    
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # 1. Initial Configuration
    rng = np.random.default_rng(size * 1000 + 42)
    init_pos = _generate_icosahedral_shells(size, rng)
    
    # Ensure validity of IC (shells should be valid, but check just in case)
    if not _is_valid(init_pos):
        # Fallback to random if shell generation failed (unlikely)
        init_pos = rng.uniform(-3, 3, (size, 3))
    
    # 2. Initial Relaxation
    current_energy, current_pos = _relax_one((init_pos, potential, kind))
    
    if not np.isfinite(current_energy) or current_pos is None:
        # Fallback: try a few random starts
        for _ in range(10):
            pos = rng.uniform(-3, 3, (size, 3))
            e, p = _relax_one((pos, potential, kind))
            if np.isfinite(e) and p is not None:
                current_energy = e
                current_pos = p
                break
    
    if current_pos is None:
        # Last resort: return random valid positions
        return _generate_random_valid(size, rng)

    best_energy = current_energy
    best_pos = current_pos.copy()
    
    # 3. Basin Hopping Loop
    # Strategy: 
    # - Start with small perturbation, increase if stuck? Or fixed schedule?
    # - For global minima, larger perturbations are needed to jump basins.
    # - We have ~280s. 
    # - Each relax takes ~0.1-1s depending on N.
    # - We can run ~100-500 relaxations.
    # - Use parallelism: generate K perturbations, relax in parallel, pick best (or Metropolis).
    # - Since we want global min, "pick best" (greedy) is often better than Metropolis for limited budget, 
    #   but Metropolis helps escape. Let's use a hybrid: 
    #   - Generate batch of perturbations with varying scales.
    #   - Relax all in parallel.
    #   - If any is better, accept the best.
    #   - If none better, accept the best of the batch with some probability (or just keep current).
    #   Actually, for global optimization, "Best of N" is often superior to Metropolis when N is large.
    
    n_threads = min(4, cpu_count())
    
    # Perturbation scales: start moderate, maybe increase over time?
    # Typical interatomic distance ~ 1.0-2.0.
    # Scale 0.5-1.0 is good for basin hopping.
    
    base_scale = 0.8
    
    # Loop until time runs out
    while time.time() < deadline:
        # Generate n_threads perturbations
        # Use a mix of scales to explore different basin depths
        scales = [base_scale * rng.uniform(0.5, 1.5) for _ in range(n_threads)]
        
        args_list = []
        for i in range(n_threads):
            perturbed = current_pos + rng.normal(0, scales[i], current_pos.shape)
            args_list.append((perturbed, potential, kind))
        
        # Relax in parallel
        with Pool(processes=n_threads) as pool:
            results = pool.map(_relax_one, args_list)
        
        # Process results
        best_in_batch_energy = np.inf
        best_in_batch_pos = None
        
        for energy, pos in results:
            if np.isfinite(energy) and pos is not None and _is_valid(pos):
                if energy < best_in_batch_energy:
                    best_in_batch_energy = energy
                    best_in_batch_pos = pos
        
        # Update current/best
        if best_in_batch_pos is not None:
            # Greedy acceptance: if we found something better, take it.
            # If not, we might still want to move to a similar energy state to explore?
            # For global min, sticking to the best found so far is usually safer unless we have a specific
            # annealing schedule. Let's try: 
            # If best_in_batch < current_energy: accept
            # Else: with probability p (e.g. 0.2), accept the best_in_batch to allow wandering.
            
            if best_in_batch_energy < current_energy:
                current_energy = best_in_batch_energy
                current_pos = best_in_batch_pos
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pos = current_pos.copy()
            else:
                # No improvement. 
                # To avoid getting stuck in a local minimum, we should occasionally accept a worse state.
                # However, if we accept worse, we might drift far. 
                # A common strategy is to restart from the best found if stuck for M iterations.
                # Let's keep it simple: stick with current unless we find better.
                # This is "Steepest Descent" in basin space, which can be trapped.
                # To mitigate, we increase perturbation scale if stuck?
                # Or, we accept the best_in_batch if it's not *too* much worse?
                
                # Let's implement a simple "Restart if stuck" logic.
                # If we haven't improved in the last 5 batches, restart from best_pos with larger scale.
                # This requires tracking history.
                
                # For now, let's just try to accept the best_in_batch with a small probability 
                # to maintain diversity in the search path.
                if rng.random() < 0.1:
                    current_energy = best_in_batch_energy
                    current_pos = best_in_batch_pos
                    # Do NOT update best_pos if it's worse
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_pos = current_pos.copy()

    # Final check
    if best_pos is not None and _is_valid(best_pos):
        return best_pos
    else:
        return _generate_random_valid(size, rng)

def _generate_random_valid(n, rng):
    """Generate a valid random configuration as a last resort."""
    positions = np.zeros((n, 3))
    spacing = 0.72
    # Simple grid-like placement to ensure validity
    idx = 0
    # Fill a cube
    side = int(np.ceil(n ** (1/3))) + 2
    for i in range(side):
        if idx >= n: break
        for j in range(side):
            if idx >= n: break
            for k in range(side):
                if idx >= n: break
                positions[idx] = [i * spacing, j * spacing, k * spacing]
                idx += 1
    # If still not enough (unlikely with side+2), add random far away
    while idx < n:
        positions[idx] = rng.uniform(-10, 10, 3)
        idx += 1
    
    # Final validation fix
    for i in range(n):
        for j in range(i):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < 0.35:
                positions[i] = positions[j] + 0.4 * np.array([1, 0, 0])
    return positions
# EVOLVE-BLOCK-END
