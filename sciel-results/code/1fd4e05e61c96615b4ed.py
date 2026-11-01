"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_clearance(positions):
    """Compute the minimum clearance (radius) for a given set of positions.
    Returns the minimum of pairwise half-distances and wall distances.
    """
    n = len(positions)
    if n == 0:
        return 0.0
    
    # Wall distances: distance from each point to nearest wall
    wall_dists = 0.5 - np.abs(positions)
    min_wall = np.min(wall_dists)
    
    if min_wall <= 0:
        return 0.0
    
    # Pairwise distances using efficient computation
    # Compute all pairwise squared distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair_sq = np.min(dists_sq)
    min_pair = np.sqrt(min_pair_sq) / 2.0
    
    return min(min_wall, min_pair)


def _energy(positions):
    """Energy function: negative of the minimum clearance (to be minimized)."""
    return -_compute_clearance(positions)


def _lattice_init(n, lattice_size=5, rng=None):
    """Initialize particles on a lattice with random perturbations."""
    if rng is None:
        rng = np.random.default_rng()
    
    # Create lattice points
    ticks = (np.arange(lattice_size) + 0.5) / lattice_size - 0.5
    lattice_points = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    if n <= len(lattice_points):
        # Select n random lattice points
        idx = rng.choice(len(lattice_points), size=n, replace=False)
        positions = lattice_points[idx].copy()
    else:
        # If n > lattice_size^3, use a denser lattice
        ls = int(np.ceil(n ** (1.0/3.0)))
        ticks2 = (np.arange(ls) + 0.5) / ls - 0.5
        lattice_points2 = np.array([(x, y, z) for z in ticks2 for y in ticks2 for x in ticks2])
        if n <= len(lattice_points2):
            idx = rng.choice(len(lattice_points2), size=n, replace=False)
            positions = lattice_points2[idx].copy()
        else:
            positions = rng.uniform(-0.45, 0.45, size=(n, 3))
    
    # Add small random perturbations to break symmetry
    positions += rng.normal(0, 0.02, size=positions.shape)
    positions = np.clip(positions, -0.49, 0.49)
    
    return positions


def _metropolis_lattice_gas(positions, n, max_iters=100000, t0=0.03, t1=0.0001, 
                             step_scale=1.0, rng=None, time_limit=None, start_time=None):
    """
    Metropolis-Hastings annealing for lattice gas model.
    Moves: single particle displacement, pair swap, cluster move.
    Energy: negative minimum clearance.
    """
    if rng is None:
        rng = np.random.default_rng()
    if start_time is None:
        start_time = time.time()
    
    current = positions.copy()
    current_energy = _energy(current)
    best = current.copy()
    best_energy = current_energy
    
    accepted = 0
    iters_done = 0
    
    for it in range(max_iters):
        # Check time limit
        if time_limit is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Geometric cooling
        t = t0 * (t1 / t0) ** (it / max_iters)
        
        # Choose move type: 70% single, 20% pair, 10% cluster
        move_type = rng.random()
        
        if move_type < 0.7:
            # Single particle move
            idx = [rng.integers(0, n)]
            step = step_scale * 0.05 * (t / t0) ** 0.3
            delta = rng.normal(0, step, size=(1, 3))
            
        elif move_type < 0.9:
            # Pair swap: move two particles and swap their positions
            idx = rng.choice(n, size=2, replace=False)
            step = step_scale * 0.03 * (t / t0) ** 0.3
            delta = rng.normal(0, step, size=(2, 3))
            # Swap: move i to j's position + delta, j to i's position + delta
            proposed_idx = idx.copy()
            delta[0], delta[1] = delta[1], delta[0]
            
        else:
            # Cluster move: move 3-5 nearby particles
            cluster_size = rng.integers(3, 6)
            idx = rng.choice(n, size=min(cluster_size, n), replace=False)
            step = step_scale * 0.02 * (t / t0) ** 0.3
            delta = rng.normal(0, step, size=(len(idx), 3))
        
        proposed = current.copy()
        proposed[idx] += delta
        
        # Clamp to valid region with margin
        proposed = np.clip(proposed, -0.499, 0.499)
        
        proposed_energy = _energy(proposed)
        
        # Metropolis acceptance
        if proposed_energy <= current_energy:
            accept = True
        else:
            delta_e = proposed_energy - current_energy
            # Accept with probability exp(-delta_e / T)
            if t > 1e-15:
                accept_prob = np.exp(-delta_e / t)
                accept = rng.random() < accept_prob
            else:
                accept = False
        
        if accept:
            current = proposed
            current_energy = proposed_energy
            accepted += 1
            if current_energy < best_energy:
                best = current.copy()
                best_energy = current_energy
        
        iters_done += 1
    
    return best, -best_energy, iters_done


def _local_refine(positions, n, rng=None, time_limit=None, start_time=None):
    """Local refinement using small random perturbations."""
    if rng is None:
        rng = np.random.default_rng()
    if start_time is None:
        start_time = time.time()
    
    current = positions.copy()
    current_score = _compute_clearance(current)
    best = current.copy()
    best_score = current_score
    
    for _ in range(50000):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        
        # Move one particle with small step
        idx = rng.integers(0, n)
        step = 0.005
        delta = rng.normal(0, step, size=3)
        
        proposed = current.copy()
        proposed[idx] += delta
        proposed = np.clip(proposed, -0.499, 0.499)
        
        score = _compute_clearance(proposed)
        
        if score > current_score:
            current = proposed
            current_score = score
            if score > best_score:
                best = current.copy()
                best_score = score
    
    return best, best_score


def construct():
    """Construct positions for 100 equal spheres in a unit cube."""
    n = N
    start_time = time.time()
    time_budget = 105  # Leave 15s buffer
    
    # Try multiple lattice initializations with different sizes
    best_positions = None
    best_score = 0.0
    
    rng = np.random.default_rng(42)
    
    # Try lattice sizes 4, 5, 6
    for ls in [4, 5, 6]:
        if time.time() - start_time > time_budget - 30:
            break
        
        pos = _lattice_init(n, lattice_size=ls, rng=rng)
        score = _compute_clearance(pos)
        
        if best_positions is None or score > best_score:
            best_positions = pos
            best_score = score
    
    # Run multiple annealing rounds
    # Round 1: High temperature, large steps
    if time.time() - start_time < time_budget - 40:
        pos1, score1, _ = _metropolis_lattice_gas(
            best_positions, n,
            max_iters=80000,
            t0=0.05,
            t1=0.001,
            step_scale=1.0,
            rng=np.random.default_rng(100),
            time_limit=time_budget - 40,
            start_time=start_time
        )
        if score1 > best_score:
            best_positions = pos1
            best_score = score1
    
    # Round 2: Medium temperature, medium steps
    if time.time() - start_time < time_budget - 20:
        pos2, score2, _ = _metropolis_lattice_gas(
            best_positions, n,
            max_iters=60000,
            t0=0.02,
            t1=0.0005,
            step_scale=0.7,
            rng=np.random.default_rng(200),
            time_limit=time_budget - 20,
            start_time=start_time
        )
        if score2 > best_score:
            best_positions = pos2
            best_score = score2
    
    # Round 3: Low temperature, small steps
    if time.time() - start_time < time_budget - 10:
        pos3, score3, _ = _metropolis_lattice_gas(
            best_positions, n,
            max_iters=40000,
            t0=0.01,
            t1=0.0001,
            step_scale=0.4,
            rng=np.random.default_rng(300),
            time_limit=time_budget - 10,
            start_time=start_time
        )
        if score3 > best_score:
            best_positions = pos3
            best_score = score3
    
    # Final local refinement
    if time.time() - start_time < time_budget - 2:
        pos4, score4 = _local_refine(
            best_positions, n,
            rng=np.random.default_rng(400),
            time_limit=time_budget - 2,
            start_time=start_time
        )
        if score4 > best_score:
            best_positions = pos4
            best_score = score4
    
    return best_positions
# EVOLVE-BLOCK-END
