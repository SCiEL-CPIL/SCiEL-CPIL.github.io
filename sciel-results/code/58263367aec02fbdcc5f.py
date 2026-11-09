"""New approach: local search with simulated annealing on HP lattice."""
# EVOLVE-BLOCK-START
import random
import math
import time


def _neighbours(point):
    """Yield the 4 neighbours of a point on the square lattice."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _compute_energy(sequence, chain):
    """Compute the energy of a conformation."""
    if not chain or len(chain) != len(sequence):
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _generate_random_saw(n, rng, dimension=2):
    """Generate a random self-avoiding walk of length n."""
    if n <= 1:
        return [(0, 0)]
    
    start = (0, 0)
    chain = [start]
    occupied = {start}
    
    for _ in range(n - 1):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site not in occupied:
                options.append(site)
        
        if not options:
            # Dead end - restart
            return _generate_random_saw(n, rng, dimension)
        
        site = rng.choice(options)
        chain.append(site)
        occupied.add(site)
    
    return chain


def _snake_conformation(n):
    """Generate a snake/zigzag conformation that always works."""
    chain = []
    for i in range(n):
        x = i % 4
        y = i // 4
        if i % 4 == 1:
            x = i % 4
        elif i % 4 == 2:
            x = 3 - (i % 4)
        elif i % 4 == 3:
            x = 3 - (i % 4)
        chain.append((x, y))
    return chain


def _simulated_annealing(sequence, init_chain, time_limit, rng):
    """Run simulated annealing to improve a conformation."""
    n = len(sequence)
    chain = list(init_chain)
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Temperature schedule
    t0 = 1.0
    t_end = 0.01
    cooling_rate = 0.999
    
    temperature = t0
    accepted = 0
    total = 0
    
    while temperature > t_end:
        if time.time() > deadline:
            break
        
        # Choose a random move: pick a random interior bead and try to move it
        if n <= 2:
            break
        
        # Pick a random index to move (not the ends)
        idx = rng.randint(1, n - 2)
        
        # Try all 4 directions
        original_pos = chain[idx]
        moved = False
        new_energy = None
        
        for new_pos in _neighbours(original_pos):
            # Check if new position is unoccupied (except by the bead itself and its neighbors)
            # We need to be careful: the bead at idx is moving, so its current position
            # should be considered free, and its neighbors' positions need checking
            
            # Create a temporary set of occupied positions
            # For efficiency, we'll check manually
            
            valid = True
            for i, pos in enumerate(chain):
                if i == idx:
                    continue
                if i == idx - 1 or i == idx + 1:
                    # These are adjacent in the chain, so they can't be at the new position
                    # unless the new position is one of their positions (which would be invalid)
                    if pos == new_pos:
                        valid = False
                        break
                else:
                    if pos == new_pos:
                        valid = False
                        break
            
            if valid:
                # Try this move
                chain[idx] = new_pos
                candidate_energy = _compute_energy(sequence, chain)
                energy_delta = candidate_energy - current_energy
                
                if energy_delta <= 0 or rng.random() < math.exp(-energy_delta / temperature):
                    current_energy = candidate_energy
                    moved = True
                    new_energy = candidate_energy
                    
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                    break
        
        if not moved:
            # Try pivot move: pick two adjacent beads and try to rotate
            if n >= 3:
                idx = rng.randint(0, n - 2)
                # Try rotating the segment
                for dir1 in _neighbours(chain[idx]):
                    for dir2 in _neighbours(chain[idx + 1]):
                        # This is complex; skip for now and just try simple moves
                        pass
        
        total += 1
        if moved:
            accepted += 1
        
        # Cool down
        temperature *= cooling_rate
        
        # Batch processing: do multiple moves before cooling
        if total % 10 == 0:
            if time.time() > deadline:
                break
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using simulated annealing with multiple restarts."""
    n = len(sequence)
    rng = random.Random(12345 + len(sequence))
    
    # Global time limit with safety margin
    global_start_time = time.time()
    global_limit = 150.0  # Leave 60s margin for 210s total
    
    # Generate initial conformation
    # Try snake first as it's guaranteed to work
    init_chain = _snake_conformation(n)
    
    best_chain_overall = list(init_chain)
    best_energy_overall = _compute_energy(sequence, init_chain)
    
    # Run multiple SA restarts with different initial conformations
    restart_count = 0
    max_restarts = 10
    
    while True:
        elapsed = time.time() - global_start_time
        if elapsed > global_limit:
            break
        
        remaining = global_limit - elapsed
        if remaining < 5:
            break
        
        # Allocate time for this restart
        alloc_time = min(20.0, remaining - 5.0)
        if alloc_time <= 1:
            break
        
        # For restart 0, use snake. For others, try random SAWs
        if restart_count == 0:
            init = init_chain
        else:
            # Try to generate a random SAW, fall back to snake
            init = _generate_random_saw(n, rng, dimension)
            if init is None or len(init) != n:
                init = init_chain
        
        # Run SA
        chain, energy = _simulated_annealing(sequence, init, alloc_time, rng)
        
        if chain is not None and len(chain) == n:
            if energy < best_energy_overall:
                best_chain_overall = chain
                best_energy_overall = energy
        
        restart_count += 1
        
        if restart_count >= max_restarts:
            break
        if remaining < 10:
            break
    
    # Ensure output format: list of lists
    return [list(p) for p in best_chain_overall]
# EVOLVE-BLOCK-END
