"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math


def _neighbours2d(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours2d(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours2d(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours2d(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours2d(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _calc_delta_energy(sequence, chain, occupied, i, old_pos, new_pos):
    """Calculate energy change when moving residue i from old_pos to new_pos."""
    if sequence[i] != 'H':
        # Only H residues contribute to energy
        return 0
    
    delta = 0
    # Count contacts lost (old_pos neighbors that are H and not adjacent in chain)
    for q in _neighbours2d(old_pos):
        j = occupied.get(q)
        if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
            delta += 1  # Losing a contact increases energy (less negative)
    
    # Count contacts gained (new_pos neighbors that are H and not adjacent in chain)
    for q in _neighbours2d(new_pos):
        j = occupied.get(q)
        if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
            delta -= 1  # Gaining a contact decreases energy (more negative)
    
    return delta


def _is_valid_move(chain, occupied, i, new_pos):
    """Check if moving residue i to new_pos preserves self-avoidance and validity."""
    if new_pos in occupied:
        return False
    
    # Check that new_pos is not adjacent to any non-consecutive residue
    for q in _neighbours2d(new_pos):
        j = occupied.get(q)
        if j is not None and abs(j - i) > 1:
            return False
    
    return True


def _local_search(sequence, chain, max_iter=2000, temperature=0.0, rng=None):
    """Improve conformation via simulated annealing with incremental energy computation."""
    n = len(chain)
    if n < 4:
        return chain, _energy(sequence, chain)
    
    if rng is None:
        rng = random.Random()
    
    best_chain = list(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_energy = current_energy
    occupied = {p: i for i, p in enumerate(current_chain)}
    
    # Simulated annealing parameters
    t_start = 1.0
    t_end = 0.01
    cooling = (t_end / t_start) ** (1.0 / max(1, max_iter))
    
    for it in range(max_iter):
        temp = t_start * (cooling ** it)
        
        # Pick a random residue to move (not endpoints)
        i = rng.randint(1, n - 2)
        old_pos = current_chain[i]
        
        # Try random neighbor
        neighbors = list(_neighbours2d(old_pos))
        new_pos = neighbors[rng.randint(0, 3)]
        
        if not _is_valid_move(current_chain, occupied, i, new_pos):
            continue
        
        # Calculate energy change
        delta_e = _calc_delta_energy(sequence, current_chain, occupied, i, old_pos, new_pos)
        
        # Acceptance criterion
        if delta_e < 0 or rng.random() < math.exp(-delta_e / max(temp, 1e-10)):
            # Accept move
            current_chain[i] = new_pos
            occupied.pop(old_pos)
            occupied[new_pos] = i
            current_energy += delta_e
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using randomized greedy growths with simulated annealing."""
    start_time = time.time()
    deadline = start_time + 175.0  # Safety margin before 210s kill
    
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 1
    
    n = len(sequence)
    # Adjust number of restarts based on sequence length
    if n <= 30:
        num_restarts = 300
    elif n <= 60:
        num_restarts = 200
    else:
        num_restarts = 100
    
    # Adjust SA iterations based on sequence length
    if n <= 30:
        sa_iters = 3000
    elif n <= 60:
        sa_iters = 2000
    else:
        sa_iters = 1500
    
    for attempt in range(num_restarts):
        if time.time() > deadline:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Apply simulated annealing to improve
        chain, energy = _local_search(sequence, chain, max_iter=sa_iters, rng=rng)
        
        if energy < best_energy:
            best, best_energy = chain, energy
    
    if best is None:  # a straight line is always self-avoiding
        best = [(i, 0) for i in range(len(sequence))]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
