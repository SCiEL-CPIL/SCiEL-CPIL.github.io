"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

# Global deadline guard: must finish well before 330s wall-clock.
_TOTAL_START = time.time()
_SAFETY_MARGIN = 30.0  # seconds to leave before hard kill

def _remaining_time():
    """Return remaining seconds before our safety deadline."""
    return _TOTAL_START + 300.0 - _SAFETY_MARGIN - time.time()

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

def _local_search(sequence, chain, max_iters, rng, time_limit):
    """
    Lightweight local search: try moving individual residues to improve energy.
    Returns the best chain found.
    """
    n = len(sequence)
    if n < 3:
        return chain
    
    current_energy = _energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    occupied = {p: i for i, p in enumerate(chain)}
    
    iters = 0
    while iters < max_iters:
        if time.time() > time_limit:
            break
        iters += 1
        
        # Pick a random interior residue to try moving
        i = rng.randrange(1, n - 1)
        old_site = chain[i]
        
        improved = False
        # Try each neighbor of old_site as a new position
        for new_site in _neighbours(old_site):
            # Check if new_site is free (not occupied by any other residue)
            if new_site in occupied:
                if occupied[new_site] != i:
                    continue
                else:
                    continue
            
            # Check adjacency with neighbors
            if i > 0:
                prev = chain[i-1]
                if abs(new_site[0]-prev[0]) + abs(new_site[1]-prev[1]) + abs(new_site[2]-prev[2]) != 1:
                    continue
            if i < n - 1:
                nxt = chain[i+1]
                if abs(new_site[0]-nxt[0]) + abs(new_site[1]-nxt[1]) + abs(new_site[2]-nxt[2]) != 1:
                    continue
            
            # Compute energy change
            delta = 0
            if sequence[i] == "H":
                # Lose contacts at old_site
                for q in _neighbours(old_site):
                    j = occupied.get(q)
                    if j is not None and j != i and sequence[j] == "H" and abs(i-j) > 1:
                        delta += 1  # losing a contact increases energy (less negative)
                # Gain contacts at new_site
                for q in _neighbours(new_site):
                    j = occupied.get(q)
                    if j is not None and j != i and sequence[j] == "H" and abs(i-j) > 1:
                        delta -= 1  # gaining a contact decreases energy (more negative)
            
            new_energy = current_energy + delta
            if new_energy < current_energy:
                # Apply the move (accept any improvement)
                occupied.pop(old_site)
                occupied[new_site] = i
                chain[i] = new_site
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_chain = list(chain)
                improved = True
                break  # Only one move per iteration
        
        if not improved and iters % 100 == 0:
            if time.time() > time_limit:
                break
    
    return best_chain

def fold(sequence, dimension):
    """Best conformation of a fixed number of randomized greedy growths, with optional local search."""
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 0
    
    n = len(sequence)
    
    # Run growths with dynamic budget based on remaining time
    if n <= 48:
        num_growths = 300
    elif n <= 80:
        num_growths = 200
    else:
        num_growths = 150
    
    for attempt in range(num_growths):
        if _remaining_time() < 5.0:
            break
        
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if best is None or value < best_energy:
            best, best_energy = chain, value
    
    # Try local search on the best chain if time permits
    if best is not None and _remaining_time() > 10.0:
        remaining = _remaining_time()
        ls_time_limit = min(time.time() + remaining - 5.0, _TOTAL_START + 300.0 - _SAFETY_MARGIN)
        max_ls_iters = max(500, min(20000, 100 * len(sequence)))
        improved_chain = _local_search(sequence, best, max_ls_iters, rng, ls_time_limit)
        improved_energy = _energy(sequence, improved_chain)
        if improved_energy < best_energy:
            best, best_energy = improved_chain, improved_energy
    
    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(len(sequence))]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
