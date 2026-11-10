"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import sys

_DEADLINE = time.time() + 175.0


def _neighbours2d(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    n = len(chain)
    if n < 2:
        return 0
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i in range(n):
        if sequence[i] == 'H':
            x, y = chain[i]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                j = where.get((x + dx, y + dy))
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2


def _energy_delta_for_move(sequence, chain, i, new_pos):
    """
    Compute the energy delta (in contacts) when moving residue i from chain[i] to new_pos.
    Returns delta_contacts = new_contacts_involving_i - old_contacts_involving_i.
    """
    n = len(chain)
    if i < 1 or i > n - 2:
        return 0
    
    old_pos = chain[i]
    old_x, old_y = old_pos
    new_x, new_y = new_pos
    
    # Build pos_to_idx for O(1) lookup
    pos_to_idx = {}
    for idx, pos in enumerate(chain):
        pos_to_idx[pos] = idx
    
    # Count old contacts involving residue i
    old_contacts = 0
    if sequence[i] == 'H':
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor_pos = (old_x + dx, old_y + dy)
            j = pos_to_idx.get(neighbor_pos)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                old_contacts += 1
    
    # Count new contacts involving residue i at new_pos
    new_contacts = 0
    if sequence[i] == 'H':
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -4)):
            neighbor_pos = (new_x + dx, new_y + dy)
            j = pos_to_idx.get(neighbor_pos)
            if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                new_contacts += 1
    
    return new_contacts - old_contacts


def _fast_grow(sequence, rng, temperature=0.0):
    """
    Fast randomized greedy chain growth.
    Prefer sites that create new H-H contacts, then sites with more free space.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    res_type = [1 if c == 'H' else 0 for c in sequence]
    
    for i in range(1, n):
        last = chain[-1]
        x, y = last
        options = []
        
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            site = (x + dx, y + dy)
            if site in occupied:
                continue
            
            gain = 0
            if res_type[i]:
                sx, sy = site
                for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    j = occupied.get((sx + ddx, sy + ddy))
                    if j is not None and j < i - 1 and res_type[j]:
                        gain += 1
            
            free = 0
            for ddx, ddy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (site[0] + ddx, site[1] + ddy) not in occupied:
                    free += 1
            
            score = gain * 10.0 + free * 0.1 + rng.random() * temperature
            options.append((score, site))
        
        if not options:
            return None
        
        best_score = -1e18
        best_site = None
        for s, st in options:
            if s > best_score:
                best_score = s
                best_site = st
        
        occupied[best_site] = i
        chain.append(best_site)

    return chain


def _local_improve_fast(sequence, chain, rng, max_moves=2000):
    """
    Improve conformation via kink flips with incremental energy computation.
    Uses simulated annealing-like acceptance to escape local minima.
    """
    n = len(chain)
    if n < 3:
        return chain, _compute_energy(sequence, chain)
    
    # Build occupied map
    occupied = {p: i for i, p in enumerate(chain)}
    res_type = [1 if c == 'H' else 0 for c in sequence]
    
    current_energy = _compute_energy(sequence, chain)
    best_energy = current_energy
    best_chain = list(chain)
    
    # Temperature for simulated annealing
    T_start = 1.0
    T_end = 0.01
    total_moves = max_moves
    
    for move in range(total_moves):
        if time.time() >= _DEADLINE:
            break
        
        # Temperature decay
        T = T_start * (T_end / T_start) ** (move / max(1, total_moves))
        
        # Pick a random internal residue
        i = rng.randint(1, n - 2)
        
        # Try all 4 possible moves for residue i
        old_pos = chain[i]
        ox, oy = old_pos
        
        best_move_site = None
        best_delta = 0
        
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            new_pos = (ox + dx, oy + dy)
            
            # Check if new_pos is free (not occupied by another residue)
            if new_pos in occupied and occupied[new_pos] != i:
                continue
            
            # Check chain continuity: new_pos must be adjacent to i-1 and i+1
            px, py = chain[i - 1]
            if abs(new_pos[0] - px) + abs(new_pos[1] - py) != 1:
                continue
            px, py = chain[i + 1]
            if abs(new_pos[0] - px) + abs(new_pos[1] - py) != 1:
                continue
            
            # Compute energy delta
            delta = _energy_delta_for_move(sequence, chain, i, new_pos)
            
            if delta < best_delta:
                best_delta = delta
                best_move_site = new_pos
        
        if best_move_site is None:
            continue
        
        # Accept or reject based on simulated annealing
        if best_delta <= 0 or rng.random() < (2.718281828 ** (-best_delta / max(T, 1e-10))):
            # Perform the move
            old_pos = chain[i]
            chain[i] = best_move_site
            del occupied[old_pos]
            occupied[best_move_site] = i
            
            current_energy += -best_delta // 2  # Note: this is approximate
            
            # Recompute exact energy periodically to avoid drift
            if move % 100 == 0:
                current_energy = _compute_energy(sequence, chain)
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
    
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation with deadline-aware search."""
    n = len(sequence)
    if n == 0:
        return []
    
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 0
    
    # First, do a quick greedy grow to get a valid starting point
    chain = _fast_grow(sequence, rng, temperature=0.0)
    if chain is None:
        # Fallback: straight line
        chain = [(i, 0) for i in range(n)]
        best = chain
        best_energy = _compute_energy(sequence, chain)
        return [list(p) for p in best]
    
    best = chain
    best_energy = _compute_energy(sequence, chain)
    
    # Local optimization
    chain, energy = _local_improve_fast(sequence, chain, rng, max_moves=1000)
    if energy < best_energy:
        best, best_energy = chain, energy
    
    # Multiple restarts with different temperatures
    restarts = 0
    while time.time() < _DEADLINE and restarts < 50:
        restarts += 1
        temp = 0.1 * (restarts % 5 + 1)
        chain = _fast_grow(sequence, rng, temperature=temp)
        if chain is None:
            continue
        
        chain, energy = _local_improve_fast(sequence, chain, rng, max_moves=500)
        if energy < best_energy:
            best, best_energy = chain, energy
        
        # Early exit if we've hit the theoretical minimum (all H-H contacts)
        h_count = sequence.count('H')
        max_possible = h_count * (h_count - 1) // 2
        if best_energy <= -max_possible:
            break
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
