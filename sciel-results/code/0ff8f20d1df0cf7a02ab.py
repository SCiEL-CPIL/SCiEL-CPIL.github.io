"""HP lattice protein folding on cubic lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math
import sys


def _neighbours(point):
    """Yield all 6 neighbours of a point in 3D."""
    x, y, z = point
    yield (x + 1, y, z)
    yield (x - 1, y, z)
    yield (x, y + 1, z)
    yield (x, y - 1, z)
    yield (x, y, z + 1)
    yield (x, y, z - 1)


def _manhattan_dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2])


def _energy(sequence, chain):
    """Calculate energy: negative number of H-H contacts."""
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == 'H':
                    if i < j:
                        contacts += 1
    return -contacts


def _delta_energy(sequence, chain, occupied, i, old_pos, new_pos):
    """
    Compute energy change when moving residue i from old_pos to new_pos.
    Returns (delta, valid).
    """
    if new_pos == old_pos:
        return 0, True
    if new_pos in occupied:
        return 0, False
    
    # Check connectivity
    if i > 0:
        if _manhattan_dist(new_pos, chain[i - 1]) != 1:
            return 0, False
    if i < len(chain) - 1:
        if _manhattan_dist(new_pos, chain[i + 1]) != 1:
            return 0, False
    
    is_h = sequence[i] == 'H'
    delta = 0
    
    if is_h:
        # Remove contacts involving old_pos
        for q in _neighbours(old_pos):
            j = occupied.get(q)
            if j is not None and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                if i < j:
                    delta += 1  # losing a contact increases energy
        
        # Add contacts involving new_pos
        for q in _neighbours(new_pos):
            j = occupied.get(q)
            if j is not None and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                if i < j:
                    delta -= 1  # gaining a contact decreases energy
    
    return delta, True


def _pivot_move(chain, occupied, i, direction, rng):
    """
    Attempt a pivot move at position i with given direction.
    direction is a tuple (dx, dy, dz) with Manhattan distance 1.
    Returns (success, new_chain, new_occupied) or (False, None, None).
    """
    n = len(chain)
    if i <= 0 or i >= n:
        return False, None, None
    
    dx, dy, dz = direction
    new_pos = (chain[i][0] + dx, chain[i][1] + dy, chain[i][2] + dz)
    
    # Check if new_pos is occupied (except by the moving segment)
    # The segment from i to n-1 will be moved
    for j in range(i, n):
        if chain[j] == new_pos:
            return False, None, None
    
    # Check connectivity at the pivot point
    if i > 0:
        prev = chain[i - 1]
        if _manhattan_dist(new_pos, prev) != 1:
            return False, None, None
    
    # Check that the new position doesn't collide with the fixed part
    if new_pos in occupied:
        occ_idx = occupied[new_pos]
        if occ_idx < i:
            return False, None, None
    
    # Perform the move: reflect the tail
    new_chain = list(chain)
    new_occupied = dict(occupied)
    
    # Remove old tail positions
    for j in range(i, n):
        del new_occupied[chain[j]]
    
    # Compute new positions by reflecting
    old_pivot = chain[i]
    new_pivot = new_pos
    
    for j in range(i, n):
        old_p = chain[j]
        # Reflect: new_p = old_p + (new_pivot - old_pivot)
        new_p = (old_p[0] + (new_pivot[0] - old_pivot[0]),
                 old_p[1] + (new_pivot[1] - old_pivot[1]),
                 old_p[2] + (new_pivot[2] - old_pivot[2]))
        new_chain[j] = new_p
        if new_p in new_occupied:
            # Collision
            return False, None, None
        new_occupied[new_p] = j
    
    return True, new_chain, new_occupied


def _anneal(sequence, chain, occupied, time_limit, rng, T_start=3.0, T_end=0.1, cooling=0.997):
    """Simulated annealing with pivot moves."""
    n = len(chain)
    current_energy = _energy(sequence, chain)
    best_energy = current_energy
    best_chain = list(chain)
    best_occupied = dict(occupied)
    
    start_time = time.time()
    T = T_start
    
    while T > T_end:
        if time.time() - start_time > time_limit:
            break
        
        # Choose a random pivot point
        i = rng.randint(1, n - 2)
        
        # Choose a random direction
        dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        direction = rng.choice(dirs)
        
        success, new_chain, new_occupied = _pivot_move(chain, occupied, i, direction, rng)
        
        if not success:
            T *= cooling
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            chain = new_chain
            occupied = new_occupied
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
                best_occupied = dict(occupied)
        
        T *= cooling
    
    return best_chain, best_occupied, best_energy


def _greedy_grow(sequence, rng):
    """Greedy growth with randomization."""
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 10.0 + free * 0.1 + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    
    return chain, occupied


def fold(sequence, dimension):
    """Find best conformation for the given sequence."""
    n = len(sequence)
    
    # Global deadline guard
    global_start = time.time()
    total_budget = 280.0  # Leave 50s margin for 330s limit
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Multiple greedy growths
    greedy_time = min(3.0, total_budget * 0.1)
    greedy_end = global_start + greedy_time
    
    for attempt in range(50):
        if time.time() > greedy_end:
            break
        rng = random.Random(42 + attempt * 7 + len(sequence))
        result = _greedy_grow(sequence, rng)
        if result is not None:
            chain, occupied = result
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
    
    # Phase 2: Annealing from best greedy result
    if best_chain is not None:
        remaining = total_budget - (time.time() - global_start)
        if remaining > 1.0:
            anneal_time = remaining * 0.6
            rng = random.Random(123 + len(sequence))
            occupied = {}
            for i, p in enumerate(best_chain):
                occupied[p] = i
            anneal_result = _anneal(sequence, best_chain, occupied, anneal_time, rng)
            if anneal_result is not None:
                a_chain, a_occ, a_energy = anneal_result
                if a_energy < best_energy:
                    best_energy = a_energy
                    best_chain = a_chain
    
    # Phase 3: More annealing from random starts if time remains
    remaining = total_budget - (time.time() - global_start)
    if remaining > 2.0:
        num_anneals = int(remaining / 3.0)
        for attempt in range(num_anneals):
            if time.time() - global_start > total_budget - 1.0:
                break
            rng = random.Random(999 + attempt * 13 + len(sequence))
            
            # Random start
            chain = [(0, 0, 0)]
            occupied = {(0, 0, 0): 0}
            success = True
            for i in range(1, n):
                options = [s for s in _neighbours(chain[-1]) if s not in occupied]
                if not options:
                    success = False
                    break
                site = rng.choice(options)
                chain.append(site)
                occupied[site] = i
            
            if success and len(chain) == n:
                anneal_time = min(2.0, total_budget - (time.time() - global_start))
                if anneal_time > 0.5:
                    anneal_result = _anneal(sequence, chain, occupied, anneal_time, rng)
                    if anneal_result is not None:
                        a_chain, a_occ, a_energy = anneal_result
                        if a_energy < best_energy:
                            best_energy = a_energy
                            best_chain = a_chain
    
    # Fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
