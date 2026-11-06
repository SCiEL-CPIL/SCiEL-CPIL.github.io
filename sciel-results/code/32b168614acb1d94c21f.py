"""Improved HP lattice protein folding with delta-energy local search and simulated annealing."""
# EVOLVE-BLOCK-START
import random
import time

_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    # Budget is 120s CPU. Safety margin to 105s.
    return 105.0 - (time.time() - _START_TIME)

def _neighbours(point):
    """Yield 4-neighbours on square lattice."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _grow(sequence, dimension, rng):
    """
    One randomized greedy chain growth: prefer sites with new H-H contacts, then free space.
    """
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        current = chain[-1]
        options = []
        
        for site in _neighbours(current):
            if site in occupied:
                continue
            
            # Count immediate H-H contacts this placement would make
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            
            # Count free neighbours (for future growth flexibility)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            # Score: prioritize contacts, then free space, then random tie-breaker
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
        
        # Pick the best option
        options.sort(key=lambda x: x[0], reverse=True)
        site = options[0][1]
        occupied[site] = i
        chain.append(site)
    
    return chain

def _compute_contacts(sequence, chain):
    """Compute contact count and build position-to-index map."""
    n = len(chain)
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = pos_to_idx.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts, pos_to_idx

def _delta_energy(sequence, chain, pos_to_idx, i, new_pos):
    """
    Compute delta in contact energy when moving residue i from chain[i] to new_pos.
    Only considers contacts involving residue i and its neighbors.
    Returns delta_energy (new - old).
    """
    curr = chain[i]
    is_h = sequence[i] == 'H'
    delta = 0
    
    # Remove contacts involving curr
    if is_h:
        for q in _neighbours(curr):
            j = pos_to_idx.get(q)
            if j is not None and j != i and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                delta += 1  # removing a contact increases energy by 1
    
    # Add contacts involving new_pos
    if is_h:
        for q in _neighbours(new_pos):
            j = pos_to_idx.get(q)
            if j is not None and j != i and j != i - 1 and j != i + 1 and sequence[j] == 'H':
                delta -= 1  # adding a contact decreases energy by 1
    
    # Also check if moving residue i creates/breaks contacts with i-1 or i+1
    # Actually, contacts between adjacent residues are NOT counted, so we exclude them.
    # But we need to check if new_pos is adjacent to i-1 or i+1 AND both are H
    # No - contacts between chain-adjacent residues don't count.
    # However, if new_pos is adjacent to some other residue j (not i-1, i+1), and both are H,
    # that's a new contact. We already handled that above since we check all neighbors.
    
    # Wait - the above logic has a subtlety. When we remove curr, we remove contacts between
    # curr and its neighbors. But some of those neighbors might be i-1 or i+1, which don't
    # count anyway. Let me be more careful.
    
    # Actually, let me redo this more carefully.
    # Contacts are between H-H pairs at lattice-adjacent positions that are NOT chain-adjacent.
    
    # Old contacts involving residue i (at curr):
    # For each neighbor q of curr, if pos_to_idx[q] = j, j != i, |j-i| > 1, and sequence[j]=='H',
    # then there's a contact.
    
    # New contacts involving residue i (at new_pos):
    # For each neighbor q of new_pos, if pos_to_idx[q] = j, j != i, |j-i| > 1, and sequence[j]=='H',
    # then there's a new contact.
    
    # Note: pos_to_idx still has curr mapped to i. We need to handle that.
    # When checking new_pos neighbors, if q == curr, that's still residue i, so skip.
    
    # Let me recompute properly:
    delta = 0
    
    if is_h:
        # Old contacts: neighbors of curr that are H and not chain-adjacent to i
        for q in _neighbours(curr):
            if q == new_pos:
                continue  # This is the new position, not currently occupied by another residue
            j = pos_to_idx.get(q)
            if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                delta += 1  # removing this contact
        
        # New contacts: neighbors of new_pos that are H and not chain-adjacent to i
        for q in _neighbours(new_pos):
            if q == curr:
                continue  # This is the old position, will be freed
            j = pos_to_idx.get(q)
            if j is not None and j != i and abs(j - i) > 1 and sequence[j] == 'H':
                delta -= 1  # adding this contact
    
    return delta

def _local_search_sa(sequence, chain, rng, max_steps=2000):
    """
    Refine the chain using kink moves with simulated annealing and delta energy.
    """
    n = len(chain)
    if n < 3:
        energy, _ = _compute_contacts(sequence, chain)
        return chain, energy
    
    current_energy, pos_to_idx = _compute_contacts(sequence, chain)
    best_energy = current_energy
    best_chain = list(chain)
    
    # Simulated annealing parameters
    T = 1.0
    cooling = 0.9995
    T_min = 0.01
    
    for step in range(max_steps):
        if _time_remaining() < 2.0:
            break
        
        # Pick a random index to move (1 to n-2)
        i = rng.randint(1, n - 2)
        prev = chain[i-1]
        curr = chain[i]
        next_ = chain[i+1]
        
        # Find valid new positions for curr
        # New pos must be neighbor of prev AND neighbor of next_
        candidates = []
        for p in _neighbours(prev):
            if p == curr:
                continue
            dist = abs(p[0] - next_[0]) + abs(p[1] - next_[1])
            if dist == 1:
                candidates.append(p)
        
        if not candidates:
            continue
            
        new_pos = rng.choice(candidates)
        
        # Check if new_pos is occupied by another part of the chain
        if new_pos in pos_to_idx and pos_to_idx[new_pos] != i:
            continue
            
        # Compute delta energy
        de = _delta_energy(sequence, chain, pos_to_idx, i, new_pos)
        
        # Accept if energy improves, or with probability exp(-de/T) if it worsens
        if de <= 0 or (T > T_min and rng.random() < 2.718281828459045 ** (-de / T)):
            # Apply move
            del pos_to_idx[curr]
            pos_to_idx[new_pos] = i
            chain[i] = new_pos
            current_energy += de
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        else:
            # Revert (no-op since we didn't apply)
            pass
        
        # Cool down
        T *= cooling
    
    return best_chain, best_energy

def fold(sequence, dimension):
    """
    Fold an HP sequence on the square lattice.
    Uses randomized greedy growth followed by local search with simulated annealing.
    """
    _init_time()
    
    n = len(sequence)
    if n == 0:
        return []
    
    # Emergency fallback if no time left
    if _time_remaining() < 1.0:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(12345 + len(sequence))
    
    best_chain = None
    best_energy = 0  # Energy is negative, so we look for lower (more negative) values
    
    # Perform many restarts
    max_iterations = 10000
    iterations = 0
    
    while _time_remaining() > 3.0 and iterations < max_iterations:
        iterations += 1
        
        # 1. Grow
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # 2. Local Search with SA
        # Adjust steps based on remaining time
        time_left = _time_remaining()
        if time_left > 60:
            steps = 3000
        elif time_left > 30:
            steps = 2000
        elif time_left > 10:
            steps = 1000
        else:
            steps = 500
        
        chain, energy = _local_search_sa(sequence, chain, rng, max_steps=steps)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
