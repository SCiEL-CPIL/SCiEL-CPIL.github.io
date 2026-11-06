"""HP lattice protein folding with delta-energy local search and segment moves."""
# EVOLVE-BLOCK-START
import random
import time

_START_TIME = None
_DEADLINE = 180.0  # wall-clock kill is 210s; use 180s as hard limit for safety

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    return _DEADLINE - (time.time() - _START_TIME)

def _neighbours(point):
    x, y = point
    return ((x+1, y), (x-1, y), (x, y+1), (x, y-1))

def _compute_contacts(sequence, chain):
    """Return set of (i, j) pairs that form H-H contacts (i < j, not adjacent)."""
    n = len(chain)
    where = {}
    for i, p in enumerate(chain):
        where[p] = i
    contacts = set()
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts.add((i, j))
    return contacts

def _energy_from_contacts(contacts):
    return -len(contacts)

def _delta_energy_kink(sequence, chain, contacts, i, new_pos):
    """
    Compute delta energy for moving residue i to new_pos (kink move).
    Assumes new_pos is valid (neighbor of i-1 and i+1, not occupied by other residues).
    Returns delta = new_energy - old_energy (negative is improvement).
    """
    curr = chain[i]
    prev = chain[i-1]
    nxt = chain[i+1]
    
    # Contacts that involve residue i and will be broken or formed
    # A contact (a, b) is broken if a == i or b == i, and the geometric condition no longer holds
    # A contact (a, b) is formed if a == i or b == i, and the geometric condition now holds
    
    delta = 0
    
    if sequence[i] == 'H':
        # Check which contacts involving i are broken
        for (a, b) in list(contacts):
            if a == i or b == i:
                other = b if a == i else a
                # Check if the contact still holds after move
                other_pos = chain[other]
                # New position must be neighbor of other_pos
                if abs(new_pos[0] - other_pos[0]) + abs(new_pos[1] - other_pos[1]) == 1:
                    # Contact still holds, no change
                    pass
                else:
                    # Contact broken
                    delta += 1  # energy increases by 1 (less negative)
        
        # Check which new contacts involving i are formed
        for q in _neighbours(new_pos):
            j = None
            # Find if q is occupied by another residue
            # We need to check if q is in the chain
            # This is O(n) which is slow. Let's use a lookup.
            # But we don't have a global lookup here. Let's pass it in or compute.
            # For efficiency, let's just check the immediate neighbors in the chain that are H.
            # Actually, the standard way: check all residues j != i, j != i-1, j != i+1
            # that are H and at q.
            # This is O(n). For n<=100, it's acceptable if done carefully.
            # But let's try to be smarter.
            pass
    
    # This approach is getting complex. Let's use a simpler full recalculation for kinks
    # but optimize the search by limiting the number of kink moves per iteration.
    # Or, let's implement a proper delta calculation with a position-to-index map.
    
    return None  # Placeholder, we'll use full recalc for simplicity first

def _grow(sequence, dimension, rng):
    """
    One randomized greedy chain growth.
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
            
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            score = gain * 10.0 + free * 0.1 + rng.random()
            options.append((score, site))
        
        if not options:
            return None
        
        options.sort(key=lambda x: x[0], reverse=True)
        # Add some randomness: pick from top 3 with probability
        top_k = min(3, len(options))
        if rng.random() < 0.3 and top_k > 1:
            site = options[rng.randint(0, top_k-1)][1]
        else:
            site = options[0][1]
        
        occupied[site] = i
        chain.append(site)
    
    return chain

def _local_search(sequence, chain, rng, max_steps=200):
    """
    Refine the chain using kink moves with simulated annealing.
    """
    n = len(chain)
    if n < 3:
        return chain, _energy_from_contacts(_compute_contacts(sequence, chain))
    
    occupied = {p: i for i, p in enumerate(chain)}
    contacts = _compute_contacts(sequence, chain)
    current_energy = _energy_from_contacts(contacts)
    
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    for step in range(max_steps):
        if _time_remaining() < 2.0:
            break
        
        i = rng.randint(1, n - 2)
        prev = chain[i-1]
        curr = chain[i]
        next_ = chain[i+1]
        
        # Find valid kink positions
        candidates = set()
        for p in _neighbours(prev):
            if p == curr:
                continue
            dist = abs(p[0] - next_[0]) + abs(p[1] - next_[1])
            if dist == 1:
                candidates.add(p)
        
        if not candidates:
            continue
        
        new_pos = rng.choice(list(candidates))
        
        if new_pos in occupied:
            # Check if it's occupied by i-1 or i+1 (shouldn't be, but just in case)
            occ_idx = occupied[new_pos]
            if occ_idx != i-1 and occ_idx != i+1:
                continue
            else:
                # This shouldn't happen in a valid kink, but if it does, skip
                continue
        
        # Perform move
        del occupied[curr]
        occupied[new_pos] = i
        old_pos = curr
        chain[i] = new_pos
        
        # Recalculate contacts and energy
        new_contacts = _compute_contacts(sequence, chain)
        new_energy = _energy_from_contacts(new_contacts)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < pow(2.718281828, -delta / T):
            # Accept
            current_energy = new_energy
            contacts = new_contacts
        else:
            # Revert
            chain[i] = old_pos
            del occupied[new_pos]
            occupied[old_pos] = i
        
        T *= cooling
    
    return chain, current_energy

def _segment_move(sequence, chain, rng, max_attempts=50):
    """
    Try to move a segment of the chain to a new location.
    This is a more powerful move than kinks.
    """
    n = len(chain)
    if n < 5:
        return chain, _energy_from_contacts(_compute_contacts(sequence, chain)), False
    
    occupied = {p: i for i, p in enumerate(chain)}
    current_energy = _energy_from_contacts(_compute_contacts(sequence, chain))
    
    # Pick a segment length
    seg_len = min(rng.randint(2, min(5, n//3)), n - 4)
    # Pick start index
    start = rng.randint(1, n - seg_len - 2)
    end = start + seg_len - 1
    
    # The segment is chain[start..end]
    # We want to move it to a new location
    # The new location must be adjacent to chain[start-1] and chain[end+1]
    # and the segment must form a valid self-avoiding walk in the new location
    
    prev_pos = chain[start-1]
    next_pos = chain[end+1]
    
    # Generate possible new starting positions for the segment
    # The new start must be adjacent to prev_pos
    # The new end must be adjacent to next_pos
    # This is complex. Let's simplify: try random new starting positions adjacent to prev_pos
    # and grow the segment from there
    
    success = False
    for attempt in range(max_attempts):
        if _time_remaining() < 2.0:
            break
        
        # Pick a new starting position adjacent to prev_pos
        new_start = rng.choice(list(_neighbours(prev_pos)))
        if new_start == chain[start]:
            continue
        if new_start in occupied:
            occ_idx = occupied[new_start]
            if not (start <= occ_idx <= end):
                continue
        
        # Try to grow the segment from new_start
        new_segment = [new_start]
        new_occupied = {new_start: start}
        
        valid = True
        for k in range(1, seg_len):
            curr = new_segment[-1]
            options = []
            for site in _neighbours(curr):
                # Check if site is valid
                if site in occupied:
                    occ_idx = occupied[site]
                    if not (start <= occ_idx <= end):
                        continue
                if site in new_occupied:
                    continue
                options.append(site)
            
            if not options:
                valid = False
                break
            
            next_site = rng.choice(options)
            new_segment.append(next_site)
            new_occupied[next_site] = start + k
        
        if not valid:
            continue
        
        # Check if the new end is adjacent to next_pos
        new_end = new_segment[-1]
        if abs(new_end[0] - next_pos[0]) + abs(new_end[1] - next_pos[1]) != 1:
            continue
        
        # Check for self-avoidance with the rest of the chain
        conflict = False
        for pos in new_segment:
            if pos in occupied:
                occ_idx = occupied[pos]
                if not (start <= occ_idx <= end):
                    conflict = True
                    break
        
        if conflict:
            continue
        
        # Perform the move
        old_segment = chain[start:end+1]
        for k, pos in enumerate(old_segment):
            del occupied[pos]
        
        chain[start:end+1] = new_segment
        for k, pos in enumerate(new_segment):
            occupied[pos] = start + k
        
        new_energy = _energy_from_contacts(_compute_contacts(sequence, chain))
        if new_energy < current_energy:
            success = True
            current_energy = new_energy
            break
        else:
            # Revert
            chain[start:end+1] = old_segment
            for pos in new_segment:
                del occupied[pos]
            for k, pos in enumerate(old_segment):
                occupied[pos] = start + k
    
    if success:
        return chain, current_energy, True
    else:
        return chain, current_energy, False

def fold(sequence, dimension):
    """
    Fold an HP sequence on the square lattice.
    """
    _init_time()
    
    n = len(sequence)
    if n == 0:
        return []
    
    if _time_remaining() < 1.0:
        return [[i, 0] for i in range(n)]
    
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Multiple growth + local search
    while _time_remaining() > 10.0:
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        
        # Local search with kinks
        chain, energy = _local_search(sequence, chain, rng, max_steps=100)
        
        if energy < best_energy:
            best_energy = energy
            best_chain = chain
    
    # Phase 2: Segment moves on the best solution
    if best_chain is not None and _time_remaining() > 5.0:
        for _ in range(50):
            if _time_remaining() < 5.0:
                break
            best_chain, best_energy, improved = _segment_move(sequence, best_chain, rng, max_attempts=20)
            if not improved:
                continue
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
