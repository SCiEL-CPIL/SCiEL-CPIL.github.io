"""HP lattice protein folding using simulated annealing with efficient energy updates and segment moves."""
# EVOLVE-BLOCK-START
import random
import math
import time


def _neighbours(point):
    """Yield the 6 neighbours of a point in 3D (or 2^(dim-1) in dim dimensions)."""
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _is_adjacent(p, q):
    """Check if two points are adjacent (Manhattan distance 1)."""
    return sum(abs(a - b) for a, b in zip(p, q)) == 1


def _count_contacts(sequence, chain, where):
    """Count H-H non-consecutive contacts."""
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return contacts


def _energy_from_contacts(contacts):
    return -contacts


def _initial_chain(sequence, dimension, rng):
    """Generate a random self-avoiding walk."""
    n = len(sequence)
    if n == 1:
        return [(0,) * dimension]
    
    for attempt in range(100):
        chain = [(0,) * dimension]
        occupied = {chain[0]}
        success = True
        for i in range(1, n):
            neighbors = list(_neighbours(chain[-1]))
            rng.shuffle(neighbors)
            placed = False
            for site in neighbors:
                if site not in occupied:
                    chain.append(site)
                    occupied.add(site)
                    placed = True
                    break
            if not placed:
                success = False
                break
        if success and len(chain) == n:
            return chain
    
    # Fallback: straight line
    return [(i,) + (0,) * (dimension - 1) for i in range(n)]


def _compute_contacts_involving(i, sequence, chain, where):
    """Compute number of H-H contacts involving residue i (non-consecutive)."""
    if sequence[i] != 'H':
        return 0
    count = 0
    p = chain[i]
    for q in _neighbours(p):
        j = where.get(q)
        if j is not None and j != i:
            # Non-consecutive: |j - i| > 1
            if abs(j - i) > 1 and sequence[j] == 'H':
                count += 1
    return count


def _simulated_annealing(sequence, dimension, rng, time_limit=5.0):
    """
    Simulated annealing with local moves including segment moves.
    Returns the best chain found.
    """
    n = len(sequence)
    if n <= 2:
        return _initial_chain(sequence, dimension, rng)
    
    chain = _initial_chain(sequence, dimension, rng)
    where = {p: i for i, p in enumerate(chain)}
    
    contacts = _count_contacts(sequence, chain, where)
    current_energy = _energy_from_contacts(contacts)
    
    best_chain = list(chain)
    best_energy = current_energy
    
    start_time = time.time()
    max_steps = 100000
    start_temp = 8.0
    end_temp = 0.005
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
        
        progress = step / max_steps
        temp = start_temp * math.exp(-8.0 * progress)
        if temp < end_temp:
            temp = end_temp
        
        move_type = rng.random()
        
        if move_type < 0.6:
            # Single-residue move: pick interior residue, move to adjacent empty site
            if n <= 3:
                continue
            i = rng.randint(1, n - 2)
            old_pos = chain[i]
            neighbors = list(_neighbours(old_pos))
            rng.shuffle(neighbors)
            
            moved = False
            for new_pos in neighbors:
                # Must be empty or occupied by i itself
                if new_pos in where and where[new_pos] != i:
                    continue
                # Must be adjacent to both i-1 and i+1
                if not _is_adjacent(new_pos, chain[i - 1]):
                    continue
                if not _is_adjacent(new_pos, chain[i + 1]):
                    continue
                
                # Compute energy delta before modifying state
                old_contacts = _compute_contacts_involving(i, sequence, chain, where)
                
                # Temporarily update to compute new contacts
                del where[old_pos]
                where[new_pos] = i
                chain[i] = new_pos
                
                new_contacts = _compute_contacts_involving(i, sequence, chain, where)
                
                delta_contacts = new_contacts - old_contacts
                delta_energy = _energy_from_contacts(delta_contacts)
                
                if delta_energy <= 0 or rng.random() < math.exp(-delta_energy / temp):
                    current_energy += delta_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                    moved = True
                    break
                else:
                    # Revert
                    del where[new_pos]
                    where[old_pos] = i
                    chain[i] = old_pos
            
            if not moved:
                continue
        
        elif move_type < 0.85:
            # Segment move: pick a segment of length 2-6 and try to reposition it
            if n <= 6:
                continue
            
            seg_len = rng.randint(2, min(6, n - 2))
            i_start = rng.randint(1, n - seg_len - 1)
            i_end = i_start + seg_len - 1
            
            # Save segment
            seg_positions = [chain[i_start + k] for k in range(seg_len)]
            seg_seq = sequence[i_start:i_end + 1]
            
            # Remove segment from where
            for k in range(seg_len):
                del where[chain[i_start + k]]
            
            # Try to reinsert segment at a new location
            # The segment must be adjacent to chain[i_start-1] and chain[i_end+1]
            anchor_left = chain[i_start - 1]
            anchor_right = chain[i_end + 1]
            
            # Generate possible new positions for the segment
            # Start from anchor_left, do a random walk of seg_len steps
            found = False
            for attempt in range(20):
                # Try to place segment starting adjacent to anchor_left
                candidates = [q for q in _neighbours(anchor_left) 
                             if q not in where and q != anchor_left]
                if not candidates:
                    break
                
                new_seg = [rng.choice(candidates)]
                occupied_temp = set(new_seg)
                success = True
                for k in range(1, seg_len):
                    next_candidates = [q for q in _neighbours(new_seg[-1]) 
                                      if q not in where and q not in occupied_temp]
                    if not next_candidates:
                        success = False
                        break
                    new_seg.append(rng.choice(next_candidates))
                    occupied_temp.add(new_seg[-1])
                
                if not success:
                    continue
                
                # Check if last position is adjacent to anchor_right
                if not _is_adjacent(new_seg[-1], anchor_right):
                    continue
                
                # Check self-avoidance with rest of chain
                valid = True
                for pos in new_seg:
                    if pos in where:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Compute energy delta
                # Old contacts: contacts involving residues in [i_start, i_end]
                old_contacts = 0
                for k in range(seg_len):
                    idx = i_start + k
                    if sequence[idx] == 'H':
                        p = seg_positions[k]
                        for q in _neighbours(p):
                            j = where.get(q)
                            if j is not None and abs(j - idx) > 1 and sequence[j] == 'H':
                                old_contacts += 1
                        # Also count contacts between segment residues
                        for m in range(k + 2, seg_len):
                            if _is_adjacent(seg_positions[k], seg_positions[m]) and sequence[i_start + m] == 'H':
                                old_contacts += 1
                
                # New contacts
                new_contacts = 0
                for k in range(seg_len):
                    idx = i_start + k
                    if sequence[idx] == 'H':
                        p = new_seg[k]
                        for q in _neighbours(p):
                            j = where.get(q)
                            if j is not None and abs(j - idx) > 1 and sequence[j] == 'H':
                                new_contacts += 1
                        for m in range(k + 2, seg_len):
                            if _is_adjacent(new_seg[k], new_seg[m]) and sequence[i_start + m] == 'H':
                                new_contacts += 1
                
                delta_contacts = new_contacts - old_contacts
                delta_energy = _energy_from_contacts(delta_contacts)
                
                if delta_energy <= 0 or rng.random() < math.exp(-delta_energy / temp):
                    # Accept move
                    for k in range(seg_len):
                        chain[i_start + k] = new_seg[k]
                        where[new_seg[k]] = i_start + k
                    current_energy += delta_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(chain)
                    found = True
                    break
                else:
                    # Revert: restore original segment
                    for k in range(seg_len):
                        where[seg_positions[k]] = i_start + k
                    break
            
            if not found:
                # Restore original segment
                for k in range(seg_len):
                    where[seg_positions[k]] = i_start + k
        
        else:
            # Random restart
            if rng.random() < 0.05:
                chain = _initial_chain(sequence, dimension, rng)
                where = {p: i for i, p in enumerate(chain)}
                contacts = _count_contacts(sequence, chain, where)
                current_energy = _energy_from_contacts(contacts)
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
    
    return best_chain


def fold(sequence, dimension):
    """
    Find the best conformation using simulated annealing.
    Returns a list of len(sequence) integer lattice points (lists of length dimension).
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    rng = random.Random(42 + len(sequence))
    
    # Time budget: 240s total for 26 sequences, so ~9s per sequence
    # Use 8s per sequence to be safe
    best_chain = _simulated_annealing(sequence, dimension, rng, time_limit=8.0)
    
    # Verify self-avoidance
    seen = set()
    valid = True
    for p in best_chain:
        if p in seen:
            valid = False
            break
        seen.add(p)
    
    # Verify unit steps
    if valid:
        for i in range(1, n):
            diff = sum(abs(a - b) for a, b in zip(best_chain[i], best_chain[i - 1]))
            if diff != 1:
                valid = False
                break
    
    if not valid:
        best_chain = [(i,) + (0,) * (dimension - 1) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
