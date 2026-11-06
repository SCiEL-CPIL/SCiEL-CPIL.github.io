"""HP lattice protein folding on the square lattice - contact-graph guided approach."""
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 195.0  # Safety margin for 210s kill


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _energy(sequence, chain):
    """Compute the HP contact energy of a conformation."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _energy_delta(sequence, chain, old_positions, new_positions):
    """Compute energy change when replacing old_positions with new_positions.
    old_positions and new_positions are lists of (index, position) pairs.
    Returns the change in energy (negative is improvement)."""
    where = {p: i for i, p in enumerate(chain)}
    
    # Remove old positions
    for idx, pos in old_positions:
        where.pop(pos, None)
    
    # Add new positions
    for idx, pos in new_positions:
        where[pos] = idx
    
    # Compute contacts involving changed positions
    # We need to count H-H contacts that are new minus those that are lost
    
    # Build set of changed indices
    changed_indices = set()
    for idx, _ in old_positions:
        changed_indices.add(idx)
    for idx, _ in new_positions:
        changed_indices.add(idx)
    
    # Count contacts before (using original where map would be complex, so let's recompute)
    # Actually, let's just recompute energy for simplicity if the change is small
    # For now, return None to indicate we should recompute
    return None


def _build_contact_graph(sequence):
    """Build a contact graph of potential H-H contacts based on sequence distance.
    Returns a list of (i, j) pairs where i < j, both are H, and they could potentially
    be in contact (not adjacent in chain)."""
    h_positions = [i for i, c in enumerate(sequence) if c == "H"]
    contacts = []
    for a in range(len(h_positions)):
        for b in range(a + 1, len(h_positions)):
            i, j = h_positions[a], h_positions[b]
            if j - i > 1:  # Not adjacent in chain
                contacts.append((i, j))
    return contacts


def _contact_compatibility(contacts, chain):
    """Check which contacts in the list are actually realized in the given chain."""
    where = {p: i for i, p in enumerate(chain)}
    realized = set()
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    realized.add((i, j))
    return realized


def _grow_with_contact_targets(sequence, dimension, rng, target_contacts):
    """Grow chain with bias toward satisfying target contacts.
    target_contacts: list of (i, j) pairs we'd like to realize."""
    n = len(sequence)
    
    # Create a target map: for each position i, which positions j do we want it to be near?
    target_map = {}
    for i, j in target_contacts:
        target_map.setdefault(i, []).append(j)
        target_map.setdefault(j, []).append(i)
    
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            
            gain = 0
            if sequence[i] == "H":
                # Check if placing here creates a contact with any target
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 2  # Strong preference for actual contacts
                        # Bonus if this is a target contact
                        if (j, i) in target_contacts or (i, j) in target_contacts:
                            gain += 3
            
            # Free space heuristic
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            # Small random perturbation for diversity
            noise = rng.random() * 0.5
            
            options.append((gain + 0.1 * free + noise, site))
        
        if not options:
            return None
        
        # Sort by score descending, pick from top few with some randomness
        options.sort(key=lambda x: -x[0])
        top_k = min(3, len(options))
        if rng.random() < 0.7:
            site = options[0][1]
        else:
            site = options[rng.randint(0, top_k - 1)][1]
        
        occupied[site] = i
        chain.append(site)
    
    return chain


def _pivot_move(chain, rng):
    """Perform a single pivot move."""
    if len(chain) < 4:
        return None
    
    pivot_idx = rng.randint(1, len(chain) - 2)
    pivot = chain[pivot_idx]
    segment = chain[pivot_idx:]
    
    for angle in [90, -90, 180]:
        rotated = []
        for p in segment:
            dx, dy = p[0] - pivot[0], p[1] - pivot[1]
            if angle == 90:
                ndx, ndy = -dy, dx
            elif angle == -90:
                ndx, ndy = dy, -dx
            else:
                ndx, ndy = -dx, -dy
            rotated.append((pivot[0] + ndx, pivot[1] + ndy))
        
        new_chain = chain[:pivot_idx] + rotated
        new_occupied = set(new_chain)
        if len(new_occupied) == len(new_chain):
            valid = True
            for i in range(1, len(new_chain)):
                dx = abs(new_chain[i][0] - new_chain[i-1][0])
                dy = abs(new_chain[i][1] - new_chain[i-1][1])
                if dx + dy != 1:
                    valid = False
                    break
            if valid:
                return new_chain
    
    return None


def _crankshaft_move(chain, rng):
    """Perform a crankshaft move: rotate a 4-residue segment."""
    if len(chain) < 6:
        return None
    
    # Choose start index for the 4-residue segment
    start = rng.randint(1, len(chain) - 5)
    segment = chain[start:start+4]
    pivot = chain[start]
    
    # Try 180-degree rotation of the 4-residue segment
    rotated = []
    for p in segment:
        dx, dy = p[0] - pivot[0], p[1] - pivot[1]
        rotated.append((pivot[0] - dx, pivot[1] - dy))
    
    new_chain = chain[:start] + rotated + chain[start+4:]
    new_occupied = set(new_chain)
    if len(new_occupied) == len(new_chain):
        valid = True
        for i in range(1, len(new_chain)):
            dx = abs(new_chain[i][0] - new_chain[i-1][0])
            dy = abs(new_chain[i][1] - new_chain[i-1][1])
            if dx + dy != 1:
                valid = False
                break
        if valid:
            return new_chain
    
    return None


def _slither_move(chain, rng):
    """Slither move: move the end of the chain to an adjacent empty site."""
    if len(chain) < 3:
        return None
    
    end = chain[-1]
    for site in _neighbours(end):
        if site not in set(chain[:-1]):
            new_chain = chain[:-1] + [site]
            return new_chain
    
    return None


def _local_search(sequence, chain, max_iterations=2000, time_limit=5.0):
    """Simulated annealing with multiple move types."""
    if chain is None:
        return None
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    temperature = 3.0
    cooling = 0.998
    
    rng = random.Random(hash(tuple(current_chain)) % (2**32))
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 10:
            break
        
        # Choose move type
        move_type = rng.random()
        new_chain = None
        
        if move_type < 0.5:
            new_chain = _pivot_move(current_chain, rng)
        elif move_type < 0.75:
            new_chain = _crankshaft_move(current_chain, rng)
        else:
            new_chain = _slither_move(current_chain, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < math.exp(-delta / max(temperature, 0.1)):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        temperature *= cooling
    
    return best_chain


def _polish(sequence, chain, max_iterations=500):
    """Deterministic polish: only accept strictly improving moves."""
    if chain is None:
        return None
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    rng = random.Random(42)
    
    for iteration in range(max_iterations):
        if _time_left() < 10:
            break
        
        improved = False
        for _ in range(20):
            move_type = rng.random()
            new_chain = None
            
            if move_type < 0.6:
                new_chain = _pivot_move(current_chain, rng)
            else:
                new_chain = _crankshaft_move(current_chain, rng)
            
            if new_chain is None:
                continue
            
            new_energy = _energy(sequence, new_chain)
            if new_energy < current_energy:
                current_chain = new_chain
                current_energy = new_energy
                improved = True
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
        
        if not improved:
            break
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using contact-graph guided growth + local search."""
    seq_len = len(sequence)
    
    # Build contact graph and select target contacts
    all_contacts = _build_contact_graph(sequence)
    
    # Select a core set of compatible contacts using greedy heuristic
    # We'll use a simple greedy: sort contacts by some priority and pick non-conflicting ones
    # For simplicity, we'll just use all contacts as soft targets
    
    # Adjust strategy based on sequence length
    if seq_len <= 30:
        num_grow = 400
        local_search_iters = 3000
        local_search_time = 8.0
        polish_iters = 500
    elif seq_len <= 50:
        num_grow = 300
        local_search_iters = 2000
        local_search_time = 6.0
        polish_iters = 400
    elif seq_len <= 70:
        num_grow = 200
        local_search_iters = 1500
        local_search_time = 4.5
        polish_iters = 300
    else:
        num_grow = 150
        local_search_iters = 1000
        local_search_time = 3.5
        polish_iters = 200
    
    rng = random.Random(12345 + seq_len)
    best_chain = None
    best_energy = 0
    
    # First pass: greedy growth with contact targets
    for i in range(num_grow):
        if _time_left() < 20:
            break
        
        # Occasionally use a subset of contacts as targets
        if i % 3 == 0 and all_contacts:
            # Use all contacts
            targets = all_contacts
        elif i % 3 == 1 and all_contacts:
            # Use half the contacts
            targets = all_contacts[:len(all_contacts)//2]
        else:
            # No specific targets
            targets = []
        
        chain = _grow_with_contact_targets(sequence, dimension, rng, targets)
        if chain is None:
            continue
        
        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = chain
    
    # Second pass: local search on best chains
    if best_chain is not None and _time_left() > 30:
        # Try local search on the best chain
        improved = _local_search(sequence, best_chain, max_iterations=local_search_iters, 
                                  time_limit=local_search_time)
        if improved is not None:
            improved_energy = _energy(sequence, improved)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved
        
        # Polish
        polished = _polish(sequence, best_chain, max_iterations=polish_iters)
        if polished is not None:
            polished_energy = _energy(sequence, polished)
            if polished_energy < best_energy:
                best_energy = polished_energy
                best_chain = polished
        
        # Try local search on a few other good chains
        for i in range(min(20, num_grow // 10)):
            if _time_left() < 25:
                break
            
            chain = _grow_with_contact_targets(sequence, dimension, rng, all_contacts[:10])
            if chain is None:
                continue
            
            value = _energy(sequence, chain)
            if value < best_energy + 3:
                improved = _local_search(sequence, chain, max_iterations=local_search_iters // 2,
                                          time_limit=local_search_time / 2)
                if improved is not None:
                    improved_energy = _energy(sequence, improved)
                    if improved_energy < best_energy:
                        best_energy = improved_energy
                        best_chain = improved
                
                polished = _polish(sequence, improved, max_iterations=polish_iters // 2)
                if polished is not None:
                    polished_energy = _energy(sequence, polished)
                    if polished_energy < best_energy:
                        best_energy = polished_energy
                        best_chain = polished
    
    # Final fallback: straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(seq_len)]
    
    return [list(p) for p in best_chain]
