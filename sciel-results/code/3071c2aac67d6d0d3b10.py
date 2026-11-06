"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys

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


def _neighbours_tuple(point):
    """Return the 4 neighbours of a 2D lattice point as a tuple."""
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


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


def _build_contact_map(sequence, chain):
    """Build a set of H-H contact pairs (i, j) with i < j."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts.add((i, j))
    return contacts


def _energy_from_contacts(contacts):
    """Compute energy from a set of contact pairs."""
    return -len(contacts)


def _grow(sequence, dimension, rng, priority_contacts=None):
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
            # Add small bonus if this placement creates a contact with a high-priority partner
            bonus = 0
            if priority_contacts and sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        if (min(i, j), max(i, j)) in priority_contacts:
                            bonus += 0.5
            options.append((gain + 0.1 * free + bonus + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot_move(chain, rng):
    """Perform a single pivot move: rotate a subchain by 90, 180, or 270 degrees around a pivot point."""
    if len(chain) < 4:
        return None
    
    # Choose a random pivot index (not the first or last)
    pivot_idx = rng.randint(1, len(chain) - 2)
    pivot = chain[pivot_idx]
    
    # Get the segment to rotate (from pivot_idx to end)
    segment = chain[pivot_idx:]
    
    # Try different rotations
    for angle in [90, 180, 270, -90]:
        # Rotate segment around pivot
        rotated = []
        for p in segment:
            dx, dy = p[0] - pivot[0], p[1] - pivot[1]
            if angle == 90:
                ndx, ndy = -dy, dx
            elif angle == -90:
                ndx, ndy = dy, -dx
            elif angle == 180:
                ndx, ndy = -dx, -dy
            else:
                ndx, ndy = dx, dy
            rotated.append((pivot[0] + ndx, pivot[1] + ndy))
        
        # Check if the rotated segment is self-avoiding and doesn't collide with the fixed part
        new_chain = chain[:pivot_idx] + rotated
        new_occupied = set(new_chain)
        if len(new_occupied) == len(new_chain):
            # Check unit steps
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
    """Perform a slither move: move the last residue to an adjacent free site."""
    if len(chain) < 3:
        return None
    
    last = chain[-1]
    prev = chain[-2]
    
    # Try moving last to each of the 4 neighbours of prev (excluding current position)
    for site in _neighbours(prev):
        if site == last:
            continue
        if site in set(chain[:-1]):
            continue
        # Check unit step from new last to nothing (it's the end)
        new_chain = chain[:-1] + [site]
        # Verify self-avoidance
        if len(set(new_chain)) == len(new_chain):
            return new_chain
    
    return None


def _crankshaft_move(chain, rng):
    """Perform a crankshaft move: rotate an internal segment of length 3 or 4."""
    if len(chain) < 6:
        return None
    
    # Choose a random segment start (at least 2 from start and end)
    seg_start = rng.randint(2, len(chain) - 4)
    seg_len = rng.choice([3, 4])
    
    if seg_start + seg_len > len(chain):
        return None
    
    segment = chain[seg_start:seg_start + seg_len]
    pivot = chain[seg_start]
    
    # Try rotations of the segment
    for angle in [90, 180, 270, -90]:
        rotated = []
        for p in segment:
            dx, dy = p[0] - pivot[0], p[1] - pivot[1]
            if angle == 90:
                ndx, ndy = -dy, dx
            elif angle == -90:
                ndx, ndy = dy, -dx
            elif angle == 180:
                ndx, ndy = -dx, -dy
            else:
                ndx, ndy = dx, dy
            rotated.append((pivot[0] + ndx, pivot[1] + ndy))
        
        new_chain = chain[:seg_start] + rotated + chain[seg_start + seg_len:]
        
        # Check self-avoidance
        new_occupied = set(new_chain)
        if len(new_occupied) != len(new_chain):
            continue
        
        # Check unit steps at boundaries
        valid = True
        # Check connection before segment
        if seg_start > 0:
            dx = abs(new_chain[seg_start][0] - new_chain[seg_start - 1][0])
            dy = abs(new_chain[seg_start][1] - new_chain[seg_start - 1][1])
            if dx + dy != 1:
                valid = False
        # Check connection after segment
        if seg_start + seg_len < len(new_chain):
            dx = abs(new_chain[seg_start + seg_len][0] - new_chain[seg_start + seg_len - 1][0])
            dy = abs(new_chain[seg_start + seg_len][1] - new_chain[seg_start + seg_len - 1][1])
            if dx + dy != 1:
                valid = False
        
        if valid:
            return new_chain
    
    return None


def _deterministic_polish(sequence, chain, max_rounds=500, time_limit=3.0):
    """Deterministic hill-climbing polish: try all pivot/crankshaft/slither moves, accept improvements."""
    if chain is None:
        return None
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    rng = random.Random(42)
    
    for _ in range(max_rounds):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 5:
            break
        
        improved = False
        
        # Try pivot moves
        for pivot_idx in range(1, len(current_chain) - 1):
            pivot = current_chain[pivot_idx]
            segment = current_chain[pivot_idx:]
            
            for angle in [90, 180, 270, -90]:
                rotated = []
                for p in segment:
                    dx, dy = p[0] - pivot[0], p[1] - pivot[1]
                    if angle == 90:
                        ndx, ndy = -dy, dx
                    elif angle == -90:
                        ndx, ndy = dy, -dx
                    elif angle == 180:
                        ndx, ndy = -dx, -dy
                    else:
                        ndx, ndy = dx, dy
                    rotated.append((pivot[0] + ndx, pivot[1] + ndy))
                
                new_chain = current_chain[:pivot_idx] + rotated
                if len(set(new_chain)) == len(new_chain):
                    # Check unit steps
                    valid = True
                    for i in range(1, len(new_chain)):
                        dx = abs(new_chain[i][0] - new_chain[i-1][0])
                        dy = abs(new_chain[i][1] - new_chain[i-1][1])
                        if dx + dy != 1:
                            valid = False
                            break
                    if valid:
                        new_energy = _energy(sequence, new_chain)
                        if new_energy < current_energy:
                            current_chain = new_chain
                            current_energy = new_energy
                            if current_energy < best_energy:
                                best_energy = current_energy
                                best_chain = list(current_chain)
                            improved = True
                            break
            if improved:
                break
        
        if improved:
            continue
        
        # Try crankshaft moves
        for seg_start in range(2, len(current_chain) - 3):
            for seg_len in [3, 4]:
                if seg_start + seg_len > len(current_chain):
                    continue
                
                segment = current_chain[seg_start:seg_start + seg_len]
                pivot = current_chain[seg_start]
                
                for angle in [90, 180, 270, -90]:
                    rotated = []
                    for p in segment:
                        dx, dy = p[0] - pivot[0], p[1] - pivot[1]
                        if angle == 90:
                            ndx, ndy = -dy, dx
                        elif angle == -90:
                            ndx, ndy = dy, -dx
                        elif angle == 180:
                            ndx, ndy = -dx, -dy
                        else:
                            ndx, ndy = dx, dy
                        rotated.append((pivot[0] + ndx, pivot[1] + ndy))
                    
                    new_chain = current_chain[:seg_start] + rotated + current_chain[seg_start + seg_len:]
                    if len(set(new_chain)) != len(new_chain):
                        continue
                    
                    valid = True
                    if seg_start > 0:
                        dx = abs(new_chain[seg_start][0] - new_chain[seg_start - 1][0])
                        dy = abs(new_chain[seg_start][1] - new_chain[seg_start - 1][1])
                        if dx + dy != 1:
                            valid = False
                    if seg_start + seg_len < len(new_chain):
                        dx = abs(new_chain[seg_start + seg_len][0] - new_chain[seg_start + seg_len - 1][0])
                        dy = abs(new_chain[seg_start + seg_len][1] - new_chain[seg_start + seg_len - 1][1])
                        if dx + dy != 1:
                            valid = False
                    
                    if valid:
                        new_energy = _energy(sequence, new_chain)
                        if new_energy < current_energy:
                            current_chain = new_chain
                            current_energy = new_energy
                            if current_energy < best_energy:
                                best_energy = current_energy
                                best_chain = list(current_chain)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try slither move
        slither = _slither_move(current_chain, rng)
        if slither is not None:
            new_energy = _energy(sequence, slither)
            if new_energy < current_energy:
                current_chain = slither
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                improved = True
        
        if not improved:
            break
    
    return best_chain


def _local_search(sequence, chain, max_iterations=1000, time_limit=5.0):
    """Simulated annealing local search to improve a conformation."""
    if chain is None:
        return None
    
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    start_time = time.time()
    temperature = 3.0
    cooling = 0.997
    
    rng = random.Random(hash(tuple(current_chain)) % (2**32))
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 10:
            break
        
        # Choose move type: pivot, crankshaft, or slither
        r = rng.random()
        if r < 0.5:
            new_chain = _pivot_move(current_chain, rng)
        elif r < 0.8:
            new_chain = _crankshaft_move(current_chain, rng)
        else:
            new_chain = _slither_move(current_chain, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0 or rng.random() < pow(2.718281828, -delta / max(temperature, 0.1)):
            current_chain = new_chain
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        
        temperature *= cooling
    
    # Final deterministic polish
    if _time_left() > 15:
        polished = _deterministic_polish(sequence, best_chain, max_rounds=200, time_limit=min(2.0, max(0.5, _time_left() - 10)))
        if polished is not None:
            polished_energy = _energy(sequence, polished)
            if polished_energy < best_energy:
                best_energy = polished_energy
                best_chain = polished
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using multiple strategies: greedy growth + local search."""
    seq_len = len(sequence)
    
    # Adjust strategy based on sequence length
    if seq_len <= 30:
        num_grow = 400
        local_search_iters = 3000
        local_search_time = 10.0
    elif seq_len <= 50:
        num_grow = 300
        local_search_iters = 2000
        local_search_time = 8.0
    elif seq_len <= 70:
        num_grow = 200
        local_search_iters = 1500
        local_search_time = 6.0
    else:
        num_grow = 150
        local_search_iters = 1000
        local_search_time = 4.0
    
    rng = random.Random(12345 + seq_len)
    best_chain = None
    best_energy = 0
    
    # Build priority contacts: H-H pairs that are close in sequence (likely to be contacts)
    priority_contacts = set()
    h_positions = [i for i, c in enumerate(sequence) if c == 'H']
    for idx_a, i in enumerate(h_positions):
        for j in h_positions[idx_a+1:idx_a+10]:
            if j - i > 1:
                priority_contacts.add((i, j))
    
    # First pass: greedy growth, track top few
    top_chains = []
    for i in range(num_grow):
        if _time_left() < 25:
            break
        
        chain = _grow(sequence, dimension, rng, priority_contacts if i % 3 == 0 else None)
        if chain is None:
            continue
        
        value = _energy(sequence, chain)
        if best_chain is None or value < best_energy:
            best_energy = value
            best_chain = chain
            top_chains = [(value, chain)]
        elif len(top_chains) < 5 and value < best_energy + 5:
            top_chains.append((value, chain))
            top_chains.sort(key=lambda x: x[0])
    
    # Second pass: local search on top chains
    if best_chain is not None and _time_left() > 35:
        # Local search on the best chain with most iterations
        improved = _local_search(sequence, best_chain, max_iterations=local_search_iters, 
                                  time_limit=local_search_time)
        if improved is not None:
            improved_energy = _energy(sequence, improved)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved
        
        # Local search on other good chains
        for val, ch in top_chains[1:]:
            if _time_left() < 20:
                break
            if val >= best_energy + 3:
                continue
            
            improved = _local_search(sequence, ch, max_iterations=local_search_iters // 2,
                                      time_limit=local_search_time / 2)
            if improved is not None:
                improved_energy = _energy(sequence, improved)
                if improved_energy < best_energy:
                    best_energy = improved_energy
                    best_chain = improved
    
    # Final fallback: straight line
    if best_chain is None:
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(seq_len)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
