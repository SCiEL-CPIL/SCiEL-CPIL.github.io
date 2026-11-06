"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
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


def _contact_map(sequence):
    """Pre-identify high-priority H-H pairs based on sequence position."""
    n = len(sequence)
    h_positions = [i for i in range(n) if sequence[i] == 'H']
    pairs = []
    for i, pos1 in enumerate(h_positions):
        for pos2 in h_positions[i+1:]:
            dist = pos2 - pos1
            if 3 <= dist <= 10:
                priority = 1.0 / (dist - 2)
                pairs.append((priority, pos1, pos2))
    pairs.sort(key=lambda x: -x[0])
    return pairs


def _contact_satisfaction(sequence, chain, priority_pairs):
    """Count how many high-priority H-H pairs are satisfied (in contact)."""
    where = {p: i for i, p in enumerate(chain)}
    satisfied = 0
    for _, i, j in priority_pairs:
        if sequence[i] == 'H' and sequence[j] == 'H':
            pi = chain[i]
            for q in _neighbours(pi):
                if where.get(q) == j:
                    satisfied += 1
                    break
    return satisfied


def _grow_constrained(sequence, dimension, rng, bias=1.0, priority_pairs=None):
    """Greedy growth with constraint satisfaction for high-priority H-H pairs."""
    if priority_pairs is None:
        priority_pairs = _contact_map(sequence)
    
    start = (0, 0)
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
                for _, j, k in priority_pairs:
                    if k == i and sequence[j] == 'H':
                        pos_j = chain[j]
                        for q in _neighbours(pos_j):
                            if q == site:
                                gain += 2.0
                                break
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            # Compactness heuristic
            if len(chain) > 10:
                cx = sum(p[0] for p in chain[-10:]) / 10.0
                cy = sum(p[1] for p in chain[-10:]) / 10.0
                dist = abs(site[0] - cx) + abs(site[1] - cy)
            else:
                dist = 0.0
            score = bias * gain - 0.3 * dist + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot_move(chain, rng):
    """Try a pivot move: rotate a subchain. Returns new chain or None if invalid."""
    n = len(chain)
    if n < 4:
        return None
    pivot_idx = rng.randint(1, n - 2)
    pivot = chain[pivot_idx]
    segment_start = pivot_idx + 1
    if segment_start >= n:
        return None
    segment = chain[segment_start:]
    fixed_set = set(chain[:pivot_idx + 1])
    
    for angle in (90, -90, 180):
        rotated = []
        valid = True
        for p in segment:
            dx, dy = p[0] - pivot[0], p[1] - pivot[1]
            if angle == 90:
                ndx, ndy = -dy, dx
            elif angle == -90:
                ndx, ndy = dy, -dx
            else:
                ndx, ndy = -dx, -dy
            new_p = (pivot[0] + ndx, pivot[1] + ndy)
            if new_p in fixed_set:
                valid = False
                break
            rotated.append(new_p)
        
        if not valid:
            continue
        
        new_chain = chain[:segment_start] + rotated
        # Check connection at pivot
        if abs(rotated[0][0] - pivot[0]) + abs(rotated[0][1] - pivot[1]) != 1:
            continue
        # Check internal self-avoidance of rotated segment (should be fine, but just in case)
        if len(set(new_chain)) != n:
            continue
        return new_chain
    return None


def _segment_move(chain, rng):
    """Crankshaft move: shift a short segment by one unit perpendicular to its direction."""
    n = len(chain)
    if n < 5:
        return None
    
    seg_len = rng.randint(2, min(4, n - 4))
    start = rng.randint(1, n - seg_len - 1)
    end = start + seg_len
    
    segment = chain[start:end]
    head = chain[:start]
    tail = chain[end:]
    
    dx = segment[-1][0] - segment[0][0]
    dy = segment[-1][1] - segment[0][1]
    
    perp_dirs = [(-dy, dx), (dy, -dx)]
    if dx == 0 and dy == 0:
        perp_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    elif dx == 0:
        perp_dirs = [(1, 0), (-1, 0)]
    elif dy == 0:
        perp_dirs = [(0, 1), (0, -1)]
    
    for pdx, pdy in perp_dirs:
        new_segment = [(x + pdx, y + pdy) for x, y in segment]
        new_chain = head + new_segment + tail
        
        if len(set(new_chain)) != n:
            continue
        
        if head:
            dx1 = new_segment[0][0] - head[-1][0]
            dy1 = new_segment[0][1] - head[-1][1]
            if abs(dx1) + abs(dy1) != 1:
                continue
        if tail:
            dx2 = tail[0][0] - new_segment[-1][0]
            dy2 = tail[0][1] - new_segment[-1][1]
            if abs(dx2) + abs(dy2) != 1:
                continue
        
        return new_chain
    return None


def _local_search(sequence, chain, rng, max_steps=500, priority_pairs=None):
    """Local search using pivot and segment moves with constraint awareness."""
    if priority_pairs is None:
        priority_pairs = _contact_map(sequence)
    
    current = chain[:]
    current_energy = _energy(sequence, current)
    best = current[:]
    best_energy = current_energy
    best_contacts = _contact_satisfaction(sequence, current, priority_pairs)
    
    for step in range(max_steps):
        if _time_left() < 5.0:
            break
        
        if rng.random() < 0.5:
            new_chain = _pivot_move(current, rng)
        else:
            new_chain = _segment_move(current, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        new_contacts = _contact_satisfaction(sequence, new_chain, priority_pairs)
        
        if new_energy < current_energy:
            current = new_chain
            current_energy = new_energy
            if current_energy < best_energy or (current_energy == best_energy and new_contacts > best_contacts):
                best = current[:]
                best_energy = current_energy
                best_contacts = new_contacts
        elif new_energy == current_energy and new_contacts > best_contacts:
            current = new_chain
            current_energy = new_energy
            best = current[:]
            best_energy = current_energy
            best_contacts = new_contacts
        else:
            delta = current_energy - new_energy
            accept_prob = math.exp(-delta / max(1, 100.0 / (step + 1)))
            if rng.random() < accept_prob:
                current = new_chain
                current_energy = new_energy
    
    return best, best_energy


def _polish(sequence, chain, max_iterations=1000, time_limit=2.0, rng=None):
    """Deterministic hill-climbing polish: accept only improving moves."""
    if chain is None:
        return None
    if rng is None:
        rng = random.Random(42)

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 15:
            break

        n = len(current_chain)
        best_move = None
        best_delta = 0

        # Try pivot moves
        for pivot_idx in range(1, n - 1):
            pivot = current_chain[pivot_idx]
            segment_start = pivot_idx + 1
            if segment_start >= n:
                continue

            segment = current_chain[segment_start:]
            fixed_set = set(current_chain[:pivot_idx + 1])

            for angle in (90, -90, 180):
                rotated = []
                valid = True
                for p in segment:
                    dx, dy = p[0] - pivot[0], p[1] - pivot[1]
                    if angle == 90:
                        ndx, ndy = -dy, dx
                    elif angle == -90:
                        ndx, ndy = dy, -dx
                    else:
                        ndx, ndy = -dx, -dy
                    new_p = (pivot[0] + ndx, pivot[1] + ndy)
                    if new_p in fixed_set:
                        valid = False
                        break
                    rotated.append(new_p)

                if not valid:
                    continue

                new_chain = current_chain[:segment_start] + rotated
                if len(set(new_chain)) != n:
                    continue
                # Check pivot connection
                if abs(rotated[0][0] - pivot[0]) + abs(rotated[0][1] - pivot[1]) != 1:
                    continue
                
                new_energy = _energy(sequence, new_chain)
                delta = new_energy - current_energy
                if delta < best_delta:
                    best_delta = delta
                    best_move = new_chain

        # Try slither move
        last = current_chain[-1]
        prev = current_chain[-2]
        occupied = set(current_chain[:-1])
        for site in _neighbours(prev):
            if site in occupied:
                continue
            new_chain = current_chain[:-1] + [site]
            new_energy = _energy(sequence, new_chain)
            delta = new_energy - current_energy
            if delta < best_delta:
                best_delta = delta
                best_move = new_chain

        if best_move is None:
            break

        current_chain = best_move
        current_energy += best_delta
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = list(current_chain)

    return best_chain


def fold(sequence, dimension):
    """Best conformation using constrained growth + local search with time budget."""
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    priority_pairs = _contact_map(sequence)
    
    best = None
    best_energy = 1
    
    # Phase 1: Generate initial solutions via constrained greedy growth
    num_growths = 100
    for i in range(num_growths):
        if _time_left() < 10.0:
            break
        
        bias = 0.5 + 0.5 * (i % 10) / 10.0
        chain = _grow_constrained(sequence, dimension, rng, bias=bias, priority_pairs=priority_pairs)
        if chain is None:
            continue
        
        energy = _energy(sequence, chain)
        if energy < best_energy:
            best = chain[:]
            best_energy = energy
    
    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best)
    
    # Phase 2: Local search from the best solution
    if _time_left() > 15.0:
        improved, improved_energy = _local_search(sequence, best, rng, max_steps=3000, priority_pairs=priority_pairs)
        if improved_energy < best_energy:
            best = improved
            best_energy = improved_energy
    
    # Phase 3: Deterministic polish
    if _time_left() > 10.0:
        polished = _polish(sequence, best, max_iterations=1000, time_limit=2.0)
        if polished is not None:
            polished_energy = _energy(sequence, polished)
            if polished_energy < best_energy:
                best = polished
                best_energy = polished_energy

    # Phase 4: Try more growths with different biases if time permits
    if _time_left() > 20.0:
        for i in range(150):
            if _time_left() < 10.0:
                break
            
            bias = 0.3 + 0.7 * rng.random()
            chain = _grow_constrained(sequence, dimension, rng, bias=bias, priority_pairs=priority_pairs)
            if chain is None:
                continue
            
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best = chain[:]
                best_energy = energy
                
                # Quick local search from this new best
                if _time_left() > 15.0:
                    improved, improved_energy = _local_search(sequence, best, rng, max_steps=800, priority_pairs=priority_pairs)
                    if improved_energy < best_energy:
                        best = improved
                        best_energy = improved_energy
                
                # Quick polish
                if _time_left() > 10.0:
                    polished = _polish(sequence, best, max_iterations=500, time_limit=1.0)
                    if polished is not None:
                        polished_energy = _energy(sequence, polished)
                        if polished_energy < best_energy:
                            best = polished
                            best_energy = polished_energy
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
