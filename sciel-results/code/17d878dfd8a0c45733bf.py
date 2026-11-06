"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 170.0  # Safety margin for 210s kill (evaluator says 180s per call, but 210s wall-clock kill)


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


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


def _contact_map(sequence):
    """Pre-identify high-priority H-H pairs based on sequence position."""
    n = len(sequence)
    h_positions = [i for i in range(n) if sequence[i] == 'H']
    # Score pairs by how "close" they are in sequence - closer pairs are more likely to be contacts
    pairs = []
    for i, pos1 in enumerate(h_positions):
        for pos2 in h_positions[i+1:]:
            dist = pos2 - pos1
            # Pairs that are 3-10 apart in sequence are good candidates for contacts
            if 3 <= dist <= 10:
                priority = 1.0 / (dist - 2)  # Higher priority for closer pairs
                pairs.append((priority, pos1, pos2))
    # Sort by priority
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
    
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                # Count new H-H contacts
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
                # Bonus for satisfying high-priority pairs
                for _, j, k in priority_pairs:
                    if k == i and sequence[j] == 'H':
                        pos_j = chain[j]
                        for q in _neighbours(pos_j):
                            if q == site:
                                gain += 2.0  # Strong bonus for satisfying priority pair
                                break
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = bias * gain + 0.1 * free + rng.random()
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
    pivot = rng.randint(2, n - 2)
    head = chain[:pivot]
    tail = chain[pivot:]
    
    px, py = chain[pivot]
    
    # Try rotation by 90 degrees clockwise: (dx,dy) -> (dy,-dx)
    new_tail = [chain[pivot]]
    for i in range(1, len(tail)):
        dx = tail[i][0] - tail[i-1][0]
        dy = tail[i][1] - tail[i-1][1]
        new_dx = dy
        new_dy = -dx
        new_tail.append((new_tail[-1][0] + new_dx, new_tail[-1][1] + new_dy))
    
    new_chain = head + new_tail
    
    if len(set(new_chain)) != n:
        return None
    
    dx = new_tail[0][0] - head[-1][0]
    dy = new_tail[0][1] - head[-1][1]
    if abs(dx) + abs(dy) != 1:
        return None
    
    return new_chain


def _segment_move(chain, rng):
    """Crankshaft move: shift a short segment by one unit perpendicular to its direction."""
    n = len(chain)
    if n < 5:
        return None
    
    # Pick a segment of length 2-4
    seg_len = rng.randint(2, min(4, n - 4))
    start = rng.randint(1, n - seg_len - 1)
    end = start + seg_len
    
    segment = chain[start:end]
    head = chain[:start]
    tail = chain[end:]
    
    # Determine the direction of the segment
    dx = segment[-1][0] - segment[0][0]
    dy = segment[-1][1] - segment[0][1]
    
    # Perpendicular directions
    perp_dirs = [(-dy, dx), (dy, -dx)]
    if dx == 0:
        perp_dirs = [(1, 0), (-1, 0)]
    elif dy == 0:
        perp_dirs = [(0, 1), (0, -1)]
    
    for pdx, pdy in perp_dirs:
        new_segment = [(x + pdx, y + pdy) for x, y in segment]
        new_chain = head + new_segment + tail
        
        # Check self-avoidance
        if len(set(new_chain)) != n:
            continue
        
        # Check unit steps at boundaries
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
        
        # Choose move type
        if rng.random() < 0.5:
            new_chain = _pivot_move(current, rng)
        else:
            new_chain = _segment_move(current, rng)
        
        if new_chain is None:
            continue
        
        new_energy = _energy(sequence, new_chain)
        new_contacts = _contact_satisfaction(sequence, new_chain, priority_pairs)
        
        # Accept if better energy, or same energy with more contacts
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
            # Accept worse moves with decreasing probability
            delta = current_energy - new_energy
            accept_prob = math.exp(-delta / max(1, 100.0 / (step + 1)))
            if rng.random() < accept_prob:
                current = new_chain
                current_energy = new_energy
    
    return best, best_energy


def fold(sequence, dimension):
    """Best conformation using constrained growth + local search with time budget."""
    n = len(sequence)
    rng = random.Random(12345 + n)
    
    # Pre-compute priority contact pairs
    priority_pairs = _contact_map(sequence)
    
    best = None
    best_energy = 1
    
    # Phase 1: Generate initial solutions via constrained greedy growth
    num_growths = 80
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
    
    # If no valid fold found, use straight line
    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_energy = _energy(sequence, best)
    
    # Phase 2: Local search from the best solution
    if _time_left() > 15.0:
        improved, improved_energy = _local_search(sequence, best, rng, max_steps=3000, priority_pairs=priority_pairs)
        if improved_energy < best_energy:
            best = improved
            best_energy = improved_energy
    
    # Phase 3: Try more growths with different biases if time permits
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
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
