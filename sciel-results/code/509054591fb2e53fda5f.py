"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 200.0  # Safety margin for 210s kill


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _build_where(chain):
    """Build position->index map for a chain."""
    where = {}
    for i, p in enumerate(chain):
        where[(p[0], p[1])] = i
    return where


def _energy(sequence, chain):
    """Compute the HP contact energy of a conformation."""
    where = _build_where(chain)
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y = p
            j = where.get((x+1, y))
            if j is not None and j > i + 1 and sequence[j] == "H":
                contacts += 1
            j = where.get((x-1, y))
            if j is not None and j > i + 1 and sequence[j] == "H":
                contacts += 1
            j = where.get((x, y+1))
            if j is not None and j > i + 1 and sequence[j] == "H":
                contacts += 1
            j = where.get((x, y-1))
            if j is not None and j > i + 1 and sequence[j] == "H":
                contacts += 1
    return -contacts


def _energy_delta_pivot(sequence, chain, where, pivot_idx):
    """Compute energy delta for a pivot move at pivot_idx.
    Returns (new_energy_delta, new_where) if valid, or (None, None) if invalid.
    The new_where is the updated position map.
    """
    n = len(chain)
    seg_start = pivot_idx + 1
    if seg_start >= n:
        return None, None
    
    old_segment = chain[seg_start:]
    fixed_set = set(chain[:pivot_idx + 1])
    pivot = chain[pivot_idx]
    
    for angle in (90, -90, 180):
        rotated = []
        valid = True
        for p in old_segment:
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
        
        # Compute delta energy
        old_seg_positions = set(old_segment)
        new_seg_positions = set(rotated)
        
        # Contacts that are lost: H-H contacts where at least one residue was in old_segment
        # and the contact is broken by the move
        lost_contacts = 0
        # For each H in old_segment, check if it had contacts with H outside segment
        for k, pos in enumerate(old_segment):
            idx = seg_start + k
            if sequence[idx] != 'H':
                continue
            x, y = pos
            for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
                nb = (x+dx, y+dy)
                j = where.get(nb)
                if j is not None and j < seg_start and sequence[j] == 'H':
                    # This contact will be lost unless the new position also has it
                    new_pos = rotated[k]
                    nx, ny = new_pos
                    for dx2, dy2 in ((1,0), (-1,0), (0,1), (0,-1)):
                        nb2 = (nx+dx2, ny+dy2)
                        j2 = where.get(nb2)
                        if j2 == j:
                            break
                    else:
                        lost_contacts += 1
        
        # Contacts that are gained: H-H contacts where at least one residue is in new_segment
        # and the contact is new
        gained_contacts = 0
        for k, pos in enumerate(rotated):
            idx = seg_start + k
            if sequence[idx] != 'H':
                continue
            x, y = pos
            for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
                nb = (x+dx, y+dy)
                j = where.get(nb)
                if j is not None and j < seg_start and sequence[j] == 'H':
                    # Check if this contact existed before
                    old_pos = old_segment[k]
                    ox, oy = old_pos
                    for dx2, dy2 in ((1,0), (-1,0), (0,1), (0,-1)):
                        nb2 = (ox+dx2, oy+dy2)
                        j2 = where.get(nb2)
                        if j2 == j:
                            break
                    else:
                        gained_contacts += 1
        
        delta = -(gained_contacts - lost_contacts)
        
        # Build new where
        new_where = dict(where)
        for k, pos in enumerate(old_segment):
            del new_where[pos]
        for k, pos in enumerate(rotated):
            new_where[pos] = seg_start + k
        
        return delta, (new_where, rotated)
    
    return None, None


def _polish(sequence, chain, max_iterations=5000, time_limit=5.0, rng=None):
    """Deterministic hill-climbing polish: accept only improving moves.
    Uses a more efficient strategy: try random pivots first, then slither."""
    if chain is None:
        return None
    if rng is None:
        rng = random.Random(42)

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    where = _build_where(current_chain)

    start_time = time.time()
    stall_count = 0
    n = len(current_chain)

    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 15:
            break

        improved = False

        # Strategy 1: Short pivots
        for _ in range(20):
            pivot_idx = rng.randint(1, min(n - 2, max(2, n // 4)))
            delta, result = _energy_delta_pivot(sequence, current_chain, where, pivot_idx)
            if delta is None:
                continue
            if delta < 0:
                new_where, rotated = result
                seg_start = pivot_idx + 1
                current_chain = current_chain[:seg_start] + rotated
                where = new_where
                current_energy += delta
                improved = True
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                break
        if improved:
            stall_count = 0
            continue

        # Strategy 2: Medium pivots
        for _ in range(15):
            pivot_idx = rng.randint(1, min(n - 2, max(3, n // 2)))
            delta, result = _energy_delta_pivot(sequence, current_chain, where, pivot_idx)
            if delta is None:
                continue
            if delta < 0:
                new_where, rotated = result
                seg_start = pivot_idx + 1
                current_chain = current_chain[:seg_start] + rotated
                where = new_where
                current_energy += delta
                improved = True
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                break
        if improved:
            stall_count = 0
            continue

        # Strategy 3: Slither moves
        last = current_chain[-1]
        prev = current_chain[-2]
        occupied = set(current_chain[:-1])
        for site in _neighbours(prev):
            if site in occupied:
                continue
            new_chain = current_chain[:-1] + [site]
            new_energy = _energy(sequence, new_chain)
            delta = new_energy - current_energy
            if delta < 0:
                current_chain = new_chain
                current_energy = new_energy
                where = _build_where(current_chain)
                improved = True
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                break
        if improved:
            stall_count = 0
            continue

        stall_count += 1
        if stall_count > 40:
            break

    return best_chain


def _grow(sequence, dimension, rng, bias=1.0):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)
    
    # Track centroid incrementally for efficiency
    cx = 0.0
    cy = 0.0
    
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            # Distance to centroid
            dist = math.sqrt((site[0] - cx)**2 + (site[1] - cy)**2)
            
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            
            # Adaptive scoring: stronger contact bias early, more compactness later
            progress = i / max(1, n - 1)
            contact_weight = bias * (1.5 - 0.5 * progress)
            compact_weight = 0.3 + 0.7 * progress
            
            score = contact_weight * gain - compact_weight * dist + 0.1 * free + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
        # Update centroid
        cx = (cx * i + site[0]) / (i + 1)
        cy = (cy * i + site[1]) / (i + 1)
    
    return chain


def _pivot_move_fast(chain, rng):
    """Fast pivot move: rotate a subchain. Validity check is O(segment_length)."""
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
            else:  # 180
                ndx, ndy = -dx, -dy
            new_p = (pivot[0] + ndx, pivot[1] + ndy)
            if new_p in fixed_set:
                valid = False
                break
            rotated.append(new_p)
        
        if not valid:
            continue
            
        new_chain = chain[:segment_start] + rotated
        return new_chain

    return None


def _slither_move(chain, rng):
    """Try to move the end of the chain."""
    n = len(chain)
    if n < 4:
        return None
    
    last = chain[-1]
    prev = chain[-2]
    occupied = set(chain[:-1])
    
    for site in _neighbours(prev):
        if site in occupied:
            continue
        new_chain = chain[:-1] + [site]
        return new_chain
    return None


def _local_search(sequence, chain, max_iterations=10000, time_limit=8.0, rng=None):
    """Simulated annealing local search to improve a conformation."""
    if chain is None:
        return None

    if rng is None:
        rng = random.Random(hash(tuple(chain)) % (2**32))

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    where = _build_where(current_chain)

    start_time = time.time()
    n = len(current_chain)
    
    # Adaptive temperature: start higher for longer sequences
    temperature = 3.0 + 0.05 * n
    cooling = 0.9985  # Slower cooling for more exploration
    min_temp = 0.2

    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 10:
            break

        # Choose move with adaptive probabilities
        r = rng.random()
        delta = None
        new_chain = None
        new_where = None
        
        if r < 0.55:
            # Standard pivot with incremental energy
            pivot_idx = rng.randint(1, n - 2)
            delta, result = _energy_delta_pivot(sequence, current_chain, where, pivot_idx)
            if delta is not None:
                new_where, rotated = result
                seg_start = pivot_idx + 1
                new_chain = current_chain[:seg_start] + rotated
        elif r < 0.75:
            # Slither move
            last = current_chain[-1]
            prev = current_chain[-2]
            occupied = set(current_chain[:-1])
            for site in _neighbours(prev):
                if site in occupied:
                    continue
                new_chain = current_chain[:-1] + [site]
                new_energy = _energy(sequence, new_chain)
                delta = new_energy - current_energy
                new_where = _build_where(new_chain)
                break
        elif r < 0.85:
            # Short pivot
            pivot_idx = rng.randint(1, min(n - 2, max(2, n // 4)))
            delta, result = _energy_delta_pivot(sequence, current_chain, where, pivot_idx)
            if delta is not None:
                new_where, rotated = result
                seg_start = pivot_idx + 1
                new_chain = current_chain[:seg_start] + rotated
        else:
            # Long pivot for large-scale rearrangement
            if n > 15:
                pivot_idx = rng.randint(1, max(2, n // 3))
            else:
                pivot_idx = rng.randint(1, n - 2)
            delta, result = _energy_delta_pivot(sequence, current_chain, where, pivot_idx)
            if delta is not None:
                new_where, rotated = result
                seg_start = pivot_idx + 1
                new_chain = current_chain[:seg_start] + rotated

        if new_chain is None or delta is None:
            continue

        # Metropolis criterion with adaptive temperature
        if delta < 0:
            accept = True
        elif temperature > min_temp:
            if delta < 50:
                accept_prob = math.exp(-delta / temperature)
            else:
                accept_prob = 0.0
            accept = rng.random() < accept_prob
        else:
            accept = False

        if accept:
            current_chain = new_chain
            current_energy += delta
            where = new_where

            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)

        temperature = max(min_temp, temperature * cooling)

    return best_chain


def _spiral_fold(sequence):
    """Create a compact spiral conformation as a deterministic starting point."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    if n == 2:
        return [(0, 0), (1, 0)]
    
    # Build a spiral that stays compact
    chain = [(0, 0)]
    # Directions: right, up, left, down, repeat but expanding
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    dir_idx = 0
    step_count = 0
    steps_per_dir = 1
    
    while len(chain) < n:
        dx, dy = directions[dir_idx]
        # Take steps_per_dir steps in this direction
        for _ in range(steps_per_dir):
            if len(chain) >= n:
                break
            last_x, last_y = chain[-1]
            new_x = last_x + dx
            new_y = last_y + dy
            chain.append((new_x, new_y))
        
        dir_idx = (dir_idx + 1) % 4
        if dir_idx == 2:  # After going left, increase steps
            steps_per_dir += 1
    
    return chain[:n]


def _snake_fold(sequence):
    """Create a snake-like conformation."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    # Snake pattern: go right, down, left, down, right, down...
    width = max(1, int(math.sqrt(n)) + 1)
    
    x, y = 0, 0
    chain.append((x, y))
    
    for i in range(1, n):
        if i % (2 * width) < width:
            # Going right
            x = (i % width) if (i // width) % 2 == 0 else width - 1 - (i % width)
            y = i // width
        else:
            # Going left
            x = width - 1 - ((i - width) % width) if (i // width) % 2 == 1 else (i - width) % width
            y = i // width
        
        # Simplified: just use a proper snake
        row = i // width
        col = i % width
        if row % 2 == 0:
            x = col
        else:
            x = width - 1 - col
        y = row
        
        chain.append((x, y))
    
    return chain


def fold(sequence, dimension):
    """Best conformation using multiple strategies: greedy growth + local search."""
    seq_len = len(sequence)

    # Adjust strategy based on sequence length
    # Allocate more time to local search for better results
    if seq_len <= 30:
        num_grow = 500
        local_search_time = 18.0
        polish_time = 10.0
        extra_search_time = 12.0
    elif seq_len <= 50:
        num_grow = 400
        local_search_time = 15.0
        polish_time = 8.0
        extra_search_time = 10.0
    elif seq_len <= 70:
        num_grow = 300
        local_search_time = 12.0
        polish_time = 7.0
        extra_search_time = 8.0
    else:
        num_grow = 200
        local_search_time = 10.0
        polish_time = 5.0
        extra_search_time = 7.0

    rng = random.Random(12345 + seq_len)
    best_chain = None
    best_energy = 0

    # Pass 0: Try deterministic structures first
    for init_chain in [_spiral_fold(sequence), _snake_fold(sequence)]:
        if init_chain is not None and len(init_chain) == seq_len:
            # Verify self-avoidance
            if len(set(init_chain)) == seq_len:
                # Verify unit steps
                valid = True
                for i in range(1, seq_len):
                    dx = abs(init_chain[i][0] - init_chain[i-1][0])
                    dy = abs(init_chain[i][1] - init_chain[i-1][1])
                    if dx + dy != 1:
                        valid = False
                        break
                if valid:
                    value = _energy(sequence, init_chain)
                    if value < best_energy:
                        best_energy = value
                        best_chain = init_chain

    # Pass 1: Greedy Growth
    # We use a range of biases to explore different compactness levels
    for i in range(num_grow):
        if _time_left() < 35:
            break

        # Bias between 0.5 and 3.0
        bias = 0.5 + rng.random() * 2.5
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is None:
            continue

        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = chain

    # Pass 2: Local Search on the best chain found
    if best_chain is not None and _time_left() > 45:
        improved = _local_search(sequence, best_chain, max_iterations=50000,
                                  time_limit=local_search_time, rng=rng)
        if improved is not None:
            improved_energy = _energy(sequence, improved)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved

        # Pass 3: Deterministic polish
        if _time_left() > 40:
            polished = _polish(sequence, best_chain, max_iterations=20000,
                               time_limit=polish_time, rng=rng)
            if polished is not None:
                polished_energy = _energy(sequence, polished)
                if polished_energy < best_energy:
                    best_energy = polished_energy
                    best_chain = polished

        # Pass 4: Additional local search with different seed if time permits
        if _time_left() > 50:
            rng2 = random.Random(99999 + seq_len)
            improved2 = _local_search(sequence, best_chain, max_iterations=30000,
                                       time_limit=extra_search_time, rng=rng2)
            if improved2 is not None:
                e2 = _energy(sequence, improved2)
                if e2 < best_energy:
                    best_energy = e2
                    best_chain = improved2

            # Final polish
            if _time_left() > 40:
                final_polish = _polish(sequence, best_chain, max_iterations=15000,
                                       time_limit=polish_time * 0.8, rng=rng2)
                if final_polish is not None:
                    fp_energy = _energy(sequence, final_polish)
                    if fp_energy < best_energy:
                        best_energy = fp_energy
                        best_chain = final_polish

        # Pass 5: Another round of growth + search if time permits
        if _time_left() > 55:
            extra_grow = 60
            for i in range(extra_grow):
                if _time_left() < 45:
                    break
                bias = 0.5 + rng2.random() * 2.5
                chain = _grow(sequence, dimension, rng2, bias=bias)
                if chain is None:
                    continue
                value = _energy(sequence, chain)
                if value < best_energy:
                    best_energy = value
                    best_chain = chain
                    # Quick local search
                    if _time_left() > 40:
                        quick_ls = _local_search(sequence, best_chain, max_iterations=10000,
                                                 time_limit=4.0, rng=rng2)
                        if quick_ls is not None:
                            qe = _energy(sequence, quick_ls)
                            if qe < best_energy:
                                best_energy = qe
                                best_chain = quick_ls
                    # Quick polish
                    if _time_left() > 35:
                        polished = _polish(sequence, best_chain, max_iterations=8000,
                                           time_limit=4.0, rng=rng2)
                        if polished is not None:
                            pe = _energy(sequence, polished)
                            if pe < best_energy:
                                best_energy = pe
                                best_chain = polished

    # Final fallback: straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(seq_len)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
