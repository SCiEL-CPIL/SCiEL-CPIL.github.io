"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 195.0


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _neighbours(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y = p
            for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


# Direction vectors: 0=East, 1=North, 2=West, 3=South
_DX = [1, 0, -1, 0]
_DY = [0, 1, 0, -1]


def _beam_search_init(sequence, beam_width=50, compactness=0.3, rng=None):
    """Beam search over turn-state sequences to build a self-avoiding chain.
    
    State: (direction, x, y, occupied_set, turn_sequence)
    Transitions: L (left turn), R (right turn), S (straight)
    Score: negative contact count - compactness * bounding_box_area
    """
    if rng is None:
        rng = random.Random()

    n = len(sequence)
    if n <= 1:
        return [(0, 0)] if n == 1 else []

    # Initial beam: 4 possible starting directions
    beam = []
    for d in range(4):
        beam.append({
            'dir': d,
            'x': 0,
            'y': 0,
            'occupied': {(0, 0)},
            'chain': [(0, 0)],
            'contacts': 0,
            'score': 0.0
        })

    for i in range(1, n):
        new_beam = []
        for state in beam:
            d = state['dir']
            x, y = state['x'], state['y']
            occ = state['occupied']
            chain = state['chain']
            contacts = state['contacts']
            seq_i = sequence[i]

            for turn in (0, 1, 2):  # 0=straight, 1=left, 2=right
                new_d = (d + turn) % 4
                nx = x + _DX[new_d]
                ny = y + _DY[new_d]
                if (nx, ny) in occ:
                    continue

                new_occ = occ | {(nx, ny)}
                new_chain = chain + [(nx, ny)]

                new_contacts = contacts
                if seq_i == 'H':
                    for q in ((nx + 1, ny), (nx - 1, ny), (nx, ny + 1), (nx, ny - 1)):
                        if q in occ:
                            # Find index of q in chain
                            # For efficiency, we track this implicitly
                            # We need to check if the occupant is H and not adjacent
                            pass
                    # Recompute contacts incrementally: check H neighbours in occupied
                    for q in ((nx + 1, ny), (nx - 1, ny), (nx, ny + 1), (nx, ny - 1)):
                        if q in occ:
                            # Find which index q corresponds to
                            # We need a position->index map for efficiency
                            pass

                # For beam search, we compute a heuristic score
                # bounding box area as compactness measure
                xs = [p[0] for p in new_chain]
                ys = [p[1] for p in new_chain]
                area = (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)

                # Contact estimate: count H-H adjacencies in new_chain
                # (expensive but beam is small)
                h_contacts = 0
                pos_map = {p: idx for idx, p in enumerate(new_chain)}
                for j in range(len(new_chain)):
                    if sequence[j] == 'H':
                        px, py = new_chain[j]
                        for q in ((px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)):
                            k = pos_map.get(q)
                            if k is not None and k > j + 1 and sequence[k] == 'H':
                                h_contacts += 1

                score = -h_contacts - compactness * area + rng.random() * 0.1

                new_beam.append({
                    'dir': new_d,
                    'x': nx,
                    'y': ny,
                    'occupied': new_occ,
                    'chain': new_chain,
                    'contacts': h_contacts,
                    'score': score
                })

        if not new_beam:
            break

        new_beam.sort(key=lambda s: s['score'], reverse=True)
        beam = new_beam[:beam_width]

    if not beam:
        return None
    return beam[0]['chain']


def _beam_search_init_fast(sequence, beam_width=30, compactness=0.2, rng=None):
    """Faster beam search: avoid recomputing contacts from scratch each step."""
    if rng is None:
        rng = random.Random()

    n = len(sequence)
    if n <= 1:
        return [(0, 0)] if n == 1 else []

    beam = []
    for d in range(4):
        beam.append({
            'dir': d,
            'x': 0,
            'y': 0,
            'occupied': {(0, 0): 0},  # point -> index
            'chain': [(0, 0)],
            'score': 0.0
        })

    for i in range(1, n):
        new_beam = []
        for state in beam:
            d = state['dir']
            x, y = state['x'], state['y']
            occ = state['occupied']
            chain = state['chain']
            score = state['score']
            seq_i = sequence[i]

            for turn in (0, 1, 2):
                new_d = (d + turn) % 4
                nx = x + _DX[new_d]
                ny = y + _DY[new_d]
                if (nx, ny) in occ:
                    continue

                new_occ = dict(occ)
                new_occ[(nx, ny)] = i
                new_chain = chain + [(nx, ny)]

                # Incremental contact count
                contact_gain = 0
                if seq_i == 'H':
                    for q in ((nx + 1, ny), (nx - 1, ny), (nx, ny + 1), (nx, ny - 1)):
                        j = new_occ.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            contact_gain += 1

                # Compactness: bounding box
                # Approximate with Manhattan distance from origin
                cx = sum(p[0] for p in new_chain) / len(new_chain)
                cy = sum(p[1] for p in new_chain) / len(new_chain)
                # Use a cheaper proxy: max |x| + max |y|
                max_x = max(abs(p[0]) for p in new_chain)
                max_y = max(abs(p[1]) for p in new_chain)
                compact = max_x + max_y

                new_score = score + contact_gain - compactness * compact + rng.random() * 0.05

                new_beam.append({
                    'dir': new_d,
                    'x': nx,
                    'y': ny,
                    'occupied': new_occ,
                    'chain': new_chain,
                    'score': new_score
                })

        if not new_beam:
            break

        new_beam.sort(key=lambda s: s['score'], reverse=True)
        beam = new_beam[:beam_width]

    if not beam:
        return None
    return beam[0]['chain']


def _pivot_delta(sequence, chain, pivot_idx, angle):
    """Compute energy delta for a pivot move without building the full new chain.
    
    Returns (delta, valid) where delta is new_energy - old_energy.
    """
    n = len(chain)
    if pivot_idx < 1 or pivot_idx >= n - 1:
        return 0, False

    pivot = chain[pivot_idx]
    seg_start = pivot_idx + 1
    if seg_start >= n:
        return 0, False

    # Build set of fixed points (indices 0..pivot_idx)
    fixed_set = set(chain[:pivot_idx + 1])

    # Compute new positions of moving segment
    new_positions = []
    valid = True
    for p in chain[seg_start:]:
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
        new_positions.append(new_p)

    if not valid:
        return 0, False

    # Compute energy delta:
    # Old contacts involving the moving segment (indices seg_start..n-1)
    # New contacts involving the moved segment
    # Contacts entirely within fixed part (0..pivot_idx) are unchanged
    # Contacts between fixed and moving change

    # Build old position map for moving segment
    old_pos = {i: chain[i] for i in range(seg_start, n)}
    new_pos = {i: new_positions[i - seg_start] for i in range(seg_start, n)}

    # H indices in moving segment
    h_moving = [i for i in range(seg_start, n) if sequence[i] == 'H']
    # H indices in fixed part
    h_fixed = [i for i in range(0, pivot_idx + 1) if sequence[i] == 'H']

    # Old contacts:
    # 1) H-H pairs both in moving segment (internal)
    # 2) H-H pairs one in moving, one in fixed
    # New contacts: same structure with new positions

    # For internal moving contacts, rotation preserves relative positions,
    # so H-H pairs within the moving segment that were at distance 1 remain at distance 1.
    # So internal contacts are unchanged!
    # Only cross contacts (fixed-moving) change.

    old_cross = 0
    for i in h_moving:
        px, py = old_pos[i]
        for q in ((px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)):
            # Check if q is in fixed part
            # We need to know which index q corresponds to in fixed part
            pass

    # This approach is complex. Let's just compute full energy for the new chain.
    # But we can do it faster by only checking relevant positions.

    # Alternative: compute old and new energy for the full chain but using a sparse approach.
    # Given the complexity, let's just build the new chain and compute energy.
    # The speedup comes from doing this in C-optimized code via numpy or similar.
    # For now, let's use a hybrid: build new chain, compute energy with early exit.

    new_chain = chain[:seg_start] + new_positions
    return None, True  # Signal to compute full energy


def _energy_fast(sequence, chain, where=None):
    """Compute energy with optional precomputed position map."""
    if where is None:
        where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y = p
            for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _local_search(sequence, chain, max_iterations=20000, time_limit=5.0, rng=None):
    """Simulated annealing with fast energy computation."""
    if chain is None:
        return None
    if rng is None:
        rng = random.Random(hash(tuple(chain)) % (2**32))

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    temperature = 3.0
    cooling = 0.9995

    n = len(current_chain)

    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 10:
            break

        r = rng.random()
        if r < 0.7:
            # Pivot move
            pivot_idx = rng.randint(1, n - 2)
            pivot = current_chain[pivot_idx]
            seg_start = pivot_idx + 1
            if seg_start >= n:
                continue
            fixed_set = set(current_chain[:pivot_idx + 1])
            angle = rng.choice((90, -90, 180))
            rotated = []
            valid = True
            for p in current_chain[seg_start:]:
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
            new_chain = current_chain[:seg_start] + rotated
        else:
            # Slither move
            if n < 4:
                continue
            last = current_chain[-1]
            prev = current_chain[-2]
            occupied = set(current_chain[:-1])
            site = None
            for s in _neighbours(prev):
                if s not in occupied:
                    site = s
                    break
            if site is None:
                continue
            new_chain = current_chain[:-1] + [site]

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


def _polish(sequence, chain, max_iterations=3000, time_limit=3.0, rng=None):
    """Deterministic hill-climbing polish."""
    if chain is None:
        return None
    if rng is None:
        rng = random.Random(42)

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    n = len(current_chain)

    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 12:
            break

        best_move = None
        best_delta = 0

        # Try pivots
        for pivot_idx in range(1, n - 1):
            pivot = current_chain[pivot_idx]
            seg_start = pivot_idx + 1
            if seg_start >= n:
                continue
            fixed_set = set(current_chain[:pivot_idx + 1])
            for angle in (90, -90, 180):
                rotated = []
                valid = True
                for p in current_chain[seg_start:]:
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
                new_chain = current_chain[:seg_start] + rotated
                new_energy = _energy(sequence, new_chain)
                delta = new_energy - current_energy
                if delta < best_delta:
                    best_delta = delta
                    best_move = new_chain

        # Try slither
        if n >= 4:
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


def _greedy_grow(sequence, rng, bias=1.0):
    """Greedy chain growth as fallback initialization."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H')
            cx = sum(p[0] for p in chain) / len(chain)
            cy = sum(p[1] for p in chain) / len(chain)
            dist = math.sqrt((site[0] - cx)**2 + (site[1] - cy)**2)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = bias * gain - 0.5 * dist + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def fold(sequence, dimension):
    seq_len = len(sequence)

    # Time allocation based on sequence length
    if seq_len <= 30:
        num_beam = 8
        beam_width = 40
        local_time = 6.0
        polish_time = 3.0
        num_grow = 100
    elif seq_len <= 50:
        num_beam = 6
        beam_width = 30
        local_time = 5.0
        polish_time = 2.5
        num_grow = 80
    elif seq_len <= 70:
        num_beam = 5
        beam_width = 25
        local_time = 4.0
        polish_time = 2.0
        num_grow = 60
    else:
        num_beam = 4
        beam_width = 20
        local_time = 3.5
        polish_time = 1.5
        num_grow = 40

    rng = random.Random(12345 + seq_len)
    best_chain = None
    best_energy = 0

    # Phase 1: Beam search initialization (diverse compactness)
    for i in range(num_beam):
        if _time_left() < 25:
            break
        compactness = 0.1 + rng.random() * 0.4
        bw = beam_width + rng.randint(-5, 5)
        chain = _beam_search_init_fast(sequence, beam_width=max(10, bw),
                                        compactness=compactness, rng=rng)
        if chain is None or len(chain) != seq_len:
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = chain

    # Phase 2: Greedy growth initialization (diverse biases)
    for i in range(num_grow):
        if _time_left() < 20:
            break
        bias = 0.5 + rng.random() * 2.0
        chain = _greedy_grow(sequence, rng, bias=bias)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = chain

    # Phase 3: Local search on best chain
    if best_chain is not None and _time_left() > 20:
        improved = _local_search(sequence, best_chain, max_iterations=30000,
                                  time_limit=local_time, rng=rng)
        if improved is not None:
            improved_energy = _energy(sequence, improved)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved

        # Phase 4: Polish
        if _time_left() > 15:
            polished = _polish(sequence, best_chain, max_iterations=5000,
                               time_limit=polish_time, rng=rng)
            if polished is not None:
                polished_energy = _energy(sequence, polished)
                if polished_energy < best_energy:
                    best_energy = polished_energy
                    best_chain = polished

        # Phase 5: Second local search with different seed
        if _time_left() > 25:
            rng2 = random.Random(99999 + seq_len)
            improved2 = _local_search(sequence, best_chain, max_iterations=15000,
                                       time_limit=local_time * 0.5, rng=rng2)
            if improved2 is not None:
                e2 = _energy(sequence, improved2)
                if e2 < best_energy:
                    best_energy = e2
                    best_chain = improved2

            if _time_left() > 15:
                final_polish = _polish(sequence, best_chain, max_iterations=3000,
                                       time_limit=polish_time * 0.5, rng=rng2)
                if final_polish is not None:
                    fp_energy = _energy(sequence, final_polish)
                    if fp_energy < best_energy:
                        best_energy = fp_energy
                        best_chain = final_polish

    # Fallback
    if best_chain is None:
        best_chain = [(i, 0) for i in range(seq_len)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
