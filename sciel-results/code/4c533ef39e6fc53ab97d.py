# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 200.0


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _neighbours(point):
    x, y = point
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _energy(sequence, chain, pos_map=None):
    if pos_map is None:
        pos_map = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            x, y = p
            j = pos_map.get((x + 1, y))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
            j = pos_map.get((x - 1, y))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
            j = pos_map.get((x, y + 1))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
            j = pos_map.get((x, y - 1))
            if j is not None and j > i + 1 and sequence[j] == 'H':
                contacts += 1
    return -contacts


_DX = [1, 0, -1, 0]
_DY = [0, 1, 0, -1]


def _beam_search_init_fast(sequence, beam_width=30, compactness=0.2, rng=None):
    """Fast beam search with incremental bounding box tracking."""
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
            'occupied': {(0, 0): 0},
            'chain': [(0, 0)],
            'score': 0.0,
            'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0
        })

    h_positions = set()  # Will be rebuilt each step is too slow; use sequence check instead

    for i in range(1, n):
        new_beam = []
        for state in beam:
            d = state['dir']
            x, y = state['x'], state['y']
            occ = state['occupied']
            chain = state['chain']
            score = state['score']
            seq_i = sequence[i]
            min_x = state['min_x']
            max_x = state['max_x']
            min_y = state['min_y']
            max_y = state['max_y']

            for turn in (0, 1, 2):
                new_d = (d + turn) % 4
                nx = x + _DX[new_d]
                ny = y + _DY[new_d]
                if (nx, ny) in occ:
                    continue

                new_occ = dict(occ)
                new_occ[(nx, ny)] = i
                new_chain = chain + [(nx, ny)]

                # Update bounding box incrementally
                new_min_x = min(min_x, nx)
                new_max_x = max(max_x, nx)
                new_min_y = min(min_y, ny)
                new_max_y = max(max_y, ny)

                # Incremental contact count
                contact_gain = 0
                if seq_i == 'H':
                    for q in ((nx + 1, ny), (nx - 1, ny), (nx, ny + 1), (nx, ny - 1)):
                        j = new_occ.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            contact_gain += 1

                # Compactness: bounding box area
                compact = (new_max_x - new_min_x + 1) * (new_max_y - new_min_y + 1)

                new_score = score + contact_gain - compactness * compact + rng.random() * 0.05

                new_beam.append({
                    'dir': new_d,
                    'x': nx,
                    'y': ny,
                    'occupied': new_occ,
                    'chain': new_chain,
                    'score': new_score,
                    'min_x': new_min_x,
                    'max_x': new_max_x,
                    'min_y': new_min_y,
                    'max_y': new_max_y
                })

        if not new_beam:
            break

        new_beam.sort(key=lambda s: s['score'], reverse=True)
        beam = new_beam[:beam_width]

    if not beam:
        return None
    return beam[0]['chain']


def _local_search(sequence, chain, max_iterations=20000, time_limit=5.0, rng=None):
    """Simulated annealing with precomputed H sets for fast pivot deltas."""
    if chain is None:
        return None
    if rng is None:
        rng = random.Random(hash(tuple(chain)) % (2**32))

    current_chain = list(chain)
    pos_map = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain, pos_map)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    temperature = 3.0
    cooling = 0.9995

    n = len(current_chain)
    h_indices = [i for i, s in enumerate(sequence) if s == 'H']
    h_set = set(h_indices)

    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 10:
            break

        r = rng.random()
        if r < 0.5:
            # Slither move
            if n < 4:
                continue
            last_idx = n - 1
            last = current_chain[last_idx]
            prev = current_chain[last_idx - 1]

            candidate_sites = []
            for s in _neighbours(prev):
                if s == last:
                    continue
                if s not in pos_map:
                    candidate_sites.append(s)

            if not candidate_sites:
                continue

            new_site = rng.choice(candidate_sites)

            delta = 0
            if last_idx in h_set:
                for q in _neighbours(last):
                    if q == prev:
                        continue
                    j = pos_map.get(q)
                    if j is not None and j < last_idx and j in h_set:
                        delta += 1
                for q in _neighbours(new_site):
                    if q == prev:
                        continue
                    j = pos_map.get(q)
                    if j is not None and j < last_idx and j in h_set:
                        delta -= 1

            if delta < 0 or rng.random() < math.exp(-delta / max(temperature, 0.1)):
                del pos_map[last]
                pos_map[new_site] = last_idx
                current_chain[last_idx] = new_site
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)

        else:
            # Pivot move with precomputed H info
            pivot_idx = rng.randint(1, n - 2)
            if pivot_idx >= n - 1:
                continue
            pivot = current_chain[pivot_idx]
            seg_start = pivot_idx + 1
            if seg_start >= n:
                continue

            angle = rng.choice((0, 1, 2))  # 0=90, 1=-90, 2=180

            # Build fixed positions set
            fixed_positions = set(current_chain[:seg_start])

            # H in moving segment
            h_moving = [i for i in range(seg_start, n) if i in h_set]
            # H in fixed segment
            h_fixed_pos = set()
            for i in range(seg_start):
                if i in h_set:
                    h_fixed_pos.add(current_chain[i])

            # Compute new positions
            new_positions = []
            valid = True
            for i in range(seg_start, n):
                p = current_chain[i]
                dx, dy = p[0] - pivot[0], p[1] - pivot[1]
                if angle == 0:
                    ndx, ndy = -dy, dx
                elif angle == 1:
                    ndx, ndy = dy, -dx
                else:
                    ndx, ndy = -dx, -dy
                new_p = (pivot[0] + ndx, pivot[1] + ndy)
                if new_p in fixed_positions:
                    valid = False
                    break
                new_positions.append(new_p)

            if not valid:
                continue

            # Energy delta: only cross contacts change
            delta = 0
            for i in h_moving:
                old_p = current_chain[i]
                for q in _neighbours(old_p):
                    if q in h_fixed_pos:
                        delta += 1
            for k, i in enumerate(h_moving):
                new_p = new_positions[k]
                for q in _neighbours(new_p):
                    if q in h_fixed_pos:
                        delta -= 1

            if delta < 0 or rng.random() < math.exp(-delta / max(temperature, 0.1)):
                for i in range(seg_start, n):
                    del pos_map[current_chain[i]]
                for k, i in enumerate(range(seg_start, n)):
                    pos_map[new_positions[k]] = i
                    current_chain[i] = new_positions[k]

                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)

        temperature *= cooling

    return best_chain


def _polish(sequence, chain, max_iterations=3000, time_limit=3.0, rng=None):
    """Deterministic hill-climbing with exhaustive slither + sampled pivots."""
    if chain is None:
        return None
    if rng is None:
        rng = random.Random(42)

    current_chain = list(chain)
    pos_map = {p: i for i, p in enumerate(current_chain)}
    current_energy = _energy(sequence, current_chain, pos_map)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    n = len(current_chain)
    h_set = set(i for i, s in enumerate(sequence) if s == 'H')

    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 12:
            break

        best_delta = 0
        best_new_chain = None
        best_new_map = None

        # Try all valid slithers
        if n >= 4:
            last_idx = n - 1
            last = current_chain[last_idx]
            prev = current_chain[last_idx - 1]

            for site in _neighbours(prev):
                if site == last or site in pos_map:
                    continue

                delta = 0
                if last_idx in h_set:
                    for q in _neighbours(last):
                        if q == prev:
                            continue
                        j = pos_map.get(q)
                        if j is not None and j < last_idx and j in h_set:
                            delta += 1
                    for q in _neighbours(site):
                        if q == prev:
                            continue
                        j = pos_map.get(q)
                        if j is not None and j < last_idx and j in h_set:
                            delta -= 1

                if delta < best_delta:
                    best_delta = delta
                    best_new_chain = current_chain[:-1] + [site]
                    best_new_map = dict(pos_map)
                    del best_new_map[last]
                    best_new_map[site] = last_idx

        # Try a few random pivots
        for _ in range(8):
            pivot_idx = rng.randint(1, n - 2)
            if pivot_idx >= n - 1:
                continue
            pivot = current_chain[pivot_idx]
            seg_start = pivot_idx + 1
            if seg_start >= n:
                continue

            angle = rng.choice((0, 1, 2))
            fixed_positions = set(current_chain[:seg_start])
            h_moving = [i for i in range(seg_start, n) if i in h_set]
            h_fixed_pos = set()
            for i in range(seg_start):
                if i in h_set:
                    h_fixed_pos.add(current_chain[i])

            new_positions = []
            valid = True
            for i in range(seg_start, n):
                p = current_chain[i]
                dx, dy = p[0] - pivot[0], p[1] - pivot[1]
                if angle == 0:
                    ndx, ndy = -dy, dx
                elif angle == 1:
                    ndx, ndy = dy, -dx
                else:
                    ndx, ndy = -dx, -dy
                new_p = (pivot[0] + ndx, pivot[1] + ndy)
                if new_p in fixed_positions:
                    valid = False
                    break
                new_positions.append(new_p)

            if not valid:
                continue

            delta = 0
            for i in h_moving:
                old_p = current_chain[i]
                for q in _neighbours(old_p):
                    if q in h_fixed_pos:
                        delta += 1
            for k, i in enumerate(h_moving):
                new_p = new_positions[k]
                for q in _neighbours(new_p):
                    if q in h_fixed_pos:
                        delta -= 1

            if delta < best_delta:
                best_delta = delta
                best_new_chain = current_chain[:seg_start] + new_positions
                best_new_map = dict(pos_map)
                for i in range(seg_start, n):
                    del best_new_map[current_chain[i]]
                for k, i in enumerate(range(seg_start, n)):
                    best_new_map[new_positions[k]] = i

        if best_new_chain is None:
            break

        current_chain = best_new_chain
        pos_map = best_new_map
        current_energy += best_delta
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = list(current_chain)

    return best_chain


def _greedy_grow(sequence, rng, bias=1.0):
    """Greedy chain growth with running center of mass."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    cx = 0.0
    cy = 0.0
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == 'H')
            dist = math.sqrt((site[0] - cx)**2 + (site[1] - cy)**2)
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = bias * gain - 0.5 * dist + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
        cx = (cx * i + site[0]) / (i + 1)
        cy = (cy * i + site[1]) / (i + 1)
    return chain


def _spiral_fold(sequence):
    """Generate a compact spiral conformation as a strong initialization."""
    n = len(sequence)
    if n <= 1:
        return [(0, 0)] if n == 1 else []

    chain = [(0, 0)]
    occupied = {(0, 0): 0}

    # Spiral direction sequence: E, N, W, W, S, S, E, E, N, N, ...
    # This creates a growing square spiral
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]
    dir_idx = 0
    step_count = 1
    steps_in_dir = 1
    steps_needed = 1
    current_dir = 0

    x, y = 0, 0
    for i in range(1, n):
        placed = False
        for attempt in range(4):
            nx = x + dx[current_dir]
            ny = y + dy[current_dir]
            if (nx, ny) not in occupied:
                occupied[(nx, ny)] = i
                chain.append((nx, ny))
                x, y = nx, ny
                placed = True
                break
            else:
                current_dir = (current_dir + 1) % 4

        if not placed:
            # Spiral is stuck, try to extend in any free direction
            for d in range(4):
                nx = x + dx[d]
                ny = y + dy[d]
                if (nx, ny) not in occupied:
                    occupied[(nx, ny)] = i
                    chain.append((nx, ny))
                    x, y = nx, ny
                    current_dir = d
                    placed = True
                    break
            if not placed:
                return None

    return chain


def fold(sequence, dimension):
    seq_len = len(sequence)
    tl = _time_left()

    # Adaptive time allocation
    if seq_len <= 30:
        num_beam = 12
        beam_width = 40
        local_time = min(8.0, tl * 0.3)
        polish_time = min(4.0, tl * 0.15)
        num_grow = 120
    elif seq_len <= 50:
        num_beam = 10
        beam_width = 35
        local_time = min(7.0, tl * 0.3)
        polish_time = min(3.5, tl * 0.15)
        num_grow = 100
    elif seq_len <= 70:
        num_beam = 8
        beam_width = 30
        local_time = min(6.0, tl * 0.3)
        polish_time = min(3.0, tl * 0.12)
        num_grow = 80
    else:
        num_beam = 6
        beam_width = 25
        local_time = min(5.0, tl * 0.3)
        polish_time = min(2.5, tl * 0.1)
        num_grow = 60

    rng = random.Random(12345 + seq_len)
    best_chain = None
    best_energy = 0

    # Phase 1: Beam search initialization
    for i in range(num_beam):
        if _time_left() < 30:
            break
        compactness = 0.05 + rng.random() * 0.35
        bw = beam_width + rng.randint(-5, 5)
        chain = _beam_search_init_fast(sequence, beam_width=max(10, bw),
                                        compactness=compactness, rng=rng)
        if chain is None or len(chain) != seq_len:
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = chain

    # Phase 1b: Spiral initialization
    if _time_left() > 35:
        spiral = _spiral_fold(sequence)
        if spiral is not None and len(spiral) == seq_len:
            sv = _energy(sequence, spiral)
            if sv < best_energy:
                best_energy = sv
                best_chain = spiral

    # Phase 2: Greedy growth initialization
    for i in range(num_grow):
        if _time_left() < 25:
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
    if best_chain is not None and _time_left() > 25:
        improved = _local_search(sequence, best_chain, max_iterations=50000,
                                  time_limit=local_time, rng=rng)
        if improved is not None:
            improved_energy = _energy(sequence, improved)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved

        # Phase 4: Polish
        if _time_left() > 18:
            polished = _polish(sequence, best_chain, max_iterations=5000,
                               time_limit=polish_time, rng=rng)
            if polished is not None:
                polished_energy = _energy(sequence, polished)
                if polished_energy < best_energy:
                    best_energy = polished_energy
                    best_chain = polished

        # Phase 5: Second local search with different seed
        if _time_left() > 30:
            rng2 = random.Random(99999 + seq_len)
            improved2 = _local_search(sequence, best_chain, max_iterations=20000,
                                       time_limit=local_time * 0.5, rng=rng2)
            if improved2 is not None:
                e2 = _energy(sequence, improved2)
                if e2 < best_energy:
                    best_energy = e2
                    best_chain = improved2

            # Phase 6: Final polish
            if _time_left() > 18:
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
