"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 205.0  # Safety margin for 210s kill


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
    where = {}
    for i, p in enumerate(chain):
        where[(p[0], p[1])] = i
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y = p
            j = where.get((x + 1, y))
            if j is not None and j > i + 1 and sequence[j] == "H":
                contacts += 1
            j = where.get((x - 1, y))
            if j is not None and j > i + 1 and sequence[j] == "H":
                contacts += 1
            j = where.get((x, y + 1))
            if j is not None and j > i + 1 and sequence[j] == "H":
                contacts += 1
            j = where.get((x, y - 1))
            if j is not None and j > i + 1 and sequence[j] == "H":
                contacts += 1
    return -contacts


def _build_where(chain):
    """Build position->index map."""
    where = {}
    for i, p in enumerate(chain):
        where[(p[0], p[1])] = i
    return where


def _count_contacts_for_residue(sequence, where, idx, pos):
    """Count H-H contacts involving residue idx at position pos (each contact counted once via i<j)."""
    if sequence[idx] != "H":
        return 0
    x, y = pos
    count = 0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        j = where.get((x + dx, y + dy))
        if j is not None and j > idx + 1 and sequence[j] == "H":
            count += 1
    return count


def _total_contacts(sequence, chain, where):
    """Total number of H-H contacts."""
    total = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            total += _count_contacts_for_residue(sequence, where, i, p)
    return total


def _delta_pivot_energy(sequence, chain, where, pivot_idx, angle):
    """Compute delta energy for a pivot move. Returns (delta, new_segment) or (None, None)."""
    n = len(chain)
    seg_start = pivot_idx + 1
    if seg_start >= n:
        return None, None

    pivot = chain[pivot_idx]
    old_segment = chain[seg_start:]
    fixed_set = set(chain[:seg_start])

    # Rotate
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
        return None, None

    new_segment = rotated

    # Compute delta energy incrementally
    # Contacts that change are those involving residues in [seg_start, n) or their neighbors
    # Old contacts involving moved residues:
    old_c = 0
    for k in range(len(old_segment)):
        idx = seg_start + k
        if sequence[idx] == "H":
            old_c += _count_contacts_for_residue(sequence, where, idx, old_segment[k])
    # Also contacts from fixed residues to old segment positions
    old_seg_set = set(old_segment)
    for i in range(seg_start):
        if sequence[i] == "H":
            x, y = chain[i]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (x + dx, y + dy)
                if nb in old_seg_set:
                    # Find which index in old_segment
                    for k in range(len(old_segment)):
                        if old_segment[k] == nb:
                            j = seg_start + k
                            if j > i + 1 and sequence[j] == "H":
                                old_c += 1
                            break

    # New contacts:
    # Build a temporary where for new positions
    new_where = dict(where)
    for pos in old_segment:
        if pos in new_where:
            del new_where[pos]
    for k, pos in enumerate(new_segment):
        new_where[pos] = seg_start + k

    new_c = 0
    for k in range(len(new_segment)):
        idx = seg_start + k
        if sequence[idx] == "H":
            x, y = new_segment[k]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                j = new_where.get((x + dx, y + dy))
                if j is not None and j > idx + 1 and sequence[j] == "H":
                    new_c += 1
    # Contacts from fixed residues to new segment positions
    new_seg_set = set(new_segment)
    for i in range(seg_start):
        if sequence[i] == "H":
            x, y = chain[i]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (x + dx, y + dy)
                if nb in new_seg_set:
                    for k in range(len(new_segment)):
                        if new_segment[k] == nb:
                            j = seg_start + k
                            if j > i + 1 and sequence[j] == "H":
                                new_c += 1
                            break

    delta = -(new_c) - (-(old_c))  # new_energy - old_energy = -new_c - (-old_c) = old_c - new_c
    return old_c - new_c, new_segment


def _apply_pivot(chain, pivot_idx, new_segment):
    """Apply a pivot move to the chain."""
    seg_start = pivot_idx + 1
    return chain[:seg_start] + new_segment


def _update_where(where, chain, pivot_idx, old_segment, new_segment):
    """Update the where map after a pivot move."""
    for pos in old_segment:
        if pos in where:
            del where[pos]
    seg_start = pivot_idx + 1
    for k, pos in enumerate(new_segment):
        where[pos] = seg_start + k


def _priority_contacts(sequence, top_k=15):
    """Return top-k H-H pairs by sequence distance (longer range = higher priority)."""
    n = len(sequence)
    cmap = []
    for i in range(n):
        if sequence[i] == "H":
            for j in range(i + 2, n):
                if sequence[j] == "H":
                    cmap.append((i, j))
    cmap.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
    return cmap[:top_k]


def _grow(sequence, dimension, rng, bias=1.0):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    n = len(sequence)

    priority_pairs = _priority_contacts(sequence, top_k=15)
    satisfied_priority = set()

    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")

            priority_bonus = 0
            for (pi, pj) in priority_pairs:
                if pj == i and pi < i - 1:
                    pi_x, pi_y = chain[pi]
                    if abs(site[0] - pi_x) + abs(site[1] - pi_y) == 1:
                        if (pi, pj) not in satisfied_priority:
                            priority_bonus += 3.0
                        else:
                            priority_bonus += 1.0

            cx = sum(p[0] for p in chain) / len(chain)
            cy = sum(p[1] for p in chain) / len(chain)
            dist = math.sqrt((site[0] - cx) ** 2 + (site[1] - cy) ** 2)

            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = bias * gain + priority_bonus - 0.5 * dist + 0.1 * free + rng.random()
            options.append((score, site))

        if not options:
            return None
        best_score, site = max(options)
        occupied[site] = i
        chain.append(site)

        # Update satisfied priority
        if sequence[i] == "H":
            for (pi, pj) in priority_pairs:
                if pj == i and pi < i - 1:
                    pi_x, pi_y = chain[pi]
                    if abs(site[0] - pi_x) + abs(site[1] - pi_y) == 1:
                        satisfied_priority.add((pi, pj))

    return chain


def _local_search_incremental(sequence, chain, max_iterations=50000, time_limit=8.0, rng=None):
    """Simulated annealing with incremental energy tracking via pivot moves."""
    if chain is None:
        return None

    if rng is None:
        rng = random.Random(hash(tuple(chain)) % (2 ** 32))

    n = len(chain)
    current_chain = list(chain)
    where = _build_where(current_chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    temperature = 5.0
    cooling = 0.997
    stall_count = 0

    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 12:
            break

        # Choose pivot index and angle
        pivot_idx = rng.randint(1, n - 2)
        angle = rng.choice((90, -90, 180))

        delta, new_segment = _delta_pivot_energy(sequence, current_chain, where, pivot_idx, angle)
        if new_segment is None:
            continue

        new_energy = current_energy + delta

        # Metropolis criterion
        if delta < 0 or (temperature > 0.1 and rng.random() < math.exp(-delta / temperature)):
            # Accept
            current_chain = _apply_pivot(current_chain, pivot_idx, new_segment)
            _update_where(where, current_chain, pivot_idx, current_chain[pivot_idx + 1:] if False else [], new_segment)
            # Rebuild where properly
            where = _build_where(current_chain)
            current_energy = new_energy

            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
                stall_count = 0
            else:
                stall_count += 1
        else:
            stall_count += 1

        temperature *= cooling

        if stall_count > 200:
            # Try slither move to escape local minimum
            last = current_chain[-1]
            prev = current_chain[-2]
            occupied = set(current_chain[:-1])
            for site in _neighbours(prev):
                if site in occupied:
                    continue
                new_chain = current_chain[:-1] + [site]
                new_e = _energy(sequence, new_chain)
                if new_e < current_energy or (temperature > 0.1 and rng.random() < math.exp(-(new_e - current_energy) / temperature)):
                    current_chain = new_chain
                    where = _build_where(current_chain)
                    current_energy = new_e
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    stall_count = 0
                break

    return best_chain


def _polish(sequence, chain, max_iterations=10000, time_limit=5.0, rng=None):
    """Deterministic hill-climbing polish: accept only improving moves."""
    if chain is None:
        return None
    if rng is None:
        rng = random.Random(42)

    n = len(chain)
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    stall_count = 0

    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 15:
            break

        improved = False

        # Try batch of random pivot moves
        for _ in range(30):
            pivot_idx = rng.randint(1, n - 2)
            angle = rng.choice((90, -90, 180))

            delta, new_segment = _delta_pivot_energy(sequence, current_chain, _build_where(current_chain), pivot_idx, angle)
            if new_segment is None:
                continue

            if delta < 0:
                current_chain = _apply_pivot(current_chain, pivot_idx, new_segment)
                current_energy += delta
                improved = True
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(current_chain)
                break

        if not improved:
            # Try slither
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
                    improved = True
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = list(current_chain)
                    break

        if not improved:
            stall_count += 1
            if stall_count > 100:
                break
        else:
            stall_count = 0

    return best_chain


def _spiral_fold(sequence):
    """Create a compact spiral conformation."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]
    if n == 2:
        return [(0, 0), (1, 0)]

    chain = [(0, 0)]
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    dir_idx = 0
    steps_per_dir = 1

    while len(chain) < n:
        dx, dy = directions[dir_idx]
        for _ in range(steps_per_dir):
            if len(chain) >= n:
                break
            last_x, last_y = chain[-1]
            chain.append((last_x + dx, last_y + dy))

        dir_idx = (dir_idx + 1) % 4
        if dir_idx == 2:
            steps_per_dir += 1

    return chain[:n]


def _snake_fold(sequence):
    """Create a snake-like conformation."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]

    width = max(1, int(math.sqrt(n)) + 1)
    chain = [(0, 0)]
    for i in range(1, n):
        row = i // width
        col = i % width
        if row % 2 == 0:
            x = col
        else:
            x = width - 1 - col
        y = row
        chain.append((x, y))

    return chain


def _backtrack_search(sequence, time_limit=5.0, rng=None):
    """Backtracking search with contact map guidance."""
    if rng is None:
        rng = random.Random(42)

    n = len(sequence)
    if n < 4:
        return None

    priority_pairs = _priority_contacts(sequence, top_k=10)
    best_chain = None
    best_energy = 0
    start_time = time.time()

    for attempt in range(15):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 20:
            break

        chain = [(0, 0)]
        occupied = {(0, 0): 0}

        prefix_len = min(8, n // 2)
        valid_prefix = True
        for i in range(1, prefix_len):
            options = []
            for site in _neighbours(chain[-1]):
                if site not in occupied:
                    options.append(site)
            if not options:
                valid_prefix = False
                break
            site = options[rng.randint(0, len(options) - 1)]
            occupied[site] = i
            chain.append(site)

        if not valid_prefix:
            continue

        def _backtrack(i):
            nonlocal best_chain, best_energy
            if time.time() - start_time > time_limit:
                return
            if _time_left() < 15:
                return

            if i == n:
                e = _energy(sequence, chain)
                if e < best_energy:
                    best_energy = e
                    best_chain = list(chain)
                return

            candidates = []
            for site in _neighbours(chain[-1]):
                if site in occupied:
                    continue
                score = 0
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and sequence[j] == 'H':
                            score += 3.0
                    for (pi, pj) in priority_pairs:
                        if pj == i and pi < i - 1:
                            pi_x, pi_y = chain[pi]
                            if abs(site[0] - pi_x) + abs(site[1] - pi_y) == 1:
                                score += 4.0
                cx = sum(p[0] for p in chain) / len(chain)
                cy = sum(p[1] for p in chain) / len(chain)
                dist = (site[0] - cx) ** 2 + (site[1] - cy) ** 2
                score -= 0.1 * dist
                candidates.append((score, site))

            if not candidates:
                return

            candidates.sort(key=lambda x: -x[0])
            trial_count = min(4, len(candidates))
            for k in range(trial_count):
                if time.time() - start_time > time_limit:
                    return
                if _time_left() < 15:
                    return

                idx = 0 if k == 0 else rng.randint(0, len(candidates) - 1)
                site = candidates[idx][1]
                chain.append(site)
                occupied[site] = i
                _backtrack(i + 1)
                chain.pop()
                del occupied[site]

            return

        _backtrack(prefix_len)

    return best_chain


def fold(sequence, dimension):
    """Best conformation using multiple strategies: greedy growth + local search."""
    seq_len = len(sequence)

    if seq_len <= 30:
        num_grow = 500
        local_search_time = 10.0
        polish_time = 5.0
        backtrack_time = 5.0
    elif seq_len <= 50:
        num_grow = 350
        local_search_time = 9.0
        polish_time = 4.5
        backtrack_time = 4.0
    elif seq_len <= 70:
        num_grow = 250
        local_search_time = 8.0
        polish_time = 4.0
        backtrack_time = 3.0
    else:
        num_grow = 180
        local_search_time = 7.0
        polish_time = 3.5
        backtrack_time = 2.5

    rng = random.Random(12345 + seq_len)
    best_chain = None
    best_energy = 0

    # Pass 0: Deterministic structures
    for init_chain in [_spiral_fold(sequence), _snake_fold(sequence)]:
        if init_chain is not None and len(init_chain) == seq_len:
            if len(set(init_chain)) == seq_len:
                valid = True
                for i in range(1, seq_len):
                    dx = abs(init_chain[i][0] - init_chain[i - 1][0])
                    dy = abs(init_chain[i][1] - init_chain[i - 1][1])
                    if dx + dy != 1:
                        valid = False
                        break
                if valid:
                    value = _energy(sequence, init_chain)
                    if value < best_energy:
                        best_energy = value
                        best_chain = init_chain

    # Pass 1: Backtracking
    if _time_left() > 35:
        bt_chain = _backtrack_search(sequence, time_limit=backtrack_time, rng=rng)
        if bt_chain is not None and len(bt_chain) == seq_len:
            bt_energy = _energy(sequence, bt_chain)
            if bt_energy < best_energy:
                best_energy = bt_energy
                best_chain = bt_chain

    # Pass 2: Greedy Growth
    for i in range(num_grow):
        if _time_left() < 28:
            break
        bias = 0.5 + rng.random() * 2.5
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = chain

    # Pass 3: Incremental local search on best chain
    if best_chain is not None and _time_left() > 30:
        improved = _local_search_incremental(sequence, best_chain, max_iterations=80000,
                                              time_limit=local_search_time, rng=rng)
        if improved is not None:
            improved_energy = _energy(sequence, improved)
            if improved_energy < best_energy:
                best_energy = improved_energy
                best_chain = improved

        # Pass 4: Polish
        if _time_left() > 25:
            polished = _polish(sequence, best_chain, max_iterations=15000,
                               time_limit=polish_time, rng=rng)
            if polished is not None:
                polished_energy = _energy(sequence, polished)
                if polished_energy < best_energy:
                    best_energy = polished_energy
                    best_chain = polished

        # Pass 5: Another local search round
        if _time_left() > 35:
            rng2 = random.Random(99999 + seq_len)
            improved2 = _local_search_incremental(sequence, best_chain, max_iterations=50000,
                                                   time_limit=local_search_time * 0.6, rng=rng2)
            if improved2 is not None:
                e2 = _energy(sequence, improved2)
                if e2 < best_energy:
                    best_energy = e2
                    best_chain = improved2

            if _time_left() > 25:
                final_polish = _polish(sequence, best_chain, max_iterations=10000,
                                       time_limit=polish_time * 0.5, rng=rng2)
                if final_polish is not None:
                    fp_energy = _energy(sequence, final_polish)
                    if fp_energy < best_energy:
                        best_energy = fp_energy
                        best_chain = final_polish

        # Pass 6: Extra growth + search
        if _time_left() > 40:
            extra_grow = 30
            for i in range(extra_grow):
                if _time_left() < 30:
                    break
                bias = 0.5 + rng2.random() * 2.5
                chain = _grow(sequence, dimension, rng2, bias=bias)
                if chain is None:
                    continue
                value = _energy(sequence, chain)
                if value < best_energy:
                    best_energy = value
                    best_chain = chain
                    if _time_left() > 25:
                        polished = _polish(sequence, best_chain, max_iterations=5000,
                                           time_limit=2.0, rng=rng2)
                        if polished is not None:
                            pe = _energy(sequence, polished)
                            if pe < best_energy:
                                best_energy = pe
                                best_chain = polished

    # Fallback
    if best_chain is None:
        best_chain = [(i, 0) for i in range(seq_len)]

    # Safety check
    if len(set(best_chain)) != seq_len:
        best_chain = []
        x, y = 0, 0
        best_chain.append((x, y))
        occupied = {(x, y)}
        for i in range(1, seq_len):
            placed = False
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in occupied:
                    best_chain.append((nx, ny))
                    occupied.add((nx, ny))
                    x, y = nx, ny
                    placed = True
                    break
            if not placed:
                best_chain.append((x + i, y))
                occupied.add((x + i, y))
                x = x + i

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
