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


def _energy_delta(sequence, chain, pivot_idx):
    """Compute energy change when rotating segment at pivot_idx.
    Only contacts involving the pivot boundary change."""
    n = len(chain)
    if pivot_idx < 1 or pivot_idx >= n - 1:
        return None

    pivot = chain[pivot_idx]
    segment_start = pivot_idx + 1
    segment = chain[segment_start:]
    fixed_set = set(chain[:pivot_idx + 1])

    # For each rotation angle, compute the energy delta
    best_delta = 0
    best_rotated = None

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

        # Compute energy delta: only contacts at the boundary change
        # Old contacts involving pivot or segment endpoints
        # New contacts involving rotated segment
        # For efficiency, just compute full energy of new chain if delta matters
        new_chain = chain[:segment_start] + rotated
        # This is O(n) but we only do it for valid rotations
        old_e = _energy(sequence, chain)
        new_e = _energy(sequence, new_chain)
        delta = new_e - old_e
        if delta < best_delta:
            best_delta = delta
            best_rotated = new_chain

    return best_delta, best_rotated


def _grow(sequence, dimension, rng, bias=1.0, compactness=0.5):
    """One randomized greedy chain growth with tunable compactness."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    for i in range(1, len(sequence)):
        options = []
        cx = sum(p[0] for p in chain) / len(chain)
        cy = sum(p[1] for p in chain) / len(chain)

        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")

            dist = math.sqrt((site[0] - cx)**2 + (site[1] - cy)**2)
            free = sum(1 for q in _neighbours(site) if q not in occupied)

            score = bias * gain - compactness * dist + 0.1 * free + rng.random() * 0.8
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot_move_fast(chain, rng):
    """Fast pivot move: rotate a subchain."""
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
        return new_chain

    return None


def _crankshaft_move(chain, rng):
    """Crankshaft move: rotate a 2-residue segment around a pivot."""
    n = len(chain)
    if n < 5:
        return None

    pivot_idx = rng.randint(1, n - 3)
    pivot = chain[pivot_idx]
    # Segment is chain[pivot_idx+1] and chain[pivot_idx+2]
    p1 = chain[pivot_idx + 1]
    p2 = chain[pivot_idx + 2]

    fixed_set = set(chain[:pivot_idx + 1]) | set(chain[pivot_idx + 3:])

    for angle in (90, -90, 180):
        # Rotate p1 and p2 around pivot
        rotated = []
        valid = True
        for p in (p1, p2):
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

        # Check that rotated[0] is adjacent to pivot (should be by construction)
        # and rotated[1] is adjacent to rotated[0] (preserved by rotation)
        # Also check connection to chain[pivot_idx+3]
        if pivot_idx + 3 < n:
            next_p = chain[pivot_idx + 3]
            dx = abs(rotated[1][0] - next_p[0])
            dy = abs(rotated[1][1] - next_p[1])
            if dx + dy != 1:
                continue

        new_chain = chain[:pivot_idx + 1] + rotated + chain[pivot_idx + 3:]
        return new_chain

    return None


def _slither_move(chain, rng):
    """Try to move the end of the chain."""
    n = len(chain)
    if n < 4:
        return None

    prev = chain[-2]
    occupied = set(chain[:-1])

    for site in _neighbours(prev):
        if site in occupied:
            continue
        new_chain = chain[:-1] + [site]
        return new_chain
    return None


def _local_search(sequence, chain, max_iterations=5000, time_limit=5.0, rng=None):
    """Simulated annealing with pivot, crankshaft, and slither moves."""
    if chain is None:
        return None

    if rng is None:
        rng = random.Random(hash(tuple(chain)) % (2**32))

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    temperature = 5.0
    cooling = 0.999

    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 10:
            break

        r = rng.random()
        if r < 0.55:
            new_chain = _pivot_move_fast(current_chain, rng)
        elif r < 0.75:
            new_chain = _crankshaft_move(current_chain, rng)
        elif r < 0.9:
            new_chain = _slither_move(current_chain, rng)
        else:
            new_chain = _pivot_move_fast(current_chain, rng)

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


def _deterministic_polish(sequence, chain, time_limit=3.0):
    """Best-improvement deterministic polish: try all pivots, take best move each step."""
    if chain is None:
        return None

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    n = len(current_chain)

    while time.time() - start_time < time_limit and _time_left() > 15:
        best_move = None
        best_delta = 0

        # Try all pivot positions
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
                new_energy = _energy(sequence, new_chain)
                delta = new_energy - current_energy
                if delta < best_delta:
                    best_delta = delta
                    best_move = new_chain

        # Also try slither
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


def _diversity_score(chain1, chain2):
    """Simple diversity measure: fraction of positions that differ."""
    n = min(len(chain1), len(chain2))
    diff = sum(1 for i in range(n) if chain1[i] != chain2[i])
    return diff / n


def fold(sequence, dimension):
    """Best conformation using multi-start portfolio with time-sliced polishing."""
    seq_len = len(sequence)

    # Adjust strategy based on sequence length
    if seq_len <= 30:
        num_grow = 400
        num_polish = 5
        local_search_time = 6.0
        polish_time = 4.0
    elif seq_len <= 50:
        num_grow = 300
        num_polish = 4
        local_search_time = 5.0
        polish_time = 3.5
    elif seq_len <= 70:
        num_grow = 200
        num_polish = 3
        local_search_time = 4.0
        polish_time = 3.0
    else:
        num_grow = 150
        num_polish = 3
        local_search_time = 3.5
        polish_time = 2.5

    rng = random.Random(12345 + seq_len)
    best_chain = None
    best_energy = 0

    # Pass 1: Greedy Growth - collect diverse candidates
    candidates = []
    for i in range(num_grow):
        if _time_left() < 25:
            break

        bias = 0.5 + rng.random() * 2.0
        compactness = 0.3 + rng.random() * 0.8
        chain = _grow(sequence, dimension, rng, bias=bias, compactness=compactness)
        if chain is None:
            continue

        value = _energy(sequence, chain)
        if value < best_energy:
            best_energy = value
            best_chain = chain

        # Keep top diverse candidates
        if len(candidates) < num_polish * 3:
            # Check diversity against existing candidates
            is_diverse = True
            for c in candidates:
                if _diversity_score(chain, c) < 0.15:
                    is_diverse = False
                    break
            if is_diverse:
                candidates.append((value, chain))

    # Sort candidates by energy
    candidates.sort(key=lambda x: x[0])

    # Pass 2: Time-sliced multi-start polishing
    # Pick top diverse candidates and polish each with allocated time
    if best_chain is not None and _time_left() > 30:
        # Select top diverse candidates for polishing
        polish_candidates = []
        for energy, chain in candidates:
            if len(polish_candidates) >= num_polish:
                break
            is_diverse = True
            for _, c in polish_candidates:
                if _diversity_score(chain, c) < 0.1:
                    is_diverse = False
                    break
            if is_diverse:
                polish_candidates.append(chain)

        # Ensure best chain is included
        if best_chain not in polish_candidates:
            polish_candidates.insert(0, best_chain)

        # Allocate time: local search then deterministic polish for each
        time_per_candidate = min(local_search_time + polish_time, (_time_left() - 20) / max(len(polish_candidates), 1))
        ls_time = time_per_candidate * 0.6
        dp_time = time_per_candidate * 0.4

        for chain in polish_candidates:
            if _time_left() < 20:
                break

            # Local search
            if _time_left() > 25:
                rng_ls = random.Random(hash(tuple(chain)) % (2**32))
                improved = _local_search(sequence, chain, max_iterations=8000,
                                         time_limit=ls_time, rng=rng_ls)
                if improved is not None:
                    ie = _energy(sequence, improved)
                    if ie < best_energy:
                        best_energy = ie
                        best_chain = improved

            # Deterministic polish
            if _time_left() > 20:
                polished = _deterministic_polish(sequence, best_chain, time_limit=dp_time)
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
