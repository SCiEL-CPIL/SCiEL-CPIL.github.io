"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time

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


def _energy_delta(sequence, chain, fixed_end, start, end):
    """
    Energy change when positions [start, end] (inclusive) of the chain move to
    new points, while positions before `start` and after `end` stay fixed.

    Returns new_energy - old_energy computed in O(end-start + a few).
    """
    n = len(chain)
    # Build the old and new position lists for the moved segment and the
    # boundary neighbours, then count H-H contacts that cross the boundary.
    # A contact is an H-H pair on neighbouring lattice sites that are not
    # adjacent along the chain. We only need to re-examine contacts that
    # involve a moved residue or a residue adjacent (in lattice space) to a
    # moved residue.

    # Positions of moved segment before and after.
    old_seg = chain[start:end + 1]
    # new_seg is passed implicitly: caller has already built it; we recompute
    # here from the rotated points the caller will use. Instead, accept the
    # new segment as an argument.
    raise RuntimeError("unused")


def _grow(sequence, dimension, rng, bias=1.0):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
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
            score = bias * gain + 0.1 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot_move(chain, rng):
    """Perform a single pivot move: rotate a subchain by 90, 180, or 270 degrees around a pivot point."""
    n = len(chain)
    if n < 4:
        return None

    pivot_idx = rng.randint(1, n - 2)
    pivot = chain[pivot_idx]
    segment = chain[pivot_idx:]

    for angle in (90, -90, 180):
        rotated = []
        for p in segment:
            dx, dy = p[0] - pivot[0], p[1] - pivot[1]
            if angle == 90:
                ndx, ndy = -dy, dx
            elif angle == -90:
                ndx, ndy = dy, -dx
            else:  # 180
                ndx, ndy = -dx, -dy
            rotated.append((pivot[0] + ndx, pivot[1] + ndy))

        new_chain = chain[:pivot_idx] + rotated
        if len(set(new_chain)) != len(new_chain):
            continue
        valid = True
        for i in range(1, len(new_chain)):
            dx = abs(new_chain[i][0] - new_chain[i - 1][0])
            dy = abs(new_chain[i][1] - new_chain[i - 1][1])
            if dx + dy != 1:
                valid = False
                break
        if valid:
            return new_chain
    return None


def _crankshaft_move(chain, rng):
    """Perform a crankshaft move: rotate a short subchain (length 2 or 3)."""
    n = len(chain)
    if n < 4:
        return None

    sub_len = rng.choice([2, 3])
    if n < sub_len + 2:
        return None

    start_idx = rng.randint(1, n - sub_len - 1)
    pivot = chain[start_idx - 1]
    segment = chain[start_idx:start_idx + sub_len]

    for angle in (90, -90, 180):
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

        new_chain = chain[:start_idx] + rotated + chain[start_idx + sub_len:]
        if len(set(new_chain)) != len(new_chain):
            continue
        valid = True
        for i in range(1, len(new_chain)):
            dx = abs(new_chain[i][0] - new_chain[i - 1][0])
            dy = abs(new_chain[i][1] - new_chain[i - 1][1])
            if dx + dy != 1:
                valid = False
                break
        if valid:
            return new_chain
    return None


def _sa_run(sequence, chain, time_limit, rng, T0=3.0, cooling=0.999, reheat_T=1.5):
    """
    Simulated annealing with a hot-restart phase.

    Phase 1: cool from T0.  Phase 2: reheat to reheat_T and cool again, which
    lets the chain escape the basin found in phase 1.
    """
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    temperature = T0

    for phase in range(2):
        if phase == 1:
            temperature = reheat_T
        while time.time() - start_time < time_limit:
            if _time_left() < 8:
                break
            if rng.random() < 0.65:
                new_chain = _pivot_move(current_chain, rng)
            else:
                new_chain = _crankshaft_move(current_chain, rng)
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

    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation using multiple strategies: greedy growth + SA local search."""
    seq_len = len(sequence)

    # Per-sequence time budget: split the global guard across the 10 sequences.
    # Longer sequences get a slightly larger share because they need more work.
    seq_budget = max(6.0, min(22.0, 150.0 / 10.0 * (seq_len / 50.0 + 0.5)))
    seq_start = time.time()

    def _seq_left():
        return seq_budget - (time.time() - seq_start)

    # Parameters scaled by length
    if seq_len <= 30:
        num_grow = 400
        sa_time = 6.0
        sa_starts = 25
    elif seq_len <= 50:
        num_grow = 300
        sa_time = 5.0
        sa_starts = 18
    elif seq_len <= 70:
        num_grow = 200
        sa_time = 4.0
        sa_starts = 12
    else:
        num_grow = 150
        sa_time = 3.0
        sa_starts = 8

    rng = random.Random(12345 + seq_len)
    best_chain = None
    best_energy = 0

    # Pass 1: greedy growth with varying bias, keep the top few.
    candidates = []
    for i in range(num_grow):
        if _time_left() < 20 or _seq_left() < seq_budget * 0.4:
            break
        bias = 0.5 + rng.random() * 1.5
        chain = _grow(sequence, dimension, rng, bias=bias)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if best_chain is None or value < best_energy:
            best_energy = value
            best_chain = chain
        candidates.append((value, chain))

    if best_chain is None:
        return [list(p) for p in [(i, 0) for i in range(seq_len)]]

    # Pass 2: SA on the best chain, then on the next-best distinct chains.
    candidates.sort(key=lambda x: x[0])
    seen = set()
    sa_candidates = []
    for value, chain in candidates:
        key = tuple(chain)
        if key in seen:
            continue
        seen.add(key)
        sa_candidates.append(chain)
        if len(sa_candidates) >= sa_starts:
            break

    for idx, chain in enumerate(sa_candidates):
        if _time_left() < 15 or _seq_left() < 2.0:
            break
        # Give the best candidate the full SA time, others a reduced time.
        t = sa_time if idx == 0 else sa_time * 0.5
        improved, improved_energy = _sa_run(sequence, chain, t, rng)
        if improved is not None and improved_energy < best_energy:
            best_energy = improved_energy
            best_chain = improved

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
