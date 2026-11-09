"""HP lattice protein folding: greedy growth + efficient local search with time guard."""
# EVOLVE-BLOCK-START
import random
import time

_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 115.0


def _remaining_time():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbours(point):
    x, y = point
    for dx, dy in _DELTAS:
        yield (x + dx, y + dy)


def _energy(sequence, chain):
    """Compute energy: -1 for each H-H contact on non-adjacent lattice neighbours."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(i - j) > 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts // 2


def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0, 0)
    chain, occupied = [start], {start: 0}
    n = len(sequence)
    for i in range(1, n):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _local_search(sequence, chain, time_limit):
    """Move-based local search with incremental energy updates. Mutates chain in place."""
    n = len(chain)
    if n < 4:
        return _energy(sequence, chain)

    where = {p: i for i, p in enumerate(chain)}
    best = _energy(sequence, chain)
    deadline = time.time() + time_limit

    while time.time() < deadline:
        improved = False

        # --- Reversal Moves ---
        for a in range(1, n - 2):
            for b in range(a + 1, n - 1):
                if time.time() > deadline:
                    break
                p_prev = chain[a - 1]
                p_end = chain[b]
                p_start = chain[a]
                p_next = chain[b + 1]

                # Check new junctions are valid (distance 1)
                if abs(p_prev[0] - p_end[0]) + abs(p_prev[1] - p_end[1]) != 1:
                    continue
                if abs(p_start[0] - p_next[0]) + abs(p_start[1] - p_next[1]) != 1:
                    continue

                # Apply reversal
                seg = chain[a:b + 1]
                seg.reverse()
                chain[a:b + 1] = seg
                for i, p in enumerate(seg, start=a):
                    where[p] = i

                e = _energy(sequence, chain)
                if e < best:
                    best = e
                    improved = True
                else:
                    seg.reverse()
                    chain[a:b + 1] = seg
                    for i, p in enumerate(seg, start=a):
                        where[p] = i
            if time.time() > deadline:
                break
        if improved:
            continue

        # --- Pivot Moves ---
        for i in range(1, n - 2):
            if time.time() > deadline:
                break
            pivot = chain[i]
            tail = chain[i + 1:]
            prefix_points = set(chain[:i])

            for k in (1, 2, 3):
                if time.time() > deadline:
                    break
                rotated = []
                valid = True
                for p in tail:
                    x = p[0] - pivot[0]
                    y = p[1] - pivot[1]
                    for _ in range(k):
                        x, y = -y, x
                    new_p = (pivot[0] + x, pivot[1] + y)
                    rotated.append(new_p)
                    if new_p in prefix_points:
                        valid = False
                        break

                if not valid:
                    continue

                if rotated[0] == chain[i - 1]:
                    continue

                chain[i + 1:] = rotated
                for j, p in enumerate(rotated, start=i + 1):
                    where[p] = j

                e = _energy(sequence, chain)
                if e < best:
                    best = e
                    improved = True
                    break
                else:
                    chain[i + 1:] = tail
                    for j, p in enumerate(tail, start=i + 1):
                        where[p] = j

            if improved:
                break

        if not improved:
            break

    return best


def fold(sequence, dimension):
    """Best conformation from many randomized greedy growths followed by local search."""
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 1
    n = len(sequence)

    # Phase 1: Many fast greedy growths to find good starting points
    growth_count = 0
    while _remaining_time() > 1.0:
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            continue
        growth_count += 1
        value = _energy(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value

        # Stop early if we've found an excellent conformation
        if best_energy <= -n // 3:
            break
        # Limit growths to avoid spending too much time on one sequence
        if growth_count >= 500:
            break

    # Phase 2: Local search on the best conformation with remaining time
    if best is not None and _remaining_time() > 0.5:
        ls_time = _remaining_time() * 0.8
        _local_search(sequence, best, ls_time)
        best_energy = _energy(sequence, best)

    if best is None:
        best = [(i, 0) for i in range(n)]

    return [list(p) for p in best]
# EVOLVE-BLOCK-END
