"""HP lattice protein folding: greedy growth + local-search refinement."""
# EVOLVE-BLOCK-START
import random
import time

_START = time.time()
_DEADLINE = _START + 195.0  # safety margin before the 210s wall-clock kill


def _neighbours(point):
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


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


def _grow(sequence, rng):
    """Randomized greedy chain growth: prefer sites making new H-H contacts."""
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
                           if q in occupied and occupied[q] < i - 1
                           and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain


def _pivot(sequence, chain, i, j, rng):
    """Rotate the sub-chain chain[i..j] by 90 deg; return new chain or None."""
    n = len(chain)
    if j - i < 2 or j - i > 12:
        return None
    a = chain[i - 1]
    b = chain[i]
    c = chain[j]
    d = chain[j + 1] if j + 1 < n else None
    ox, oy = a[0] - b[0], a[1] - b[1]
    if rng.random() < 0.5:
        nx, ny = -oy, ox
    else:
        nx, ny = oy, -ox
    if (nx, ny) == (-oy, ox):
        def rot(p):
            rx, ry = p[0] - b[0], p[1] - b[1]
            return (b[0] - ry, b[1] + rx)
    else:
        def rot(p):
            rx, ry = p[0] - b[0], p[1] - b[1]
            return (b[0] + ry, b[1] - rx)
    new_b = (b[0] + nx, b[1] + ny)
    new_c = rot(c)
    if abs(new_b[0] - a[0]) + abs(new_b[1] - a[1]) != 1:
        return None
    if abs(new_c[0] - new_b[0]) + abs(new_c[1] - new_b[1]) != 1:
        return None
    if d is not None and abs(new_c[0] - d[0]) + abs(new_c[1] - d[1]) != 1:
        return None
    new_chain = list(chain)
    new_chain[i] = new_b
    new_chain[j] = new_c
    for p in (new_b, new_c):
        for k, q in enumerate(chain):
            if k == i or k == j:
                continue
            if q == p:
                return None
    return new_chain


def _refine(sequence, chain, budget):
    """Local search: accept energy-improving pivots while budget remains."""
    energy = _energy(sequence, chain)
    n = len(chain)
    if n < 4:
        return chain, energy
    t_end = min(time.time() + budget, _DEADLINE - 0.5)
    positions = []
    for i in range(1, n - 2):
        for j in range(i + 2, min(i + 13, n - 1)):
            positions.append((i, j))
    rng = random.Random(999)
    while time.time() < t_end and positions:
        i, j = rng.choice(positions)
        new_chain = _pivot(sequence, chain, i, j, rng)
        if new_chain is None:
            continue
        new_energy = _energy(sequence, new_chain)
        if new_energy < energy:
            chain, energy = new_chain, new_energy
    return chain, energy


def fold(sequence, dimension):
    """Best conformation over several greedy growths each refined by local search."""
    n = len(sequence)
    rng = random.Random(12345 + n)
    best, best_energy = None, 1
    n_grow = 40
    for _ in range(n_grow):
        if time.time() > _DEADLINE - 5.0:
            break
        chain = _grow(sequence, rng)
        if chain is None:
            continue
        value = _energy(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value
    if best is None:
        best = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best)
    # collect a few distinct alternates to refine
    seen = {tuple(best)}
    candidates = []
    rng2 = random.Random(777 + n)
    for _ in range(n_grow):
        if time.time() > _DEADLINE - 5.0:
            break
        chain = _grow(sequence, rng2)
        if chain is None:
            continue
        key = tuple(chain)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(chain)
    to_refine = [best] + candidates
    for idx, chain in enumerate(to_refine):
        if time.time() > _DEADLINE - 3.0:
            break
        rem = _DEADLINE - 3.0 - time.time()
        b = rem * 0.6 if idx == 0 else rem * 0.05
        if b < 0.2:
            break
        chain2, e2 = _refine(sequence, chain, b)
        if e2 < best_energy:
            best, best_energy = chain2, e2
    if best is None:
        best = [(i, 0) for i in range(n)]
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
