# EVOLVE-BLOCK-START
import time
import random
import sys

_TIME_BUDGET = 315.0  # safety margin below the 330s wall-clock kill
_DEADLINE = time.monotonic() + _TIME_BUDGET
_TIME_CHECK = 250


def _check():
    return time.monotonic() > _DEADLINE


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


def _grow(sequence, dimension, rng):
    """Greedy chain growth preferring new H-H contacts, then open space."""
    n = len(sequence)
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, n):
        cur = chain[-1]
        cands = []
        for site in _neighbours(cur):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == "H":
                        gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            cands.append((gain * 10 + free + rng.random(), site))
        if not cands:
            return None
        cands.sort(reverse=True)
        k = min(len(cands), 3)
        _, site = cands[rng.randrange(k)]
        occupied[site] = i
        chain.append(site)
    return chain


def _move(sequence, chain, pos, site):
    """Return (delta, new_chain) if residue pos can move to empty site, else None.

    delta = new_energy - old_energy (both negative)."""
    if pos == 0 or pos == len(chain) - 1:
        return None
    p = chain[pos]
    if site == p:
        return None
    if site in chain:
        return None
    a = pos - 1
    b = pos + 1
    if site == chain[a] or site == chain[b]:
        return None
    s = sequence
    isH = s[pos] == "H"
    delta = 0
    if isH:
        # break contacts between pos and every H within lattice distance 1
        for q in _neighbours(p):
            j = chain.index(q) if q in chain else -1
            if j >= 0 and j != a and j != b and s[j] == "H":
                delta += 1
        # form new contacts
        for q in _neighbours(site):
            j = chain.index(q) if q in chain else -1
            if j >= 0 and j != a and j != b and s[j] == "H":
                delta -= 1
    new_chain = list(chain)
    new_chain[pos] = site
    return delta, new_chain


def _local_search(sequence, chain, max_iter, rng):
    n = len(chain)
    if n < 4:
        return chain
    energy = _energy(sequence, chain)
    it = 0
    no_improve = 0
    while it < max_iter and no_improve < 60:
        if _check():
            break
        it += 1
        pos = rng.randrange(1, n - 1)
        p = chain[pos]
        for site in _neighbours(p):
            res = _move(sequence, chain, pos, site)
            if res is None:
                continue
            delta, new_chain = res
            if delta < 0:
                chain = new_chain
                energy += delta
                no_improve = 0
                break
        else:
            no_improve += 1
    return chain


def fold(sequence, dimension):
    n = len(sequence)
    rng = random.Random(12345 + n)
    best = None
    best_energy = 1
    total_iters = 0
    max_iter_per = 400
    while True:
        if _check():
            break
        chain = _grow(sequence, dimension, rng)
        if chain is None:
            if total_iters % 40 == 0 and _check():
                break
            continue
        chain = _local_search(sequence, chain, max_iter_per, rng)
        total_iters += max_iter_per
        value = _energy(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value
        # adapt the per-cycle budget to the remaining time
        remaining = _DEADLINE - time.monotonic()
        est = max(0.001, total_iters / (max(1e-9, time.monotonic() - (_DEADLINE - _TIME_BUDGET))))
        max_iter_per = max(30, int(remaining * 8 / max(1, est / 400)))
        if total_iters > 200000 and remaining < 5.0:
            break
    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
