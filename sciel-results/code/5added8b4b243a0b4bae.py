"""HP lattice protein folding: randomized greedy growth + local search refinement."""
# EVOLVE-BLOCK-START
import random
import time

_T0 = time.perf_counter()
# Wall-clock safety margin: stop searching well before the evaluator kills us.
_DEADLINE = 185.0


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain):
    """Exact HP contact energy: -1 per H-H pair on neighbouring sites, not adjacent in chain."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _delta_energy(sequence, chain, where, i, old, new):
    """Energy change of moving residue i from `old` to `new` (new must be free)."""
    if sequence[i] != "H":
        return 0
    delta = 0
    # remove contacts i made at `old`
    for q in _neighbours(old):
        j = where.get(q)
        if j is not None and j != i and j != i - 1 and j != i + 1 and sequence[j] == "H":
            delta += 1
    # add contacts i makes at `new`
    for q in _neighbours(new):
        j = where.get(q)
        if j is not None and j != i and j != i - 1 and j != i + 1 and sequence[j] == "H":
            delta -= 1
    return delta


def _grow(sequence, dimension, rng, variant):
    """Randomized greedy growth. Returns (chain, where, energy) or None if it traps."""
    n = len(sequence)
    if n <= 2:
        chain = [(0,) * dimension for _ in range(n)]
        where = {p: i for i, p in enumerate(chain)}
        return chain, where, 0

    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    # Choose an initial direction to give variety.
    dirs = list(_neighbours(start))
    rng.shuffle(dirs)

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
            # variant 0: contact-first; variant 1: more exploration via free-space weight
            if variant == 0:
                score = gain + 0.05 * free + rng.random()
            else:
                score = gain + 0.35 * free + rng.random()
            options.append((score, site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)

    where = dict(occupied)
    energy = _energy(sequence, chain)
    return chain, where, energy


def _local_search(sequence, chain, where, energy, rng, time_budget):
    """Refine with move + flip moves under simulated annealing. Returns (chain, where, energy)."""
    n = len(chain)
    if n < 4:
        return chain, where, energy

    best_chain = list(chain)
    best_where = dict(where)
    best_energy = energy

    # Temperature schedule scaled to the problem size.
    t_start = max(0.6, 0.08 * n)
    t_end = 0.02
    # Estimate a per-iteration cost to pick a step count that fits the budget.
    step_cost = 4e-5 * n + 2e-4  # rough seconds per iteration
    max_iter = max(2000, int(time_budget / step_cost)) if step_cost > 0 else 100000
    max_iter = min(max_iter, 200000)

    for it in range(max_iter):
        if time.perf_counter() - _T0 > _DEADLINE:
            break
        if it and (it % 1000 == 0):
            temp = t_start * (t_end / t_start) ** (it / max_iter)
        else:
            temp = t_start * (t_end / t_start) ** (it / max_iter)

        accepted = False
        # Flip move: reverse a 2-residue segment (i, i+1).
        if n > 3 and rng.random() < 0.5:
            i = rng.randint(1, n - 3)
            a = chain[i]
            b = chain[i + 1]
            # After flip, residue i goes to b and i+1 goes to a.
            # Check self-avoidance: b must not collide with chain[i-1]; a must not collide with chain[i+2].
            ok = True
            if b == chain[i - 1] or a == chain[i + 2]:
                ok = False
            if ok:
                d = _delta_energy(sequence, chain, where, i, a, b) + \
                    _delta_energy(sequence, chain, where, i + 1, b, a)
                if d <= 0 or rng.random() < min(1.0, 0.5 * (-d) / (temp + 1e-9)):
                    # commit
                    chain[i], chain[i + 1] = b, a
                    where[a] = i + 1
                    where[b] = i
                    energy += d
                    accepted = True
                    if energy < best_energy:
                        best_energy = energy
                        best_chain = list(chain)
                        best_where = dict(where)
        # Move move: shift a non-endpoint residue to an adjacent free cell.
        if not accepted and n > 3:
            i = rng.randint(1, n - 2)
            old = chain[i]
            # candidate neighbours that are not the two chain neighbours
            cands = [q for q in _neighbours(old) if q != chain[i - 1] and q != chain[i + 1]]
            if cands:
                new = rng.choice(cands)
                if new not in where:
                    d = _delta_energy(sequence, chain, where, i, old, new)
                    if d <= 0 or rng.random() < min(1.0, 0.5 * (-d) / (temp + 1e-9)):
                        chain[i] = new
                        where.pop(old)
                        where[new] = i
                        energy += d
                        accepted = True
                        if energy < best_energy:
                            best_energy = energy
                            best_chain = list(chain)
                            best_where = dict(where)

    # Final exact sanity check on the best conformation.
    exact = _energy(sequence, best_chain)
    if exact < best_energy:
        best_energy = exact
    return best_chain, best_where, best_energy


def fold(sequence, dimension):
    """Best conformation found under a strict wall-clock deadline."""
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]

    # Per-sequence budget, leaving headroom across all sequences.
    seq_budget = 16.0
    rng = random.Random(12345 + n)

    best_chain = None
    best_where = None
    best_energy = 0

    # 1) Best of several randomized greedy constructions (a few variants).
    for attempt in range(6):
        if time.perf_counter() - _T0 > _DEADLINE:
            break
        variant = attempt % 2
        res = _grow(sequence, dimension, rng, variant)
        if res is None:
            continue
        chain, where, energy = res
        if best_chain is None or energy < best_energy:
            best_chain, best_where, best_energy = chain, where, energy

    if best_chain is None:
        # Fallback: a straight line is always self-avoiding.
        best_chain = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
        best_where = {p: i for i, p in enumerate(best_chain)}
        best_energy = _energy(sequence, best_chain)

    # 2) Local-search refinement on the best construction.
    if time.perf_counter() - _T0 < _DEADLINE:
        remaining = _DEADLINE - (time.perf_counter() - _T0) - 1.0
        if remaining > 0.2:
            best_chain, best_where, best_energy = _local_search(
                sequence, best_chain, best_where, best_energy, rng, remaining)

    # 3) If time remains, try refining a couple of alternate constructions too.
    if time.perf_counter() - _T0 < _DEADLINE:
        for attempt in range(2):
            if time.perf_counter() - _T0 > _DEADLINE:
                break
            res = _grow(sequence, dimension, rng, (attempt + 1) % 2)
            if res is None:
                continue
            chain, where, energy = res
            if energy < best_energy:
                remaining = _DEADLINE - (time.perf_counter() - _T0) - 1.0
                if remaining > 0.2:
                    chain, where, energy = _local_search(
                        sequence, chain, where, energy, rng, remaining)
                if energy < best_energy:
                    best_chain, best_where, best_energy = chain, where, energy

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
