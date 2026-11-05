"""HP lattice protein folding: diversified greedy seeds + fast incremental local search."""
# EVOLVE-BLOCK-START
import random
import time

# Global wall-clock deadline with a safety margin (evaluator kills at 210s).
_DEADLINE = time.time() + 200.0


def _neighbours(p):
    return ((p[0] - 1, p[1]), (p[0] + 1, p[1]), (p[0], p[1] - 1), (p[0], p[1] + 1))


def _greedy_grow(sequence, rng, diversity=0.5):
    """One randomized greedy chain growth: prefer new H-H contacts, then free space."""
    n = len(sequence)
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    h = {start: 1 if sequence[0] == 'H' else 0}
    for i in range(1, n):
        best = None
        best_score = None
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and j < i - 1 and sequence[j] == 'H':
                        gain += 1
            free = 0
            for q in _neighbours(site):
                if q not in occupied:
                    free += 1
            score = gain + 0.1 * free + diversity * rng.random()
            if best_score is None or score > best_score:
                best_score = score
                best = site
        if best is None:
            return None
        chain.append(best)
        occupied[best] = i
        h[best] = 1 if sequence[i] == 'H' else 0
    return chain


def _local_search(sequence, chain, iters, rng):
    """Fast incremental pivot-move local search. Returns best chain and energy."""
    n = len(chain)
    if n < 3:
        return chain, 0
    pos = list(chain)
    pt = {p: i for i, p in enumerate(pos)}
    h = [1 if sequence[i] == 'H' else 0 for i in range(n)]
    hp = {p for p, v in zip(pos, h) if v}

    def contacts_at(p):
        c = 0
        for q in _neighbours(p):
            j = pt.get(q)
            if j is not None and h[j]:
                c += 1
        return c

    energy = -sum(contacts_at(p) for p in hp) // 2

    best_chain = list(pos)
    best_energy = energy
    it = 0
    while it < iters and time.time() < _DEADLINE:
        it += 1
        if n > 4:
            i = rng.randint(2, n - 3)
        else:
            i = 1
        old = pos[i]
        for d in rng.sample(_neighbours(old), 4):
            if d in pt:
                continue
            # remove old site
            pt.pop(old)
            h[old] = 0
            hp.discard(old)
            e_old = energy
            energy -= contacts_at(old)
            # place at new site
            pt[d] = i
            pos[i] = d
            h[d] = 1
            if h[i]:
                hp.add(d)
            e_new = energy
            energy += contacts_at(d)
            if energy <= e_new:
                if energy < best_energy:
                    best_energy = energy
                    best_chain = list(pos)
            else:
                # revert
                pos[i] = old
                pt.pop(d)
                h[d] = 0
                hp.discard(d)
                energy -= contacts_at(d)
                pt[old] = i
                h[old] = 1
                if h[i]:
                    hp.add(old)
                energy += contacts_at(old)
    return best_chain, best_energy


def fold(sequence, dimension):
    """Best conformation over diversified greedy seeds refined by local search."""
    n = len(sequence)
    # Per-sequence budget: grow over sequences, keep a final safety margin.
    seq_budget = min(16.0, max(1.0, (_DEADLINE - time.time()) / 2.0 - 0.5))
    seq_deadline = time.time() + seq_budget

    rng = random.Random(12345 + len(sequence))
    best = None
    best_energy = 0

    # Try a handful of greedy seeds, refining each with local search.
    seeds = 6
    for s in range(seeds):
        if time.time() >= seq_deadline:
            break
        diversity = 0.5 * (s + 1)
        chain = _greedy_grow(sequence, rng, diversity)
        if chain is None:
            continue
        chain, energy = _local_search(sequence, chain, 2000, rng)
        if best is None or energy < best_energy:
            best, best_energy = chain, energy

    if best is None:
        # A straight line is always self-avoiding.
        best = [(i, 0) for i in range(n)]

    return [list(p) for p in best]
# EVOLVE-BLOCK-END
