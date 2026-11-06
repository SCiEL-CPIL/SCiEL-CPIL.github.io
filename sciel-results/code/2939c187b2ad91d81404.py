"""HP lattice protein folding: beam-search growth + pivot local polish."""
# EVOLVE-BLOCK-START
import random
import time

# --- global time budget shared across all fold() calls (evaluator kills at 210s) ---
_START = None
_GLOBAL_DEADLINE = None
_GLOBAL_SPAN = 205.0      # usable budget for all 10 sequences combined
_PER_CALL_CAP = 18.0      # no single call hogs the budget
_MARGIN = 2.0             # safety margin before the hard kill


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _energy(sequence, chain, isH=None):
    if isH is None:
        isH = [c == "H" for c in sequence]
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if isH[i]:
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and isH[j]:
                    contacts += 1
    return -contacts


def _beam_grow(sequence, isH, dimension, rng, width):
    """Grow the chain keeping a beam of the top `width` partial conformations.
    Score = H-H contacts so far + free-space term + small noise (diversity)."""
    n = len(sequence)
    start = (0,) * dimension
    # beam entries: (score, chain(list), occupied(dict))
    beam = [(0.0, [start], {start: 0})]
    for i in range(1, n):
        cands = []
        for score, chain, occupied in beam:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                gain = 0
                if isH[i]:
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and j < i - 1 and isH[j]:
                            gain += 1
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                s = gain + 0.15 * free + 0.3 * rng.random()
                cands.append((s, chain + [site], occupied, site))
        if not cands:
            return None
        cands.sort(key=lambda x: x[0], reverse=True)
        beam = []
        for s, chain, occ, site in cands[:width]:
            new_occ = occ.copy()
            new_occ[site] = i
            beam.append((s, chain, new_occ))
    return beam[0][1]  # best full conformation from the beam


def _polish(sequence, isH, chain, rng, iters, dimension):
    """Local search via pivot moves (rigid +/-90 tail rotation). Returns (chain, energy)."""
    if dimension != 2:
        return chain, _energy(sequence, chain, isH)
    n = len(chain)
    best = [tuple(p) for p in chain]
    best_e = _energy(sequence, best, isH)
    no_improve = 0
    for _ in range(iters):
        h = rng.randint(1, n - 2)
        dirn = rng.choice((-1, 1))
        head = best[:h + 1]
        tail = best[h + 1:]
        hx, hy = head[-1]
        head_set = set(head)
        new_tail = []
        ok = True
        for (x, y) in tail:
            dx, dy = x - hx, y - hy
            if dirn == 1:
                nx, ny = hx - dy, hy + dx
            else:
                nx, ny = hx + dy, hy - dx
            if (nx, ny) in head_set:
                ok = False
                break
            new_tail.append((nx, ny))
        if not ok:
            no_improve += 1
            if no_improve > 300:
                break
            continue
        cand = head + new_tail
        e = _energy(sequence, cand, isH)
        if e < best_e:
            best, best_e = cand, e
            no_improve = 0
        elif e == best_e and rng.random() < 0.3:
            best = cand  # neutral move to explore
    return best, best_e


def fold(sequence, dimension):
    """Best conformation via beam-search growth restarts + pivot local polish."""
    global _START, _GLOBAL_DEADLINE
    if _START is None:
        _START = time.monotonic()
        _GLOBAL_DEADLINE = _START + _GLOBAL_SPAN
    call_start = time.monotonic()
    call_end = min(call_start + _PER_CALL_CAP, _GLOBAL_DEADLINE - _MARGIN)

    rng = random.Random(12345 + len(sequence))
    isH = [c == "H" for c in sequence]
    n = len(sequence)
    width = 12
    polish_iters = 400

    best, best_energy = None, 0
    while time.monotonic() < call_end - 1.0:
        chain = _beam_grow(sequence, isH, dimension, rng, width)
        if chain is None:
            continue
        if n > 2:
            chain, e = _polish(sequence, isH, chain, rng, polish_iters, dimension)
        else:
            e = _energy(sequence, chain, isH)
        if e < best_energy:
            best, best_energy = chain, e

    # final high-precision polish pass on the single best if time remains
    if best is not None and time.monotonic() < call_end - 0.3 and n > 2:
        best, best_energy = _polish(sequence, isH, best, rng, 1500, dimension)

    if best is None:  # a straight line is always self-avoiding
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
