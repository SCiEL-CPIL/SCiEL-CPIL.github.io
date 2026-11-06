"""Cap set: produce a vector ordering whose greedy cap-set insertion is large.

Structural change from the seed: instead of a static per-vector heuristic, we
PRECOMPUTE a maximal cap set (fast random greedy insertion using integer
encodings + a set for O(1) line membership), then place its members at the
FRONT of the ordering (so the evaluator's greedy pass reproduces that maximal
set) and append the remaining vectors.  We run many random restarts and
hill-climb the seed to find a large maximal set, with a deterministic fallback
and an explicit wall-clock deadline guard.
"""
import time
import random
import itertools

_DEADLINE = 325.0          # safety margin below the 330 s kill
_DEADLINE_START = None
_CACHE = {}                # (vector, n) -> priority float


def _elapsed():
    global _DEADLINE_START
    if _DEADLINE_START is None:
        _DEADLINE_START = time.time()
    return time.time() - _DEADLINE_START


def _deadline():
    return _elapsed() < _DEADLINE


def _encode(v):
    e = 0
    for x in v:
        e = e * 3 + x
    return e


def _all_vectors(n):
    return list(itertools.product((0, 1, 2), repeat=n))


def _maximal_greedy(n, seed, vectors, encs):
    """Random greedy insertion.  vectors: list of tuples; encs: list of int
    encodings aligned with vectors.  Returns (chosen_indices, size)."""
    rng = random.Random(seed)
    order = list(range(len(vectors)))
    rng.shuffle(order)
    present = set()
    chosen = []
    for idx in order:
        e = encs[idx]
        v = vectors[idx]
        bad = False
        # a line through v and any existing u requires c = (3-v-u) mod 3 present
        for u in chosen:
            # compute encoding of c quickly using digit-wise op
            c = 0
            for k in range(n):
                c = c * 3 + (3 - v[k] - u[k]) % 3
            if c in present:
                bad = True
                break
        if not bad:
            chosen.append(v)
            present.add(e)
    return chosen, len(chosen)


def _maximal_greedy_fast(n, seed, vectors, encs, digits):
    """digits: list of digit arrays (tuples) for fast c-encoding."""
    rng = random.Random(seed)
    order = list(range(len(vectors)))
    rng.shuffle(order)
    present = set()
    chosen_digits = []
    size = 0
    for idx in order:
        e = encs[idx]
        d = digits[idx]
        bad = False
        for ud in chosen_digits:
            c = 0
            for k in range(n):
                c = c * 3 + (3 - d[k] - ud[k]) % 3
            if c in present:
                bad = True
                break
        if not bad:
            chosen_digits.append(d)
            present.add(e)
            size += 1
    return size


def _deterministic_capset(n, vectors, encs, digits):
    """Greedy over vectors with coordinate sum == 0 mod 3 (a natural cap-set
    pool) in lexicographic order.  Returns list of tuples."""
    pool = [i for i, v in enumerate(vectors) if sum(v) % 3 == 0]
    present = set()
    chosen = []
    for idx in pool:
        e = encs[idx]
        d = digits[idx]
        bad = False
        for ud in chosen:
            c = 0
            for k in range(n):
                c = c * 3 + (3 - d[k] - ud[k]) % 3
            if c in present:
                bad = True
                break
        if not bad:
            chosen.append(d)
            present.add(e)
    return [tuple(x) for x in chosen]


def _compute(n):
    vectors = _all_vectors(n)
    encs = [_encode(v) for v in vectors]
    digits = [list(v) for v in vectors]
    pos = {}
    # deterministic fallback first (always available, small cost)
    capset = _deterministic_capset(n, vectors, encs, digits)
    best_size = len(capset)
    best_capset = capset
    # random restarts + hill climb
    base_seed = 12345
    tries = 0
    while _deadline() and tries < 50000:
        seed = (base_seed + tries) % (10 ** 9)
        sz = _maximal_greedy_fast(n, seed, vectors, encs, digits)
        if sz > best_size:
            # rebuild the actual capset for this winning seed
            rng = random.Random(seed)
            order = list(range(len(vectors)))
            rng.shuffle(order)
            present = set()
            chosen = []
            for idx in order:
                e = encs[idx]
                d = digits[idx]
                bad = False
                for ud in chosen:
                    c = 0
                    for k in range(n):
                        c = c * 3 + (3 - d[k] - ud[k]) % 3
                    if c in present:
                        bad = True
                        break
                if not bad:
                    chosen.append(d)
                    present.add(e)
            best_size = sz
            best_capset = [tuple(x) for x in chosen]
            base_seed = seed
        tries += 1
    capset_set = set(best_capset)
    ordering = list(best_capset) + [v for v in vectors if v not in capset_set]
    for i, v in enumerate(ordering):
        pos[v] = float(len(ordering) - i)
    return pos


def priority(vector, n):
    key = (vector, n)
    if key not in _CACHE:
        _CACHE[key] = _compute(n)
    return _CACHE[key][vector]


__all__ = ["priority"]
