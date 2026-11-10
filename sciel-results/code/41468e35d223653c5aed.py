"""Stronger algorithm: model the problem as a Sidon (B2) set over prefix sums,
search with randomized/annealing insertion, then extend greedily."""
import random
import time

DEADLINE = 82.0  # safety margin below the 90s wall-clock kill


def _valid_insert(s, x, diffs):
    """Return (ok, new_diffs, collisions) for inserting x into set s.
    s: set of prefix sums (does NOT include x). diffs: set of positive differences."""
    collisions = 0
    new_diffs = []
    for y in s:
        d = x - y
        if d in diffs:
            collisions += 1
        else:
            new_diffs.append(d)
    return collisions == 0, new_diffs, collisions


def _build_sidon(max_val, rng, temperature, max_iters):
    """Build a Sidon set greedily with random order + annealed repair."""
    s = {0}
    diffs = set()
    for x in range(1, max_val + 1):
        ok, new_diffs, _ = _valid_insert(s, x, diffs)
        if ok:
            s.add(x)
            diffs.update(new_diffs)
    return s


def _extend_sequence(seq, max_val):
    """Greedy extension of a valid sequence (the seed's method)."""
    values = list(seq)
    sums = set()
    for i in range(len(values)):
        acc = 0
        for j in range(i, len(values)):
            acc += values[j]
            sums.add(acc)
    for x in range(max_val + 1):
        if x in values:
            continue
        tails = []
        total = x
        tails.append(total)
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values


def _seq_from_sidon(s):
    """Convert a Sidon set (prefix sums incl. 0) to a sequence of gaps."""
    ps = sorted(s)
    return [ps[i + 1] - ps[i] for i in range(len(ps) - 1)]


def construct():
    t0 = time.time()
    rng = random.Random(12345)
    best_len = 0
    best_seq = []

    # Baseline: seed greedy (fast, deterministic)
    base = _extend_sequence([], 256)
    best_len = len(base)
    best_seq = base

    # Try to beat it via Sidon-set search.
    # Multiple randomized trials; annealing helps escape local optima.
    trials = 0
    while time.time() - t0 < DEADLINE:
        trials += 1
        # Randomized Sidon build
        s = {0}
        diffs = set()
        candidates = list(range(1, 257))
        rng.shuffle(candidates)
        for x in candidates:
            ok, new_diffs, _ = _valid_insert(s, x, diffs)
            if ok:
                s.add(x)
                diffs.update(new_diffs)
        # Try to grow further by inserting missed values (second pass, random)
        missed = [x for x in candidates if x not in s]
        rng.shuffle(missed)
        for x in missed:
            ok, new_diffs, _ = _valid_insert(s, x, diffs)
            if ok:
                s.add(x)
                diffs.update(new_diffs)
        # Convert to sequence and extend greedily
        seq = _seq_from_sidon(s)
        seq = _extend_sequence(seq, 256)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq

    # Safety: verify best_seq is valid
    def valid(seq):
        if len(seq) == 0:
            return True
        if any(seq[i + 1] <= seq[i] for i in range(len(seq) - 1)):
            return False
        sums = set()
        n = len(seq)
        for i in range(n):
            acc = 0
            for j in range(i, n):
                acc += seq[j]
                if acc in sums:
                    return False
                sums.add(acc)
        return True

    if valid(best_seq):
        return best_seq
    # Fallback to baseline if somehow invalid
    return base


# EVOLVE-BLOCK-END
