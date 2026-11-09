"""Construct a strictly increasing sequence in [1,256] whose non-empty
consecutive-block sums are all distinct, maximizing length.

Design: keep the seed's constructive family (pick increasing values, keep all
consecutive-block sums distinct) but replace the single greedy pass with a
backtracking search plus randomized restarts and branch ordering.  The greedy
pass alone stalls at length 39; backtracking lets the search escape the local
trap.  An explicit wall-clock deadline guard returns the best sequence found
with a safety margin.
"""
import time
import random

DEADLINE = 80.0          # seconds; evaluator kills at 90s, keep a margin
MAXV = 256               # upper bound on each value
TARGET = 40              # known-feasible length; aim to prove it


def _extend(seq, sums):
    """Return the smallest integer > seq[-1] (<= MAXV) that keeps every
    non-empty consecutive-block sum distinct, or None if none exists.
    `sums` is the set of all block sums for `seq`."""
    last = seq[-1]
    n = len(seq)
    # Candidate block sums if we append x: x, x+seq[-1], ..., x+sum(seq)
    # i.e. x + prefix_sums where prefix_sums = [0, seq[-1], seq[-1]+seq[-2], ...]
    # We precompute the suffix sums of seq.
    suffix = [0] * (n + 1)
    s = 0
    for i in range(n - 1, -1, -1):
        s += seq[i]
        suffix[n - i] = s
    # suffix[k] for k in 1..n is the sum of the last k elements; suffix[0]=0
    cand_offsets = suffix  # offsets 0..sum(seq)
    for x in range(last + 1, MAXV + 1):
        ok = True
        for off in cand_offsets:
            if (x + off) in sums:
                ok = False
                break
        if ok:
            return x
    return None


def _backtrack(seq, sums, target, deadline):
    """Depth-first backtracking trying to reach `target` length.
    Branch order: prefer the smallest admissible value (greedy bias) which
    tends to keep values small and leaves room.  Returns seq if it reaches
    target, else the best (longest) prefix found."""
    best = list(seq)
    best_len = len(seq)
    n = len(seq)
    if n >= target:
        return seq
    last = seq[-1]
    # Precompute offsets once per node for candidate-sum testing.
    suffix = [0] * (n + 1)
    s = 0
    for i in range(n - 1, -1, -1):
        s += seq[i]
        suffix[n - i] = s
    cand_offsets = suffix

    for x in range(last + 1, MAXV + 1):
        if time.time() >= deadline:
            break
        # Test admissibility
        ok = True
        for off in cand_offsets:
            if (x + off) in sums:
                ok = False
                break
        if not ok:
            continue
        # Commit
        new_sums = sums | {x + off for off in cand_offsets}
        new_seq = seq + [x]
        if len(new_seq) > best_len:
            best = list(new_seq)
            best_len = len(new_seq)
        if len(new_seq) >= target:
            return new_seq
        res = _backtrack(new_seq, new_sums, target, deadline)
        if len(res) >= target:
            return res
        if len(res) > best_len:
            best = res
            best_len = len(res)
        if time.time() >= deadline:
            break
    return best


def construct():
    start = time.time()
    deadline = start + DEADLINE
    rng = random.Random(12345)

    best = []
    best_len = 0

    # Phase 1: plain greedy (deterministic, fast) to get a strong base.
    seq = []
    sums = set()
    while True:
        if time.time() >= deadline:
            break
        x = _extend(seq, sums)
        if x is None:
            break
        seq.append(x)
        sums |= {x + off for off in [0]}
        # recompute full sums for seq
        # (rebuild to be safe)
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        best = list(seq)
        best_len = len(seq)
        if best_len >= TARGET:
            return best
        if time.time() >= deadline:
            break

    # Phase 2: randomized restarts with backtracking to push past the greedy
    # plateau.  Each restart builds a short random prefix then backtracks.
    attempt = 0
    while time.time() < deadline - 2.0:
        attempt += 1
        # Random prefix length between 0 and min(6, best_len)
        plen = rng.randint(0, min(6, best_len))
        # Build a random increasing prefix
        prefix = []
        lo = 1
        for _ in range(plen):
            hi = MAXV - (plen - len(prefix))  # leave room
            if hi < lo:
                break
            v = rng.randint(lo, hi)
            prefix.append(v)
            lo = v + 1
        # Validate prefix block sums
        psums = set()
        ok = True
        n = len(prefix)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += prefix[j]
                if s in psums:
                    ok = False
                    break
                psums.add(s)
            if not ok:
                break
        if not ok:
            continue
        # Backtrack from this prefix
        res = _backtrack(prefix, psums, TARGET, deadline)
        if len(res) > best_len:
            best = res
            best_len = len(res)
            if best_len >= TARGET:
                return best

    return best


if __name__ == "__main__":
    out = construct()
    print(len(out), out)
