import time
import random
from math import isqrt
from collections import Counter

N = 5000
DEADLINE = 305.0  # safety margin before 330s kill

def _primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, isqrt(limit) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(sieve[p * p::p]))
    return [p for p in range(limit + 1) if sieve[p]]

ALL_PRIMES = _primes(N)
PRIME_SET = set(ALL_PRIMES)


def _admissible_from_counts(counters, size):
    """True iff every prime p <= size has an empty residue class."""
    for p in ALL_PRIMES:
        if p > size:
            break
        c = counters.get(p)
        if c is None or len(c) < p:
            continue  # some residue class empty -> OK
        return False
    return True


def _greedy_construct(deadline, seed):
    """Build a large admissible set by random greedy insertion."""
    rnd = random.Random(seed)
    counters = {}
    points = set()
    remaining = list(range(N + 1))
    rnd.shuffle(remaining)
    idx = 0
    while idx < len(remaining):
        if time.time() > deadline:
            break
        x = remaining[idx]
        idx += 1
        new_size = len(points) + 1
        ok = True
        for p in ALL_PRIMES:
            if p > new_size:
                break
            c = counters.get(p)
            if c is None:
                c = counters[p] = Counter()
            r = x % p
            if c[r] == 0 and len(c) == p - 1:
                ok = False
                break
        if ok:
            points.add(x)
            for p in ALL_PRIMES:
                if p > new_size:
                    break
                c = counters.get(p)
                if c is None:
                    c = counters[p] = Counter()
                c[x % p] += 1
    return points, counters


def _local_search(points, counters, deadline, seed):
    """Swap-based local search to grow the admissible set."""
    rnd = random.Random(seed)
    pts = set(points)
    ctrs = {p: Counter(c) for p, c in counters.items()}
    size = len(pts)
    iters = 0
    while time.time() < deadline:
        iters += 1
        tight = []
        for p in ALL_PRIMES:
            if p > size:
                break
            c = ctrs.get(p)
            if c is not None and len(c) == p:
                tight.append(p)
        if not tight:
            break
        p = rnd.choice(tight)
        c = ctrs[p]
        m = max(c.values())
        top = [r for r, v in c.items() if v == m]
        r = rnd.choice(top)
        # candidate points to remove: in pts with residue r mod p
        rem = [x for x in pts if x % p == r]
        if not rem:
            continue
        rnd.shuffle(rem)
        best = None
        for x in rem:
            ns = size - 1
            ok = True
            for q in ALL_PRIMES:
                if q > ns:
                    break
                cq = ctrs.get(q)
                if cq is None:
                    continue
                if cq[x % q] == 1 and len(cq) == q:
                    ok = False
                    break
            if not ok:
                continue
            # simulate removal then greedy add
            for q in ALL_PRIMES:
                if q > ns:
                    break
                cq = ctrs.get(q)
                if cq is None:
                    continue
                cq[x % q] -= 1
                if cq[x % q] == 0:
                    del cq[x % q]
            # greedy add
            added = 0
            while True:
                ns2 = len(pts) + 1
                bestadd = None
                bestkey = None
                for y in range(N + 1):
                    if y in pts:
                        continue
                    ok2 = True
                    relax = 0
                    for q in ALL_PRIMES:
                        if q > ns2:
                            break
                        cq = ctrs.get(q)
                        if cq is None:
                            continue
                        ry = y % q
                        if cq[ry] == 0 and len(cq) == q - 1:
                            ok2 = False
                            break
                        if len(cq) == q:
                            relax += 1
                    if not ok2:
                        continue
                    key = (ns2, -relax)
                    if bestkey is None or key < bestkey:
                        bestkey = key
                        bestadd = y
                        if ns2 == size:
                            break
                if bestadd is None:
                    break
                for q in ALL_PRIMES:
                    if q > ns2:
                        break
                    cq = ctrs.get(q)
                    if cq is None:
                        cq = ctrs[q] = Counter()
                    cq[bestadd % q] += 1
                pts.add(bestadd)
                added += 1
            new_size = len(pts)
            # commit or rollback
            if added > 0:
                size = new_size
                continue
            else:
                # rollback
                pts.discard(x)
                for q in ALL_PRIMES:
                    if q > size:
                        break
                    cq = ctrs.get(q)
                    if cq is None:
                        continue
                    cq[x % q] += 1
                if iters > 400:
                    break
    return pts


def construct():
    start = time.time()
    deadline = start + DEADLINE
    best = None
    best_size = 0
    seed = 12345
    while True:
        if time.time() > deadline - 3:
            break
        remaining_budget = deadline - time.time()
        # give each restart a proportional slice
        slice_deadline = time.time() + min(remaining_budget * 0.5, 40.0)
        pts, counters = _greedy_construct(slice_deadline, seed)
        if not pts:
            seed += 1
            continue
        # local search
        opt_deadline = min(deadline - 1.0, time.time() + max(5.0, remaining_budget * 0.4))
        pts = _local_search(pts, counters, opt_deadline, seed)
        if len(pts) > best_size:
            best_size = len(pts)
            best = pts
        seed += 1
        if time.time() > deadline - 2:
            break
    if best is None:
        best = {0}
    return list(best)
