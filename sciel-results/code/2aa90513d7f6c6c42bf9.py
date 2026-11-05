"""Hypergraph-MIS style solver for no-equal-gcd triples on [1,128]."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

N = 128
ALL = list(range(1, N + 1))


def _build_gcd_table():
    table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            table[i][j] = g
            table[j][i] = g
    return table


def _build_conflict_ds(gcd_table):
    """conflict_d[x] = frozenset of d>=2 such that x participates in some bad triple."""
    conflict_d = [None] * (N + 1)
    for x in range(1, N + 1):
        ds = set()
        for a in range(1, N + 1):
            if a == x:
                continue
            g = gcd_table[x][a]
            if g >= 2:
                for b in range(a + 1, N + 1):
                    if b == x:
                        continue
                    if gcd_table[x][b] == g and gcd_table[a][b] == g:
                        ds.add(g)
        conflict_d[x] = frozenset(ds)
    return conflict_d


def _bad_count_with(x, selected, gcd_table, conflict_d):
    """Number of bad triples involving x given selected (x not in selected)."""
    ds = conflict_d[x]
    if not ds:
        return 0
    cnt = 0
    for a in selected:
        for b in selected:
            if a >= b:
                continue
            g = gcd_table[a][b]
            if g in ds and gcd_table[x][a] == g and gcd_table[x][b] == g:
                cnt += 1
    return cnt


def _greedy_add(base, conflict_d, gcd_table, order):
    """Add every candidate that doesn't create a bad triple, in order."""
    selected = list(base)
    selset = set(selected)
    for x in order:
        if x in selset:
            continue
        if _bad_count_with(x, selected, gcd_table, conflict_d) == 0:
            selected.append(x)
            selset.add(x)
    return selected


def _local_search(selected, gcd_table, conflict_d, deadline, seed):
    """Add blocked elements by removing the cheapest offending member."""
    selected = list(selected)
    selset = set(selected)
    rng = random.Random(seed)
    candidates = [x for x in ALL if x not in selset]
    rng.shuffle(candidates)

    def _triples_of(y, sel):
        ds = conflict_d[y]
        if not ds:
            return []
        out = []
        for a in sel:
            for b in sel:
                if a >= b:
                    continue
                g = gcd_table[a][b]
                if g in ds and gcd_table[y][a] == g and gcd_table[y][b] == g:
                    out.append((a, b))
        return out

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in list(candidates):
            if time.time() >= deadline:
                break
            if x in selset:
                continue
            tris = _triples_of(x, selected)
            if not tris:
                selected.append(x)
                selset.add(x)
                candidates.remove(x)
                improved = True
                continue
            # find a member whose removal clears all these triples and is cheapest
            # cost(y) = number of bad triples y is in (with current selected, excluding x)
            involved = set()
            for (a, b) in tris:
                involved.add(a)
                involved.add(b)
            best_y = None
            best_cost = None
            for y in involved:
                # count bad triples containing y (without x)
                dsy = conflict_d[y]
                c = 0
                for a in selected:
                    if a == y:
                        continue
                    for b in selected:
                        if b <= a or b == y:
                            continue
                        g = gcd_table[a][b]
                        if g in dsy and gcd_table[y][a] == g and gcd_table[y][b] == g:
                            c += 1
                if best_cost is None or c < best_cost:
                    best_cost = c
                    best_y = y
            # removing best_y and adding x: net change in size is 0 (remove 1 add 1)
            # only do it if it reduces total bad triples overall (i.e. best_y is "worse")
            # we accept the swap if best_y's removal eliminates >= the triples x would create
            # Since x is not in set, removing best_y removes best_cost triples;
            # adding x after removal creates 0 (by construction). Net bad-triple delta = -best_cost.
            # Size unchanged, but we may unblock future adds. Accept if best_cost>0.
            if best_y is None or best_cost <= 0:
                continue
            selected.remove(best_y)
            selset.discard(best_y)
            selected.append(x)
            selset.add(x)
            candidates.remove(x)
            improved = True

    return selected


def construct():
    start = time.time()
    deadline = start + 78.0  # safety margin before 90s kill

    gcd_table = _build_gcd_table()
    conflict_d = _build_conflict_ds(gcd_table)

    best_set = [1]
    best_size = 1

    # Precompute prime membership and parity for seeds
    def has_prime(x, p):
        return x % p == 0

    # Candidate seed families
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]

    def make_seed(family):
        s = [x for x in ALL if family(x)]
        return s

    families = []
    for p in primes:
        families.append(lambda x, p=p: has_prime(x, p))
    families.append(lambda x: x % 2 == 0)
    families.append(lambda x: x % 3 == 0)
    families.append(lambda x: x % 2 == 0 or x % 3 == 0)
    families.append(lambda x: x % 2 == 0 or x % 5 == 0)
    families.append(lambda x: x % 3 == 0 or x % 5 == 0)
    families.append(lambda x: x % 2 == 0 or x % 3 == 0 or x % 5 == 0)
    families.append(lambda x: x >= 64)
    families.append(lambda x: x <= 64)
    families.append(lambda x: x % 4 == 0)
    families.append(lambda x: x % 6 == 0)
    families.append(lambda x: x % 2 == 0 and x % 3 != 0)
    families.append(lambda x: x % 3 == 0 and x % 2 != 0)

    rng = random.Random(12345)

    # Order strategies for greedy add
    def orders():
        base = list(ALL)
        yield base
        r = base[:]
        rng.shuffle(r)
        yield r
        yield sorted(ALL, key=lambda x: len(conflict_d[x]))
        r = base[:]
        rng.shuffle(r)
        yield r
        yield sorted(ALL, key=lambda x: -len(conflict_d[x]))

    attempt = 0
    for fam in families:
        if time.time() >= deadline - 20:
            break
        seed = make_seed(fam)
        # ensure seed itself is valid (it should be, but guard)
        valid = True
        for a, b, c in combinations(seed, 3):
            if gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]:
                valid = False
                break
        if not valid:
            continue

        for od in orders():
            if time.time() >= deadline - 15:
                break
            attempt += 1
            gset = _greedy_add(seed, conflict_d, gcd_table, od)
            if len(gset) > best_size:
                best_set = gset[:]
                best_size = len(gset)
            # local search
            if time.time() < deadline - 8:
                ls = _local_search(gset, gcd_table, conflict_d, deadline - 8, seed=attempt)
                if len(ls) > best_size:
                    best_set = ls[:]
                    best_size = len(ls)

    # Final local search on the best set found
    if time.time() < deadline - 2:
        ls = _local_search(best_set, gcd_table, conflict_d, deadline - 2, seed=99999)
        if len(ls) > best_size:
            best_set = ls
            best_size = len(ls)

    if not best_set:
        best_set = [1]
    return best_set
# EVOLVE-BLOCK-END
