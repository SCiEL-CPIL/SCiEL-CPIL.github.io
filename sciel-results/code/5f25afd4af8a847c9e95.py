"""Rewritten program: maximum subset of [1,128] with no equal-lcm triple."""
import time
import random
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s kill

    N = 128
    # Precompute LCM table for [1,128]
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            lcm_table[i][j] = _lcm(i, j)

    # Find all bad triples: (a,b,c) with a<b<c and lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, N + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))

    # For each element, list of bad triples it participates in
    from collections import defaultdict
    triples_by_elem = defaultdict(list)
    for t in bad_triples:
        for x in t:
            triples_by_elem[x].append(t)

    # Greedy initial solution
    def greedy_order(order):
        selected = set()
        for x in order:
            # Check if adding x creates any bad triple with two already-selected
            ok = True
            for t in triples_by_elem[x]:
                others = [y for y in t if y != x]
                if len(set(others) & selected) == 2:
                    ok = False
                    break
            if ok:
                selected.add(x)
        return selected

    # Try a few greedy orders
    best = set()
    rng = random.Random(42)
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        [x for x in range(1, N + 1) if x % 2 == 0],  # evens first
        [x for x in range(1, N + 1) if x % 2 == 1],  # odds first
    ]
    for _ in range(4):
        o = list(range(1, N + 1))
        rng.shuffle(o)
        orders.append(o)

    for o in orders:
        s = greedy_order(o)
        if len(s) > len(best):
            best = s

    # Local search: hill climbing with random restarts and swaps
    def is_valid(s):
        for t in bad_triples:
            if len(set(t) & s) == 3:
                return False
        return True

    def try_improve(s, rng, time_limit):
        s = set(s)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try adding each non-member
            for x in range(1, N + 1):
                if x in s:
                    continue
                new_s = s | {x}
                # Check validity
                bad = None
                for t in triples_by_elem[x]:
                    others = [y for y in t if y != x]
                    if len(set(others) & s) == 2:
                        bad = t
                        break
                if bad is None:
                    s = new_s
                    improved = True
                    break
                else:
                    # Try removing one element from bad triple to allow x
                    for y in bad:
                        if y == x:
                            continue
                        if y in s:
                            cand = (s - {y}) | {x}
                            if is_valid(cand):
                                s = cand
                                improved = True
                                break
                    if improved:
                        break
            # Try removing and re-adding (swap)
            if not improved:
                for x in range(1, N + 1):
                    if x in s:
                        continue
                    for y in list(s):
                        cand = (s - {y}) | {x}
                        if is_valid(cand):
                            # Only accept if it doesn't reduce size (same size, may enable future)
                            # Accept to escape plateaus
                            s = cand
                            improved = True
                            break
                    if improved:
                        break
        return s

    # Multiple local search iterations with time budget
    best_size = len(best)
    best_set = set(best)
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        # Perturb best set
        s = set(best_set)
        # Randomly remove a few and add a few
        for _ in range(rng.randint(1, 3)):
            if s:
                rem = rng.choice(list(s))
                s.discard(rem)
                # Try to fill with random
                for _ in range(3):
                    x = rng.randint(1, N)
                    if x not in s:
                        if is_valid(s | {x}):
                            s.add(x)
        # Local improve
        s = try_improve(s, rng, min(deadline, time.time() + 5.0))
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            # Reset iteration counter implicitly by continuing

    # Final greedy polish on best_set
    # Try to add any remaining element
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, N + 1):
            if x in best_set:
                continue
            if is_valid(best_set | {x}):
                best_set.add(x)
                changed = True

    if not best_set:
        best_set = {1}

    return list(best_set)
