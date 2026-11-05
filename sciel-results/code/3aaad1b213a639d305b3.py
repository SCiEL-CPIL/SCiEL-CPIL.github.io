"""Structural rewrite: use a randomised local-search (flip) heuristic to find a large
valid 3-uniform hypergraph 2-colouring avoiding red K4^(3) and blue K5^(3).

Design:
- Start with a random colouring for a target n (try from 34 down to 10).
- Use a fast violation counter: for every 4-set track red-triple count (violation if 4),
  for every 5-set track blue-triple count (violation if 10).
- Flip random triples; accept if total violations decrease (or equal with small prob).
- Keep the best (largest n with 0 violations) found before a deadline.

We precompute for each triple the list of 4-sets and 5-sets it belongs to, so a flip
updates counters in O(1) per affected set.
"""
import time
import random
from itertools import combinations

# Deadline guard
_DEADLINE = time.time() + 140.0  # 10s safety margin under 150s kill
_BEST = {'n': 0, 'colors': None}


def _triple_list(n):
    """Return list of triples (a,b,c) in lex order and a mapping triple->index."""
    triples = list(combinations(range(n), 3))
    tmap = {t: i for i, t in enumerate(triples)}
    return triples, tmap


def _build_membership(n, triples, tmap):
    """
    For each 4-set, list the 4 triples it contains.
    For each 5-set, list the 10 triples it contains.
    """
    # 4-sets
    four_sets = list(combinations(range(n), 4))
    four_triples = []
    for fs in four_sets:
        ts = []
        for i in range(4):
            t = tuple(fs[j] for j in range(4) if j != i)
            ts.append(tmap[t])
        four_triples.append(ts)

    # 5-sets
    five_sets = list(combinations(range(n), 5))
    five_triples = []
    for fs in five_sets:
        ts = []
        for combo in combinations(range(5), 3):
            t = tuple(fs[j] for j in combo)
            ts.append(tmap[t])
        five_triples.append(ts)

    return four_sets, four_triples, five_sets, five_triples


def _count_violations(colors, four_triples, five_triples):
    """Count red K4 violations (all 4 triples red) + blue K5 violations (all 10 blue)."""
    v = 0
    for ts in four_triples:
        if all(colors[t] == 0 for t in ts):
            v += 1
    for ts in five_triples:
        if all(colors[t] == 1 for t in ts):
            v += 1
    return v


def _try_n(n, rng, time_budget):
    """Try to find a valid colouring for n vertices within time_budget seconds."""
    triples, tmap = _triple_list(n)
    four_sets, four_triples, five_sets, five_triples = _build_membership(n, triples, tmap)
    m = len(triples)

    # Start random
    colors = [rng.randint(0, 1) for _ in range(m)]
    # Initialize counters
    red4 = [0] * len(four_sets)  # count of red triples in each 4-set
    blue5 = [0] * len(five_sets)  # count of blue triples in each 5-set
    for fi, ts in enumerate(four_triples):
        c = 0
        for t in ts:
            if colors[t] == 0:
                c += 1
        red4[fi] = c
    for fi, ts in enumerate(five_triples):
        c = 0
        for t in ts:
            if colors[t] == 1:
                c += 1
        blue5[fi] = c
    violations = sum(1 for x in red4 if x == 4) + sum(1 for x in blue5 if x == 10)

    if violations == 0:
        return colors

    # Local search: flip triples
    # Precompute for each triple which 4-sets and 5-sets it's in
    triple_to_fours = [[] for _ in range(m)]
    for fi, ts in enumerate(four_triples):
        for t in ts:
            triple_to_fours[t].append(fi)
    triple_to_fives = [[] for _ in range(m)]
    for fi, ts in enumerate(five_triples):
        for t in ts:
            triple_to_fives[t].append(fi)

    start = time.time()
    best_v = violations
    best_colors = colors[:]
    no_improve = 0
    max_no_improve = 50000

    while time.time() - start < time_budget and no_improve < max_no_improve:
        t = rng.randrange(m)
        old = colors[t]
        new = 1 - old
        # Compute delta violations
        delta = 0
        for fi in triple_to_fours[t]:
            old_cnt = red4[fi]
            if old == 0:  # was red, now blue: red count decreases
                new_cnt = old_cnt - 1
                if old_cnt == 4:
                    delta -= 1
                if new_cnt == 4:
                    delta += 1
            else:  # was blue, now red: red count increases
                new_cnt = old_cnt + 1
                if old_cnt == 4:
                    delta -= 1
                if new_cnt == 4:
                    delta += 1
            red4[fi] = new_cnt
        for fi in triple_to_fives[t]:
            old_cnt = blue5[fi]
            if old == 1:  # was blue, now red: blue count decreases
                new_cnt = old_cnt - 1
                if old_cnt == 10:
                    delta -= 1
                if new_cnt == 10:
                    delta += 1
            else:  # was red, now blue: blue count increases
                new_cnt = old_cnt + 1
                if old_cnt == 10:
                    delta -= 1
                if new_cnt == 10:
                    delta += 1
            blue5[fi] = new_cnt
        colors[t] = new
        violations += delta
        if violations < best_v:
            best_v = violations
            best_colors = colors[:]
            no_improve = 0
        else:
            no_improve += 1
            # Revert if not improving and random says so (simulated annealing lite)
            if no_improve > 1000 and rng.random() < 0.5:
                # revert
                colors[t] = old
                for fi in triple_to_fours[t]:
                    red4[fi] = red4[fi] + (1 if old == 0 else -1)
                for fi in triple_to_fives[t]:
                    blue5[fi] = blue5[fi] + (1 if old == 1 else -1)
                violations -= delta

        if best_v == 0:
            return best_colors

    return None if best_v > 0 else best_colors


def construct():
    global _BEST, _DEADLINE
    start = time.time()
    rng = random.Random(42)

    # Try n from 34 down; spend more time on larger n
    for n in range(34, 9, -1):
        remaining = _DEADLINE - time.time()
        if remaining < 5:
            break
        # Allocate time proportional to n
        budget = min(remaining * 0.6, 20.0)
        result = _try_n(n, rng, budget)
        if result is not None:
            _BEST = {'n': n, 'colors': result}
            # Continue trying larger? No, we went down. Return best found.
            break
        # If n failed, try smaller
        if time.time() > _DEADLINE - 10:
            break

    if _BEST['n'] == 0:
        # Fallback: small valid construction
        n = 9
        from itertools import combinations as _comb
        _BEST = {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in _comb(range(n), 3)]}

    return {'n': _BEST['n'], 'colors': _BEST['colors']}
