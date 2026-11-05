"""Triple system on 16 points avoiding K(2,2,2) via incremental local search."""
import random
import time
import sys
from itertools import combinations

def construct():
    deadline = time.time() + 110
    rng = random.Random(12345)

    n = 16
    triples = []
    triple_idx = {}
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                triple_idx[(a, b, c)] = len(triples)
                triples.append((a, b, c))
    T = len(triples)  # 560

    # Precompute pairings of 6 points into 3 pairs (15 ways)
    def all_pairings_6(pts):
        pts = list(pts)
        results = []
        def rec(rem, current):
            if not rem:
                results.append(tuple(sorted(current, key=lambda x: min(x))))
                return
            first = rem[0]
            for i in range(1, len(rem)):
                pair = frozenset({first, rem[i]})
                new_rem = rem[1:i] + rem[i+1:]
                rec(new_rem, current + [pair])
        rec(pts, [])
        return results

    # Build configs and triple_to_configs
    triple_to_configs = [[] for _ in range(T)]
    configs = []

    for six_pts in combinations(range(n), 6):
        for pairing in all_pairings_6(six_pts):
            p1, p2, p3 = pairing
            trans = []
            for x in p1:
                for y in p2:
                    for z in p3:
                        t = tuple(sorted((x, y, z)))
                        trans.append(t)
            if len(set(trans)) != 8:
                continue
            trans_idx = tuple(sorted(triple_idx[t] for t in trans))
            ci = len(configs)
            configs.append(trans_idx)
            for ti in trans_idx:
                triple_to_configs[ti].append(ci)

    C = len(configs)

    included = [False] * T
    viol = [0] * C
    current_count = 0

    def add_triple(ti):
        nonlocal current_count
        included[ti] = True
        current_count += 1
        for ci in triple_to_configs[ti]:
            viol[ci] += 1

    def remove_triple(ti):
        nonlocal current_count
        included[ti] = False
        current_count -= 1
        for ci in triple_to_configs[ti]:
            viol[ci] -= 1

    def is_valid_add(ti):
        for ci in triple_to_configs[ti]:
            if viol[ci] >= 7:
                return False
        return True

    def greedy_init():
        nonlocal current_count
        for ti in range(T):
            if included[ti]:
                remove_triple(ti)
        order = list(range(T))
        rng.shuffle(order)
        for ti in order:
            if is_valid_add(ti):
                add_triple(ti)

    def local_search():
        nonlocal current_count
        while time.time() < deadline - 2:
            made_progress = False
            # Try adding random valid triples
            for _ in range(100):
                ti = rng.randint(0, T - 1)
                if not included[ti] and is_valid_add(ti):
                    add_triple(ti)
                    made_progress = True
                    break
            # Shake to escape local optima
            if not made_progress or rng.random() < 0.05:
                k = rng.randint(5, 15)
                removed = []
                for _ in range(k):
                    ti = rng.randint(0, T - 1)
                    if included[ti]:
                        remove_triple(ti)
                        removed.append(ti)
                # Re-add greedily
                for _ in range(300):
                    ti = rng.randint(0, T - 1)
                    if not included[ti] and is_valid_add(ti):
                        add_triple(ti)

    best_count = 0
    best_state = None

    for restart in range(500):
        if time.time() > deadline - 5:
            break
        greedy_init()
        local_search()
        if current_count > best_count:
            best_count = current_count
            best_state = included[:]

    if best_state is None:
        best_state = included[:]

    result = []
    for ti in range(T):
        if best_state[ti]:
            result.append(list(triples[ti]))
    return result
