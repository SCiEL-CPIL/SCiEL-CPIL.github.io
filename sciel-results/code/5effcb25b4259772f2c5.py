import time
import random

def construct():
    start = time.time()
    deadline = start + 85.0

    def elapsed():
        return time.time() - start

    def inc(a, b):
        # return True if (a+b) divides a*b  (i.e. a,b must NOT both be chosen)
        x, y = a, b
        while y:
            x, y = y, x % y
        g = x
        u = a // g
        v = b // g
        return (u + v) % g == 0

    odds = [x for x in range(1, 257) if x % 2 == 1]
    evens = [x for x in range(2, 257) if x % 2 == 0]

    # Check each even against all odds
    odds_ok = []
    for e in evens:
        ok = True
        for o in odds:
            if inc(e, o):
                ok = False
                break
        if ok:
            odds_ok.append(e)

    n_ev = len(odds_ok)
    # Precompute conflict matrix as bitmasks for speed
    conflict_mask = [0] * n_ev
    for i in range(n_ev):
        for j in range(i + 1, n_ev):
            if inc(odds_ok[i], odds_ok[j]):
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)

    def greedy_is(order=None):
        if order is None:
            order = list(range(n_ev))
        remaining = set(order)
        result = []
        while remaining:
            best = None
            best_deg = None
            for v in remaining:
                deg = bin(conflict_mask[v] & (sum(1 << u for u in remaining if u != v))).count('1')
                if best is None or deg < best_deg or (deg == best_deg and v > best):
                    best = v
                    best_deg = deg
            result.append(best)
            remaining.discard(best)
            to_remove = [u for u in remaining if (conflict_mask[best] >> u) & 1]
            for u in to_remove:
                remaining.discard(u)
        return set(result)

    # Simulated annealing with neutral moves
    def simulated_annealing(initial_set, max_iters=20000):
        chosen = set(initial_set)
        best = set(chosen)
        best_size = len(chosen)
        temp = 1.0
        cooling = 0.9995

        for it in range(max_iters):
            if elapsed() > deadline - 5:
                break
            not_chosen = [i for i in range(n_ev) if i not in chosen]
            if not not_chosen:
                break

            # Randomly pick a move type
            if random.random() < 0.4 and len(chosen) > 0:
                # Swap: remove one, add one
                remove_candidate = random.choice(list(chosen))
                add_candidate = random.choice(not_chosen)
                # Check if add_candidate conflicts with (chosen - {remove_candidate})
                new_conflict = False
                for c in chosen:
                    if c != remove_candidate and (conflict_mask[add_candidate] >> c) & 1:
                        new_conflict = True
                        break
                if not new_conflict:
                    chosen.discard(remove_candidate)
                    chosen.add(add_candidate)
                    # Neutral or improving move
                    if len(chosen) > best_size:
                        best = set(chosen)
                        best_size = len(chosen)
                else:
                    # Accept with probability based on temperature
                    if random.random() < temp:
                        chosen.discard(remove_candidate)
                        chosen.add(add_candidate)
            elif random.random() < 0.7:
                # Add only
                add_candidate = random.choice(not_chosen)
                no_conflict = True
                for c in chosen:
                    if (conflict_mask[add_candidate] >> c) & 1:
                        no_conflict = False
                        break
                if no_conflict:
                    chosen.add(add_candidate)
                    if len(chosen) > best_size:
                        best = set(chosen)
                        best_size = len(chosen)
                elif random.random() < temp * 0.5:
                    # Accept conflict (will need to remove later)
                    pass
            else:
                # Remove only (to allow escape)
                if chosen and random.random() < temp * 0.3:
                    remove_candidate = random.choice(list(chosen))
                    chosen.discard(remove_candidate)

            temp *= cooling

        return best

    # Generate multiple diverse initial solutions
    best_ev_set = set()
    best_size = 0

    # Deterministic greedy
    g0 = greedy_is()
    if len(g0) > best_size:
        best_ev_set = g0
        best_size = len(g0)

    # Randomized greedy passes
    for _ in range(30):
        if elapsed() > deadline - 30:
            break
        random.seed()
        order = list(range(n_ev))
        random.shuffle(order)
        g = greedy_is(order)
        if len(g) > best_size:
            best_ev_set = g
            best_size = len(g)

    # Simulated annealing from best initial
    if elapsed() < deadline - 10:
        sa_result = simulated_annealing(best_ev_set, max_iters=50000)
        if len(sa_result) > best_size:
            best_ev_set = sa_result
            best_size = len(sa_result)

    # More SA from different starting points
    for _ in range(10):
        if elapsed() > deadline - 10:
            break
        random.seed(random.randint(0, 10**6))
        order = list(range(n_ev))
        random.shuffle(order)
        g = greedy_is(order)
        sa_result = simulated_annealing(g, max_iters=20000)
        if len(sa_result) > best_size:
            best_ev_set = sa_result
            best_size = len(sa_result)

    final = set(odds)
    for e in best_ev_set:
        final.add(odds_ok[e])

    return list(final)
