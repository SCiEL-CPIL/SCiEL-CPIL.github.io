import random
import time

N = 1000

def _greedy(order, rep, chosen, cap=2):
    """Try to build a set by scanning `order` and adding x if valid.
    rep[s] = number of representations s = a+b (a<=b) among `chosen`.
    Returns (new_chosen, new_rep)."""
    chosen = list(chosen)
    rep = dict(rep)
    for x in order:
        ok = rep.get(2 * x, 0) < cap
        if ok:
            for y in chosen:
                if rep.get(x + y, 0) >= cap:
                    ok = False
                    break
        if ok:
            chosen.append(x)
            rep[2 * x] = rep.get(2 * x, 0) + 1
            for y in chosen[:-1]:
                s = x + y
                rep[s] = rep.get(s, 0) + 1
    return chosen, rep

def _rebuild_rep(chosen):
    rep = {}
    m = len(chosen)
    for i in range(m):
        a = chosen[i]
        rep[2 * a] = rep.get(2 * a, 0) + 1
        for j in range(i + 1, m):
            s = a + chosen[j]
            rep[s] = rep.get(s, 0) + 1
    return rep

def _can_add(chosen, x, rep, cap=2):
    if rep.get(2 * x, 0) >= cap:
        return False
    for y in chosen:
        if rep.get(x + y, 0) >= cap:
            return False
    return True

def _local_search(chosen, rep, order, deadline, cap=2, max_passes=40):
    """Polish: repeatedly try insertions (scan order) then 1-for-1 swaps.
    Returns best (chosen, rep, size) found."""
    chosen = list(chosen)
    rep = dict(rep)
    best_size = len(chosen)
    best_chosen = list(chosen)
    best_rep = dict(rep)

    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False

        # --- insertion pass ---
        for x in order:
            if time.time() > deadline:
                break
            if x in rep:  # already present (use a set for membership)
                continue
            if _can_add(chosen, x, rep, cap):
                # add
                chosen.append(x)
                rep[2 * x] = rep.get(2 * x, 0) + 1
                for y in chosen[:-1]:
                    s = x + y
                    rep[s] = rep.get(s, 0) + 1
                improved = True
                if len(chosen) > best_size:
                    best_size = len(chosen)
                    best_chosen = list(chosen)
                    best_rep = dict(rep)

        # --- swap pass: try replacing a with b ---
        if time.time() > deadline:
            break
        # candidate pool = all numbers not in set
        present = set(chosen)
        pool = [x for x in order if x not in present]
        for a in list(chosen):
            if time.time() > deadline:
                break
            # remove a
            for y in chosen:
                if y == a:
                    continue
                s = a + y
                rep[s] -= 1
                if rep[s] == 0:
                    del rep[s]
            rep[2 * a] -= 1
            if rep[2 * a] == 0:
                del rep[2 * a]
            chosen.remove(a)
            # try to add a better b
            done = False
            for b in pool:
                if time.time() > deadline:
                    break
                if _can_add(chosen, b, rep, cap):
                    # add b
                    chosen.append(b)
                    rep[2 * b] = rep.get(2 * b, 0) + 1
                    for y in chosen[:-1]:
                        s = b + y
                        rep[s] = rep.get(s, 0) + 1
                    improved = True
                    done = True
                    break
            if not done:
                # put a back
                chosen.append(a)
                rep[2 * a] = rep.get(2 * a, 0) + 1
                for y in chosen[:-1]:
                    s = a + y
                    rep[s] = rep.get(s, 0) + 1

        if not improved:
            break

    if len(best_chosen) > len(chosen):
        return best_chosen, best_rep, best_size
    return chosen, rep, len(chosen)

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    best_chosen = []
    best_size = 0
    best_rep = {}

    def consider(chosen, rep):
        nonlocal best_chosen, best_size, best_rep
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_chosen = list(chosen)
            best_rep = dict(rep)

    # ---- Phase 1: diverse structured greedy seeds ----
    orderings = []
    orderings.append(list(range(1, N + 1)))                       # ascending
    orderings.append(list(range(N, 0, -1)))                       # descending
    # sparse / weighted
    rng = random.Random(12345)
    for seed in range(12):
        r = random.Random(seed * 7919 + 13)
        w = {x: r.random() for x in range(1, N + 1)}
        orderings.append(sorted(range(1, N + 1), key=lambda x: w[x]))
    # arithmetic-progression-ish orderings
    for step in (2, 3, 5, 7):
        seq = list(range(1, N + 1, step))
        seq += list(range(1, N + 1, step))  # duplicates removed below
        orderings.append(list(dict.fromkeys(seq)))
    # block orderings (small block first, then large)
    for bl in (10, 25, 50):
        seq = []
        for b in range(0, N, bl):
            seq.extend(range(b + 1, min(b + bl, N) + 1))
        orderings.append(seq)

    for oi, order in enumerate(orderings):
        if time.time() > deadline - 30:
            break
        chosen, rep = _greedy(order, {}, [], cap=2)
        consider(chosen, rep)

    # ---- Phase 2: local-search polish on the best seeds ----
    # keep a few distinct good seeds
    seeds = []
    seen = set()
    for oi, order in enumerate(orderings):
        if time.time() > deadline - 20:
            break
        chosen, rep = _greedy(order, {}, [], cap=2)
        key = frozenset(chosen)
        if key not in seen:
            seen.add(key)
            seeds.append((len(chosen), chosen, rep))
    seeds.sort(key=lambda t: -t[0])
    seeds = seeds[:6]

    for size, chosen, rep in seeds:
        if time.time() > deadline - 10:
            break
        # build a good order for local search: prefer unused small numbers
        present = set(chosen)
        ls_order = [x for x in range(1, N + 1) if x not in present]
        c2, r2, sz2 = _local_search(chosen, rep, ls_order, deadline, cap=2)
        consider(c2, r2)

    # ---- Phase 3: random-restart greedy + polish while time remains ----
    base = best_size
    for it in range(2000):
        if time.time() > deadline - 8:
            break
        r = random.Random(it * 31 + 1)
        # random permutation biased
        order = list(range(1, N + 1))
        r.shuffle(order)
        chosen, rep = _greedy(order, {}, [], cap=2)
        if len(chosen) > best_size:
            consider(chosen, rep)
        # quick polish if it's competitive
        if len(chosen) >= best_size - 2:
            present = set(chosen)
            ls_order = [x for x in range(1, N + 1) if x not in present]
            c2, r2, sz2 = _local_search(chosen, rep, ls_order, deadline, cap=2, max_passes=8)
            consider(c2, r2)

    # ---- Final safety: ensure we return something valid ----
    if not best_chosen:
        # fallback: plain ascending greedy
        best_chosen, best_rep = _greedy(list(range(1, N + 1)), {}, [], cap=2)

    # validate
    rep = _rebuild_rep(best_chosen)
    for s, c in rep.items():
        assert c <= 2, "invalid representation count"
    assert all(1 <= x <= N for x in best_chosen)
    assert len(set(best_chosen)) == len(best_chosen)

    return best_chosen
