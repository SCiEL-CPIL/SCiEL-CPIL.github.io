import time
import random

def construct():
    start = time.time()
    DEADLINE = start + 140.0  # safety margin under 150s wall-clock

    n = 20
    full = (1 << n) - 1
    rng = random.Random(12345)

    # Precompute bitmasks for all triples (a<b<c) and all pairs (a<b)
    triples = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                triples.append((a, b, c))
    pair_mask = {}
    for a in range(n):
        for b in range(a + 1, n):
            pair_mask[(a, b)] = (1 << a) | (1 << b)

    def common3(ADJ, a, b, c):
        return ADJ[a] & ADJ[b] & ADJ[c]

    def is_free(ADJ):
        for (a, b, c) in triples:
            if (ADJ[a] & ADJ[b] & ADJ[c]) & (ADJ[a] & ADJ[b] & ADJ[c]) != 0:
                pass
        # proper count
        for (a, b, c) in triples:
            cm = ADJ[a] & ADJ[b] & ADJ[c]
            if cm & (cm - 1) and ((cm & (cm - 1)) & (cm & (cm - 1) & (cm & (cm - 1) & 0))):
                pass
        # simplest correct count via popcount
        for (a, b, c) in triples:
            cm = ADJ[a] & ADJ[b] & ADJ[c]
            if cm & (cm - 1) and (cm & (cm - 1)) & (cm & (cm - 1) & (cm - 1)):
                pass
        # Just use bit_count
        for (a, b, c) in triples:
            if ((ADJ[a] & ADJ[b] & ADJ[c]).bit_count()) >= 3:
                return False
        return True

    def can_add(ADJ, a, b):
        # Adding (a,b): only triples containing a or b can gain a common neighbor.
        # Triple (a,b,c): new common = old + [c in N(a) and c in N(b)]
        #   old = ADJ[a]&ADJ[b]&ADJ[c] (a,b not neighbors yet so a not in ADJ[b], b not in ADJ[a])
        #   Actually a is not in ADJ[b] currently. After add, b in ADJ[a], a in ADJ[b].
        #   common(a,b,c) gains a (if a in ADJ[c]) and b (if b in ADJ[c]).
        # Triple (a,c,d): gains b (if b in ADJ[c] and b in ADJ[d]).
        # Triple (b,c,d): gains a (if a in ADJ[c] and a in ADJ[d]).
        # Check triples (a,b,c):
        for c in range(n):
            if c == a or c == b:
                continue
            old = ADJ[a] & ADJ[b] & ADJ[c]
            gain = 0
            if ADJ[c] >> a & 1:
                gain += 1
            if ADJ[c] >> b & 1:
                gain += 1
            if (old.bit_count() + gain) >= 3:
                return False
        # Triples (a,c,d) with c,d != b, c<d, and at least one != a
        # We need c,d both != a, and c<d, and (c,d) != (a,b) handled. Also exclude the (a,b,c) case where one is b.
        for c in range(n):
            if c == a or c == b:
                continue
            for d in range(c + 1, n):
                if d == a or d == b:
                    continue
                # triple (a,c,d): gains b if b in N(c) and b in N(d)
                old = ADJ[a] & ADJ[c] & ADJ[d]
                gain = 1 if ((ADJ[c] >> b & 1) and (ADJ[d] >> b & 1)) else 0
                if (old.bit_count() + gain) >= 3:
                    return False
        # Triples (b,c,d) with c,d != a, c<d
        for c in range(n):
            if c == a or c == b:
                continue
            for d in range(c + 1, n):
                if d == a or d == b:
                    continue
                old = ADJ[b] & ADJ[c] & ADJ[d]
                gain = 1 if ((ADJ[c] >> a & 1) and (ADJ[d] >> a & 1)) else 0
                if (old.bit_count() + gain) >= 3:
                    return False
        return True

    def add_edge(ADJ, a, b):
        ADJ[a] |= (1 << b)
        ADJ[b] |= (1 << a)

    def remove_edge(ADJ, a, b):
        ADJ[a] &= ~(1 << b)
        ADJ[b] &= ~(1 << a)

    def edge_count(ADJ):
        return sum(ADJ[i].bit_count() for i in range(n)) // 2

    def edges_list(ADJ):
        out = []
        for a in range(n):
            m = ADJ[a]
            while m:
                lsb = m & (-m)
                b = lsb.bit_length() - 1
                if b > a:
                    out.append([a, b])
                m &= m - 1
        return out

    def greedy(ADJ, pairs, rng):
        for (a, b) in pairs:
            if ADJ[a] >> b & 1:
                continue
            if can_add(ADJ, a, b):
                add_edge(ADJ, a, b)

    def local_search(ADJ, max_iters, rng):
        # Repeatedly try random edge additions and 2-opt swaps.
        for _ in range(max_iters):
            if time.time() > DEADLINE - 5:
                break
            improved = False
            # try a batch of random single additions
            for _ in range(200):
                a = rng.randrange(n)
                b = rng.randrange(n)
                if a == b or (ADJ[a] >> b & 1):
                    continue
                if can_add(ADJ, a, b):
                    add_edge(ADJ, a, b)
                    improved = True
            # try a batch of 2-opt swaps: remove (x,y), add (a,b)
            for _ in range(150):
                a = rng.randrange(n)
                b = rng.randrange(n)
                if a == b or (ADJ[a] >> b & 1):
                    continue
                x = rng.randrange(n)
                y = rng.randrange(n)
                if x == y or not (ADJ[x] >> y & 1):
                    continue
                if {a, b} == {x, y}:
                    continue
                # remove (x,y)
                remove_edge(ADJ, x, y)
                if can_add(ADJ, a, b):
                    add_edge(ADJ, a, b)
                    improved = True
                else:
                    add_edge(ADJ, x, y)  # revert
            if not improved:
                # try a few more rounds of pure additions with different order
                for _ in range(100):
                    a = rng.randrange(n)
                    b = rng.randrange(n)
                    if a == b or (ADJ[a] >> b & 1):
                        continue
                    if can_add(ADJ, a, b):
                        add_edge(ADJ, a, b)
                        improved = True
                if not improved:
                    break
        return ADJ

    # all possible edges
    all_edges = [(a, b) for a in range(n) for b in range(a + 1, n)]

    best_ADJ = None
    best_cnt = 0

    def update_best(ADJ):
        nonlocal best_ADJ, best_cnt
        c = edge_count(ADJ)
        if c > best_cnt:
            best_cnt = c
            best_ADJ = [ADJ[i] for i in range(n)]

    # Seed: K_{2,18}
    seed = [0] * n
    for a in range(2):
        for b in range(2, 20):
            seed[a] |= (1 << b)
            seed[b] |= (1 << a)
    update_best(seed)

    # Multiple randomized restarts: greedy fill + local search
    restarts = 0
    while time.time() < DEADLINE - 20:
        restarts += 1
        ADJ = [0] * n
        # random subset of a random ordering to seed, then greedy fill
        pairs = all_edges[:]
        rng.shuffle(pairs)
        # optional: start from a random sparse seed to diversify
        greedy(ADJ, pairs, rng)
        local_search(ADJ, 30, rng)
        update_best(ADJ)
        # early exit if we hit theoretical-ish high values
        if best_cnt >= 150:
            break

    # One more pass: take best, do longer local search
    if best_ADJ is not None:
        ADJ = [best_ADJ[i] for i in range(n)]
        if time.time() < DEADLINE - 10:
            local_search(ADJ, 40, rng)
            update_best(ADJ)

    if best_ADJ is None:
        return [[a, b] for a in range(2) for b in range(2, 20)]

    return edges_list(best_ADJ)
