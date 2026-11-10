import time
import random
import sys

def construct():
    start = time.time()
    deadline = start + 85.0
    N = 180

    # Precompute all forbidden triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    # a = bc/(b+c), so (b+c) | bc
    triples = []
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            s = b + c
            prod = b * c
            if prod % s == 0:
                a = prod // s
                if 1 <= a < b:
                    triples.append((a, b, c))

    # For each element x, list of (y, z) such that {x,y,z} is a forbidden triple
    # (x can play any role). We store pairs of the OTHER two elements.
    partners = [set() for _ in range(N + 1)]  # partners[x] = set of frozenset-pairs? 
    # Better: partners[x] = list of (y, z) where {x,y,z} forbidden
    partners_list = [[] for _ in range(N + 1)]
    for a, b, c in triples:
        partners_list[a].append((b, c))
        partners_list[b].append((a, c))
        partners_list[c].append((a, b))

    # conflict degree
    degree = [len(partners_list[i]) for i in range(N + 1)]

    def has_triple(S):
        Sset = set(S)
        for a, b, c in triples:
            if a in Sset and b in Sset and c in Sset:
                return True
        return False

    def can_add(Sset, x):
        for y, z in partners_list[x]:
            if y in Sset and z in Sset:
                return False
        return True

    def greedy_build(order):
        S = set()
        for x in order:
            if can_add(S, x):
                S.add(x)
        return S

    def local_search_add(S, max_iters=5000):
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() >= deadline - 5.0:
                break
            improved = False
            iters += 1
            for x in range(1, N + 1):
                if x not in S and can_add(S, x):
                    S.add(x)
                    improved = True
                    break
        return S

    def kick_and_fill(S, rng, k):
        """Remove k random elements, then greedily refill with best-addable."""
        S = set(S)
        elems = list(S)
        if len(elems) <= k:
            return S
        removed = set(rng.sample(elems, k))
        S -= removed
        # greedy refill: repeatedly add the addable element that... just add any
        order = list(range(1, N + 1))
        rng.shuffle(order)
        for x in order:
            if x not in S and can_add(S, x):
                S.add(x)
        return S

    def simulated_annealing(S, rng, steps):
        """Anneal with 1-removal-2-add and 2-removal-2-add moves, accepting worse with prob."""
        S = set(S)
        cur_size = len(S)
        best = set(S)
        best_size = cur_size
        T = 1.5
        cool = 0.999
        missing_all = list(range(1, N + 1))
        for it in range(steps):
            if time.time() >= deadline - 5.0:
                break
            # choose move type
            if len(S) < 2:
                break
            if rng.random() < 0.5:
                # 1 remove, up to 2 add
                y = rng.choice(list(S))
                S2 = S - {y}
                # try adding random missing elements
                added = 0
                for x in rng.sample(missing_all, min(6, len(missing_all))):
                    if x in S2:
                        continue
                    if can_add(S2, x):
                        S2.add(x)
                        added += 1
                new_size = len(S2)
                delta = new_size - cur_size
                if delta >= 0 or rng.random() < pow(2.718281828, delta / T):
                    S = S2
                    cur_size = new_size
                    if cur_size > best_size:
                        best = set(S)
                        best_size = cur_size
            else:
                # 2 remove, 2 add
                y1, y2 = rng.sample(list(S), 2)
                S2 = S - {y1, y2}
                for x in rng.sample(missing_all, min(8, len(missing_all))):
                    if x in S2:
                        continue
                    if can_add(S2, x):
                        S2.add(x)
                new_size = len(S2)
                delta = new_size - cur_size
                if delta >= 0 or rng.random() < pow(2.718281828, delta / T):
                    S = S2
                    cur_size = new_size
                    if cur_size > best_size:
                        best = set(S)
                        best_size = cur_size
            T *= cool
            if T < 0.02:
                T = 0.02
        return best, best_size

    # Strategy 1: top half (trivially valid, size 90)
    S1 = set(range(91, N + 1))
    S1 = local_search_add(S1)
    best = S1
    best_size = len(S1)

    rng = random.Random(12345)

    # Strategy 2: degree-based greedy
    order_by_degree = sorted(range(1, N + 1), key=lambda x: degree[x])
    S2 = greedy_build(order_by_degree)
    S2 = local_search_add(S2)
    if len(S2) > best_size:
        best, best_size = S2, len(S2)

    # Strategy 3: descending
    S3 = greedy_build(list(range(N, 0, -1)))
    S3 = local_search_add(S3)
    if len(S3) > best_size:
        best, best_size = S3, len(S3)

    # Strategy 4: ascending
    S4 = greedy_build(list(range(1, N + 1)))
    S4 = local_search_add(S4)
    if len(S4) > best_size:
        best, best_size = S4, len(S4)

    # Main loop: random greedy + kick+fill + anneal
    while time.time() < deadline - 8.0:
        order = list(range(1, N + 1))
        rng.shuffle(order)
        S_try = greedy_build(order)
        S_try = local_search_add(S_try, max_iters=2000)
        # kick
        S_try = kick_and_fill(S_try, rng, k=rng.randint(3, 10))
        S_try = local_search_add(S_try, max_iters=2000)
        if len(S_try) > best_size:
            best, best_size = S_try, len(S_try)
        # anneal from current best
        if time.time() < deadline - 6.0:
            S_an, size_an = simulated_annealing(best, rng, steps=4000)
            S_an = local_search_add(S_an, max_iters=1000)
            if len(S_an) > best_size:
                best, best_size = S_an, len(S_an)

    # Final polish
    best = local_search_add(best, max_iters=3000)
    # a few final kicks
    for _ in range(20):
        if time.time() >= deadline - 3.0:
            break
        S_try = kick_and_fill(best, rng, k=rng.randint(2, 6))
        S_try = local_search_add(S_try, max_iters=1000)
        if len(S_try) > best_size:
            best, best_size = S_try, len(S_try)

    best = local_search_add(best, max_iters=2000)

    # Verify
    if has_triple(best):
        best = set(range(91, N + 1))

    return list(best)
