"""Linear triple systems on 16 points with small independence number."""
import random
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    N = 16
    FULL = (1 << N) - 1
    rng = random.Random(12345)

    all_triples = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                mask = (1 << a) | (1 << b) | (1 << c)
                all_triples.append((mask, (a, b, c)))
    num_triples = len(all_triples)

    popcount = [0] * (1 << N)
    for i in range(1, 1 << N):
        popcount[i] = popcount[i >> 1] + (i & 1)

    def independence_number(triples):
        indep = [True] * (1 << N)
        for tmask, _ in all_triples:
            comp = FULL ^ tmask
            sub = comp
            while True:
                s = tmask | sub
                indep[s] = False
                if sub == 0:
                    break
                sub = (sub - 1) & comp
        best = 0
        for mask in range(1 << N):
            if indep[mask]:
                pc = popcount[mask]
                if pc > best:
                    best = pc
        return best

    def is_linear(triples):
        n = len(triples)
        for i in range(n):
            for j in range(i + 1, n):
                if popcount[triples[i][0] & triples[j][0]] > 1:
                    return False
        return True

    def greedy_construct(rng):
        points = list(range(N))
        rng.shuffle(points)
        triples = []
        for p in points:
            candidates = [t for t in all_triples if (t[0] >> p) & 1]
            rng.shuffle(candidates)
            for tmask, tpoints in candidates:
                ok = True
                for emask, _ in triples:
                    if popcount[tmask & emask] > 1:
                        ok = False
                        break
                if ok:
                    triples.append((tmask, tpoints))
                    break
        return triples

    best_triples = None
    best_alpha = N + 1

    for i in range(200):
        if time.time() > deadline:
            break
        triples = greedy_construct(rng)
        if not is_linear(triples):
            continue
        alpha = independence_number(triples)
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples
        if best_alpha <= 5:
            break

    if best_triples is not None:
        for step in range(50000):
            if time.time() > deadline:
                break
            idx = rng.randrange(len(best_triples))
            old_mask, _ = best_triples[idx]
            
            for _ in range(30):
                new_mask, new_points = all_triples[rng.randrange(num_triples)]
                if new_mask == old_mask:
                    continue
                ok = True
                for j, (emask, _) in enumerate(best_triples):
                    if j == idx:
                        continue
                    if popcount[new_mask & emask] > 1:
                        ok = False
                        break
                if not ok:
                    continue
                new_triples = list(best_triples)
                new_triples[idx] = (new_mask, new_points)
                new_alpha = independence_number(new_triples)
                if new_alpha < best_alpha:
                    best_triples = new_triples
                    best_alpha = new_alpha
                    break

    if best_triples is None:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    
    return [list(tpoints) for tmask, tpoints in best_triples]
# EVOLVE-BLOCK-END
