"""Combined program: maximum-clique search on the gcd-compatibility graph."""
import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 82  # safety margin before 90s wall-clock kill

    # Candidates: numbers in [1,255] sharing a factor with 255 = 3*5*17
    candidates = [i for i in range(1, 256)
                  if i % 3 == 0 or i % 5 == 0 or i % 17 == 0]
    n = len(candidates)
    idx_255 = candidates.index(255)

    # Compatibility: compat[i][j] = True iff gcd(candidates[i], candidates[j]) > 1
    compat = [[False] * n for _ in range(n)]
    for i in range(n):
        compat[i][i] = True
        ci = candidates[i]
        for j in range(i + 1, n):
            if math.gcd(ci, candidates[j]) > 1:
                compat[i][j] = True
                compat[j][i] = True

    # ---- Maximum clique via Bron-Kerbosch with pivot (clique = all pairs gcd>1) ----
    best = [set()]

    def bronk(R, P, X):
        if len(R) > len(best[0]):
            best[0] = set(R)
        if not P and not X:
            return
        # pick pivot u from P ∪ X maximizing |P ∩ N(u)|
        u = -1
        maxc = -1
        for cand in list(P) + list(X):
            c = 0
            for v in P:
                if compat[cand][v]:
                    c += 1
            if c > maxc:
                maxc = c
                u = cand
        # iterate over P \ N(u)
        for v in list(P):
            if u != -1 and compat[u][v]:
                continue
            bronk(R | {v}, P & {w for w in P if compat[v][w]},
                  X & {w for w in X if compat[v][w]})
            P.remove(v)
            X.add(v)

    # Force 255 in the clique: R = {255}, P = N(255), X = empty
    bronk({idx_255}, {i for i in range(n) if i != idx_255 and compat[idx_255][i]}, set())

    best_set = set(best[0])

    # ---- Polish pass: try to add any compatible element, then swaps ----
    def try_add(S):
        for i in range(n):
            if i in S:
                continue
            if all(compat[i][y] for y in S):
                S.add(i)
                return True
        return False

    improved = True
    while improved and (time.time() - start_time < deadline - 5):
        improved = False
        if try_add(best_set):
            improved = True
            continue
        # remove-one-add-many
        for rem in list(best_set):
            if rem == idx_255:
                continue
            S_t = best_set - {rem}
            base = len(S_t)
            for add in range(n):
                if add in S_t:
                    continue
                if all(compat[add][y] for y in S_t):
                    S_t.add(add)
            if len(S_t) > len(best_set):
                best_set = S_t
                improved = True
                break

    # ---- Random-restart local search to explore mixed structures ----
    rng = random.Random(12345)
    for _ in range(2000):
        if time.time() - start_time > deadline - 3:
            break
        core = {idx_255}
        others = [i for i in range(n) if i != idx_255]
        rng.shuffle(others)
        for i in others[:rng.randint(0, 4)]:
            if all(compat[i][y] for y in core):
                core.add(i)
        S = set(core)
        # greedy expand
        rem = [i for i in range(n) if i not in S]
        rng.shuffle(rem)
        for i in rem:
            if all(compat[i][y] for y in S):
                S.add(i)
        # local improvement
        imp = True
        while imp and (time.time() - start_time < deadline - 3):
            imp = False
            if try_add(S):
                imp = True
                continue
            for rem_el in list(S):
                if rem_el == idx_255:
                    continue
                S_t = S - {rem_el}
                for add in range(n):
                    if add in S_t:
                        continue
                    if all(compat[add][y] for y in S_t):
                        S_t.add(add)
                if len(S_t) > len(S):
                    S = S_t
                    imp = True
                    break
        if len(S) > len(best_set):
            best_set = S

    return sorted(candidates[i] for i in best_set)

# EVOLVE-BLOCK-END
