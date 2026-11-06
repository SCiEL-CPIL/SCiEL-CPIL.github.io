"""Structural rewrite: conflict-graph independent-set search for large sum-free subsets."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 145.0

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        s = set(S)
        for i, a in enumerate(S):
            for b in S[i:]:
                if a + b in s:
                    return False
        return True

    def repair(S):
        """Drop elements until sum-free (greedy: keep as many as possible)."""
        s = set(S)
        changed = True
        while changed:
            changed = False
            bad = set()
            for i, a in enumerate(S):
                for b in S[i:]:
                    if a + b in s:
                        bad.add(a)
                        bad.add(b)
                        bad.add(a + b)
            if bad:
                for x in bad:
                    if x in s:
                        s.discard(x)
                S = [x for x in S if x in s]
                changed = True
        return S

    def solve(S):
        n = len(S)
        idx = {v: i for i, v in enumerate(S)}
        # conflict[a][b] True iff a+b in set
        conflict = [[False] * n for _ in range(n)]
        for i in range(n):
            ai = S[i]
            row = conflict[i]
            for j in range(i, n):
                if ai + S[j] in idx:
                    row[j] = True
                    conflict[j][i] = True

        def neighbors(i):
            return [j for j in range(n) if conflict[i][j]]

        def try_remove(i, chosen, chosen_set):
            # returns new (chosen, chosen_set) if removing i keeps sum-free, else None
            newc = chosen_set - {i}
            for j in neighbors(i):
                if j in newc:
                    return None
            return chosen - [i], newc

        def improve(chosen, chosen_set):
            changed = True
            while changed:
                changed = False
                for i in chosen[:]:
                    if i not in chosen_set:
                        continue
                    if is_sum_free([S[j] for j in chosen_set | {i}]):
                        chosen.append(i)
                        chosen_set.add(i)
                        changed = True
                for i in chosen[:]:
                    if i not in chosen_set:
                        continue
                    res = try_remove(i, chosen, chosen_set)
                    if res is not None:
                        chosen, chosen_set = res
                        changed = True
            return chosen, chosen_set

        def greedy(order):
            chosen = []
            chosen_set = set()
            for i in order:
                if i in chosen_set:
                    continue
                if is_sum_free([S[j] for j in chosen_set | {i}]):
                    chosen.append(i)
                    chosen_set.add(i)
            return improve(chosen, chosen_set)

        def random_order():
            o = list(range(n))
            random.shuffle(o)
            return o

        def degree_order():
            deg = [0] * n
            for i in range(n):
                for j in range(n):
                    if conflict[i][j]:
                        deg[i] += 1
            return sorted(range(n), key=lambda i: deg[i])

        orders = [
            list(range(n)),
            list(range(n - 1, -1, -1)),
            degree_order(),
            sorted(range(n), key=lambda i: S[i]),
            sorted(range(n), key=lambda i: -S[i]),
        ]

        best = None
        best_len = -1
        for o in orders:
            ch, cs = greedy(o)
            if len(ch) > best_len:
                best_len = len(ch)
                best = ch

        # random restarts until deadline
        while time.time() < DEADLINE - 1.0:
            ch, cs = greedy(random_order())
            if len(ch) > best_len:
                best_len = len(ch)
                best = ch
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        best = solve(S)
        if best is None:
            best = []
        best = repair(best)
        if not is_sum_free(best):
            # absolute safety: fall back to a known-valid sum-free set
            best = [x for x in S if x % 2]
        results.append(best)
    return results
# EVOLVE-BLOCK-END
