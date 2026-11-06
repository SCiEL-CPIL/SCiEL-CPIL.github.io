import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin within 90s wall-clock
    n = 256

    def time_ok(margin=0.0):
        return time.time() < deadline - margin

    def validate(A, B):
        """Return True if both are Sidon and nonzero-diff sets are disjoint."""
        def sidon(S):
            seen = set()
            L = list(S)
            for i in range(len(L)):
                for j in range(i, len(L)):
                    s = L[i] + L[j]
                    if s in seen:
                        return False
                    seen.add(s)
            return True
        if not (sidon(A) and sidon(B)):
            return False
        dA = set()
        LA = list(A)
        for i in range(len(LA)):
            for j in range(i + 1, len(LA)):
                dA.add(abs(LA[i] - LA[j]))
        dB = set()
        LB = list(B)
        for i in range(len(LB)):
            for j in range(i + 1, len(LB)):
                dB.add(abs(LB[i] - LB[j]))
        return len(dA & dB) == 0

    def full_diffs(S):
        S = list(S)
        d = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d.add(abs(S[i] - S[j]))
        return d

    def build(order, prefer, max_each=64):
        """Greedy incremental builder.

        prefer: 'A', 'B', or 'bal' (assign to the smaller set first).
        """
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()

        def try_add(x, S, sums, diffs, other_diffs):
            # Sidon check: all pair sums x+s (s in S) plus x+x must be new
            new_sums = set()
            for s in S:
                ns = x + s
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            if (2 * x) in sums or (2 * x) in new_sums:
                return False
            # diff check: |x - s| for s in S must not hit other_diffs or existing diffs
            for s in S:
                d = abs(x - s)
                if d in other_diffs or d in diffs:
                    return False
            return True

        for x in order:
            if len(A) >= max_each and len(B) >= max_each:
                break
            if not time_ok(5.0):
                break
            # decide order of preference
            if prefer == 'A':
                orderAB = ['A', 'B']
            elif prefer == 'B':
                orderAB = ['B', 'A']
            else:  # bal
                if len(A) <= len(B):
                    orderAB = ['A', 'B']
                else:
                    orderAB = ['B', 'A']

            placed = False
            for tag in orderAB:
                if tag == 'A':
                    if len(A) >= max_each:
                        continue
                    if try_add(x, A, sums_A, diffs_A, diffs_B):
                        for s in A:
                            sums_A.add(x + s)
                        sums_A.add(2 * x)
                        for s in A:
                            diffs_A.add(abs(x - s))
                        A.append(x)
                        placed = True
                        break
                else:
                    if len(B) >= max_each:
                        continue
                    if try_add(x, B, sums_B, diffs_B, diffs_A):
                        for s in B:
                            sums_B.add(x + s)
                        sums_B.add(2 * x)
                        for s in B:
                            diffs_B.add(abs(x - s))
                        B.append(x)
                        placed = True
                        break
            # if not placed, skip x
        return A, B

    # ---- Local search: try to add a missing candidate or swap to improve size ----
    def local_search(A, B, max_each=64, max_rounds=4000):
        A = list(A)
        B = list(B)
        sums_A = set()
        sums_B = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums_A.add(A[i] + A[j])
        for i in range(len(B)):
            for j in range(i, len(B)):
                sums_B.add(B[i] + B[j])
        diffs_A = full_diffs(A)
        diffs_B = full_diffs(B)

        def can_add(x, S, sums, diffs, other_diffs):
            for s in S:
                ns = x + s
                if ns in sums:
                    return False
            if (2 * x) in sums:
                return False
            for s in S:
                d = abs(x - s)
                if d in other_diffs or d in diffs:
                    return False
            return True

        def do_add(x, S, sums, diffs):
            for s in S:
                sums.add(x + s)
            sums.add(2 * x)
            for s in S:
                diffs.add(abs(x - s))
            S.append(x)

        def remove(x, S, sums, diffs):
            S.remove(x)
            # recompute is simplest and safe
            new_sums = set()
            for i in range(len(S)):
                for j in range(i, len(S)):
                    new_sums.add(S[i] + S[j])
            sums.clear()
            sums.update(new_sums)
            diffs.clear()
            diffs.update(full_diffs(S))

        all_nums = set(range(1, n + 1))
        rng = random.Random(7)

        for _ in range(max_rounds):
            if not time_ok(4.0):
                break
            progress = False

            # 1) try to add a new element to the smaller set (or either if room)
            targets = []
            if len(A) < max_each:
                targets.append(('A', A, sums_A, diffs_A, diffs_B))
            if len(B) < max_each:
                targets.append(('B', B, sums_B, diffs_B, diffs_A))
            if len(A) < len(B):
                targets.sort(key=lambda t: 0 if t[0] == 'A' else 1)
            else:
                targets.sort(key=lambda t: 0 if t[0] == 'B' else 1)

            for tag, S, sums, diffs, other in targets:
                if len(S) >= max_each:
                    continue
                cand = [v for v in all_nums if v not in S]
                rng.shuffle(cand)
                added = False
                for v in cand:
                    if can_add(v, S, sums, diffs, other):
                        do_add(v, S, sums, diffs)
                        progress = True
                        added = True
                        break
                if added:
                    break

            if progress:
                continue

            # 2) swap-based improvement: remove one element from a set and
            #    try to insert two new ones (net +1).
            #    Try on the smaller set first.
            sets = [('A', A, sums_A, diffs_A, diffs_B),
                    ('B', B, sums_B, diffs_B, diffs_A)]
            if len(A) < len(B):
                sets.reverse()
            done = False
            for tag, S, sums, diffs, other in sets:
                if done:
                    break
                idx = rng.randrange(len(S))
                victim = S[idx]
                # snapshot
                S_snap = list(S)
                sums_snap = set(sums)
                diffs_snap = set(diffs)
                remove(victim, S, sums, diffs)
                # try to add up to 2 new elements
                added = 0
                cand = [v for v in all_nums if v not in S]
                rng.shuffle(cand)
                for v in cand:
                    if added >= 2:
                        break
                    if can_add(v, S, sums, diffs, other):
                        do_add(v, S, sums, diffs)
                        added += 1
                if added >= 1:
                    progress = True
                    done = True
                else:
                    # revert
                    S[:] = S_snap
                    sums.clear()
                    sums.update(sums_snap)
                    diffs.clear()
                    diffs.update(diffs_snap)

            if not progress:
                break

        return A, B

    # ---- Generate candidate orderings ----
    base = list(range(1, n + 1))
    strategies = []
    strategies.append(('asc', list(base)))
    strategies.append(('desc', list(base[::-1])))
    # banding: low for A, high for B (interleave so bal keeps them separate)
    low = list(range(1, 129))
    high = list(range(256, 128, -1))
    strategies.append(('band', low + high))
    strategies.append(('band2', high + low))
    # odd/even
    odd = list(range(1, n + 1, 2))
    even = list(range(2, n + 1, 2))
    strategies.append(('odd_even', odd + even))
    strategies.append(('even_odd', even + odd))

    rng = random.Random(123)
    for _ in range(6):
        s = list(base)
        rng.shuffle(s)
        strategies.append(('rand', s))

    best_total = 0
    best = None

    for name, order in strategies:
        if not time_ok(8.0):
            break
        for prefer in ['bal', 'A', 'B']:
            if not time_ok(6.0):
                break
            A, B = build(order, prefer)
            if not (A and B):
                continue
            A, B = local_search(A, B)
            if not validate(A, B):
                continue
            total = len(A) + len(B)
            if total > best_total:
                best_total = total
                best = (list(A), list(B))

    # Final: ensure at least one element each
    if best is None:
        A = [1]
        B = [2]
        if validate(A, B):
            best = (A, B)
        else:
            best = ([1], [3])

    A, B = best
    # cap sizes
    if len(A) > 64:
        A = A[:64]
    if len(B) > 64:
        B = B[:64]
    # final safety validation; if broken, fall back to trivial
    if not validate(A, B):
        A, B = [1], [2]
    return [A, B]
